import argparse
import os
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from rdkit import Chem


def maybe_add_unimol_repo():
    unimol_repo = os.getenv("UNIMOL_REPO")
    if unimol_repo:
        repo_path = Path(unimol_repo).expanduser().resolve()
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))


maybe_add_unimol_repo()

from unimol_tools.data import Dictionary  # noqa: E402
from unimol_tools.data.datahub import DataHub  # noqa: E402
from unimol_tools.models.unimol import (  # noqa: E402
    BACKBONE,
    GaussianLayer,
    LinearHead,
    NonLinearHead,
    molecule_architecture,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SDF_DIR = REPO_ROOT / "unimol_out" / "sdf"
DEFAULT_SDF_PATH = DEFAULT_SDF_DIR / "smiles_unimol.sdf"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "unimol_out" / "pair_rep"


def env_path(name, default=None, required=False):
    value = os.getenv(name)
    if value:
        return Path(value).expanduser().resolve()
    if default is not None:
        return Path(default).expanduser().resolve()
    if required:
        raise ValueError(f"Environment variable {name} is required")
    return None


def resolve_sdf_inputs(sdf_path_arg=None):
    if sdf_path_arg:
        sdf_input = Path(sdf_path_arg).expanduser().resolve()
    elif os.getenv("SDF_PATH"):
        sdf_input = Path(os.environ["SDF_PATH"]).expanduser().resolve()
    else:
        sdf_input = env_path("SDF_DIR", DEFAULT_SDF_DIR)

    if sdf_input.is_dir():
        sdf_paths = sorted(sdf_input.glob("*.sdf"))
    else:
        sdf_paths = [sdf_input]
    if not sdf_paths:
        raise FileNotFoundError(f"No SDF files found under {sdf_input}")
    return sdf_paths


def canonicalize_smiles(smiles, isomeric=False):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid smiles: {smiles}")
    return Chem.MolToSmiles(mol, isomericSmiles=isomeric)


class MolDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __getitem__(self, idx):
        return self.items[idx], 0

    def __len__(self):
        return len(self.items)


class ExplicitUniMolModel(nn.Module):
    def __init__(self, ckpt_path, dict_path, output_dim=1, remove_hs=False):
        super().__init__()
        self.args = molecule_architecture()
        self.output_dim = output_dim
        self.remove_hs = remove_hs
        self.dictionary = Dictionary.load(str(dict_path))
        self.mask_idx = self.dictionary.add_symbol("[MASK]", is_special=True)
        self.padding_idx = self.dictionary.pad()
        self.embed_tokens = nn.Embedding(
            len(self.dictionary), self.args.encoder_embed_dim, self.padding_idx
        )
        self.encoder = BACKBONE[self.args.backbone](
            encoder_layers=self.args.encoder_layers,
            embed_dim=self.args.encoder_embed_dim,
            ffn_embed_dim=self.args.encoder_ffn_embed_dim,
            attention_heads=self.args.encoder_attention_heads,
            emb_dropout=self.args.emb_dropout,
            dropout=self.args.dropout,
            attention_dropout=self.args.attention_dropout,
            activation_dropout=self.args.activation_dropout,
            max_seq_len=self.args.max_seq_len,
            activation_fn=self.args.activation_fn,
            no_final_head_layer_norm=self.args.delta_pair_repr_norm_loss < 0,
        )
        kernel_dim = 128
        n_edge_type = len(self.dictionary) * len(self.dictionary)
        self.gbf_proj = NonLinearHead(kernel_dim, self.args.encoder_attention_heads, self.args.activation_fn)
        self.gbf = GaussianLayer(kernel_dim, n_edge_type)
        self.classification_head = LinearHead(
            input_dim=self.args.encoder_embed_dim,
            num_classes=self.output_dim,
            pooler_dropout=self.args.pooler_dropout,
        )
        self.load_pretrained_weights(ckpt_path)

    def load_pretrained_weights(self, path):
        state_dict = torch.load(path, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        self.load_state_dict(state_dict, strict=False)

    def batch_collate_fn(self, samples):
        from unimol_tools.utils import pad_1d_tokens, pad_2d, pad_coords

        batch = {}
        for key in samples[0][0].keys():
            if key == "src_coord":
                value = pad_coords([torch.tensor(s[0][key]).float() for s in samples], pad_idx=0.0)
            elif key == "src_edge_type":
                value = pad_2d(
                    [torch.tensor(s[0][key]).long() for s in samples],
                    pad_idx=self.padding_idx,
                )
            elif key == "src_distance":
                value = pad_2d(
                    [torch.tensor(s[0][key]).float() for s in samples],
                    pad_idx=0.0,
                )
            elif key == "src_tokens":
                value = pad_1d_tokens(
                    [torch.tensor(s[0][key]).long() for s in samples],
                    pad_idx=self.padding_idx,
                )
            else:
                raise KeyError(f"Unexpected unimol input key: {key}")
            batch[key] = value
        return batch, None

    def forward(self, src_tokens, src_distance, src_coord, src_edge_type):
        padding_mask = src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        x = self.embed_tokens(src_tokens)

        n_node = src_distance.size(-1)
        gbf_feature = self.gbf(src_distance, src_edge_type)
        gbf_result = self.gbf_proj(gbf_feature)
        graph_attn_bias = gbf_result.permute(0, 3, 1, 2).contiguous().view(-1, n_node, n_node)

        encoder_rep, encoder_pair_rep, _, _, _ = self.encoder(
            x,
            padding_mask=padding_mask,
            attn_mask=graph_attn_bias,
        )
        return {
            "encoder_rep": encoder_rep,
            "encoder_pair_rep": encoder_pair_rep,
            "src_tokens": src_tokens,
            "src_coord": src_coord,
        }


def select_atom_tokens(model, src_tokens, src_coord, encoder_pair_rep):
    bos_idx = model.dictionary.bos()
    eos_idx = model.dictionary.eos()
    pad_idx = model.dictionary.pad()
    keep_mask = (src_tokens != bos_idx) & (src_tokens != eos_idx) & (src_tokens != pad_idx)
    keep_indices = keep_mask.nonzero(as_tuple=False).view(-1)
    atomic_symbols = [model.dictionary.symbols[token.item()] for token in src_tokens[keep_indices]]
    encoder_pair_rep = encoder_pair_rep.index_select(0, keep_indices).index_select(1, keep_indices)
    atomic_coords = src_coord.index_select(0, keep_indices)
    heavy_atom_indices = [idx for idx, symbol in enumerate(atomic_symbols) if symbol != "H"]
    return encoder_pair_rep, atomic_symbols, atomic_coords, heavy_atom_indices


def summarize_pair_rep(pair_rep):
    return {
        "mean": float(pair_rep.mean().item()),
        "std": float(pair_rep.std(unbiased=False).item()),
        "min": float(pair_rep.min().item()),
        "max": float(pair_rep.max().item()),
    }


def verify_alignment(mol_index, mol, input_smiles, atomic_symbols):
    sdf_smiles = canonicalize_smiles(Chem.MolToSmiles(Chem.RemoveHs(mol)), isomeric=False)
    input_smiles_canonical = canonicalize_smiles(input_smiles, isomeric=False)
    sdf_no_h = Chem.RemoveHs(mol)
    sdf_symbols = [atom.GetSymbol() for atom in sdf_no_h.GetAtoms()]
    heavy_symbols = [symbol for symbol in atomic_symbols if symbol != "H"]

    print(
        f"[extract][mol={mol_index}] canonical_smiles_sdf={sdf_smiles} "
        f"canonical_smiles_input={input_smiles_canonical}"
    )
    print(
        f"[extract][mol={mol_index}] atom_count_sdf_no_h={len(sdf_symbols)} "
        f"atom_count_pair_no_h={len(heavy_symbols)}"
    )
    print(
        f"[extract][mol={mol_index}] atom_symbols_head_sdf={sdf_symbols[:5]} "
        f"atom_symbols_head_pair={heavy_symbols[:5]}"
    )
    print(
        f"[extract][mol={mol_index}] atom_symbols_tail_sdf={sdf_symbols[-5:]} "
        f"atom_symbols_tail_pair={heavy_symbols[-5:]}"
    )

    if sdf_smiles != input_smiles_canonical:
        raise AssertionError(
            f"canonical smiles mismatch for molecule {mol_index}: {sdf_smiles} vs {input_smiles_canonical}"
        )
    if len(sdf_symbols) != len(heavy_symbols):
        raise AssertionError(
            f"atom count mismatch for molecule {mol_index}: {len(sdf_symbols)} vs {len(heavy_symbols)}"
        )
    if sdf_symbols != heavy_symbols:
        raise AssertionError(
            f"atom symbol sequence mismatch for molecule {mol_index}: {sdf_symbols} vs {heavy_symbols}"
        )


def extract_for_sdf(
    model,
    sdf_path,
    output_dir,
    batch_size,
    device,
    target_indices=None,
    input_smiles_by_index=None,
):
    datahub = DataHub(
        data=str(sdf_path),
        task="repr",
        is_train=False,
        model_name="unimolv1",
        remove_hs=False,
    )
    dataset = MolDataset(datahub.data["unimol_input"])
    mols = datahub.data["mols"]
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=model.batch_collate_fn,
    )

    sdf_output_dir = output_dir / sdf_path.stem
    sdf_output_dir.mkdir(parents=True, exist_ok=True)

    total_samples = len(dataset)
    target_indices = set(range(total_samples)) if target_indices is None else set(target_indices)
    global_index = 0
    print(f"[extract] model_training_before_eval={model.training}")
    model.eval()
    print(f"[extract] model_training_after_eval={model.training}")
    print(f"[extract] infer_mode=torch.no_grad")
    sample_pair_info = None

    with torch.no_grad():
        for batch_inputs, _ in dataloader:
            batch_inputs = {key: value.to(device) for key, value in batch_inputs.items()}
            outputs = model(**batch_inputs)
            for row in range(outputs["src_tokens"].size(0)):
                mol_index = global_index
                pair_rep, atomic_symbols, atomic_coords, heavy_atom_indices = select_atom_tokens(
                    model,
                    outputs["src_tokens"][row].cpu(),
                    outputs["src_coord"][row].cpu(),
                    outputs["encoder_pair_rep"][row].cpu(),
                )
                pair_stats = summarize_pair_rep(pair_rep)
                if mol_index in target_indices:
                    print(
                        f"[extract][mol={mol_index}] encoder_pair_rep_stats "
                        f"mean={pair_stats['mean']:.6f} std={pair_stats['std']:.6f} "
                        f"min={pair_stats['min']:.6f} max={pair_stats['max']:.6f}"
                    )
                    if input_smiles_by_index is not None:
                        verify_alignment(
                            mol_index=mol_index,
                            mol=mols[mol_index],
                            input_smiles=input_smiles_by_index[mol_index],
                            atomic_symbols=atomic_symbols,
                        )

                mol_output_dir = sdf_output_dir / f"{mol_index:06d}"
                mol_output_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "encoder_pair_rep": pair_rep.to(torch.float32),
                        "atomic_symbols": atomic_symbols,
                        "atomic_coords": atomic_coords.to(torch.float32),
                        "keep_atom_indices": torch.tensor(heavy_atom_indices, dtype=torch.long),
                        "pair_stats": pair_stats,
                        "molecule_index": mol_index,
                        "sdf_path": str(sdf_path),
                    },
                    mol_output_dir / "pair_rep.pt",
                )
                if sample_pair_info is None:
                    sample_pair_info = {
                        "mol_index": int(mol_index),
                        "pair_shape": tuple(pair_rep.shape),
                    }
                global_index += 1

    print(f"extracted {global_index} pair representations from {sdf_path} to {sdf_output_dir}")
    if sample_pair_info is not None:
        print(
            f"[extract] strict_unimol_pair_source=True sample_mol_index={sample_pair_info['mol_index']} "
            f"sample_pair_shape={sample_pair_info['pair_shape']}"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--ckpt-path", type=str, default=None)
    parser.add_argument("--dict-path", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt_path = Path(args.ckpt_path).expanduser().resolve() if args.ckpt_path else env_path("UNIMOL_CKPT", required=True)
    dict_path = Path(args.dict_path).expanduser().resolve() if args.dict_path else env_path("UNIMOL_DICT", required=True)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else env_path("OUTPUT_DIR", DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    sdf_paths = resolve_sdf_inputs(args.sdf_path)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = ExplicitUniMolModel(ckpt_path=ckpt_path, dict_path=dict_path).to(device)
    print(f"[extract] unimol_repo={os.getenv('UNIMOL_REPO', '<site-packages:unimol_tools>')}")
    print(f"[extract] ckpt_path={ckpt_path}")
    print(f"[extract] dict_path={dict_path}")
    print(f"[extract] model_training={model.training}")
    print(f"[extract] infer_mode=torch.no_grad")
    for sdf_path in sdf_paths:
        extract_for_sdf(
            model=model,
            sdf_path=sdf_path,
            output_dir=output_dir,
            batch_size=args.batch_size,
            device=device,
        )


if __name__ == "__main__":
    main()
