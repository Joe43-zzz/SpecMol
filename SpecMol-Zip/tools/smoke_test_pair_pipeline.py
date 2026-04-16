import argparse
import math
import os
import random
import sys
from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from make_bace_from_unimol import (  # noqa: E402
    DEFAULT_CSV,
    attach_unimol_pair_to_data,
    build_graph_data,
    load_pair_payload,
    resolve_sdf_path,
    validate_all_pairs_edge_index,
)
from model_gnn_pre import LH_Direct  # noqa: E402
from tools.extract_unimol_pair import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_PAIR_OUTPUT_DIR,
    ExplicitUniMolModel,
    canonicalize_smiles,
    env_path,
    extract_for_sdf,
)


OPTIONAL_WARNING_MARKER = "[optional-warning][ignorable]"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=str, default=None)
    parser.add_argument("--sdf-dir", type=str, default=None)
    parser.add_argument("--sdf-path", type=str, default=None)
    parser.add_argument("--pair-output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--ckpt-path", type=str, default=None)
    parser.add_argument("--dict-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--pair-update-layers", type=int, default=2)
    return parser.parse_args()


def corrcoef(x, y):
    x = x.to(torch.float64)
    y = y.to(torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt(torch.sum(x * x) * torch.sum(y * y))
    if float(denom.item()) == 0.0:
        return float("nan")
    return float((torch.sum(x * y) / denom).item())


def sdf_heavy_symbols_and_coords(sdf_path, mol_index):
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mol = suppl[mol_index]
    if mol is None:
        raise ValueError(f"failed to read mol {mol_index} from {sdf_path}")
    mol_no_h = Chem.RemoveHs(mol)
    conf = mol_no_h.GetConformer()
    heavy_symbols = [atom.GetSymbol() for atom in mol_no_h.GetAtoms()]
    coords = torch.tensor(
        [
            [
                conf.GetAtomPosition(atom_idx).x,
                conf.GetAtomPosition(atom_idx).y,
                conf.GetAtomPosition(atom_idx).z,
            ]
            for atom_idx in range(mol_no_h.GetNumAtoms())
        ],
        dtype=torch.float32,
    )
    canonical_smiles = canonicalize_smiles(Chem.MolToSmiles(mol_no_h), isomeric=False)
    return canonical_smiles, heavy_symbols, coords


def pair_scalar_from_payload(pair_payload):
    pair_rep = pair_payload["encoder_pair_rep"]
    keep_atom_indices = torch.as_tensor(pair_payload["keep_atom_indices"], dtype=torch.long)
    pair_rep = pair_rep.index_select(0, keep_atom_indices).index_select(1, keep_atom_indices)
    return torch.nn.functional.softplus(pair_rep.mean(dim=-1))


def directed_pair_vectors(coords, pair_scalar):
    num_nodes = coords.size(0)
    dist_values = []
    inv_dist_values = []
    weight_values = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            dist = torch.norm(coords[i] - coords[j], p=2)
            dist_values.append(dist)
            inv_dist_values.append(1.0 / (1.0 + dist))
            weight_values.append(pair_scalar[i, j])
    return (
        torch.stack(dist_values).to(torch.float32),
        torch.stack(inv_dist_values).to(torch.float32),
        torch.stack(weight_values).to(torch.float32),
    )


def validate_alignment_with_graph(mol_index, smile, atoms_from_graph, canonical_sdf_smiles, heavy_symbols):
    input_canonical_smiles = canonicalize_smiles(smile, isomeric=False)
    normalized_graph_atoms = [atom.capitalize() for atom in atoms_from_graph]
    print(
        f"[align][mol={mol_index}] canonical_smiles_sdf={canonical_sdf_smiles} "
        f"canonical_smiles_input={input_canonical_smiles}"
    )
    print(
        f"[align][mol={mol_index}] graph_atom_count={len(atoms_from_graph)} "
        f"sdf_atom_count={len(heavy_symbols)}"
    )
    print(
        f"[align][mol={mol_index}] graph_atom_head={normalized_graph_atoms[:5]} "
        f"sdf_atom_head={heavy_symbols[:5]}"
    )
    print(
        f"[align][mol={mol_index}] graph_atom_tail={normalized_graph_atoms[-5:]} "
        f"sdf_atom_tail={heavy_symbols[-5:]}"
    )
    if input_canonical_smiles != canonical_sdf_smiles:
        raise AssertionError("canonical smiles mismatch between input graph and SDF")
    if normalized_graph_atoms != heavy_symbols:
        raise AssertionError("heavy atom symbol sequence mismatch between graph and SDF")


def build_v0_graph(smile, label_value):
    graph_data, atoms = build_graph_data(
        smile=smile,
        label=[float(label_value)],
        weight=[1.0],
        fingerprint=torch.zeros(1489).tolist(),
    )
    return graph_data, atoms


def build_v1_graph(mol_index, smile, label_value, pair_output_dir, sdf_path):
    graph_data, atoms = build_v0_graph(smile, label_value)
    pair_payload = load_pair_payload(pair_output_dir, sdf_path, mol_index)
    graph_data.mol_id = int(mol_index)
    graph_data = attach_unimol_pair_to_data(graph_data, atoms, pair_payload)
    validate_all_pairs_edge_index(graph_data.edge_index, graph_data.x.size(0))
    return graph_data, atoms, pair_payload


def check_v0_graph(data):
    assert data.edge_index.dim() == 2 and data.edge_index.size(0) == 2
    assert torch.isfinite(data.x).all()
    assert torch.isfinite(data.edge_attr).all()
    print(
        f"[v0-check] mol_id={getattr(data, 'mol_id', 'na')} N={data.x.size(0)} "
        f"bond_E={data.edge_index.size(1)}"
    )


def build_checked_pair_result(mol_index, smile, label_value, pair_output_dir, full_sdf_path, pair_sdf_path, pair_mol_index):
    graph_data, atoms, pair_payload = build_v1_graph(
        mol_index=pair_mol_index,
        smile=smile,
        label_value=label_value,
        pair_output_dir=pair_output_dir,
        sdf_path=pair_sdf_path,
    )
    graph_data.mol_id = int(mol_index)
    canonical_sdf_smiles, heavy_symbols, coords = sdf_heavy_symbols_and_coords(full_sdf_path, mol_index)
    validate_alignment_with_graph(mol_index, smile, atoms, canonical_sdf_smiles, heavy_symbols)

    pair_scalar = pair_scalar_from_payload(pair_payload)
    dist_vec, inv_dist_vec, weight_vec = directed_pair_vectors(coords, pair_scalar)
    corr_dist = corrcoef(dist_vec, weight_vec)
    corr_inv_dist = corrcoef(inv_dist_vec, weight_vec)
    if abs(corr_inv_dist) >= 0.999:
        raise AssertionError(
            f"corr(1/(1+d), w_ij) is too close to 1, got {corr_inv_dist:.6f}"
        )

    return {
        "graph_data": graph_data,
        "pair_payload": pair_payload,
        "corr_dist": corr_dist,
        "corr_inv_dist": corr_inv_dist,
        "N": graph_data.x.size(0),
        "E": graph_data.edge_index.size(1),
        "pair_shape": tuple(pair_payload["encoder_pair_rep"].shape),
    }


def write_subset_sdf(source_sdf_path, selected_indices, subset_sdf_path):
    supplier = Chem.SDMolSupplier(str(source_sdf_path), removeHs=False)
    writer = Chem.SDWriter(str(subset_sdf_path))
    subset_records = []
    try:
        for subset_index, original_index in enumerate(selected_indices):
            mol = supplier[original_index]
            if mol is None:
                raise ValueError(f"failed to read mol {original_index} from {source_sdf_path}")
            writer.write(mol)
            subset_records.append({"original_index": original_index, "subset_index": subset_index})
    finally:
        writer.close()
    return subset_records


def summarize_pair_rows(rows):
    header = "| mol_index | N | E | pair_shape | corr(dist,w) | corr(1/(1+d),w) | passed |"
    sep = "| --- | --- | --- | --- | --- | --- | --- |"
    lines = [header, sep]
    for row in rows:
        lines.append(
            f"| {row['mol_index']} | {row['N']} | {row['E']} | {row['pair_shape']} | "
            f"{row['corr_dist']:.6f} | {row['corr_inv_dist']:.6f} | YES |"
        )
    return "\n".join(lines)


def summarize_variant_rows(rows):
    header = "| variant | batch_size | total_nodes | total_edges | low_x | high_x | spec_x | x_fp | delta_pair_mean | delta_pair_std | passed |"
    sep = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    lines = [header, sep]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['batch_size']} | {row['total_nodes']} | {row['total_edges']} | "
            f"{row['low_x_shape']} | {row['high_x_shape']} | {row['spec_x_shape']} | {row['x_fp_shape']} | "
            f"{row['delta_pair_mean']} | {row['delta_pair_std']} | YES |"
        )
    return "\n".join(lines)


def run_variant_forward(name, data_list, device, use_pair_update, pair_update_layers):
    loader = DataLoader(data_list, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    total_nodes = int(batch.x.size(0))
    total_edges = int(batch.edge_index.size(1))

    if hasattr(batch, "edge_weight_3d"):
        if batch.edge_weight_3d.shape != (total_edges,):
            raise AssertionError(
                f"{name}: batch.edge_weight_3d.shape mismatch: {tuple(batch.edge_weight_3d.shape)} vs {(total_edges,)}"
            )
        if not torch.isfinite(batch.edge_weight_3d).all():
            raise AssertionError(f"{name}: batch.edge_weight_3d contains NaN/inf")

    model = LH_Direct(
        in_dim=93,
        hid_dim=512,
        K=10,
        dprate=0.5,
        dropout=0.0,
        is_bns=False,
        act_fn="relu",
        type="tri",
        use_pair_update=use_pair_update,
        pair_update_layers=pair_update_layers,
    ).to(device)
    model.eval()
    with torch.no_grad():
        low_x, high_x, spec_x, x_fp = model(batch, str(device))

    for tensor_name, tensor in {
        "low_x": low_x,
        "high_x": high_x,
        "spec_x": spec_x,
        "x_fp": x_fp,
    }.items():
        if not torch.isfinite(tensor).all():
            raise AssertionError(f"{name}: {tensor_name} contains NaN/inf")

    expected_shape = (2, 512)
    if tuple(low_x.shape) != expected_shape:
        raise AssertionError(f"{name}: low_x shape mismatch {tuple(low_x.shape)}")
    if tuple(high_x.shape) != expected_shape:
        raise AssertionError(f"{name}: high_x shape mismatch {tuple(high_x.shape)}")
    if tuple(spec_x.shape) != expected_shape:
        raise AssertionError(f"{name}: spec_x shape mismatch {tuple(spec_x.shape)}")
    if tuple(x_fp.shape) != expected_shape:
        raise AssertionError(f"{name}: x_fp shape mismatch {tuple(x_fp.shape)}")

    delta_pair_mean = "NA"
    delta_pair_std = "NA"
    if use_pair_update:
        if model.latest_pair_update_stats is None:
            raise AssertionError(f"{name}: pair update enabled but no stats were recorded")
        delta_pair_mean_value = model.latest_pair_update_stats["delta_pair_mean"]
        delta_pair_std_value = math.sqrt(model.latest_pair_update_stats["delta_pair_var"])
        delta_pair_mean = f"{delta_pair_mean_value:.6f}"
        delta_pair_std = f"{delta_pair_std_value:.6f}"
        print(f"[{name}] pair_bias_shape={model.latest_pair_update_stats['pair_bias_shape']}")
        print(f"[{name}] pair_bias_updated_shape={model.latest_pair_update_stats['pair_bias_updated_shape']}")
        print(f"[{name}] delta_pair_shape={model.latest_pair_update_stats['delta_pair_shape']}")
        print(f"[{name}] delta_pair_mean={delta_pair_mean}")
        print(f"[{name}] delta_pair_std={delta_pair_std}")

    print(f"[{name}] total_nodes={total_nodes}")
    print(f"[{name}] total_edges={total_edges}")
    print(f"[{name}] low_x.shape={tuple(low_x.shape)}")
    print(f"[{name}] high_x.shape={tuple(high_x.shape)}")
    print(f"[{name}] spec_x.shape={tuple(spec_x.shape)}")
    print(f"[{name}] x_fp.shape={tuple(x_fp.shape)}")
    print(f"[{name}] forward_passed=True")

    return {
        "variant": name,
        "batch_size": 2,
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "low_x_shape": tuple(low_x.shape),
        "high_x_shape": tuple(high_x.shape),
        "spec_x_shape": tuple(spec_x.shape),
        "x_fp_shape": tuple(x_fp.shape),
        "delta_pair_mean": delta_pair_mean,
        "delta_pair_std": delta_pair_std,
    }


def main():
    args = parse_args()
    random.seed(args.seed)

    csv_path = (
        Path(args.csv_path).expanduser().resolve()
        if args.csv_path
        else Path(os.getenv("CSV_PATH", DEFAULT_CSV)).resolve()
    )
    sdf_path = resolve_sdf_path(sdf_dir_arg=args.sdf_dir, sdf_path_arg=args.sdf_path)
    pair_output_dir = (
        Path(args.pair_output_dir).expanduser().resolve()
        if args.pair_output_dir
        else env_path("OUTPUT_DIR", DEFAULT_PAIR_OUTPUT_DIR)
    )
    ckpt_path = (
        Path(args.ckpt_path).expanduser().resolve()
        if args.ckpt_path
        else env_path("UNIMOL_CKPT", required=True)
    )
    dict_path = (
        Path(args.dict_path).expanduser().resolve()
        if args.dict_path
        else env_path("UNIMOL_DICT", required=True)
    )

    df = pd.read_csv(csv_path)
    selected_indices = random.sample(range(len(df)), 2)
    selected_indices.sort()
    print(f"[smoke] selected_indices={selected_indices}")
    print(
        f"{OPTIONAL_WARNING_MARKER} deepchem may print optional dependency warnings "
        f"(tensorflow/dgl/transformers/lightning/jax); they do not affect this Uni-Mol -> PyG -> SpecMol chain."
    )

    device = torch.device(args.device)
    extractor = ExplicitUniMolModel(ckpt_path=ckpt_path, dict_path=dict_path).to(device)
    print(f"[smoke] extractor_device={device}")
    print(f"[smoke] extractor_training={extractor.training}")
    smoke_tmp_dir = pair_output_dir / "smoke_subset"
    smoke_tmp_dir.mkdir(parents=True, exist_ok=True)
    subset_sdf_path = smoke_tmp_dir / "selected_two_mols.sdf"
    subset_records = write_subset_sdf(sdf_path, selected_indices, subset_sdf_path)
    print(f"[smoke] subset_sdf_path={subset_sdf_path}")

    extract_for_sdf(
        model=extractor,
        sdf_path=subset_sdf_path,
        output_dir=smoke_tmp_dir,
        batch_size=4,
        device=device,
        target_indices=[record["subset_index"] for record in subset_records],
        input_smiles_by_index=[str(df.iloc[record["original_index"]]["smiles"]) for record in subset_records],
    )

    v0_data_list = []
    pair_rows = []
    v1_data_list = []
    for record in subset_records:
        mol_index = record["original_index"]
        smile = str(df.iloc[mol_index]["smiles"])
        label_value = float(df.iloc[mol_index]["label"])

        v0_graph, _ = build_v0_graph(smile, label_value)
        v0_graph.mol_id = int(mol_index)
        check_v0_graph(v0_graph)
        v0_data_list.append(v0_graph)

        checked = build_checked_pair_result(
            mol_index=mol_index,
            smile=smile,
            label_value=label_value,
            pair_output_dir=smoke_tmp_dir,
            full_sdf_path=sdf_path,
            pair_sdf_path=subset_sdf_path,
            pair_mol_index=record["subset_index"],
        )
        print(
            f"[pair-source][mol={mol_index}] strict_unimol_pair_source=True "
            f"corr(dist_ij,w_ij)={checked['corr_dist']:.6f} "
            f"corr(1/(1+d),w_ij)={checked['corr_inv_dist']:.6f}"
        )
        pair_rows.append({"mol_index": mol_index, **checked})
        v1_data_list.append(checked["graph_data"])

    variant_rows = []
    variant_rows.append(
        run_variant_forward(
            name="V0_2D",
            data_list=v0_data_list,
            device=device,
            use_pair_update=False,
            pair_update_layers=args.pair_update_layers,
        )
    )
    variant_rows.append(
        run_variant_forward(
            name="V1_UniMolStatic",
            data_list=v1_data_list,
            device=device,
            use_pair_update=False,
            pair_update_layers=args.pair_update_layers,
        )
    )
    variant_rows.append(
        run_variant_forward(
            name="V2_UniMolUpdate",
            data_list=v1_data_list,
            device=device,
            use_pair_update=True,
            pair_update_layers=args.pair_update_layers,
        )
    )

    print("[pair-summary]")
    print(summarize_pair_rows(pair_rows))
    print("[variant-summary]")
    print(summarize_variant_rows(variant_rows))
    print("[smoke] all required checks passed for V0/V1/V2")


if __name__ == "__main__":
    main()
