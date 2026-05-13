"""Split a single SDF into N chunks for parallel Uni-Mol pair extraction.

Used for large datasets (e.g. HIV ~41k mols) where one extraction job
exceeds the QOS wall-time limit. Each chunk is processed independently
via Slurm array job (see hpc/run_pair_extract.sbatch), then merged with
tools/merge_pair_rep.py.

Usage:
    python tools/split_sdf.py --sdf unimol_out_hiv/sdf/smiles_unimol.sdf --chunk-size 5000
    python tools/split_sdf.py --task hiv --chunk-size 5000

Outputs:
    <sdf_dir>/chunks/<stem>_chunk_00.sdf
    <sdf_dir>/chunks/<stem>_chunk_01.sdf
    ...
    <sdf_dir>/chunks/manifest.json
"""

import argparse
import json
from pathlib import Path

from rdkit import Chem


def split_sdf(sdf_path: Path, chunk_size: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = sdf_path.stem

    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    chunks = []
    current_writer = None
    current_chunk_idx = 0
    current_chunk_count = 0
    total = 0

    def open_chunk(idx):
        path = out_dir / f"{stem}_chunk_{idx:02d}.sdf"
        return path, Chem.SDWriter(str(path))

    chunk_path, current_writer = open_chunk(current_chunk_idx)
    chunks.append({"chunk_index": current_chunk_idx, "path": str(chunk_path), "count": 0,
                   "start_index": 0})

    for mol in supplier:
        if mol is None:
            mol = Chem.Mol()
        current_writer.write(mol)
        current_chunk_count += 1
        total += 1

        if current_chunk_count >= chunk_size:
            current_writer.close()
            chunks[-1]["count"] = current_chunk_count
            current_chunk_idx += 1
            current_chunk_count = 0
            chunk_path, current_writer = open_chunk(current_chunk_idx)
            chunks.append({"chunk_index": current_chunk_idx, "path": str(chunk_path),
                           "count": 0, "start_index": total})

    current_writer.close()
    chunks[-1]["count"] = current_chunk_count
    if chunks[-1]["count"] == 0:
        Path(chunks[-1]["path"]).unlink(missing_ok=True)
        chunks.pop()

    manifest = {
        "source_sdf": str(sdf_path),
        "stem": stem,
        "chunk_size": chunk_size,
        "total_mols": total,
        "n_chunks": len(chunks),
        "chunks": chunks,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(chunks)} chunks, {total} total mols")
    print(f"Manifest: {manifest_path}")
    print(f"Use SLURM_ARRAY_TASK_ID range: 0-{len(chunks)-1}")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf", type=str, default=None,
                        help="Path to source SDF (overrides --task)")
    parser.add_argument("--task", type=str, default=None,
                        help="Task name, auto-resolves unimol_out_<task>/sdf/smiles_unimol.sdf")
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output dir (default: <sdf_dir>/chunks)")
    args = parser.parse_args()

    if args.sdf:
        sdf_path = Path(args.sdf).resolve()
    elif args.task:
        sdf_path = Path(f"unimol_out_{args.task}/sdf/smiles_unimol.sdf").resolve()
    else:
        parser.error("must provide --sdf or --task")
        return 1

    if not sdf_path.exists():
        parser.error(f"SDF not found: {sdf_path}")
        return 1

    out_dir = Path(args.out_dir).resolve() if args.out_dir else sdf_path.parent / "chunks"
    split_sdf(sdf_path, args.chunk_size, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
