"""Merge chunked Uni-Mol pair_rep outputs back into a single per-SDF namespace.

After tools/split_sdf.py + chunked extraction via hpc/run_pair_extract.sbatch,
each chunk's pair_rep lives at:
    unimol_out_<task>/pair_rep_chunks/chunk_NN/<stem>_chunk_NN/{local_idx:06d}/pair_rep.pt

This script renumbers them to the global namespace expected by
make_task_from_unimol.py:
    unimol_out_<task>/pair_rep/<stem>/{global_idx:06d}/pair_rep.pt

Global index = chunk.start_index + local_idx (start_index from manifest.json).

Uses symlinks by default (cheap on Linux); falls back to file copy if symlinks
fail (e.g. Windows without admin privileges).

Usage:
    python tools/merge_pair_rep.py --task hiv
    python tools/merge_pair_rep.py --manifest unimol_out_hiv/sdf/chunks/manifest.json \
        --chunks-root unimol_out_hiv/pair_rep_chunks \
        --target unimol_out_hiv/pair_rep
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def make_link_or_copy(src: Path, dst: Path, copy_mode: bool):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_mode:
        shutil.copy2(src, dst)
    else:
        try:
            os.symlink(src, dst)
        except (OSError, NotImplementedError):
            shutil.copy2(src, dst)


def merge(manifest_path: Path, chunks_root: Path, target_root: Path, copy_mode: bool):
    manifest = json.loads(manifest_path.read_text())
    stem = manifest["stem"]
    target_stem_dir = target_root / stem
    target_stem_dir.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_missing = 0
    for chunk in manifest["chunks"]:
        chunk_idx = chunk["chunk_index"]
        start = chunk["start_index"]
        count = chunk["count"]

        # extractor writes to: chunks_root/chunk_NN/<chunk_stem>/{local:06d}/pair_rep.pt
        chunk_stem = f"{stem}_chunk_{chunk_idx:02d}"
        candidates = [
            chunks_root / f"chunk_{chunk_idx:02d}" / chunk_stem,
            chunks_root / f"chunk_{chunk_idx:02d}",
            chunks_root / chunk_stem,
        ]
        chunk_dir = None
        for c in candidates:
            if c.is_dir():
                chunk_dir = c
                break
        if chunk_dir is None:
            print(f"  WARNING: chunk dir not found for chunk {chunk_idx:02d}. Tried: {candidates}")
            n_missing += count
            continue

        for local_idx in range(count):
            src = chunk_dir / f"{local_idx:06d}" / "pair_rep.pt"
            global_idx = start + local_idx
            dst_dir = target_stem_dir / f"{global_idx:06d}"
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / "pair_rep.pt"
            if not src.exists():
                print(f"  WARNING: missing {src}")
                n_missing += 1
                continue
            make_link_or_copy(src.resolve(), dst, copy_mode)
            n_total += 1

    print(f"Merged {n_total} pair_rep entries into {target_stem_dir}")
    if n_missing > 0:
        print(f"WARNING: {n_missing} entries missing or unresolved")
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default=None,
                        help="Task name; auto-resolves paths")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Override manifest path")
    parser.add_argument("--chunks-root", type=str, default=None,
                        help="Override chunks root (default: unimol_out_<task>/pair_rep_chunks)")
    parser.add_argument("--target", type=str, default=None,
                        help="Override target dir (default: unimol_out_<task>/pair_rep)")
    parser.add_argument("--copy", action="store_true",
                        help="Copy files instead of symlink (default: symlink)")
    args = parser.parse_args()

    if args.task:
        manifest_path = Path(args.manifest) if args.manifest else \
            Path(f"unimol_out_{args.task}/sdf/chunks/manifest.json")
        chunks_root = Path(args.chunks_root) if args.chunks_root else \
            Path(f"unimol_out_{args.task}/pair_rep_chunks")
        target_root = Path(args.target) if args.target else \
            Path(f"unimol_out_{args.task}/pair_rep")
    else:
        if not (args.manifest and args.chunks_root and args.target):
            parser.error("Either --task or all of --manifest/--chunks-root/--target required")
            return 1
        manifest_path = Path(args.manifest)
        chunks_root = Path(args.chunks_root)
        target_root = Path(args.target)

    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 1
    if not chunks_root.exists():
        print(f"ERROR: chunks root not found: {chunks_root}", file=sys.stderr)
        return 1

    return merge(manifest_path, chunks_root, target_root, copy_mode=args.copy)


if __name__ == "__main__":
    raise SystemExit(main())
