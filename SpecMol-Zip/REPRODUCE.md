# Reproducing the results

This is the top-level reproduce guide for the ICDE-2027 EAB submission. Every
number in the paper traces to a `*_results.json` / `paper/audits/*.json` on disk;
this file says which command produced each one, where the per-seed JSONs live,
and how to regenerate the LaTeX tables. Run everything from the inner
`SpecMol-Zip/` directory (the one this file is in).

---

## 1. Environment

Two interpreters are used; both are pinned.

**Local table regeneration / RF baselines (CPU, no GPU needed)** — system Python
3.12:
```
C:\Users\zhoutianyang\AppData\Local\Programs\Python\Python312\python.exe
```
plus a venv that carries the science stack (pandas / scikit-learn / rdkit):
```
D:\SpecMol-Zip\.venv\Scripts\python.exe   # Python 3.8.10
```
Key local packages: torch 1.13.1+cu117, PyG 2.3.1 (or 2.7.0 under Python 3.12),
rdkit 2023.3.2, scikit-learn, deepchem 2.8.0, pandas.

**Deep training (GPU)** — MBZUAI HPC, conda env `specmol`: Python 3.10, torch
2.4.1, PyG 2.6.1, rdkit, deepchem 2.8.0, `unimol_tools` (ships the Uni-Mol
`mol_pre_all_h_220816.pt` checkpoint + `mol.dict.txt`). GPU: RTX 5000 Ada 32GB,
CUDA 12.4, Slurm. Do NOT set `CUDA_VISIBLE_DEVICES` by hand — Slurm owns it; pass
`--gpu 0`.

Hardware/runtime reference for the cells below:
- Classification/regression deep arm (1 seed, finetune 40 ep / patience 15,
  batch 1024, K 10, hid_dim 512): ~10-40 min on one RTX 5000 Ada.
- QM9-mu 7-arm pilot (3 seeds each): ~1 GPU-day serial (jobs 138929-138935 in the
  DFT run). FreeSolv is tiny (~50 min for V0+V2-T5 to n=9).
- Matched RF (`qm9mu_rf_matched.py`, strongFull ~210 descriptors, 3 seeds): a few
  minutes to ~1 h on CPU depending on subset size (~12.5k-14.8k molecules).

---

## 2. One-command table regeneration

All paper tables are generated from the on-disk `*_results.json` by one script:
```
python paper/make_tables.py
```
Writes `paper/tables/{classification,regression,main_results}.tex`. `main.tex`
`\input`s these; it never hard-codes a cell. The script reads (and silently skips
any missing) the result JSONs listed in its module docstring — the canonical ones
are `bbbp_all_results.json`, `bace_all_results.json`,
`baselines_ml_results_deng30_unimol_fold.json` (30-seed RF),
`baselines_chemprop_results_unimol_fold.json`, `baselines_matched_results.json`
(matched-split RF + frozen-Uni-Mol control), and the regression
`<task>_<variant>_results.json` files.

Regression cells are first aggregated from per-seed files:
```
python paper/aggregate_regression_seeds.py        # all tasks/variants
```
which reads `hpc/results/<prefix>_seed{N}.json` and writes the
`<task>_<variant>_results.json` files `make_tables.py` consumes. FreeSolv V0 /
V2-T5 are at n=9 (seeds 9,19,29,39,49,59,69,79,89); other regression cells are
n=3 (seeds 9,19,29) — `aggregate_regression_seeds.py` reports `n` from the files
that actually exist.

---

## 3. Where the per-seed JSONs live

| Result | File(s) |
|---|---|
| BBBP deep (n=9) | `bbbp_all_results.json` (also archived `n9_archive/bbbp_all_results.json`) |
| BACE deep (frozen, n=9) | `bace_all_results.json` |
| BACE finetune pilot (n=3, split=all) | `paper/audits/inductive_sweep_2026-06-01.json` |
| FreeSolv V0 / V2-T5 (n=9) | `freesolv_v0_results.json` / `freesolv_unimol_v2t5_results.json`; per-seed at `hpc/results/<prefix>_seed{N}.json` |
| RF 30-seed (Deng2023) | `baselines_ml_results_deng30_unimol_fold.json` |
| Chemprop | `baselines_chemprop_results_unimol_fold.json` |
| Matched-split RF + Uni-Mol-direct | `baselines_matched_results.json` |
| QM7 + QM9-mu MMFF pilots | `paper/audits/qm_pilot_2026-06-02.json` |
| QM9-mu TRUE-DFT 7-arm cell + gate-probe | `paper/audits/stage2_qm9_dft_2026-06-03.json` |
| QM9-mu matched RF (per manifest) | `unimol_out_qm9mu_rf_matched.json` (MMFF) / `unimol_out_qm9mu_dft_rf_matched.json` (DFT) |

Honesty note: every n=3 cell carries a power-bound caveat in the paper (below the
team's own n=9 retraction bar); do not read an n=3 delta as a strong effect.

---

## 4. Regenerating the QM9-mu DFT cell and its matched RF

The QM9-mu TRUE-DFT cell (`paper/audits/stage2_qm9_dft_2026-06-03.json`) is the
geometry-signal boundary experiment. Two halves, both on the SAME DFT `embed_ok`
subset (12530-mol / 1256-test):

**Deep arms (GPU, HPC).** The 7 arms (V0/V2T5/random/T7/T7random/T8/T8random) are
launched by `hpc/run_qm9mu_dft_pilot.sbatch`. Build once, then one arm per job:
```
PREP_ONLY=1 sbatch hpc/run_qm9mu_dft_pilot.sbatch     # prepare DFT coords -> Uni-Mol extract -> make .pt
ARM=V0       sbatch hpc/run_qm9mu_dft_pilot.sbatch     # repeat for each of the 7 NAMES
```
The DFT coordinates come from the validated `data/qm9/qm9_dft_solB_cache.pkl`
(Solution B: graph kept identical to the MMFF cell, only the coords Uni-Mol sees
change; a per-molecule physical bond-length gate drops misaligned molecules).
Each arm writes `Best Test RMSE` to `hpc/logs/qm9mu_dft_<ARM>.log`.

**Matched RF ceiling (CPU).** RF features are geometry-blind, but RF is **re-fit
and re-scored on the `embed_ok` subset of whichever manifest you pass** — and the
DFT subset is smaller than MMFF's, so the two give DIFFERENT numbers. The
manifest dir is therefore a REQUIRED argument (no MMFF default):
```
python qm9mu_rf_matched.py unimol_out_qm9mu_dft       # DFT cell  -> weak 0.939 / strong 0.845
python qm9mu_rf_matched.py unimol_out_qm9mu           # MMFF cell -> weak 0.986 / strong 0.925
```
Each writes `<manifest_dir>_rf_matched.json` with keys
`manifest_dir, n_train, n_test, weak_rmse, strong_rmse` (+ per-seed detail).
**Do NOT reuse the MMFF 0.925 as the DFT ceiling** — they are not equal "by
construction"; only deep-vs-RF ON THE SAME manifest subset is a valid comparison.
`verify_g9_manifest_identity.py` confirms the DFT test set is a strict subset of
the MMFF one and flags any RF run that reported the wrong (MMFF) `n_test`.
Background: `paper/STAGE2_QM9_DFT_DESIGN_2026-06-02.md`.

The QM9-mu MMFF and QM7 pilots use the analogous
`hpc/run_qm9mu_pilot.sbatch` / `run_qm7_pilot.sh`; their matched RF cells are the
same `qm9mu_rf_matched.py` invoked on `unimol_out_qm9mu` / `unimol_out_qm7`.

---

## 5. The random-pair null (C1)

The central protocol contribution is a **random-pair null control**: replace the
real Uni-Mol pair representation with same-shape Gaussian noise and check whether
the "3D" gain survives. It is a single flag on the standard training entrypoint:
```
python main_pretrain.py ... --use_v2 --randomize_pair
```
`--randomize_pair` replaces `data.pair_repr_edge` with same-shape Gaussian noise
before the pair-to-edge-weight gate (requires `--use_v2`; for the raw-geometry
T9 path it randomizes the distances instead). The matched-seed `real` vs
`random` arms are what the C1 null compares (e.g. the QM9-mu DFT cell's
`random` / `T7random` / `T8random` arms above). Headline: on the saturated ADMET
sets and the geometry-pure QM9 dipole the null is NOT rejected (real ties
random); on BBBP the n=3 real-pair edge is +0.032 mean but binomial p=0.125, i.e.
below the noise floor — reported with a power-bound caveat, never as a win.

---

## 6. Classical and Chemprop baselines

```
python baselines_ml.py --n_seeds 30 ...        # 30-seed RF/SVM (Deng2023 protocol) -> baselines_ml_results_deng30_unimol_fold.json
python baselines_matched.py ...                # RF + frozen-Uni-Mol-embedding on the model's EXACT split -> baselines_matched_results.json
python baselines_chemprop.py ...               # D-MPNN -> baselines_chemprop_results_unimol_fold.json
```
`baselines_matched.py` is the one to trust when a dataset's deng30 RF was
computed on a regenerated (unmatched) split; it runs on the model's exact fold so
RF-vs-deep is apples-to-apples.

---

## 7. Caveats that travel with the numbers

- **n=3 cells** (BACE finetune pilot, QM9 pilots, ESOL/Lipo/ClinTox deep) are
  below the team's n=9 retraction bar — power-bound statement, not a strong claim.
- **Absolute MMFF vs DFT RMSE are NOT comparable** (different, easier 12.5k DFT
  subset; the geometry-free V0 dropped identically). Only deep-vs-RF on the same
  subset is valid.
- The **spectral/Chebyshev backbone is not novel** (EAGCN / SPECTRA / Specformer
  precede it); T7/T8 carry no attention-mechanism novelty. The contribution is
  the protocol + boundary, not the architecture.
- The aromatic-routing mechanism was **n=9-refuted** (was a 3-seed artifact); it
  is kept out of headline claims (`paper/audits/gate_attribution_summary_n9.md`).
