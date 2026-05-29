# Result provenance checklist

This file records where the main paper numbers should come from. It is a
submission checklist, not part of the paper build.

> **PARTIALLY SUPERSEDED (2026-05-29).** Authoritative numbers are the generated
> `paper/tables/{classification,regression}.tex` (from the committed per-seed
> JSONs); the legacy 3-dataset block and several source notes below predate the
> 6-dataset matched-baseline + contribution-slim update. Key corrections:
> - **FreeSolv V0 = 0.665 ± 0.031** (matched, `freesolv_v0_results.json`), *not* the old 0.760 (`freesolv_baseline_results.json`).
> - **FreeSolv V2-T5 = 0.638 uses the GBF (RDKit-3D + Gaussian-basis) pair surrogate, NOT Uni-Mol pair tensors** — no Uni-Mol pair run exists for FreeSolv; the file is named `freesolv_unimol_v2t5_*` but is GBF internally (the method section states this).
> - **Matched RF** (`baselines_matched_results.json`, 30 RF-seeds on the model's exact split) overrides the deng30 RF for FreeSolv/ESOL/Lipo/ClinTox (FreeSolv 0.720, ESOL 0.361, Lipo 0.643); bbbp/bace stay deng30. RF dominates BACE/ESOL/Lipo.
> - A frozen-**Uni-Mol-embedding** baseline (Uni-Mol row) was added for FreeSolv/ESOL/Lipo/ClinTox (bbbp/bace = Phase-B B2).
> - **Contributions slimmed 5→3** (matched-protocol audit / fingerprint-saturation / reproducibility); mechanism, T7, B4 demoted to caveated secondary analyses (see `main.tex`).

## Table generator

- Script: `paper/make_tables.py`
- Outputs:
  - `paper/tables/main_results.tex`
  - `paper/tables/classification.tex`
  - `paper/tables/regression.tex`
- Current local Python command:
  - `..\python38-embed\python.exe paper\make_tables.py`

## Source files by row

| Variant | BBBP source | BACE source | FreeSolv source | Notes |
|---|---|---|---|---|
| V0 | `bbbp_all_results.json` | `bace_all_results.json` overrides `baseline_unifold_results.json` | `freesolv_baseline_results.json` | BBBP V0 uses n=9; BACE/FreeSolv use n=3. |
| FP-only | none | `baseline_unifold_results.json` or `bace_all_results.json` if present | none | BACE-only control. |
| RF | `baselines_ml_results_deng30_unimol_fold.json` | `baselines_ml_results_deng30_unimol_fold.json` | `baselines_ml_results_deng30_unimol_fold.json` | Caption should keep the Deng-style 30-seed protocol explicit. |
| Chemprop | `baselines_chemprop_results_unimol_fold.json` | `baselines_chemprop_results_unimol_fold.json` | `baselines_chemprop_results_unimol_fold.json` | Supervised D-MPNN reference. |
| V2-T5 | `bbbp_all_results.json` | `bace_all_results.json` | `freesolv_unimol_v2t5_results.json` | Final FreeSolv V2-T5 cell uses Uni-Mol pair tensors; the older GBF file remains an implementation control only. |
| T6 | `bbbp_all_results.json` | `bace_all_results.json` | none | No FreeSolv T6 run is reported. |
| T7 | `bbbp_t7_bare_results.json` | `bace_t7_bare_results.json` | `freesolv_t7_bare_results.json` | Descriptive n=3 controlled exploration only; FreeSolv T7 uses the earlier GBF pair-feature pipeline. |

## Current generated main table values

| Variant | BBBP | BACE | FreeSolv |
|---|---:|---:|---:|
| V0 | 0.828 +/- 0.027 | 0.757 +/- 0.047 | 0.760 +/- 0.058 |
| FP-only | -- | 0.846 +/- 0.012 | -- |
| RF | 0.799 +/- 0.007 | 0.894 +/- 0.003 | 0.675 +/- 0.010 |
| Chemprop | 0.815 +/- 0.014 | 0.840 +/- 0.045 | 0.879 +/- 0.085 |
| V2-T5 | 0.831 +/- 0.032 | 0.763 +/- 0.010 | 0.638 +/- 0.017 |
| T6 | 0.831 +/- 0.022 | 0.768 +/- 0.088 | -- |
| T7 | 0.821 +/- 0.011 | 0.780 +/- 0.027 | 0.687 +/- 0.070 |

## Known checks before submission

- `paper/tables/main_results.tex` has a T7 row with all three values.
- `paper/tables/classification.tex` caption distinguishes BBBP n=9 from T7 n=3.
- `paper/tables/regression.tex` caption says FreeSolv V2-T5 uses Uni-Mol pair representations and FreeSolv T7 uses the earlier GBF surrogate.
- `paper/main.tex` states BBBP/BACE use Uni-Mol scaffold folds and FreeSolv uses a matched scaffold-style protocol with Uni-Mol pair tensors for the final V2-T5 cell.
- `paper/bbbp_n9_summary.md` must be treated as an analysis artifact; if it disagrees with generated tables, resolve before submission.
- The paper should not compare FreeSolv T7 directly against Uni-Mol FreeSolv V2-T5 because they use different pair-feature sources.

## Non-table claim provenance

| Claim family | Source artifact | Notes |
|---|---|---|
| Earlier FreeSolv GBF V2-T5 control `0.675` | `freesolv_v2t5_results.json` | Not the current main V2-T5 table source. Use only as an implementation/control comparison. |
| BBBP n=9 sampling statement | `bbbp_all_results.json`; `paper/bbbp_n9_summary.md` | Mechanism audit remains original BBBP seeds only. |
| BBBP random-pair ablation | `bbbp_all_results.json` (`v2_t5_random_pair_ablation`) | Matched seeds 9/19/29 only; do not generalize to n=9. |
| BACE post-B4 graph rows | `bace_all_results.json` | Table std recomputes sample std over per-seed means. |
| Gate distribution and bond-type routing | `paper/gate_audit_summary.md`; `paper/gate_attribution_summary.md`; `mlp_phi_stats/*audit.json` | Scope claims to audited checkpoints. |
| T7 gate range `[0.122,0.124]` | `freesolv_t7_bare_results.json` plus corresponding BBBP/BACE T7 audit logs/checkpoints | Keep descriptive; do not use as proof of attention mechanism benefit. |
