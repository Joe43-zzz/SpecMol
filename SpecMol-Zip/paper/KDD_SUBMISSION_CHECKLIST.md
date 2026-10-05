# KDD 2027 D&B — Submission Checklist & Handoff (2026-06-04)

**Target venue:** KDD 2027 Datasets & Benchmarks Track, **Cycle 1 ≈ early Aug 2026**
(official dates TBA — VERIFY when CFP posts ~Jun–Jul 2026; KDD 2026 C1 was 7/24/2025).
Fallback: KDD 2027 D&B Cycle 2 (~Feb 2027) fresh resubmit; or TMLR (rolling).

**Submit this file:** `paper/main_acm.tex` (acmart sigconf, double-blind).
Working/reference copy (full content, IEEEtran): `paper/main.tex` — untouched.
Rebuild ACM from main.tex anytime: `python paper/build_acm.py`.

---

## Thesis (honest, positive, locked with PI)
Contribution = **the evaluation protocol/methodology**, NOT a new predictive model:
shape-matched **random-pair null** + **frozen-probe→finetune correction** +
**matched-split RF ceiling** + **WHEN-map** + regenerable harness. The spectral
model (V2-T5/T7/T8) is the case study; the real **Uni-Mol finetune** is a
validating baseline. NOT a "our 3D method wins" / "geometry helps" paper (refuted).

## The result (real Uni-Mol finetune, same matched split as RF/frozen-probe)
| | Uni-Mol FT | matched RF | regime |
|---|---|---|---|
| FreeSolv RMSE↓ | **0.435** | 0.720 | non-saturated → FT beats |
| Lipo RMSE↓ | **0.521** | 0.643 | non-saturated → FT beats |
| ClinTox AUC↑ | 0.832±.043 | 0.812 | non-saturated → FT ties/edges |
| BBBP AUC↑ | 0.826±.008 | 0.814 | non-saturated → FT ties/edges |
| BACE AUC↑ | 0.859 | **0.906** | saturated → RF leads |
| ESOL RMSE↓ | 0.395 | **0.361** | saturated → RF leads |

Honesty caveats baked into text/captions: MolTrain has no seed control → FreeSolv/
BACE/ESOL/Lipo are deterministic single-split internal-5-fold-CV point estimates
(std=0 ≠ zero-variance claim); only BBBP/ClinTox have real n=3 variance. Regression
RMSE is standardized (NOT kcal/mol). frozen→FT jump (ClinTox .489→.832, Lipo
.794→.521) = the C2 correction confirmation.

## DONE (verified, compiles 0-err/0-undef)
- [x] Positive reframe: title/abstract/C1–C4/intro/§5.2/conclusion + RW SOTA-ceiling para
- [x] Uni-Mol FT integrated (both tables "Uni-Mol (FT)" row + WHEN-map narrative §5.2)
- [x] 3 bib DOIs verified (qiao2025scage, hussain2022egt, wang2023threedpgt) + 3 SOTA refs (MoLFormer/SpaceFormer/Equiformer)
- [x] 2 adversarial reviews (reframe + FT integration) — all blockers fixed
- [x] ACM 8-page version: `main_acm.tex` main body intro→Conclusion = 8pp; 7 supporting blocks relocated to \appendix (non-destructive)
- [x] Double-blind scan CLEAN (no names/URLs/cluster/SHAs; anonymous author)

## TODO — needs PI / future session (≈2-month runway)
- [ ] **git commit** the whole batch to `codex/t7-attn-improve` (NOT yet committed)
- [ ] `/goal clear` the obsolete "投ICDE" session goal (pivoted to KDD)
- [ ] Verify official KDD 2027 D&B Cycle-1 deadline when CFP posts
- [ ] Camera-ready acmart: drop `nonacm,anonymous`; add `\acmConference`, CCS concepts, keywords
- [ ] If KDD counts the 2 disclosure statements toward 8pp: shave ~0.5pg (Repro+AI-disclosure now spill to p9)
- [ ] Optional: re-balance main↔appendix (PI may re-promote any relocated block; see build_acm.py `specs`)
- [ ] Lu et al. SpaceFormer (lu2025spaceformer): confirm exact ICML/PMLR volume at camera-ready

## Appendix manifest (relocated by build_acm.py, all recoverable)
spectra-RW-para · Featurization-Control subsection (incl 14-repo table*) ·
V2-T5 pair-source+tensor-gate · ablation detail (bias-init/fig:gate/aromatic-routing/scope) ·
DFT gate-movement-probe · V0-finetune within-pipeline detail (incl tab:frozen-vs-finetune) ·
spectral-GNN background · 3D-SSL background · DFT motivation.

## Key files
- `paper/main_acm.tex` (submit) · `paper/main.tex` (ref) · `paper/build_acm.py` (rebuild)
- `paper/make_tables.py` → `paper/tables/{classification,regression}.tex` (from per-seed JSON)
- `unimol_finetune_results.json` (the FT numbers) · `reproduce_unimol_finetune.py` + `hpc/run_unimol_finetune.sbatch` (runner)
- `paper/ICDE_POSITIVE_REPOSITION_PLAN_2026-06-04.md` (full plan) · `paper/audits/reframe_adversarial_review_2026-06-04.json` (review)
