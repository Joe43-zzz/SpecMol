# SOTA survey + baseline run-plan (for advisor discussion) — 2026-06-03

**Framing (non-negotiable):** this is an ICDE-EAB **audit / negative-result** paper. We do **not** compete on leaderboard AUC. Recent SOTA goes in **Related Work as the ceiling we deliberately don't contest**; we **run** only baselines that defend against a specific reviewer attack. Deadline kept at **6/11 (R1)**.

**The load-bearing argument the survey hands us:** every SOTA number is on a *different* split/cleaning than our Uni-Mol-fold pipeline. **Smoking gun: canonical Uni-Mol BBBP ≈ 0.729 under its own scaffold split vs 0.907 under SpaceFormer's cleaned split — a >0.17 AUC swing from split + dedup alone, larger than most model-vs-model gaps.** That swing *is* our argument that leaderboard AUC is not the right axis. Cite it explicitly.

---

## 1. SOTA "ceiling" table → Related Work (cite, do NOT run; always with the split caveat)

| Dataset | SOTA ceiling (model, number, split) | Citation |
|---|---|---|
| BBBP (AUC↑) | MoLFormer-XL **0.937** (MoLFormer scaffold); AdaMR 0.917 (MolNet-scaffold); Uni-Mol 0.907 (SpaceFormer cleaned-OOD) | Ross NMI'22 (arXiv:2106.09553); arXiv:2401.06166; arXiv:2503.10489 |
| BACE (AUC↑) | GEM **0.985** / SpaceFormer 0.966 / Uni-Mol 0.952 (cleaned-OOD); AdaMR 0.894, MoLFormer 0.882 (other splits) | arXiv:2503.10489 (2025); Ross NMI'22 |
| ClinTox (AUC↑) | ImageMol 0.975 / AdaMR 0.969 / MoLFormer 0.948 | Zeng NMI'22; arXiv:2401.06166; Ross NMI'22 |
| FreeSolv (RMSE↓, kcal/mol) | MSHG-MAE **0.78** / DFusMol 1.08 / KANO 1.14 / Uni-Mol 1.48–1.74 | JCIM'26 5c02994; Front.Mol.Biosci.'25; NatComm'23 |
| ESOL (RMSE↓, log) | MSHG-MAE **0.465** / DFusMol 0.668 / KANO 0.670 / Uni-Mol 0.78 | JCIM'26; 2025; NatComm'23 |
| Lipo (RMSE↓, logD) | MSHG-MAE **0.501** / KANO 0.566 / Uni-Mol 0.575 / DFusMol 0.576 | JCIM'26; NatComm'23; 2025 |
| QM7 (MAE, kcal/mol) | SchNet ~0.31; FCHL kernels 0.25–0.30 | NeurIPS'17; Christensen JCP'20 |
| QM9-μ (MAE, Debye) | Equiformer / TorchMD-NET / EQGAT / PaiNN **0.011–0.012 D** (random 110k split) | arXiv:2206.11990 (ICLR'23) |

**Methodological RW anchors (already mostly cited):** Deng2023 (NatComm, 62k models, RF best-or-tied p<0.05) · Mole-BERT · Uni-Mol · 2026 survey arXiv:2604.16586 (rankings unstable across splits).

⚠️ **Never tabulate these side-by-side with our numbers** — different splits, and QM9-μ SOTA is MAE on the 11k random test set while ours is RMSE on a 1,256-mol matched DFT subset (~80× scale gap). One-line ceiling citations only.

---

## 2. ⚠️ Two honesty flags the survey caught (apply to the paper regardless of what we run)

**(A) Standardization hazard — regression RMSE.** Our FreeSolv/ESOL/Lipo RMSE are on **standardized (z-scored) labels** (σ≈1), NOT raw units. FreeSolv σ≈3.8 kcal/mol ⇒ our **0.72 standardized ≈ 2.7 kcal/mol RAW = WORSE than the field's 0.78–1.7**. We must **never imply we beat the kcal/mol literature**; the regression table caption already says "standardized scale; FreeSolv not in kcal/mol" (good) — keep SOTA citations out of that table, and ideally add a standardized→raw note. *(Our deep-vs-RF comparison is valid because both are on the same standardized scale.)*

**(B) Bound the "RF beats deep" thesis.** It is strong on **BACE / ESOL / QM7** (genuinely saturated). It is **WEAK on Lipophilicity** — strong supervised deep (KANO/Uni-Mol ~0.50–0.57) *beats* RF 0.643 — and **not clearly true on ClinTox** (deep reaches 0.95+; our RF 0.812 is mid-pack). So our "RF dominates Lipo/ESOL" is dominance over **our matched spectral variants**, not over deep SOTA. → Add a sentence bounding the saturation thesis to the genuinely-saturated endpoints; flag Lipo/ClinTox as boundary cases. **(Being applied to main.tex now.)**

---

## 3. Baselines to RUN — prioritized, 6/11-feasible

| Pri | Baseline | Answers attack | Cost | Feasible 6/11 | Code |
|---|---|---|---|---|---|
| **P1 ✅ MUST-RUN** | **Uni-Mol end-to-end FINETUNE** on our matched split, all 6 cells, 3 seeds | "Your 'Uni-Mol' row is a crippled frozen-emb+RF, not Uni-Mol" | **~4–6 GPU-h** | **Yes** | shipped (`reproduce_unimol_finetune.py` + `hpc/run_unimol_finetune.sbatch`), **never run** (VPN-gated) |
| **P2 ⚠️ scoped demo** | **Frozen SchNet RBF** as a *second 3D pair source* through our gate, **QM9-μ DFT subset only** (n=3) | "Your null is about Uni-Mol, not 3D; a real equivariant encoder might help" (caveat iii) | low (frozen fwd + existing probe) | Yes, **QM9-μ only** | **net-new extractor** (`make_qm9mu_from_schnet.py`): PyG SchNet returns node/graph, not pair tensors → hook per-edge RBF/filter → project to pair_dim=64 |
| **P3 ❌ DO NOT RUN** | 2D transformer (MolFormer/Graphormer) reproduction | (weak) | high | **No** (zero transformer code in repo) | RW-only |

**Already-run baselines to keep/foreground (no new run):** RF (Morgan+RDKit2D, Deng2023 30-seed) ✓; Chemprop D-MPNN ✓; frozen-Uni-Mol-emb+RF (rename → "Uni-Mol-emb+RF" once P1 lands).

---

## 4. Experiment steps (if greenlit)

**P1 — Uni-Mol finetune (turnkey):**
0. VPN+ssh HPC, `git pull` (reproduce_unimol_finetune.py branch). No new data (reuses fold split; bbbp/bace re-pointed at `paper/audits/{bbbp,bace}_fold_recovered.csv`).
1. **Smoke (~10min):** `DATASETS="freesolv" SEEDS="9" EPOCHS=20 sbatch hpc/run_unimol_finetune.sbatch` → catch unimol_tools API drift cheaply.
2. **Sharpest cells:** `DATASETS="bace clintox" sbatch ...` (BACE = "RF still beats finetuned Uni-Mol"; ClinTox = likely Uni-Mol *wins*, bounds the thesis honestly).
3. **Remainder → 6/6:** `DATASETS="bbbp freesolv esol lipo" sbatch ...` (resumable JSON).
4. **Paper:** rename row "Uni-Mol" → **"Uni-Mol-emb+RF"** (frozen probe), ADD **"Uni-Mol (finetune)"** from `unimol_finetune_results.json` into both tables. ⚠️ if MolTrain std is fake-0, report n=3 mean + internal-CV spread, not a real seed std.

**P2 — SchNet second-3D-source (QM9-μ only, only if P1 done + time left):**
1. Write `make_qm9mu_from_schnet.py`: load pretrained schnetpack QM9 SchNet, ONE frozen forward on `data.pos` per molecule, harvest per-edge **Gaussian-RBF distance expansion** (model-free distance control — truest analogue) and/or the learned post-interaction edge filter → project to pair_dim=64 → write into `pair_repr_edge` slot (`down_task_qm9mu_schnet/`).
2. Run the existing 7-arm gate decomposition on the **same 12,532-mol DFT subset** (n=3): real SchNet-pair vs shuffled-random vs RF. Only the pair tensor swaps; gate wiring unchanged.
3. **Paper:** add arms "SchNet-RBF (real)" / "SchNet-RBF (shuffled)" to `tab:dft` → converts caveat (iii) "untested on equivariant encoders" from a *stated limitation* into a *tested result*. **Scope guard:** ship QM9-μ SchNet cell ONLY; defer BBBP/BACE SchNet parity to future work.

---

## 5. 4-sentence brief for 学姐

> This is an audit paper, so we run only baselines that defend a specific reviewer attack, never anything to beat the leaderboard — recent SOTA (SpaceFormer BACE 0.966, MoLFormer BBBP 0.937, MSHG-MAE ESOL 0.465, Equiformer QM9-μ 0.011 D) goes in Related Work as the ceiling we don't contest, always with the caveat that every number is on a different split (the Uni-Mol BBBP 0.729→0.907 split swing is our central argument). The one high-value, low-risk run is **P1: the real end-to-end Uni-Mol finetune on our matched split** — turnkey (code shipped, never run, ~4–6 GPU-h) and it kills the sharpest attack that we "crippled Uni-Mol by freezing it." **P2** (frozen SchNet as a second 3D source through our gate, QM9-μ only) is a scoped demo that turns our weakest caveat into a tested result but needs net-new extraction code, so do it only after P1 and defer full parity. Do **not** attempt MolFormer/Graphormer reproduction (zero code, can't land in 8 days). Two honesty fixes go in regardless: our regression RMSE is **standardized** (don't imply we beat the kcal/mol literature), and "RF beats deep" must be **bounded to BACE/ESOL/QM7** (Lipo/ClinTox are boundary cases where strong deep beats RF).

---

## 6. Ready-to-paste Related-Work ceiling paragraph (for main.tex `sec:related`)

> \paragraph{The SOTA ceiling we do not contest.} Recent supervised and pretrained models post far higher leaderboard scores than our audit targets---MoLFormer-XL~\cite{...} reaches BBBP~$0.937$ and ClinTox~$0.948$, the 3D-space-modeling SpaceFormer~\cite{...} reaches BACE~$0.966$ (GEM~$0.985$), and equivariant geometry models~\cite{equiformer} reach QM9 dipole MAE~$0.011$~D. We cite these as the upper envelope, not a target: our contribution is a controlled audit of \emph{frozen} 3D-pair injection, not a leaderboard entry. Crucially, these numbers are not mutually comparable---canonical Uni-Mol scores BBBP~$0.729$ under its own scaffold split but $0.907$ under a cleaned out-of-distribution split~\cite{spaceformer}, a $>0.17$ AUC swing from split and de-duplication alone that exceeds most model-vs-model gaps. This split-sensitivity is itself part of our argument that single-split leaderboard AUC is not a robust axis for the question we ask.

*(Add bib keys for MoLFormer, SpaceFormer, Equiformer, AdaMR, KANO, DFusMol, MSHG-MAE before pasting; the 3 already-added P-cites stay.)*
