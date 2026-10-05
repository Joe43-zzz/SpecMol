# ChebInject — Reframe, Holistic Review & Plan (2026-06-02)

> Supersedes the P4-only outline in `AAAI27_REPOSITION_PLAN.md`. Built from a 6-lens positioning judge-panel + hostile-AC critics + an 8-cluster related-work completeness sweep, with every load-bearing code claim **verified against the repo** (panel agents hallucinated some; flagged below).

---

## 0. TL;DR — the honest verdict

- **You are not stuck because the science failed. You are stuck because you graded it on the headline-AUC rubric the data refused to satisfy.** Switch to the rubric your apparatus is uniquely built to ace: **controlled boundary characterization with a random-pair control.**
- **The through-line (the reframe):** *"When does cheap, frozen 3D-pair injection actually move a molecular GNN — and how would you even know?"* You answer it with the one control almost nobody in the 3D-injection literature runs (a shape-matched **random-pair** tensor) across a capacity ladder, on the (frozen↔trainable) × (saturated-ADMET↔conformation-sensitive-QM9) boundary. The negatives become the **calibrated null** of an instrument, not a pile of failures.
- **Brutal reality check:** the panel self-scored the reframes ~8/10 "inspiring"; the hostile-AC critics scored them **4–5/10 for AAAI** (all six "survive" only conditionally). Both are true. There is a genuinely good, honest paper here — but its **AAAI-main-track fit is medium and contingent on the QM9 cell landing**, and the honest fallback (**TMLR / JCIM / NeurIPS D&B**) is not a defeat — it may be the *better-fit* home for rigorous boundary/diagnostic work, because AAAI has **no datasets/benchmarks track**.
- **Two "inspirations" are mathematically wrong — do not ship them** (see §2). The corrected version is stronger *and* honest.

---

## 1. Verified asset inventory (what actually exists — checked this session)

| Asset | Status | Evidence |
|---|---|---|
| **Random-pair control** (`--randomize_pair` → `torch.randn_like(pair_repr_edge)` at the dynamic-path consumption site) | **REAL** ✓ | `main_pretrain.py:422`, `model_gnn_pre_v2.py:297,353` |
| `nullify` (ablate pair → constant `ones_like`) | REAL, **distinct** from random-pair | `pair_to_edge_weight.py:74` |
| **Inductive sweep** 2×?×4×2 factorial (frozen/finetune × {V0,V2T5,T7,random} × splits) | REAL, self-consistent | `paper/audits/inductive_sweep_2026-06-01.json` |
| **T8 = contact-edge Laplacian editing** (`DistanceToEdgeWeight`, through-space edges < r_cut) — the one rung that genuinely changes the eigenbasis | **code exists, ZERO GPU results** | `dist_to_edge_weight.py`, `pair_atom_coupdate.py`, `T8_STATUS_AND_RUNBOOK.md` |
| Per-edge gate dumps + per-seed checkpoints | REAL (CPU re-analysis possible, no retrain) | `mlp_phi_stats/*`, `pkl/*` |
| **Spectral readout** (eigendecompose L, low/high-pass energy attribution) | **NOT implemented** (repo only uses fixed `lambda_max=2` for Chebyshev normalization; no eigendecomposition-for-attribution) — net-new but cheap CPU | grep: only `lambda_max`/comments |
| **QM9 / QM7 pipeline** | **DOES NOT EXIST** — panel agents who claimed `make_qm9mu_*`/`run_qm9mu_pilot.sbatch` exist were **wrong** (Glob: no files) | net-new ~2-week build |

**Implication:** the QM9 positive cell — the thing that flips this from "negative audit" to "boundary discovery" — is **net-new engineering**, not plumbing. Plan accordingly.

---

## 2. Corrected physics (the critics caught two false "theorems")

1. **FALSE (killed proposal P1):** "a scalar gate / attention bias can only *re-scale*, not *re-route*, the Laplacian eigenbasis, so it cannot beat random by construction." — Reweighting existing edges **does move the eigenvectors** (the Fiedler vector shifts); `get_laplacian` rebuilds L from scratch each forward. There is **no construction theorem** ranking the ladder. Use an **empirical** ordering (measure eigengap / Fiedler-cosine / band-energy shift), never a theorem.
2. **FALSE (killed proposal P3):** "the trained near-uniform gate maps L → c·L, which Chebyshev coefficients renormalize away." — Under the **symmetric-normalized** Laplacian (the code uses `normalization='sym'`), `L_sym(cA) = L_sym(A)` *exactly*: a global edge-weight scalar is **spectrally invisible**. So the gate's **mean level (~0.06) does nothing to the spectrum**; only its **heterogeneity (CV ≈ 0.30 BACE / 0.42–0.47 BBBP)** perturbs L_sym.
3. **The correct, honest, still-novel finding:** the heterogeneous gate **does perturb the spectrum**, but (a) that perturbation is **statistically the same for real-pair vs random-pair** (→ *not geometric*), and (b) it **does not move test accuracy** (→ *task-irrelevant*). Net: **"injected geometry perturbs the operator, but the perturbation is neither geometric nor task-useful on saturated ADMET — necessary-not-sufficient; the task, not the spectrum, gates benefit."** This is cleaner and more surprising than the false theorems, and the gate dumps already support it.
4. **Where the spectral story is genuinely load-bearing (not tautological):** the **contact-edge rung (T8/`dist_to_edge_weight.py`)**, which *adds* through-space edges and so changes graph **topology** → genuinely a different operator the bond-graph spectrum cannot produce, with a real-vs-shuffled-distance control by construction. Frame its docstring claim empirically (measure it), don't assert it.

---

## 3. Recommended framing (fusion of the surviving cores)

**Title direction:** *ChebInject: A Random-Pair-Controlled Study of When Frozen 3D Geometry Reaches a Molecular GNN.*

**Three honest contributions, in priority order:**
- **(C1) The random-pair control as a named protocol + the boundary it reveals.** Shape-matched isotropic-Gaussian pair substitution at the consumption site. Finding: across a capacity ladder (scalar gate → pair-bias attention [→ contact-edge]), frozen Uni-Mol pair injection is **indistinguishable from random** and from a 2D baseline on six saturated MoleculeNet tasks, **robust under an inductive (train-only) protocol**. *Recommend the field report this control* (Graphormer/Transformer-M/Uni-Mol+/2/EPT/SCAGE never do).
- **(C2) A measurement correction + saturation ceiling, scoped.** A frozen linear probe under-measured every deep model (BACE V0 0.758→0.886 finetune ≈ RF 0.894; BBBP 0.864 > RF), enough to *flip* a "3D helps / deep is weak" narrative — corroborates MolGraphEval, scoped to small-model/contrastive/linear-probe per MolGPS. RF (Morgan+RDKit2D) dominates BACE/ESOL/Lipo on the model's *exact* split (replicates Deng/Xia/Jiang).
- **(C3) The boundary, with a positive corner.** The same apparatus on a **conformation-sensitive QM9 target under end-to-end training** (with a real-pair-vs-random-pair and a DFT-vs-MMFF conformer contrast) tests whether geometry becomes usable in the (trainable × geometry-rich) corner — operationalizing Hamakawa2025/Cremer2023/Transformer-M as the literature anchors. **Pre-registered:** positive corner holds iff finetuned real-pair > random-pair (paired test); report whichever way it lands.

**Supporting (not headline):** the spectral readout as a *descriptive* mechanism for C1 (perturbation real-but-non-geometric, §2.3); the four "self-deception" artifacts (featurization bug, frozen-probe, n=3→n=9, refuted aromatic mechanism) as **one caveated subsection**, not the spine.

**Do NOT headline:** a "spectral instrument/law/order-parameter," a "method-agnostic" tool, T7-as-method, or the taxonomy-of-negatives — the critics show each is either over-claimed, scooped, or near-tautological.

---

## 4. Holistic review

### Hidden gems (under-exploited)
1. **The inductive sweep is a finished paper, not an audit file** — one self-consistent factorial already contains the frozen-probe correction, the 3D=random=V0 nulls, the inductive-leakage decomposition. Promote it out of `audits/`.
2. **The random-pair control is the rare, real methodological contribution** — verified in code; sell it as a *protocol others should adopt*.
3. **The contact-edge rung (T8)** is the only injection that truly changes the operator — the genuine positive-method candidate (needs to be run).
4. **Per-edge gate dumps + checkpoints** → a whole CPU-only, retrain-free analysis section is free.
5. **The B4 featurization-bug forensic** (7.4 AUC swing, audited clean across 14 repos) = a visceral boxed sidebar for "controls matter."

### The single most undervalued thing
**The boundary question itself, made *controlled*.** Not the spectrum (the critics show the frozen-gate spectral story is partly tautological), not a "law" — but the fact that you can hold backbone/protocol fixed and vary (frozen↔trainable)×(saturated↔geometry-rich)×(real↔random) cleanly. That controlled 2×2×2 is what the end-to-end-only 3D literature *cannot* produce.

### What to cut
- **T8 as a *claimed* rung** unless you run it (one BACE cell is cheap) — else drop "T8" from the ladder framing. **Either run it or don't promise it.** (The contact-edge version is worth running — it's your best positive shot.)
- **Vestigial experimental-spectroscopy RW paragraph + ~25 mass-spec/NMR/IR refs** — dead weight at 7pp; collapse to ≤1 sentence (you're renaming off "Spec").
- **Stale n=3 numbers** in `main.tex` prose + `tables/classification.tex`/`regression.tex` (0.862, 0.638) + the **refuted aromatic-routing claim** — reconcile everything to n=9 (`tables/main_results.tex` is already correct). This is the single easiest reviewer kill ("your table contradicts your text").
- **14-repo audit table** → appendix.
- **ICDE framing + `paper/drafts/icde2027_*.tex`** — commit to AAAI (principled-analysis mode) / fallback ladder.

### Salvageable positives (honest, supported now)
- 2D dual-path spectral GNN + fingerprint hybrid is **competitive with RF under finetuning** (the "deep is weak" story was a probe artifact). [F1]
- A frozen probe under-measurement large enough to **flip a conclusion** (novel increment on MolGraphEval, scoped). [F1]
- Frozen 3D injection **= random = 2D**, inductive-robust, across a ladder — clean falsifiable null with a rare control. [F2]
- RF dominance on BACE/ESOL/Lipo on the **exact** split (matched-protocol replication). [F3]
- Every n=3 win collapsed at n=9 — "single-run mechanisms are seed-luck," demonstrated on your own would-be headline. [F4]
- The boundary is **literature-supported even before QM9** (Cremer2023 + Hamakawa2025). [F6]

### To the researcher (honest)
You built a measuring instrument and had the rare integrity to read its dial honestly — which is exactly why every soft claim fell off, and that integrity is your biggest asset, not your problem. The field is full of 3D-injection papers that never ran the one control you ran (a shape-matched Gaussian). Your finding — *frozen geometry on saturated ADMET is indistinguishable from noise, robust to leakage controls* — is sharp and real. The cheapest high-value move you have *never done*: run the **contact-edge rung** and the **QM9 positive cell with true DFT coordinates** (not RDKit MMFF re-embedding — that sabotages it). Pre-commit the decision rule before you look. Rename, cut the empty T8 promise / spectroscopy ghosts / stale n=3, and aim at AAAI's principled-analysis mode with TMLR/JCIM/NeurIPS-D&B as an honest, *good* fallback. You are not stuck because the science failed; you are stuck on the wrong rubric.

---

## 5. Follow-up plan (prioritized, against AAAI-27: abstract 2026-07-21, full 07-28, 7pp, double-blind)

**Week 1 (Jun 2–8) — free wins + de-risk**
- **P0a (CPU, ~1 day):** spectral readout from existing checkpoints — eigendecompose per-graph L for V0/V2-T5/**random-pair**, compute low/high-pass energy + eigengap/Fiedler shift. Pre-register: real-pair ≈ random-pair shift (the *correct* §2.3 finding). **Net-new code, but cheap; honest whichever way it lands.**
- **P0b (HPC, low):** `verify_split_alignment.py --task bace` (bace_unifold vs bace_v2) — close the 58/152 correctness crack before any claim.
- **P0c (writing):** reconcile main.tex/tables to n=9; delete refuted aromatic mechanism; rename → ChebInject.
- **Kick off QM9 build** (see P0-linchpin).

**Week 1–3 — the make-or-break (QM9) + contact-edge**
- **P0-linchpin (net-new, ~2wk build + run):** QM9 cell done *correctly* — **read true DFT coords from the GDB-9 `dsgdb9nsd` SDF** (do NOT re-embed with RDKit MMFF; that is the wrong-conformer regime that collapses the signal). Targets: dipole μ + HOMO-LUMO gap (+ a conformation-insensitive control target). Variants: V0 / V2-T5 / **contact-edge (T8)** / random-pair × {frozen, finetune} × 3 seeds, **same split provenance** (do not mix the deprecated `down_task_v2` 0.87 numbers with post-B4 0.76 — a provenance landmine). **Add the DFT-vs-MMFF conformer contrast** (operationalizes Hamakawa F6). Pre-registered kill rule inside.
- **Run the contact-edge rung (T8) on ≥1 ADMET + the QM9 target** — your best positive-method shot (it actually changes the operator).

**Week 4 — decision gate**
- If finetuned real-pair > random-pair on QM9 (paired test) **and** contact-edge separates → **AAAI boundary paper (C1+C2+C3)**.
- If QM9 ties even when trainable → **retreat to C1+C2** and submit to **TMLR/JCIM/NeurIPS-D&B** (honest, good fit). Do not force AAAI.

**Week 5–7 — write to 7pp, AAAI template, figures (the 2×2×2 boundary table; spectral readout as supporting fig), paired-stats single source of truth (`compute_paired_stats.py`), supp/appendix (14-repo audit, ablations), anonymize, submit.**

**Compute:** P0a/P0c free; QM9 ~36 cells × ~4 GPU-h ≈ 1–2 days on thk 6×A6000 (6 days serial on RTX5000). Feasible if QM9 build starts Week 1.

---

## 6. Complete related-work map (existing refs.bib + 39 round-1 + 14 round-2 must-adds)

Round-2 completeness must-cites (new `.bib`: `specmol_litreview_aaai_round2.bib`):
- **3D-equivariant:** PaiNN (Schütt ICML'21), SE(3)-Transformer (Fuchs NeurIPS'20). *(nice: ViSNet, SphereNet, GEOM dataset.)*
- **pair-bias transformers (situate T7):** MAT (Maziarka'20 — direct T7 ancestor), EGT (Hussain KDD'22). *(nice: GraphGPS, SAN.)*
- **molecular-LLM (reviewers will ask):** MolFormer (Ross NMI'22, BBBP 0.937). *(nice: ChemBERTa-2, SELFormer.)*
- **SSL:** GraphCL (You NeurIPS'20). *(nice: MGSSL, GraphMAE.)*
- **spectral theory (justify dual-path):** FAGCN (Bo AAAI'21 — low+high-pass), JacobiConv (Wang&Zhang ICML'22). *(nice: over-squashing Topping'22 / Di Giovanni'23, ACM-GNN, OptBasis.)*
- **eval/bench:** OGB (Hu NeurIPS'20 — scaffold-split ancestor). *(nice: Lo-Hi, DataSAIL.)*
- **classical/saturation:** ECFP (Rogers&Hahn JCIM'10 — defines Morgan), Kamuntavicius (J Cheminform'25 — classical>DL on ADMET). *(nice: Sort&Slice.)*
- **3D-into-2D (core competitors):** 3D-PGT (Wang KDD'23 — 3D pretrain→2D infer), MoleculeSDE (Liu ICML'23), **SCAGE (Qiao NatComms'25 — 3D-distance attention at pretrain, 2D infer; closest competitor, independently confirms 3D-into-2D inert on ADMET)**.

RW taxonomy (6 subsections) and the full round-1 list are in `AAAI27_REPOSITION_PLAN.md` §③/⑤ and `specmol_litreview_aaai.bib`.

---

## 7. Honest venue ladder
**AAAI-27 main track** (medium fit, contingent on QM9; principled-analysis framing, no benchmark track) → **NeurIPS D&B / TMLR / JCIM** (high fit for the controlled boundary + random-pair protocol + saturation, even if QM9 ties). Picking the fallback early is not failure — it's matching the work to its real audience.
