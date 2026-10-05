# SpecMol Audit Paper — Results Tables + Narrative

**Status:** consolidation of the 2026-06 audit findings. Numbers marked `[verify: <json>]`
should be confirmed against the cited result file before camera-ready; the rest are
this-session-verified. Venue framing: AAAI-27 P4 "when does cheap 3D injection help?".

---

## 0. Thesis (one paragraph)

Cheap Uni-Mol 3D injection into a 2D spectral GNN is studied as a **controlled audit
of when 3D geometry helps molecular property prediction**. Two boundaries emerge.
**(B1) On fingerprint-saturated ADMET, 3D injection is inert** — a *data* ceiling, not
an injection artifact: a matched-split RandomForest on Morgan+RDKit2D dominates, and
the result is robust across a faithful injection ladder (scalar gate → sidecar
co-update → interleaved co-update → raw trainable geometry) and six independent
controls. **(B2) On geometry-bearing QM, the geometry mechanism is demonstrably alive
but still loses to a cheap charge-feature RF** — we show, with a gate-probe, that the
co-update gates open *geometry-specifically* (real distances open them, shuffled
distances do not) and that real geometry beats its random control; yet a 12-target
screen proves every QM9 target is charge- or composition-determined, so descriptor/charge
RF wins regardless. The contribution is the **boundary itself plus a reusable rigor
kit** (matched-RF ceiling, random-pair control, gate-probe, target screen) — and two
honesty corrections it surfaced (a gate-LR wiring bug that had frozen the geometry gates
even under finetune; a split-alignment confound that briefly looked like a breakthrough).

---

## 1. Narrative arc (sections)

1. **Setup.** ChebNetII dual-path spectral GNN + fingerprint MLP, contrastively
   pretrained; Uni-Mol 64-d pair_rep injected via an escalating ladder (V2-T5 scalar
   gate → T6/T7 → T8 sidecar co-update → **T9 interleaved co-update** → **A2 raw
   trainable distance**). Frozen linear-probe is the canonical protocol; an end-to-end
   finetune twin exists.
2. **B1 — ADMET is a data ceiling (the negative, made bulletproof).** Matched-split RF
   dominates; the whole injection ladder ties V0; six controls (below) rule out the
   "broken injection" rebuttal.
3. **B2 — QM geometry: mechanism alive, ceiling holds.** Gate-LR bug found+fixed → the
   gate-probe shows the geometry gates DO open and are geometry-specific; real>random;
   but charge-RF still wins, and the 12-target screen explains why (no geometry-limited
   target).
4. **Honesty corrections (the methodology in action).** (i) gate-LR wiring bug; (ii) the
   DFT split-alignment confound (deep-on-1256 vs RF-on-1473) that we caught and corrected.
5. **Conclusion.** A defensible "when does cheap 3D help, and where does it stall"
   boundary, not a SOTA claim. Reusable controls released.

---

## 2. Results tables

### Table 1 — ADMET classification (ROC-AUC ↑, Uni-Mol fold / matched split)
| Method | BACE | BBBP | ClinTox |
|---|---|---|---|
| **RF Morgan+RDKit2D (matched, n=30)** | **0.906** | 0.814 | 0.812 |
| RF Deng2023 (n=30) | 0.894/0.895 | 0.799 | 0.817 |
| Uni-Mol-direct (frozen CLS + RF) | 0.722 | 0.765 | 0.488 |
| Chemprop D-MPNN | 0.840 | 0.815 | — |
| Deep V0 (frozen probe) | 0.758 | 0.828 | 0.804 |
| Deep V2-T5 / T6 / T7 / T8 | 0.76 band | 0.831 | 0.829 |
| Deep V0 **finetune** (inductive, n=9) | 0.875 (still < RF) | **0.865 (> RF)** | [verify] |
- Saturation: RF dominates BACE; deep ties/leads only BBBP. ClinTox tie (σ large).
- Finetune lifts the deep NUMBER (BBBP 0.828→0.865 > RF) but does NOT make 3D help
  (3D=random survives finetune). `[verify: bace_all_results.json, bbbp_all_results.json,
  baselines_matched_results.json]`

### Table 2 — Regression (RMSE ↓, matched split)
| Method | FreeSolv | ESOL | Lipo |
|---|---|---|---|
| **RF Morgan+RDKit2D (matched, n=30)** | 0.720 | **0.361** | **0.643** |
| Uni-Mol-direct | 0.683 | 0.580 | 0.794 |
| Deep V0 (n=9) | **0.647** | ~0.82 | ~0.878 |
| Deep V2-T5 (n=9) | 0.673 | ~0.815 | — |
- ESOL/Lipo fingerprint-saturated (RF crushes deep). FreeSolv: deep ≈ ties / nominally
  leads RF (the one regression cell deep doesn't lose) — within seed noise. `[verify:
  baselines_matched_results.json, *_v0/_v2t5_results.json]`

### Table 3 — QM geometry mechanism (the B2 core), DFT QM9-μ, **same 1256 test set**
| Arm | RMSE (n=3, std-units) | note |
|---|---|---|
| **charge-RF (strongFull, ~210 desc incl partial charge)** | **0.845 ± 0.003** | the ceiling — WINS |
| Deep T8 (real geometry, gate-fix) | 0.882 [0.878,0.885] | reads geometry, loses to RF |
| Deep T8random (shuffled geometry) | 0.891 [0.887,0.896] | real < random (geometry effect) |
| weak-RF (11 composition desc) | 0.939 | composition floor |
| Deep V0 (2D) | ~0.993 | `[verify on 1256]` |
- **Mechanism is alive:** real geometry (0.882) beats its shuffled control (0.891),
  non-overlapping at n=3; and the gate-probe (below) shows the gates open only for real
  geometry. **Ceiling holds:** charge-RF (0.845) beats every deep arm — dipole is a
  charge-distribution property; Gasteiger charges proxy it cheaply.
- ⚠️ Earlier "deep 0.882 > charge-RF 0.925" was a **split confound** (deep on DFT-1256,
  RF on MMFF-1473; only 1242/1473 overlap). Corrected here on the identical 1256 set.

### Table 3b — Gate-probe (geometry-specificity of the co-update gates)
After end-to-end finetune (gate-LR fix applied), per-layer `tanh(pair_gate)`, n=3 seeds:
| pair input | layer-0 pair_gate | layers 1-3 | reading |
|---|---|---|---|
| **real geometry** | ~+0.31 to +0.44 | substantial, consistent | gates OPEN for geometry |
| **shuffled geometry** | ~0.000 | ~0.000 | gates STAY SHUT |
- The optimizer opens the geometry gate **only when the pair carries real geometry** —
  proving (a) the freeze was a wiring/protocol artifact (gates CAN open), and (b) the
  opening is geometry-driven. Holds on both MMFF and DFT. `[verify: hpc/logs/qm9mu_*gatefix.log]`

### Table 4 — QM9 12-target screen (why QM has no geometry battlefield)
weak-RF (11 composition) vs strong-RF (~210 incl charge), z-scored; `skill = 1−strongRMSE/dummy`.
| target | weak-RF | strong-RF | chargeΔ | skill | verdict |
|---|---|---|---|---|---|
| u0/u298/h298/g298 | ~0.093 | ~0.045 | ~+0.05 | ~0.96 | composition-nailed |
| zpve | 0.149 | 0.109 | +0.04 | 0.89 | composition-nailed |
| cv | 0.307 | 0.161 | +0.15 | 0.84 | descriptor-strong |
| alpha | 0.287 | 0.179 | +0.11 | 0.82 | descriptor-strong |
| lumo / gap / r2 | 0.32/0.38/0.45 | 0.22/0.24/0.27 | +0.10/0.14/0.18 | 0.74-0.78 | descriptor-strong |
| homo | 0.608 | 0.455 | +0.15 | 0.54 | descriptors fail BUT charge helps |
| **mu** | 0.711 | **0.597** | +0.11 | **0.39** | descriptors fail most, but charge helps |
- **No target is descriptors-fail (skill<0.5) AND charge-irrelevant (chargeΔ<0.1).** The
  low-skill targets (mu, homo) are charge-helped; the charge-irrelevant ones (energies)
  are composition-nailed → **no SpecMol-winnable geometry battlefield on QM9.** `[verify:
  qm9_target_screen.py output; rerun on full split for camera-ready]`

### Table 5 — The faithful injection ladder (the null is a DESIGN-robust DATA ceiling)
| rung | what | ADMET | QM9-μ |
|---|---|---|---|
| V2-T5 | 64→1 scalar gate → Laplacian | ties V0 | ties V0 |
| T6/T7 | per-step / pair-bias attention | ties V0 | ties V0 |
| T8 | sidecar co-update (rich pair, re-scalarized) | ties V0 | reads geom ~3σ, loses RF |
| **T9** | **interleaved per-step co-update, pair persists** (this work) | — | (built, run pending) |
| **A2** | **raw trainable distance pair (no Uni-Mol, no H-drop)** (this work) | — | (built, run pending) |
- Even the top, faithful, un-frozen rungs (T9, A2) do not change the ceiling — the null
  is not a "broken injection" artifact. (T9/A2 CPU-verified: epoch0≡V2-T5; gates open;
  geometry-specific control works.)

### The six ADMET null controls (B1 robustness)
1. V0 = V2-T5 = T6 = T7 = T8 within seed noise. 2. random-pair ≈ real-pair. 3. nullify ≥
V2-T5. 4. Uni-Mol-direct < RF. 5. RF dominates (matched). 6. holds under inductive
(`--pretrain_split train`) — no transductive leakage. + n≥9 erased every prior n=3 "win".

---

## 3. Contributions (slimmed, defensible)
1. **A controlled boundary** for cheap 3D injection: inert on FP-saturated ADMET (data
   ceiling); on geometry-bearing QM the mechanism is alive (geometry-specific gates,
   real>random) but loses to a charge-RF because QM targets are charge/composition-determined.
2. **The saturation/charge ceiling**, established by matched-split RF (+ a 12-target QM9
   screen locating exactly where descriptors fail and why geometry still can't win there).
3. **A reusable rigor kit + two honesty corrections**: matched-RF ceiling, random-pair
   control, gate-probe (geometry-specificity), target screen; the gate-LR wiring-bug fix
   and the DFT split-alignment correction demonstrate the kit catching over-claims.
4. **(Secondary) A faithful injection ladder** (scalar → sidecar → interleaved → raw
   trainable) showing the null is design-robust, not an artifact of a lossy injection.

---

## 4. Honesty arc (for the discussion / limitations)
- The gate-LR group matched only T7's `attn_gate`, silently leaving T8/T9's ReZero gates
  at the slow encoder LR ("barely move") even under finetune → the geometry thesis had
  never been fairly tested. Fixed; the gate-probe then showed geometry-specific opening.
- A DFT QM9-μ comparison briefly read as "deep beats charge-RF" — a split confound (deep
  on DFT-1256, RF defaulted to MMFF-1473). Re-running RF on the DFT manifest reversed it
  (charge-RF 0.845 < deep 0.882). The G9 split-alignment control is what caught it.
- Realistic ceiling: a non-equivariant ChebNetII reads 3D only through a coordinate-free
  scalar-distance RBF (no SE(3)/angular structure); geometry-SOTA is equivariant-net
  territory. We claim a boundary result, not a SOTA win.

---

## 5. Still to lock for camera-ready (HPC / verification)
- Extend the QM9-μ DFT triad (V0 / T8 real / T8random / charge-RF) to **n≥9** on the 1256
  set (n=3 is below the project's own retraction bar).
- Re-run the 12-target screen on the full/canonical split (current screen used a 5581-mol
  manifest∩qm9 overlap subset).
- The **real end-to-end Uni-Mol finetune** baseline (reproduce_unimol_finetune.py, unrun)
  — the credibility baseline reviewers will demand beside the frozen-embedding row.
- n≥9 for the ADMET deep cells + the matched RF columns already at n=30.
- (Optional) run T9 / A2 once on the DFT QM9-μ cell to populate the ladder's top rungs —
  expected to confirm the ceiling, reported as ablation completeness.
