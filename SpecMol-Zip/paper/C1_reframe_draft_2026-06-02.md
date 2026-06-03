# Contribution reframe — DRAFT (2026-06-02, NOT applied to main.tex)

Status: **draft for review only.** `main.tex` is untouched and still carries the
ICDE-era C1/C2/C3. Nothing here is committed. Grounded in
`NEAREST_NEIGHBORS_AND_GROUNDED_PLAN_2026-06-02.md` (65-paper deep read) + the
2026-06-02 code-verified facts (QM9 pilot used MMFF not DFT; S-CGIB not a scoop;
spectral-readout Part A exact lemma).

---

## 0. The repositioning in one line

We are **not** a method paper (we cannot beat classification SOTA — MoLFormer
BBBP 0.937, SpaceFormer BACE 0.966; our V0/V2-T5 sit at 0.76–0.83). We are a
**controlled audit of when frozen 3D-pair injection helps a molecular spectral
GNN, with a reusable random-pair null protocol.** The architecture is the
*instrument*; the negatives are the *measurements*; the QM9 cell is the
*positive boundary probe*.

**Proposed title (brand the method, keep the repo name SpecMol):**
> *ChebInject: A Controlled Audit of Frozen 3D-Pair Injection in Spectral
> Molecular GNNs, with a Random-Pair Null Protocol.*

**The moat (verified unoccupied across all 65 papers).** The 5-way intersection:
(a) a **frozen, pretrained 3D pair** representation, (b) injected into a
**spectral** GNN, (c) measured against a **shape-matched random-pair null**,
(d) under a **matched inductive split**, (e) with a **frozen→finetune
absolute-AUC correction** on the same pipeline. Each ingredient is individually
precedented; the intersection is ours.

---

## 1. Current contributions (verbatim from main.tex) → what changes

| | Current (ICDE-era) | Reframed (this draft) |
|---|---|---|
| C1 | "matched-protocol audit of 3D-pair injection across six benchmarks" | **C1 = the named random-pair NULL PROTOCOL** + matched inductive audit (the genuinely unscooped piece, promoted to lead) |
| C2 | "Fingerprint saturation—not 3D geometry—is the operative ceiling" | folded into the **reference ceiling** that threads through C2/C3 (replicated under matched control, credited to xia2023/kamuntavicius2025/Praski — never claimed new) |
| C3 | "released, regenerable evaluation harness" | kept as a **supporting artifact pillar** (release the protocol + harness), not a headline bullet |
| — | (absent) | **C2(new) = frozen-probe → finetune absolute-AUC correction**, scoped per MolGPS |
| — | (absent) | **C3(new) = the frozen-saturated-vs-trainable-geometry BOUNDARY** (ADMET → QM7 → QM9-μ), with the QM9 cell as the conditional positive corner |

Net: lead with the **protocol** (C1), keep the **correction** (C2) and the
**boundary** (C3) as the two measurements, and demote saturation + harness to
the supporting role the corpus says they deserve (both are established /
expected, not novel).

---

## 2. The three contributions, paper-ready prose (draft)

**(C1) A random-pair null protocol for 3D-injection claims, under a matched
inductive split.** We argue that any claim "geometry helps my GNN" must clear a
**shape-matched random-pair null**: replace the frozen 3D pair tensor with
`randn_like` noise of identical shape and scale at the exact consumption site,
holding architecture, split, seeds, and budget fixed. We implement this for
three injection mechanisms (a static edge gate, a pair-biased attention, and an
atom↔pair co-update) and show that across six MoleculeNet endpoints under a
single Uni-Mol scaffold-fold protocol, the real frozen pair does **not** separate
from its random-pair null within seed variance. This generalizes Hamakawa et
al.'s wrong-conformer control into a reusable, architecture-agnostic null that
others can drop into any 3D-injection pipeline. *(This is the one genuinely
unoccupied contribution; everything else is a measurement made with it.)*

**(C2) A frozen-probe → finetune absolute-AUC correction that overturns a
concrete 3D-injection conclusion.** Prior work establishes that a frozen linear
probe *ranks* encoders unreliably (MolGraphEval rank-corr 0.77; Pinto2025
under-measures by 5–40%). We add the missing piece: on a single pipeline we
report the **absolute** delta (BACE 0.758 frozen → 0.886 finetuned) and use it
to **flip** a 3D-injection claim that looked real under a frozen probe. We scope
this strictly to the **small / contrastive / linear-probe** regime and cite
MolGPS (probing ≥ finetune for *large supervised* 2D models) as the explicit
opposite-regime boundary — we do not claim a universal probing pathology.

**(C3) A frozen-saturated-vs-trainable-geometry boundary.** We map where frozen
3D-pair injection is inert vs where a real geometry signal appears, along a
geometry-signal axis: **saturated ADMET** (inert; descriptor RF is the ceiling)
→ **composition-dominated QM7** (atomization energy ≈ Σ atomic contributions;
count/composition RF crushes deep; the deep gain is ~92 % injection
*architecture*, not geometry) → **geometry-pure QM9 dipole μ** (the one corner
where a geometry-isolated signal emerges, and only through the richest mechanism
— atom↔pair co-update — at ~3 % RMSE). The fingerprint/descriptor RF ceiling is
the reference line throughout. **Honest status (see §4):** the QM9 cell to date
used **MMFF** conformers; whether *real DFT* geometry pushes deep **below** the
RF ceiling is the single pre-registered deciding experiment (Stage-2).

---

## 3. Nearest-neighbor differentiators (Related Work, draft paragraph map)

Each "A did X, NOT Y; we do Z" — must-cite set, grounded in verified methods.

- **SCAGE** (Nat. Commun. 2025) — closest competitor: 2D+3D multitask pretrain
  (incl. a fingerprint-prediction objective), **end-to-end finetune**, BBBP/BACE
  + activity cliffs. *Not:* frozen 3D source / inference-time frozen-pair
  injection; no random-pair control; not spectral. **Z:** we study the
  frozen-injection regime SCAGE trains *through*, with a null it never runs.
- **3D-PGT** (KDD 2023) — 3D generative pretrain → 2D-only inference, BACE 0.809.
  **Z:** distillation-into-weights (theirs) vs frozen-pair-injection-at-inference
  (ours) + the null.
- **Praski2025** — 25 frozen embeddings incl. Uni-Mol: **Uni-Mol 76.85 < ECFP
  79.89** on ADMET. **Cite FOR us** (strongest external validation that frozen
  Uni-Mol ≈ useless); we explain *why* (random-pair tie) and show finetune
  recovers it (C2).
- **Hamakawa2025** (JCIM) — wrong-conformer control on *finetuned* Uni-Mol;
  correct > wrong ≈ ECFP. **Z:** our random-pair (shape-matched Gaussian)
  generalizes the wrong-conformer idea into a reusable null on a *frozen
  spectral* pipeline. (Single most-aligned prior; frame as our protocol's
  ancestor.)
- **Graphormer / MAT / EGT / Transformer-M / Uni-Mol2** — pair-as-attention-bias
  / 3D-distance-as-bias, **end-to-end**, big-pretrain. **Z:** they own the
  attention mechanism (T7/T8 carry *zero* novelty here) and are the upstream
  pair source we consume *frozen*; our delta is frozen-source + spectral-target +
  null. (Add **MAT** + **EGT**; they are currently absent.)
- **EAGCN** (2021) — learned edge-attention modulating the molecular Laplacian.
  **Must-cite; do NOT claim edge-gating novelty** — V2-T5 = EAGCN's edge-weighted
  Laplacian with the gate values *sourced from a frozen Uni-Mol 3D pair*.
- **FAGCN** (AAAI 2021) — per-edge scalar gate mixing low/high-pass. **Add;**
  direct antecedent of the V2-T5 scalar gate + dual-path design.
- **KA-GNN** (Nat. Mach. Intell. 2025) — 2D MP + 5 Å through-space contact
  edges, BACE 0.890. **Important caveat: KA-GNN already owns the contact-edge
  mechanism** we floated as a "positive method" — so contact-edge is a
  **completeness rung, not a headline** (and our orphaned `dist_to_edge_weight`
  stays unshipped unless used purely for ladder completeness).
- **MolGraphEval** (NeurIPS 2023 D&B) / **MolGPS** (NeurIPS 2024) /
  **Pinto2025** / **Sun2022** — the C2 precedent cluster. **Credit, don't claim
  discovery**; our increment is the *absolute-delta-that-flips-a-conclusion*,
  scoped per MolGPS.
- **S-CGIB** (AAAI 2025, Hoang & Lee) — **NOT a C1 scoop** (verified 2026-06-02):
  2D-graph SSL via subgraph-conditioned info bottleneck; no 3D, no spectral
  backbone, no contrastive NT-Xent, no null/matched/frozen-vs-finetune audit.
  Cite as adjacent molecular-SSL, not a competitor.

Currently **present** in main.tex: EAGCN, Hamakawa, Graphormer, MolGraphEval,
Praski (RW only). Currently **absent** and to add: SCAGE, 3D-PGT, MAT, EGT,
FAGCN, KA-GNN, MoleculeACE (van Tilborg), MolGPS-as-boundary.

---

## 4. Mechanism evidence to thread in (already produced, free)

**Spectral-readout Part A (exact, checkpoint-free — `tools/spectral_readout.py`).**
Under symmetric Laplacian normalization the V2-T5 gate's **mean is spectrally
invisible**: the sym-normalized Laplacian is *identical* for every uniform edge
weight (numerically max|Δeig| = 8.9 × 10⁻¹⁶ between c=1 and c=7), while a
heterogeneous weight vector shifts the spectrum (L2 = 0.18). This is a clean
analytic reason the scalar gate can act *only* through per-edge heterogeneity —
it cannot encode "more/less geometry on average." Pairs naturally with the
finding that the learned gate trends toward aggressive sparsification, not
near-identity. *(Part B — learned filter response real-vs-random — is
descriptive only: with one matched seed pair the low/high filters differ
[L2 1.10 / 0.50], but n=1 cannot separate injection-effect from
seed/optimization noise; a multi-seed version is a cheap follow-up. Also note
the code's "low/high-pass" branch names are nominal — the trained shapes do not
respect them — fix this wording in any figure caption.)*

---

## 5. Honest caveats that MUST survive the reframe (no motivated reasoning)

1. **QM9 = MMFF, not DFT.** The current QM9-μ "geometry" result is MMFF-derived.
   C3's positive corner is **conditional** on the Stage-2 DFT re-run. Until then,
   phrase C3 as "a geometry-isolated signal appears on MMFF μ through the richest
   mechanism, still below the RF ceiling; the DFT test is pre-registered."
2. **No broad performance win.** BBBP n=9 tie; BACE RF-dominated. Keep the
   abstract on protocol / correction / boundary, never "improves prediction."
3. **Mechanism (aromatic-routing) was n=9-refuted** — keep it out of headline
   claims; the spectral Part-A lemma is the robust mechanism statement now.
4. **T7/T8 attention carries no novelty** — Graphormer/EGT/Uni-Mol2 own it.
5. **Contact-edge is scooped by KA-GNN** — completeness rung only.

---

## 6. Venue framing (honest)

Best-fit for a controlled-negative + reusable-protocol paper is **TMLR / JCIM /
NeurIPS D&B**. **AAAI is a genuine stretch**, viable *only* if the Stage-2 QM9
DFT cell lands (deep < RF on a geometry-pure target) and we sell it as
"principled analysis + a reusable protocol," not as a model. The decision rule is
pre-registered in `shiny-hugging-glacier.md` §"预注册 QM9 决策规则". Do not
over-index on AAAI before Stage-2 reports.
