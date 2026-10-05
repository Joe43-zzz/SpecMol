# ICDE 2027 R1 — Submission Packet (paste-ready)

**This is the only remaining step (step 10), and it is yours to perform** — it requires
your ICDE/CMT account credentials and identity. Everything below is paste-ready for the
portal fields. The compiled PDF is at `paper/main.pdf` (IEEEtran 2-column, **10 pages**,
under the 12-page limit; 0 LaTeX errors, 0 undefined citations, 39 references).

Deadline: **2026-06-11 17:00 PT** (12 days out as of 2026-05-29).

---

## Track
ICDE 2027 Research Track, **Experiment, Analysis & Benchmark (EAB)** category.

## Title
An Empirical Study of 3D-Pair Injection Mechanisms in Pretrained Molecular Spectral GNNs

## Authors
*(fill in — set per the blind policy; see checklist #2. PDF currently reads "Anonymous".)*

## Suggested topics / keywords
molecular property prediction; graph neural networks; spectral GNN (ChebNet II); 3D pair
representations (Uni-Mol); benchmark reliability; fingerprint baselines; reproducibility;
self-supervised pretraining.

---

## Abstract — plain text (full, ~340 words)
Pretrained graph neural networks for molecular property prediction routinely incorporate
three-dimensional structural information through pair representations, yet whether this
geometry contributes anything to downstream performance under matched evaluation has not
been systematically audited. We conduct a controlled empirical study of how a frozen
Uni-Mol pair representation can be injected into a ChebNet II spectral backbone trained
with NT-Xent contrastive pretraining, centered on a static bond-level pair-to-edge gate
(V2-T5) and contrasted with a higher-capacity pair-biased multi-head attention mechanism
(T7). We evaluate on six MoleculeNet-derived benchmarks under a matched Uni-Mol
scaffold-fold protocol with three to nine training seeds per cell: BBBP, BACE, and ClinTox
for classification, and FreeSolv, ESOL, and Lipo for regression. Across all six benchmarks,
neither mechanism reliably separates from the V0 two-dimensional baseline within seed
variance at matched split. On FreeSolv — the endpoint where a 3D-pair effect is most often
reported — V2-T5 attains 0.638 ± 0.017 RMSE, nominally below both the matched V0 baseline
(0.665 ± 0.031) and a 30-seed Random Forest baseline (0.720); however, this margin is
carried by a single seed and is not significant at n=3, while ESOL and Lipo tie V0 to
within 0.006 RMSE. Classical Random Forest with Morgan and 11 RDKit 2D descriptors
dominates not only BACE (by about 13 AUC) but also the ESOL and Lipo regression endpoints
(by 0.45 and 0.24 RMSE), so fingerprint saturation is a hard ceiling across both task
types; the only endpoints where a deep variant leads a matched Random Forest are BBBP and
FreeSolv. A mechanism-level analysis nonetheless reveals interpretable, seed-stable
structure: V2-T5 learns an aromatic-routing gate on BBBP (Pearson r = +0.66 with the
aromatic-bond indicator), and T7's per-head attention gate stays in the
near-identity-residual regime of ReZero and LayerScale across every audited checkpoint. We
characterize this as a "mechanism without metric" regime. We additionally document and
quantify a silent unidirectional-bond featurization regression in our codebase that drops
about 19% of heavy atoms from the symmetric Laplacian, shifting apparent V2-T5 BACE
performance by 7.4 AUC points, and we audit fourteen widely cited public molecular GNN
repositories for the same vulnerability. We release the full evaluation harness, dataset
processing pipelines, baseline implementations, and per-seed checkpoints.

## Abstract — plain text (trimmed, ~150 words, if the portal enforces a short limit)
We empirically audit whether injecting a frozen Uni-Mol 3D pair representation into a
ChebNet II spectral GNN improves molecular property prediction under matched evaluation. We
test a static bond-level gate (V2-T5) and a pair-biased attention variant (T7) against a 2D
baseline (V0) and strong classical controls on six MoleculeNet benchmarks under one
matched Uni-Mol scaffold-fold protocol. Neither mechanism reliably separates from V0 within
seed variance on any benchmark; a 30-seed Random Forest on Morgan + RDKit descriptors
dominates BACE (about 13 AUC), ESOL, and Lipo, while deep variants lead only on BBBP and
FreeSolv. Yet the learned mechanisms are interpretable and seed-stable — V2-T5 up-weights
aromatic bonds on BBBP (Pearson +0.66); T7's attention gate stays near the residual
identity — a "mechanism without metric" regime. We also document and cross-audit (14
repositories) a unidirectional-bond featurization bug that alone shifts apparent BACE
performance by 7.4 AUC. Full harness and per-seed artifacts released.

---

## AI-use disclosure (also in the PDF as an unnumbered section)
An AI coding assistant was used under continuous author direction for software engineering
(data pipelines, baseline runners, table generation), launching/aggregating experiments,
and drafting/copy-editing. It generated no experimental data; all numbers derive from the
released code, and the authors verified all results and take full responsibility. No AI is
listed as an author. *(Confirm wording against the ICDE 2027 CFP's exact AI policy.)*

## Reproducibility statement (EAB requirement — in the PDF)
Full harness, dataset-build pipelines (deterministic scaffold-fold, seed=42), the
`LH_Direct_V2` encoder with V2-T5/T7 flags, RF + Chemprop + Uni-Mol-embedding baselines,
per-seed result JSONs, and `paper/make_tables.py` (regenerates every table) are released.

---

## Pre-submission checklist (yours)
1. [ ] Read `paper/main.pdf` end-to-end (10pp).
2. [ ] **Blind policy**: PDF reads `\author{Anonymous}`. If ICDE R1 is double-blind and the
   repo is public/identifiable, anonymize the commit SHAs (`0292a2d8`, `909966bd`) in the
   B4 audit table (§ Featurization Control) and any "our fork" phrasing. (Say the word and
   I'll do it.)
3. [ ] Confirm AI-disclosure wording vs the ICDE 2027 CFP; decide whether to name the tool.
4. [ ] Decide on abstract length (full ~340w in PDF, or paste the ~150w version above).
5. [ ] Optional: ask me to complete the Uni-Mol-direct row for BBBP/BACE (currently 4/6).
6. [ ] Create the submission in the ICDE 2027 portal, paste title/abstract/authors/topics,
   upload `paper/main.pdf`, confirm.
7. [ ] **Submit before 2026-06-11 17:00 PT.**
