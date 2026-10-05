# Baselines-to-Run + Latest SOTA — SpecMol/ChebInject (compiled 2026-06-04)

**Purpose.** An *actionable* companion to the May-14 survey PDF
(`Molecular Property Prediction Benchmarks and 2025-2026 SOTA…`). The survey is
a literature map; this file answers two operational questions:
1. **Which baselines do I still need to RUN** (vs. only cite) for *my* datasets, ranked by
   reviewer-demand × cost?
2. **What is the newest (post-May-2026) SOTA/data** I should be aware of?

It is scoped to the audit/boundary framing (ICDE-2027 EAB / AAAI-27), where the
thesis is "when does cheap 3D injection help, and is the deep>simple gap real?" —
so the baseline list is built around **fair-comparison strength**, not chasing AUC.

---

## 0. Where you already are (canonical state, do NOT re-run)

| Dataset | Type | Already-run baselines on YOUR split |
|---|---|---|
| BBBP, BACE, ClinTox | binary/multi-task cls | V0, V2-T5, T6, T7, T8; **RF (Morgan+RDKit2D, n=30)**; Chemprop D-MPNN; Uni-Mol-direct (matched) |
| FreeSolv, ESOL, Lipo | regression (RMSE) | V0, V2-T5, T6 (partial), T7 (partial); RF; Uni-Mol-direct |
| QM9-μ, QM9-12target, QM7 | quantum reg | V0/V2-T5/T7/T8 pilots; charge-RF; (MMFF conformers — DFT cell still unmade) |

Key facts you've *already established* (don't let a baseline re-litigate these):
- RF on Morgan+RDKit2D **dominates** BACE/ESOL/Lipo and ties BBBP → fingerprint saturation is real.
- All deep variants tie within seed noise on BBBP/BACE/ClinTox at n=9.
- 3D-injection gate (T7/T8) is inert on saturated ADMET; QM9-μ is the only geometry-isolated signal, and even there charge-RF (0.845) beats deep (0.882) on the matched 1256 subset.
- **Real end-to-end Uni-Mol finetune** is the one comparator still in flight (MBZUAI, VPN/GPU-blocked) — finishing it is higher priority than any new baseline below.

---

## 1. Baselines to RUN — ranked

### Tier 1 — Reviewers WILL ask; cheap & runnable (do these next)

| # | Baseline | Why it's mandatory for your story | Repo / how to run | Cost |
|---|---|---|---|---|
| 1 | **MapLight / MapLight+GIN** | The *winning* fingerprint+GNN-fusion recipe on TDC; it is the direct competitor to your MLP-fusion design. If you claim a fusion contribution you must show you're not just a weaker MapLight. One of only ~3 models that passed the 2026 TDC reproducibility audit. | TDC leaderboard code; ECFP+Avalon+ErG+RDKit2D → CatBoost/XGB (+ optional pretrained DGL-LifeSci GIN 300-d). Runs on CPU. | Low |
| 2 | **scikit-fingerprints RF sweep + MOLTOP** | Strengthens your "simple baseline" claim beyond a single Morgan+RDKit2D RF. `scikit-fingerprints` gives 30+ fingerprints in a sklearn API; MOLTOP is hyperparameter-free, low-variance, "embarrassingly strong" GNN baseline. Shows the saturation ceiling isn't a one-fingerprint artifact. | `pip install scikit-fingerprints`; MOLTOP from ECAI-2024 repo. [skfp](https://github.com/scikit-fingerprints/scikit-fingerprints) | Low |
| 3 | **GEM / GraphMVP / Mole-BERT** (pretrained reps) | These are *the* canonical pretraining comparators every MoleculeNet reviewer expects in the table. You currently lean on reported numbers; running ≥1 (GEM is the strongest 3D-pretrain comparator) on YOUR matched split removes the "cross-split" objection. | Official checkpoints exist; frozen-embed + your LogReg head keeps it cheap and split-matched. | Med (download + featurize) |

### Tier 2 — The QM geometry battlefield (this is where 3D SOTA actually lives)

Your geometry thesis lives or dies on **QM9-μ / QM7**, and there your real
comparators are the 3D atomistic nets, *not* the ADMET GNNs. Right now you have
no proper 3D-equivariant baseline on these — that's the biggest gap.

| # | Baseline | Why | Repo | Cost |
|---|---|---|---|---|
| 4 | **SchNet + PaiNN + DimeNet++** via **SchNetPack 2.0** | The reference 3D-equivariant MAE numbers on QM9-μ/QM7. Without at least SchNet+PaiNN you cannot claim anything about "3D helps on QM9-μ" — these define the bar. PaiNN is SOTA-tier on dipole specifically. | [schnetpack](https://github.com/atomistic-machine-learning/schnetpack), built-in QM9 tutorial + configs. GPU but small models. | Med |
| 5 | **Uni-Mol2** (scaling comparator) | Your "scaling pays only for physics" narrative needs the model that proved it (+27% MAE on QM9 vs Uni-Mol). Cite always; run only if you want a same-split QM9-μ point. | arXiv 2406.14969 + official repo. | High (optional run) |

Published QM9-**dipole** MAE bar to beat / contextualize (Debye):
PaiNN/equivariant ≈ best; physics-informed-descriptor MLP **0.0231**; ALIGNN 0.0248; SphereNet 0.025; DimeNet++ 0.030; SchNet 0.033. (Plain GraphConv ~0.62 — i.e., a non-3D GNN is ~20× worse → this is your cleanest "3D genuinely matters here" evidence.)

### Tier 3 — Newest models (2025-2026); cite, run only if cheap

| Baseline | Venue | Relevance | Run? |
|---|---|---|---|
| **HimNet** (Hierarchical Interaction Message Net) | arXiv 2504.20127 → **Nat. Commun. Chem. 2026** | Newest MoleculeNet GNN SOTA (8 MoleculeNet + 3 ADMET), code public. Strong "we compared to 2026 SOTA" point. | Optional — [Hugh415/HimNet](https://github.com/Hugh415/HimNet) |
| **NovoExpert-2** | ChemRxiv 2026 (15000061 v2) | TDC-ADMET SOTA, FP+GBT ensemble — reinforces your fingerprint-saturation thesis on the *newer* benchmark. | Cite only |
| **SPECTRA** | arXiv 2511.04838 (2025) | Spectral-domain graph augmentation on the Laplacian eigenbasis — the most recent prior work that explicitly uses spectral machinery like your ChebNetII. **Must-cite** for novelty positioning. | Cite |
| **MiniMol / MolGPS** | arXiv 2404.14986 / NeurIPS 2024 | "Symmetry + multitask beats LLM-scale" — bounds your contribution claims. | Cite |

---

## 2. Latest SOTA / data developments since the May-14 survey

- **HimNet promoted to Nature Commun. Chem. (Feb 2026)** — was a preprint in the survey; now peer-reviewed, so it's a fair "current SOTA" anchor with runnable code.
- **TDC reproducibility audit (bioRxiv 10.64898/2026.02.26.708193)** confirmed: *most* TDC leaderboard tops fail re-runs; only **MapLight, MapLight+GIN, CaliciBoost** survived. → If you run a TDC point, run MapLight (it's the reproducible one) and cite the audit as your justification.
- **BOOM (NeurIPS 2025, arXiv 2505.01912)** — OOD molecular-property benchmark; already in your AAAI bib. Good "robustness beyond scaffold split" experiment if a reviewer pushes.
- **OOD-evaluation paper** *Evaluating ML Models for Molecular Property Prediction… on OOD Data* (J. Chem. Inf. Model. 2025, `acs.jcim.5c00475`) — reinforces saturation/robustness narrative.
- **QM9-dipole** active in 2025-2026: physics-informed descriptors (Research Square rs-9501441, MAE 0.0231 D) and Q-DFTNet (J. Comput. Chem. 2025) — relevant if your QM9-μ cell becomes a headline; shows even on the geometry-pure target, *descriptor* models are competitive with 3D nets (mirrors your charge-RF > deep finding!).
- **Migration target benchmarks** unchanged: TDC ADMET (22 endpoints), Polaris Hub (OOD-by-design), PharmaBench. A reviewer asking "is MoleculeNet still meaningful" is answered by adding ≥2 TDC endpoints (e.g. Caco2_Wang, BBB_Martins, CYP2D6_Veith).

---

## 3. Concrete recommended run order (next compute window)

1. **Finish the real Uni-Mol end-to-end finetune** (already queued, MBZUAI) — closes the single most-cited gap; nothing below outranks it.
2. **MapLight + scikit-fingerprints RF/MOLTOP** on BBBP/BACE/ClinTox/FreeSolv/ESOL/Lipo (CPU, fast) — hardens "simple-baseline ceiling" beyond one RF.
3. **SchNet + PaiNN** on QM9-μ and QM7 (SchNetPack, your matched DFT subset) — gives the geometry battlefield real 3D comparators; this is the experiment that makes the boundary claim defensible.
4. *(optional)* GEM frozen-embed on your matched split; HimNet on 2-3 datasets if reviewers want a 2026 SOTA row.

**Decision rule to pre-register** (avoid post-hoc spin): on QM9-μ, if your 3D-injection
variant cannot beat PaiNN/SchNet *and* charge-RF still beats your deep model, the
honest claim is "3D signal is real but cheap descriptors capture it" — which is a
stronger, more publishable boundary result than a marginal AUC win.

---

## Sources
- [MoleculeNet SOTA / ADMET 2026 search hub](https://pubs.acs.org/doi/10.1021/acs.jcim.5c00475)
- [HimNet — arXiv 2504.20127](https://arxiv.org/abs/2504.20127) · [Nat. Commun. Chem. 2026](https://www.nature.com/articles/s42004-026-01922-x) · [code](https://github.com/Hugh415/HimNet)
- [NovoExpert-2 ADMET — ChemRxiv 2026](https://chemrxiv.org/doi/pdf/10.26434/chemrxiv.15000061/v2)
- [scikit-fingerprints — arXiv 2407.13291](https://arxiv.org/html/2407.13291v3) · [repo](https://github.com/scikit-fingerprints/scikit-fingerprints)
- [SchNetPack 2.0 — repo + QM9 tutorial](https://github.com/atomistic-machine-learning/schnetpack)
- [QM9 dipole physics-informed descriptors (rs-9501441)](https://www.researchsquare.com/article/rs-9501441/v1) · [Q-DFTNet J. Comput. Chem. 2025](https://onlinelibrary.wiley.com/doi/10.1002/jcc.70206)
- [OOD evaluation — J. Chem. Inf. Model. 2025](https://pubs.acs.org/doi/10.1021/acs.jcim.5c00475)
</content>
</invoke>
