# SpecMol

> Spectral-Fingerprint Dual-Fusion Network for Molecular Property Prediction

**SpecMol** is an ongoing research project exploring whether a dual-fusion architecture — combining a **spectral graph encoder (ChebNetII)** with a **molecular fingerprint MLP** — can outperform single-stream GNN baselines on small-molecule property prediction tasks.

The project is part of an undergraduate research assistantship at the **School of Electrical and Electronic Engineering, Xiamen University Malaysia**, supervised by Dr. Yau Wei Chuen (Oct 2025 – present).

---

## Motivation

Most mainstream molecular GNNs (GIN, GAT, GCN) operate purely in the spatial domain and rely on message passing over local neighborhoods. They tend to underutilize **global structural signals** that spectral methods can capture cheaply.

In parallel, classical fingerprint-based MLPs (ECFP / MACCS) remain surprisingly competitive on small datasets because they encode strong chemical priors.

**SpecMol** asks: *can a spectral GNN and a fingerprint MLP be fused in a way that recovers the strengths of both, especially under data-scarce settings like BACE?*

---

## Architecture Overview

```
   SMILES ──┬── RDKit ──── Molecular Graph ──── ChebNetII Encoder ──┐
            │                                                       │
            └── ECFP / MACCS Fingerprints ───── MLP Encoder ────────┤
                                                                    │
                                                            Dual-Fusion Layer
                                                                    │
                                                          Contrastive Pretraining
                                                                    │
                                                            Downstream Head
                                                                    │
                                                              BACE / etc.
```

- **Graph branch**: ChebNetII (Chebyshev polynomial spectral filter, learnable order)
- **Fingerprint branch**: ECFP4 / MACCS → 2-layer MLP
- **Fusion**: gated concatenation (other variants under ablation)
- **Pretraining**: SimCLR-style contrastive objective on augmented molecular views
- **Downstream**: BACE binary classification (and planned extensions to BBBP / Tox21)

---

## Tech Stack

- **Frameworks**: PyTorch 2.x, PyTorch Geometric, DeepChem, RDKit
- **Compute**: AutoDL A800 GPU (remote SSH + tmux workflow)
- **Tooling**: Git, Jupyter, Claude Code / Cursor as primary AI coding agents
- **Tracking**: Weights & Biases (planned)

---

## Repository Layout

```
SpecMol/
├── data/                  # data download + preprocessing scripts (raw data .gitignored)
├── specmol/
│   ├── encoders/          # ChebNetII implementation
│   ├── fingerprints/      # ECFP / MACCS feature extraction
│   ├── fusion/            # dual-fusion modules
│   ├── pretrain/          # contrastive pretraining loops
│   └── downstream/        # BACE classifier head
├── configs/               # training configs (YAML)
├── scripts/               # train / eval entry points
├── notebooks/             # exploratory notebooks
└── README.md
```

---

## Status

This is **active research code**, not a polished release. Expect:

- frequent rebases and architectural refactors
- partial coverage of unit tests
- ablation experiments still in progress
- some modules tagged `# WIP` or `# TODO`

Reproducible scripts and a full ablation table will be released alongside the project write-up.

---

## Roadmap

- [x] Data pipeline (BACE, ZINC subset)
- [x] ChebNetII encoder integrated with PyG
- [x] Fingerprint MLP branch
- [x] Baseline GIN reproduction
- [ ] Dual-fusion gating ablation
- [ ] Contrastive pretraining at scale
- [ ] BBBP / Tox21 transfer evaluation
- [ ] Technical report draft

---

## Acknowledgements

- Supervised by Dr. Yau Wei Chuen, XMU EEE
- Compute provided by AutoDL
- Heavy use of **Claude Code** and **Cursor** for codebase refactoring and PyG / RDKit pipeline development. Evaluating **Xiaomi MiMo-V2.5-Pro** as a long-context coding agent for this workflow.

---

## License

Code released under the MIT License. See `LICENSE` for details.
Raw datasets follow their respective original licenses (MoleculeNet / DeepChem).
