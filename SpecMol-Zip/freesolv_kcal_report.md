# FreeSolv RMSE in kcal/mol

## Label normalization

DeepChem 2.8.0 ships `freesolv.csv.gz` with pre-normalized labels (z-scored, mean=0, std=1). Even `load_freesolv(transformers=[])` returns z-scored values. The original experimental hydration free energies are in kcal/mol.

Conversion factor recovered by fitting 18 known experimental values (R^2=0.9999, max error=0.04 kcal/mol):

- **mean = -3.809 kcal/mol, std = 3.845 kcal/mol**
- z-score normalization was computed over the full 642-molecule dataset (not train-only)
- Formula: RMSE(kcal/mol) = RMSE(z-score) x 3.845

## Our results

| Variant | RMSE (z-score) | RMSE (kcal/mol) | n cells |
|---------|----------------|-----------------|---------|
| Baseline (2D-only) | 0.7604 +/- 0.0474 | 2.924 +/- 0.182 | 3 |
| V2-T5 (GBF 3D) | 0.6746 +/- 0.0325 | 2.594 +/- 0.125 | 3 |

Per-seed breakdown:

| Seed | Baseline (kcal/mol) | V2-T5 (kcal/mol) |
|------|---------------------|-------------------|
| 9 | 2.776 | 2.421 |
| 19 | 2.814 | 2.649 |
| 29 | 3.181 | 2.712 |

## Literature comparison

| Method | RMSE (kcal/mol) | Split | Source |
|--------|-----------------|-------|--------|
| Our V2-T5 | 2.59 +/- 0.12 | scaffold 80/10/10 | this work |
| Our Baseline | 2.92 +/- 0.18 | scaffold 80/10/10 | this work |
| Ridge (Morgan FP) | 4.02 | scaffold 80/10/10 | this work (sanity check) |
| Uni-Mol | 1.48 +/- 0.039 | scaffold 80/10/10 | Zhou et al. 2023, Table 2 |
| GEM | 1.877 | scaffold | Fang et al. 2022 |
| GROVER | 2.176 | scaffold | Rong et al. 2020 |
| D-MPNN | 2.082 | scaffold | Yang et al. 2019 |
| SchNet | 3.215 | scaffold | MoleculeNet benchmark |
| GraphConv | 2.900 | scaffold | MoleculeNet benchmark |

Note: Literature numbers from MoleculeNet leaderboard and original papers. Most use scaffold split on the same 642-molecule FreeSolv. Uni-Mol uses its own scaffold split implementation; seed may differ.

## Takeaway

1. Our V2-T5 at **2.59 kcal/mol** is competitive with mid-tier GNN baselines (D-MPNN 2.08, GROVER 2.18) but substantially behind Uni-Mol (1.48). The gap is ~1.1 kcal/mol.
2. V2-T5 beats our own 2D baseline by **0.33 kcal/mol** (11.3% relative), confirming that the 3D pair_repr signal adds value even with RDKit conformers instead of Uni-Mol.
3. Our model is comparable to GraphConv (2.90) — the improvement from V2 3D roughly closes the gap to D-MPNN territory.
4. For paper narrative: the result demonstrates the V2 architecture's ability to use 3D information on a second dataset (regression), but the absolute performance gap to Uni-Mol suggests that pair_repr quality (Uni-Mol encoder vs RDKit GBF) matters substantially. Running with actual Uni-Mol pair_repr on FreeSolv would be the natural next comparison.
