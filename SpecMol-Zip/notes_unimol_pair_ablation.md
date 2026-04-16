# BACE Uni-Mol Pair Ablation Notes

- Variants:
  - `V0`: pure 2D bond graph.
  - `V0-ctrl`: all-pairs graph with `edge_weight_3d = 1`, used to isolate the effect of dense all-pairs connectivity alone.
  - `V1`: static Uni-Mol pair from official `encoder_pair_rep`.
  - `V2-stable`: static Uni-Mol pair + train-time `PairUpdateBlock` with residual alpha and clamp for stability.
- Fixed split:
  - Source file: `unimol_out/splits/data_with_folds_scaffold_k10_seed42.csv`
  - `test_fold=0`
  - `val_fold=1`
- Strict Uni-Mol pair source:
  - `edge_weight_3d = softplus(mean_over_last_dim(encoder_pair_rep[i,j,:]))`
  - Not a hand-crafted `1 / (1 + d)` distance formula.
- Optional dependency warnings from `deepchem` about `tensorflow`, `dgl`, `transformers`, `lightning`, or `jax` are ignorable for this pipeline. They do not affect the Uni-Mol pair extraction, PyG data construction, or SpecMol forward path used in these experiments.
