# T6: Bidirectional Pair-Node Update — BACE Experiment

## Design

T6 extends the V2-T5 static-pair model with **dynamic pair representation updates** during Chebyshev spectral convolution.

### T5 (baseline): Static pair
- Uni-Mol pair representations are loaded once
- `PairToEdgeWeight` maps pair_repr → edge_weight (scalar per bond)
- Edge weights are fixed throughout all K Chebyshev steps

### T6: Dynamic pair-node update
- At each Chebyshev step k, `NodeToPairUpdate` refines pair_repr using node features:
  - Q = q_proj(T_k), K = k_proj(T_k)  (linear projections of Cheby state)
  - qk = (Q[src] * K[dst]).sum(-1) * scaling  (attention-like score)
  - delta = out_proj(qk)  (project scalar → pair_dim)
  - pair_repr_edge += delta  (residual update)
- After update, `PairToEdgeWeight` recomputes edge_weight from updated pair_repr
- **Zero-init**: out_proj initialized to zero → T6 starts identical to T5

### Key question
Does allowing node features to dynamically refine pair representations improve downstream BACE classification over static pair weights?

## Parameters

```bash
python main_pretrain.py \
    --task bace --path down_task_v2 \
    --batch_size 512 --epochs 1000 --gpu 0 \
    --hid_dim 512 --K 10 --patience 100 \
    --random_seed {9,19,29} \
    --use_v2 --t6
```

## Comparison Table (BACE, Uni-Mol scaffold fold split)

| Variant | Grand Mean AUC | Std | Notes |
|---------|---------------|-----|-------|
| V0 Baseline | 0.797 | 0.085 | 2D-only, no Uni-Mol |
| FP-only | 0.846 | 0.010 | Fingerprint MLP only |
| V2-T5 (static pair) | 0.837 | 0.009 | Static Uni-Mol pair weights |
| **T6 (dynamic pair)** | **TBD** | **TBD** | Dynamic pair-node update |

## Results

_Pending — training in progress on MBZUAI HPC (2026-05-12)._

Run `python collect_t6_results.py` after training completes to populate `t6_bace_results.json`.

## Files

- `node_to_pair_update.py` — NodeToPairUpdate module
- `model_gnn_pre_v2.py` — LH_Direct_V2 (T5/T6 modes)
- `LH_Direct_ChebnetII_prop_v2.py` — ChebnetII_prop_V2 (per-step pair update)
- `pair_to_edge_weight.py` — PairToEdgeWeight (pair_repr → edge_weight)
- `collect_t6_results.py` — Result collection script
