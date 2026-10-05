# T8 faithful Uni-Mol co-update — overnight status + runbook

_Built while you slept. Honest summary first: I implemented and CPU-verified the
code; I did **not** run any GPU experiments (no HPC/VPN/GPU here) and I did **not**
fabricate any results. T8 is ready to launch on HPC._

---

## TL;DR

I added **T8**, the top rung of the 3D-injection capacity ladder:

```
T5  scalar gate         64-dim pair -> 1 scalar / bond -> Laplacian
T6  per-step scalar     pair updated from node Q.K, but STILL 1 scalar to nodes
T7  pair-bias attention multi-channel pair -> nodes via attn bias, 1 block reused
                        across K steps, pair update = delta_proj(scalar logits),
                        per-head gate empirically FREEZES under contrastive loss
T8  faithful co-update  STACK of L independent layers; per-head pair->atom bias
    (NEW)               AND atom->pair update from per-head Q*K channel (not a
                        scalar); run ONCE before the Chebyshev K-loop; ReZero
                        soft-start so epoch-0 == V2-T5 bitwise.
```

**Why T8 matters for the ICDE submission (the honest framing):** three existing
controls already say the BACE/BBBP null is a *data* ceiling, not an injection
artifact (nullify >= V2-T5; random-pair ~ real pair; Uni-Mol-direct < RF). T8 is
the rung that kills the sharpest reviewer attack — *"your null is just a 64->1
bottleneck"* — by showing the result is robust even with a full multi-channel,
multi-layer, Uni-Mol-faithful pair->node path. It is also the **only** place a
*conditional positive* could still appear: geometry-driven regression
(FreeSolv/ESOL), where 3D is physically load-bearing.

**I am not promising T8 rescues a positive 3D result on BACE/BBBP.** The evidence
says it won't. Its job there is to make the negative result bullet-proof.

---

## What is DONE (code, CPU-verified)

| Item | Status |
|---|---|
| `pair_atom_coupdate.py` — `PairAtomCoUpdateLayer` + `PairAtomCoUpdateStack` | ✅ new file, 5/5 self-tests pass |
| Wired into `LH_Direct_ChebnetII_prop_v2.py` (import, `__init__` T8 branch, `forward` run-once-before-K-loop, gated node residual, tuple return) | ✅ |
| Wired into `model_gnn_pre_v2.py` (`ChebNetII_V2`/`LH_Direct_V2` kwargs, mutual-exclusion guard, `pair_dim` gating, `_encode` branch, grad diagnostics) | ✅ |
| Wired into `main_pretrain.py` (`--t8`, `--t8_num_layers/num_heads/head_dim/dropout/init_std/pair_update`, variant tag, model construction) | ✅ all 6 wirings textually verified |
| End-to-end CPU model test (construct/forward/backward on 2-mol batch, mutual-exclusion guard, logits variant) | ✅ pass |
| **Soft-start proof: T8 at init is bitwise-equal to V2-T5** | ✅ `max\|T8 − V2-T5\| = 0.00e+00` |
| Gates receive gradient at init (ReZero can start) + full co-update trains once gates open | ✅ pass |
| Batch isolation (no cross-molecule leakage) | ✅ `0.00e+00` |
| Launch script `hpc/run_t8.sh` | ✅ (bash -n clean) |

> Honesty note: the first pass of the `main_pretrain.py` argparse edits **silently
> failed** (the file uses `parser.add_argument`, the edit used `p.add_argument`) and
> the variant-tag edit missed the real `_resolve_variant` structure — which would
> have caused an `AttributeError: 'Namespace' has no attribute 't8'` on *every* run
> (the construction block referenced `args.t8` before the flag existed). Both were
> caught by the CPU smoke test and fixed; the wiring is now verified consistent.

**Design guarantees baked in and tested:**
- **Soft start (ReZero).** Each layer's node-attn / FFN / atom->pair residual is
  scaled by `tanh(gate)` with the gate init at 0, so the stack is the identity map
  at init. The co-evolved node features enter the spectral output as a **delta**
  `(x_co - x)` (which is 0 at init), NOT a separately-gated raw term — this avoids
  a chained-ReZero cold start where an outer gate at 0 would block gradient to the
  inner node gates (caught + fixed during CPU verification; inner node-gate grad
  went 0 -> 2.7 after the fix). Net: at epoch 0 T8 ≡ V2-T5 (same near-0.993 static
  gate, same Laplacian; verified `max|Δ| = 0`), with a clean gradient to *open*
  the gates only if geometry helps — unlike T7's `sigmoid(-2)` gate that started
  at 0.12 and then froze.
- **Multi-channel atom->pair.** `pair_update='qk_hadamard'` updates the pair from
  the per-head Hadamard/channel product `Q[i,h]*K[j,h] in R^d` concatenated over
  heads, **not** a scalar projection. The `logits` choice reproduces T6/T7's
  scalar-per-head update as a built-in ablation. Caveat (review): this uses the
  raw pre-softmax, pre-bias QK — it is multi-channel but NOT the full Uni-Mol
  realized-attention atom->pair map (no bias/softmax/value), so call it
  "multi-channel" rather than "fully Uni-Mol-faithful" in the paper.
- **True iteration.** A stack of `L` *independent* layers (default 4), decoupled
  from the Chebyshev K-loop (run once, before it) — this avoids the `2^K`
  recurrence blow-up that forced T7's awkward accumulation, and gives the real
  "co-evolve atom & pair" behaviour T6/T7 lack.
- **Sparse / batch-isolated.** All compute at `pair_edge_index` positions, PyG
  grouped softmax + scatter, `dim_size=N`. No dense `[B,N,N,D]`.

## What is NOT done (be aware)

- **No GPU runs.** Every number that would go in the paper still has to come off
  HPC. Nothing fabricated.
- **Table aggregation for T8 is not wired** into `paper/make_tables.py` /
  `paper/aggregate_regression_seeds.py`. I left those alone on purpose: (a) your
  parallel finetune session may be touching results/tables and I didn't want a
  collision; (b) I can't test an aggregator against a per-seed JSON schema that
  doesn't exist yet. Once the first real `*_t8_seed*.json` lands, wiring it in is
  a ~15-line additive mirror of the `t7` path — I'll do it then (or see the
  "collect" note below).
- **ESOL/Lipo data-root name is ambiguous** on disk (`down_task_esol_unimol_v2`
  vs `down_task_esol_v2`). The script defaults to `*_unimol_v2`; override with
  `DATA_PATH=...` if your V2-T5 ESOL cell used the other root. They MUST match.

## Footprint (so you can review/trust the diff)

Working tree now contains **only** the T8 deliverable on top of your parallel
session's existing changes:
- Modified: `LH_Direct_ChebnetII_prop_v2.py`, `model_gnn_pre_v2.py`,
  `main_pretrain.py` (additive T8 branches; existing T5/T6/T7/V0 paths untouched).
- New: `pair_atom_coupdate.py`, `hpc/run_t8.sh`, this file.
- **Nothing committed, nothing pushed.**

I also **reverted** some orphaned artifacts from earlier in the session (a
pre-pivot frozen-probe n=9 migration of `bbbp_all_results.json`,
`freesolv_*_results.json`, `paper/tables/*.tex`, and the
`aggregate_regression_seeds.py` SEEDS line) back to HEAD, since they were not part
of T8 and risked colliding with your finetune work. If you actually wanted that
n=9 frozen-probe migration kept, it's recoverable — tell me and I'll redo it
cleanly.

---

## How to run on HPC (after VPN + GPU node)

> **Post-review fixes (2026-06-01)** — the launcher no longer passes the bogus
> `--out` flag (it crashed every job); classification T8 now routes through the
> canonical `hpc/run_dataset_training.sbatch` (stdout->collector, matched env).
> `qk_outer` was renamed `qk_hadamard` (it's a Hadamard/channel product, not an
> outer product; `qk_outer` still works as a deprecated alias). **H3:** at the
> default `bias_init=5.0` the pair side gets ~zero gradient (saturated sigmoid),
> so the qk_hadamard-vs-logits contrast barely trains — run with `BIAS_INIT=1.0`
> and check the diagnostics. Regression T8 is intentionally NOT wired yet (H2).

From repo root on the HPC **login node** (VPN up); `run_t8.sh` submits the sbatch.

### Path B (PRIMARY — defends the negative claim): ladder robustness on BACE+BBBP
Show that going from a 1-scalar gate all the way to a multi-layer co-update does
**not** move BACE/BBBP off the V0/RF ceiling.

```bash
# multi-channel T8 (qk_hadamard), BACE + BBBP, seeds 9/19/29, default bias=5
bash hpc/run_t8.sh

# H3: escape the pair_to_edge_weight sigmoid saturation so the pair side can learn
BIAS_INIT=1.0 bash hpc/run_t8.sh

# scalar-logit ablation (isolates the multi-channel atom->pair contribution)
PAIR_UPDATE=logits bash hpc/run_t8.sh
BIAS_INIT=1.0 PAIR_UPDATE=logits bash hpc/run_t8.sh
```
The other rungs (T5/T6/T7/nullify/random-pair) are already in your existing
result JSONs — no re-run needed.

### Path A (UPSIDE bet — FreeSolv/ESOL): NOT wired yet (H2)
`run_t8.sh` deliberately **fails loudly** for regression tasks. Before running it:
1. add `--t8` (+ `--t8_*`) to `run_regression_v2t5.py`, mirroring its `--t7` path;
2. route regression T8 through that runner (it writes per-seed JSON via
   `--results-path`), NOT through `main_pretrain.py`;
3. extend `paper/aggregate_regression_seeds.py` `VARIANTS`/`_names` with `t8`.
This keeps regression T8 on the SAME harness as the V2-T5/T7 cells it's compared
to (CLAUDE.md fair-comparison rule).

### Manual single cell (sanity, ~4 GPU-h) — NO `--out` (main_pretrain has no such flag)
```bash
python main_pretrain.py --task bace --path down_task_bace_v2 \
  --use_v2 --t8 --t8_num_layers 4 --t8_num_heads 4 --t8_head_dim 32 \
  --t8_pair_update qk_hadamard --bias_init 1.0 --diagnostics \
  --batch_size 512 --epochs 1000 --K 10 --hid_dim 512 \
  --random_seed 9 --gpu 0 2>&1 | tee hpc/logs/bace_t8_seed9.log
```

### Verify the run is actually training the co-update (not dead)
Run with `--diagnostics`: `print_model_diagnostics` now prints a
`mode=current_t8` line per forward with the per-layer `node_gates`/`pair_gates`
(init 0; must move off 0) and `node_delta_norm`. Also
`capture_gradient_diagnostics()` returns `t8_L0_{q,k,v,out_proj,bias_proj,
pair_update,node_gate,pair_gate}`. **Watch `pair_gate`/`t8_L0_pair_update`
specifically** — at `bias_init=5` the review measured them at ~1e-10 (dead). If
they stay ~0 even at `bias_init=1.0`, the multi-channel atom->pair channel is not
learning and the qk_hadamard-vs-logits comparison is meaningless — fix before a
multi-seed run.

### Collect -> tables (once per-seed JSONs exist)
- Regression: extend `paper/aggregate_regression_seeds.py` `VARIANTS` with
  `"t8"` (+ `_names` map `"t8": f"{task}_t8_results.json"`), then it aggregates
  `hpc/results/<task>_t8_seed*.json`.
- Classification: collect `hpc/results/<task>_t8_seed*.json` into a
  `<task>_t8_bare_results.json` (mirror the t7 collect), then add a `t8` row to
  `paper/make_tables.py`.
- Ping me with one real per-seed JSON and I'll wire both in 15 min and verify the
  table regenerates with the existing cells unchanged.

---

## Pre-registered read of the result (decide BEFORE you see numbers)

- **BACE/BBBP (Path B):** expectation = T8 ties V0/V2-T5 (NO-GO for a positive).
  That is the *intended* outcome — it makes the negative claim robust across the
  full injection ladder. Report it as ladder-robustness, not failure.
- **FreeSolv/ESOL (Path A):** the only place a positive is plausible. Gate
  (mirror the T7 hard gate in `reviewer_risk_register.md`): only add a T8 method
  paragraph / abstract bullet if it beats V2-T5 **and** beats its own `logits`
  ablation at n>=3. Otherwise T8 joins the ladder as further robustness evidence.
- If `qk_hadamard` > `logits` on regression, that isolates a genuine
  *multi-channel atom->pair* contribution (not just capacity). That's the cleanest
  positive the data could give you.
