# Phase B runbook — HPC empirical strengthening (fire on VPN reconnect)

Prereq: FortiClient → MBZUAI-VPN connected, `ssh mbzuai-hpc` resolves.
Rules: GPU via sbatch only; 1-GPU per-user quota; CPU jobs on `-p ws-ia`; don't
push / don't merge main. Run order = priority (B1→B4); B5 is stretch.
Budget: ~1 GPU × ~12 days ≈ ~280 GPU-h — full set won't fit; stop-and-document
per Phase-C if time runs out.

UNIMOL env (for any Uni-Mol step):
```
export UNIMOL_CKPT=$HOME/zhoutianyang/SpecMol/SpecMol-Zip/unimol_weights/mol_pre_all_h_220816.pt
export UNIMOL_DICT=$HOME/miniconda3/envs/specmol/lib/python3.10/site-packages/unimol_tools/weights/mol.dict.txt
```

---

## B1 — Mechanism audit at n=9 (cheapest, decides whether the mechanism claim survives)
1. Check if BBBP V2-T5 seed {39,49,59,69,79,89} checkpoints were saved:
   `ssh mbzuai-hpc 'ls ~/zhoutianyang/SpecMol/SpecMol-Zip/pkl/ | grep -i bbbp'`
2. If saved → re-dump gate stats per seed (no retrain); else retrain those 6 with the flag:
   `TASK=bbbp; for s in 39 49 59 69 79 89; do EXTRA_ARGS="--dump_gate_stats" sbatch hpc/run_dataset_training.sbatch bbbp v2 $s; done`
3. `python paper/analyze_gate_per_edge.py` (+ `summarize_gate_audit.py`) → Pearson(gate, aromatic) for all 9 seeds.
4. **Decision:** holds across 9 → upgrade the mechanism from "secondary observation" toward a finding; does NOT hold → drop it entirely (already demoted in `main.tex`, so this is a clean removal).

## B2 — bbbp/bace Uni-Mol-direct + matched RF → 6/6 (cheap, CPU)
1. **A4 first** — find the exact bbbp/bace fold split:
   `ssh mbzuai-hpc 'ls -R ~/zhoutianyang/SpecMol/SpecMol-Zip/unimol_out_bbbp/splits ~/.../unimol_out_bace/splits'`
   (if no fold CSV, extract test SMILES from `down_task_bbbp*/processed/*.pt` / `down_task_bace_v2/processed/*.pt`).
2. Add `bbbp`/`bace` entries to `baselines_matched.py:DATASETS` pointing at that split (kind="fold").
3. Run on CPU: `sbatch -p ws-ia --cpus-per-task 16 --mem 32G --time 02:00:00 --wrap="bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate specmol && export CUDA_VISIBLE_DEVICES= && <UNIMOL env> && cd ~/.../SpecMol-Zip && python baselines_matched.py --datasets bbbp bace --out baselines_matched_bbbp_bace.json'"`
4. Merge into `baselines_matched_results.json` (or extend make_tables to read both), then `make_tables.py`.

## B3 — FreeSolv V0+V2-T5 to n=9 (settles the single-seed edge)
```
for s in 39 49 59 69 79 89; do TASK=freesolv sbatch hpc/run_regression_v2t5.sbatch $s; done     # V2-T5
# V0: use the V0 regression runner (run_regression_v0 / --t6=0 V0 path) for the same 6 seeds
```
NOTE: `paper/aggregate_regression_seeds.py` hardcodes `SEEDS=(9,19,29)` — **extend to the 9-seed list** before aggregating, then `make_tables.py`.

## B4 — Remaining deep cells toward n=9 (as budget allows; BACE+ClinTox first)
```
SEEDS="39 49 59 69 79 89" bash hpc/submit_dataset_all.sh bace "v0 v2"
SEEDS="39 49 59 69 79 89" bash hpc/submit_dataset_all.sh clintox "v0 v2"
# esol/lipo via run_regression_v2t5.sbatch for the 6 new seeds (+ extend aggregate SEEDS)
```
Collect: `python hpc/collect_results.py --task <t> --log-dir hpc/logs/<t> --output hpc/results/<t>_all_results.json` then scp + `make_tables.py`.

## B5 — Stretch: Tox21 (chunked pair-extraction to fit 48G) + HIV (~41k). Likely stays a documented limitation.

---

## A6 cleanup (do anytime, no HPC) — stale internal docs
- `paper/reviewer_risk_register.md` and `paper/result_provenance_checklist.md` still carry an old FreeSolv V0 `0.760` and a "FreeSolv V2-T5 uses Uni-Mol" line that contradicts the method (FreeSolv V2-T5 = GBF surrogate). Reconcile to the committed numbers (V0 0.665, RF 0.720 matched, V2-T5 0.638-GBF) or delete the stale lines. These are internal notes, not in the paper.

## Phase C after any B batch
`collect_results.py` → `aggregate_regression_seeds.py` (mind the SEEDS list) → `make_tables.py` → update `main.tex` n-claims/captions → `latexmk` (≤12pp, 0 undefined) → consistency grep → commit (no push).
