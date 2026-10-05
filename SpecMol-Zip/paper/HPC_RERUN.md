# HPC Post-B4 重跑指南（BACE + FreeSolv）

目标：让 paper 三个 benchmark（BBBP / BACE / FreeSolv）都用 post-B4 数据。
BBBP 已 done（HPC 上 2026-05-13 完成），本指南覆盖 BACE 和 FreeSolv。

| 数据集 | 数据目录 | 训练入口 | 结果产出 |
|--------|----------|---------|---------|
| BACE | `down_task_bace_v2/`, `down_task_bace_unifold/` | `hpc/submit_dataset_all.sh bace` | `hpc/results/bace_all_results.json` |
| FreeSolv | `down_task_freesolv_2d/`, `down_task_freesolv_v2/` | `hpc/submit_freesolv_all.sh` | `hpc/results/freesolv_{v0,v2t5}_seed*.json` |

## 前置检查

```bash
ssh mbzuai-hpc                          # 先连 FortiClient → MBZUAI-VPN
cd ~/zhoutianyang/SpecMol/SpecMol-Zip
git pull                                # 拉本地最新
conda activate specmol
squeue -u $USER                         # 看现有作业
```

按 `CLAUDE.md` 的 HPC 规则：Slurm 自动管 GPU 可见性，**不手动 export
CUDA_VISIBLE_DEVICES**，sbatch 用 `--gres=gpu:1` 即可，训练脚本传 `--gpu 0`。

---

## A. BACE 流程

### A.1 数据重生成（必跑一次）

旧 `down_task_v2/` 是 pre-B4 archive，新数据按 CLAUDE.md 约定写到 `down_task_bace_v2/`
和 `down_task_bace_unifold/`。计算节点上跑，不能在 lo-02 登录节点：

```bash
srun --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=20G --time=01:00:00 \
     --pty bash -lc '
       cd ~/zhoutianyang/SpecMol/SpecMol-Zip &&
       conda activate specmol &&
       TASK=bace bash hpc/run_dataset_pipeline.sh prep
     '
```

完成后：
- `down_task_bace_v2/processed/bace_tri_{9,19,29}_*.pt`
- `down_task_bace_unifold/processed/bace_tri_{9,19,29}_*.pt`

### A.2 三 variant × 3 seeds 训练（批量提交）

```bash
bash hpc/submit_dataset_all.sh bace     # variants 默认 "v0 v2 t6"，seeds 默认 "9 19 29"
```

总计 9 个作业，QOS 上限 4 h（默认就是这个）。监控：

```bash
squeue -u $USER
tail -f hpc/logs/bace/v2t5_seed9_*.out
```

### A.3 结果收集

```bash
python hpc/collect_results.py \
    --task bace \
    --log-dir hpc/logs/bace \
    --output hpc/results/bace_all_results.json
```

把 JSON 拷回本地（或 git push），本地跑 `python paper/make_tables.py`，
`tables/main_results.tex` 自动含 BACE post-B4 数字。

---

## B. FreeSolv 流程

### B.1 数据重生成

FreeSolv 用 RDKit 3D + GBF 而非 Uni-Mol pair_repr，所以**不**通过 `run_dataset_pipeline.sh`，
而是 `prepare_freesolv_data.py`。本地（Windows）已重生过；HPC 上也得重生一次：

```bash
srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=00:30:00 \
     --pty bash -lc '
       cd ~/zhoutianyang/SpecMol/SpecMol-Zip &&
       conda activate specmol &&
       python prepare_freesolv_data.py
     '
```

完成后：
- `down_task_freesolv_2d/processed/freesolv_{train,valid,test,all}.pt`
- `down_task_freesolv_v2/processed/freesolv_{train,valid,test,all}.pt`

### B.2 baseline + V2-T5 × 3 seeds 训练

```bash
bash hpc/submit_freesolv_all.sh         # variants 默认 "baseline v2t5"，seeds 默认 "9 19 29"
```

共 6 个作业，每个写到独立的 per-seed JSON：
- `hpc/results/freesolv_v0_seed{9,19,29}.json`
- `hpc/results/freesolv_v2t5_seed{9,19,29}.json`

### B.3 结果聚合

跑完所有 seeds 后**在本地**（git pull 把 hpc/results 拉回来后）跑：

```bash
python paper/aggregate_freesolv_seeds.py
# 产出 freesolv_baseline_results.json + freesolv_v2t5_results.json
python paper/make_tables.py             # 更新表格
```

---

## Troubleshooting

- **`QOSMaxWallDurationPerJobLimit`** — TIME 别超 4 h（默认就是）。
- **`QOSMaxMemoryPerUser`** — 全员 48 G 上限，单 job 默认 40 G，可能要错峰提交。
- **作业 PD 卡住** — `scontrol show job <jobid>` 看 Reason 列。
- **`collect_results.py` 报缺日志** — 检查 `hpc/logs/bace/*.err` 看是不是某 seed 失败。
- **`aggregate_freesolv_seeds.py` 报 missing seed** — 看对应 sbatch 的 .err / .out。

---

## V0 已有 unifold 结果可复用？

`SpecMol-Zip/baseline_unifold_results.json` 的 V0 数据**不受 B4 影响**（pretrain 损失
平坦在 18.6665，B4 fix 只改图分支而 V0 fp 占比大）。但严格 fair-comparison 仍建议
重跑——它的数据根目录是 `down_task_unifold/` 而非新约定的 `down_task_bace_unifold/`。
如果时间紧可以先把 V2-T5 / T6 跑起来，V0 留用旧的，在论文里说明。
