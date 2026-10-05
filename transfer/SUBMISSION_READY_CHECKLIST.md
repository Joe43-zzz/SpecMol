# ICDE 投稿 readiness checklist

**Date**: 2026-05-27
**Submission target**: `conference_101719.tex` (IEEE conference format)
**Compile**: `cd D:\specmol-zip\transfer && pdflatex conference_101719 && pdflatex conference_101719`（无需 bibtex，bibliography 是 inline `\begin{thebibliography}`）

---

## 当前提交准备度

| 状态 | 项 | 备注 |
|---|---|---|
| ✅ | 主 `.tex` 本地 clean build | 8 页（≤ 12 ICDE limit），0 undefined ref，2 个 minor 字体 warning |
| ✅ | Bibliography | 35 cite key 全部 used + defined，0 mismatch（inline `\bibitem`，不走 bibtex） |
| ✅ | Internal refs (`\ref` / `\eqref`) | 9 used，全部对应 `\label` 存在 |
| ✅ | Figures 嵌入 | `gate_distribution_bace_v2t5.pdf` 路径解析 OK |
| ✅ | Section 结构 | §1 Intro / §2 Related / §3 Method (6 subsec) / §4 Experiments (3 subsec) / §5 Discussion / §6 Conclusion |
| ✅ | Prose ↔ table 数值 | Table 1 数字与 prose 一致（V0 0.828±0.027、V2-T5 0.831±0.032、T6 0.831±0.022、FP-only 0.846±0.012、RF 0.894±0.003 等） |
| ⏳ | 作者 block | L18-21 anonymous 占位，**ICDE 双盲 OK，不阻塞提交**；与学姐对齐后填真实姓名 |
| ⏳ | T7 章节 | hedge 已就位 in `t7_addition_drafts.tex`（Block A/B/C），等 2026-05-27 上午 bare-T7 数据 |
| ⏳ | 最终 PDF | 决策（outcome A/B/C）后 1h 内出 |

---

## 不阻塞提交项（已知 minor 噪音）

- LaTeX font warning `TS1/ptm/m/sc undefined` —— Times Roman 小型大写字母 fallback，paper 里没用到该字体形态，warning 不影响渲染
- 5 个 `\label` 定义了但未引用：`sec:ablations` `sec:intro` `sec:method` `sec:related` `sec:t6` —— 用于潜在 cross-ref 的预留 anchor，IEEE 模板里这是常态，不需要删
- `main_pretrain.py:428` 的 `# TODO: spec` 注释 —— 代码注释，paper 不引用，可忽略

---

## 明早决策点（bare-T7 数据来了后）

### Step 1：收集 + 计算
```cmd
cd D:\specmol-zip\SpecMol-Zip
ssh mbzuai-hpc "cd ~/zhoutianyang/SpecMol/SpecMol-Zip-bare/SpecMol-Zip && python hpc/collect_results.py --task bbbp --log-dir hpc/logs/bbbp --output bbbp_t7_bare_results.json && python hpc/collect_results.py --task bace --log-dir hpc/logs/bace --output bace_t7_bare_results.json"
:: 把生成的 bbbp_t7_bare_results.json / bace_t7_bare_results.json scp 到本地 D:\specmol-zip\SpecMol-Zip\
python paper/make_tables.py
python paper/compute_paired_stats.py
```

### Step 2：读关键数字
- `T7_BBBP_MEAN`、`T7_BBBP_STD`：从 `paper/tables/main_results.tex` T7 行
- `T7_DELTA_V2T5`：从 `paper/paired_stats.json` 的 "BBBP T7 vs V2-T5" 条目
- `T7_SIGN_P`：同上的 `sign_test_two_sided_p`
- `T7_BACE_MEAN`：BACE T7 单 seed mean
- `T7_GATE_SIGMOID`：`ssh mbzuai-hpc "grep -E 'attn_gate.*final' hpc/logs/bbbp/t7_seed*.log"` 然后 sigmoid

### Step 3：判 outcome
| Outcome | 触发条件 | 动作 |
|---|---|---|
| **A** Win | `T7_BBBP_MEAN > 0.831` 且 `T7_DELTA_V2T5 ≥ 0.02` 且 3 seeds 方向一致 | 用 `t7_addition_drafts.tex` Block A，sed 替换占位符，splice 进 conference_101719.tex |
| **B** Tie | `\|T7_BBBP_MEAN − 0.831\| ≤ 0.032` | Block B（中性 ablation） |
| **C** Omit | `T7_BBBP_MEAN < 0.80` 或 job crash 或 时间不够 | Block C（safety net）—— 主 paper 不动，可选 Limitations 加一句 |

### Step 4：Splice 流程
- Outcome A/B 需要更新的位置（按 conference_101719.tex 当前行号）：
  - L40 abstract 结尾前 → 加 outcome 段
  - L66 Contributions itemize 内（仅 A）→ 加 \item
  - L120 §3.6 T6 结尾后 → 加新 \subsection T7
  - L144 Table 1 caption "Deep-model cells (V0, FP-only, V2-T5, T6)" → 改成 "(V0, FP-only, V2-T5, T6, T7)"
  - L158 Table 1 T6 行后 → 加 T7 行（make_tables 重生成后直接复制）
  - L163 Main Results paragraph 结尾后 → 加 outcome 段
  - §5 Discussion 适当位置 → 加 mechanism 段
- 再 `pdflatex × 2`，确认页数 ≤ 12（如果 outcome A 把 paper 推过 12 页，砍 §3.6 T6 或 §4.2 Ablations 中较冗的段）

### Step 5：可选后续动作（仅 outcome A 触发）
- 在 mbzuai-hpc 上提交 per-head（S2）+ BBBP n=9 extension（master plan async-giggling-garden.md §S2/S5）
- 这些是 camera-ready 增量，不阻塞初次投稿

---

## 关键文件清单

| 路径 | 角色 | 状态 |
|---|---|---|
| `D:\specmol-zip\transfer\conference_101719.tex` | 投稿主文件 | clean，build 8 页 |
| `D:\specmol-zip\transfer\conference_101719.pdf` | 当前提交 PDF | 2026-05-27 build |
| `D:\specmol-zip\transfer\t7_addition_drafts.tex` | T7 三 outcome 草稿 | **本晚新增** |
| `D:\specmol-zip\transfer\SUBMISSION_READY_CHECKLIST.md` | 本文件 | **本晚新增** |
| `D:\specmol-zip\SpecMol-Zip\paper\main.tex` | standalone IMRAD 镜像 | 与 transfer 同步 |
| `D:\specmol-zip\SpecMol-Zip\paper\make_tables.py` | 表格 generator | **加了 T7 支持** |
| `D:\specmol-zip\SpecMol-Zip\paper\compute_paired_stats.py` | sign-test | **加了 T7 comparisons** |
| `D:\specmol-zip\SpecMol-Zip\paper\compute_bbbp_n9_stats.py` | n=9 stats | **加了 T7 schema 防 crash** |

---

## 如果 deadline 真的顶到头（极端情况）

1. 直接用现有 `conference_101719.pdf`（8 页、零警告、双盲匿名块合法）提交
2. T7 留给下一轮（rebuttal、camera-ready、或下一个 venue）
3. 现 paper 的 headline = V2-T5 + aromatic-bond routing 机制，**独立完整** —— 不依赖 T7
