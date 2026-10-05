# 项目全局状态图（ICDE 2027 EAB）

> 目的：一眼看清 **哪些实验有用（进论文）/ 哪些是开发副产品（不进）/ 哪些还没做**。
> 截稿 R1：**2026-06-11**。最后更新：见 git。

---

## 0. 论文当前状态（一句话）
**框架已定**（FIRE-前置 + 分子作 case study）；**所有表格数字已逐格核对原始结果文件、全部正确**；编译 **0 error，17 页**。
**只剩 3 件硬事**：① 超 1 页（正文 13>12）；② Uni-Mol-FT 的 4 个假 ±（bootstrap 收口）；③ 单盲真实作者名。其余都是可选打磨。

---

## 1. ✅ 进论文、已完成、已验证（这些就是"有用"的实验）

| 实验 | 在论文哪 | 状态 |
|---|---|---|
| 6 数据集 matched-split 主表（2D-Only/SEG/PBA + RF + Chemprop + FP-only + frozen-Uni-Mol-probe + Uni-Mol-FT），BBBP/BACE/ClinTox + FreeSolv/ESOL/Lipo | Table I（分类）/ Table II（回归） | ✅ 数字已核 |
| WHEN-map（8 端点"谁赢"总表，含 QM7/QM9-µ） | Table（tab:whenmap） | ✅ |
| frozen→finetune 提升（V0-FT：BACE 0.886 / BBBP 0.860 / FreeSolv 0.602；微调后所有臂含 random 打平） | §IV-D + Table III | ✅ 数字已核 |
| QM7 + QM9-µ 几何压力测试（7 臂 geometry/architecture 分解，MMFF + **真 DFT** 坐标 + gate-fix 探针） | §V + DFT 表 | ✅ 已核；**"3D 到底有没有用"这个最关键的 deciding 实验已做完，结论=没有** |
| random-pair null 消融（BBBP：真 pair vs 同形状噪声） | §IV 消融(b) | ✅ |
| bias-init 消融 + 训练后门分布图 | §IV 消融(a) + Fig | ✅ |
| B4 单向键 loader bug 审计（7.4 AUC 摆动）+ 14 仓库横向审计 | §III + Table | ✅ |
| aromatic-routing 机制 | §IV（已**降级为 caveat**：n=9 不复现，主动撤回） | ✅ 如实处理 |

**核心结论（都站得住、不依赖待做项）**：
1. two-regime map：饱和端点 RF 不败；非饱和端点微调后深度有竞争力。
2. 便宜的"冻结 3D 注入"在所有任务上**测不到提升**（连真 DFT 几何也不行）。
3. 数据管线影响巨大：一个 loader bug = 7.4 AUC。

---

## 2. ✗ 跑了但不进论文（开发副产品 / 废弃 / 被取代）—— 不用管它们

| 东西 | 为什么不进 |
|---|---|
| **T6**（dynamic pair-node）| 开发阶梯的一步，非独立模型，主表不渲染 |
| **T9**（PairAtomCoUpdateStepStack）| exploratory，README 已标"not used in the paper" |
| APC/T8 在**分类**上的 frozen-probe | T8 只用在 QM9-µ 几何任务；分类上不报 |
| `*_DEPRECATED_pre_B4.json` | B4 修复前的旧数，已正确排除 |
| `baselines_ml_results_scaffold/unimol_fold`（5-seed）| 被 deng30 + matched（30-seed）取代 |
| `v2_t5_nullify_bace`（BACE nullify）| 论文用的是 **BBBP** random-pair 消融 |
| `spectral_readout.json` | 谱密度诊断，非头条数字 |
| bbbp/bace 的 n=9 archive、mlp_phi_stats per-seed | 中间产物，已聚合进主表 |

> 一句话：第 2 节这些**不需要你再操心**，它们要么是过程、要么已被更好的取代。

---

## 3. ⏳ 还没做 / 待收口（这才是真正剩下的事，按优先级）

| # | 待办 | 卡在哪 | 影响 |
|---|---|---|---|
| **P1** | **超页**：正文 13>12 页 → 压回 ≤12 | 需你拍：上 **M7 表压缩**（14 行 repo 审计表→2–3 句 + 完整表进 artifact，省 ~0.4 页） | **desk-reject 级,必须解** |
| **P2** | **Uni-Mol-FT 的 4 个假 ±**（BACE/FreeSolv/ESOL/Lipo std=0）→ 补真 SE | bootstrap **esol/lipo 还在 HPC 跑**；VPN/登录节点暂时连不上 | 数字层面唯一没收口的 |
| **P3** | **单盲真实作者名 + 单位**（现在是 `Anonymous`）| 需你给真名；AI 声明并入 Acknowledgment 段 | 投稿前必填 |
| **P4** | **2 个新实验**（去重/泄漏审计 + scaffold-vs-random RF）已写好脚本、待提交 | VPN/登录节点连不上 | **加分项,非必须**；为"data-engineering 主线"补硬证据 |
| **P5** | **artifact 仓库 push** + 把真实 URL 填进 §Reproducibility | 你建空 public repo 后照命令清单 push（已给）| EAB 强制项 |

> P4 的两个实验：**做了能让"数据管线影响大"这条主线更硬**，但**不做也不影响现有结论**。是否做取决于你要不要把论文更往 data-engineering 方向推。

---

## 4. 外部依赖（不在我手里、等你/等恢复）
- **VPN / HPC 登录节点**：现在 pre-auth 被拒（疑似限流/抽风）。恢复后我立刻：收 bootstrap → 做 P2；提交 P4 两个实验。
- **真实作者名 / 单位**（P3）。
- **建空 public GitHub repo + push**（P5）。

---

## 5. 我建议的下一步顺序（你只需点头）
1. **现在能做、不依赖 VPN**：P1 超页（上 M7 表压缩）—— 这是最该先解的硬伤。**说"压"我就做。**
2. **VPN 一通**：P2（bootstrap 收口 B6）+ P4（提交两实验）一起。
3. **你有空时**：P3 给真名、P5 建 repo。

> 没在这张表上的 = 已完成或不用做。如果你担心某个具体实验"是不是漏了"，把名字告诉我，我对着第 1/2 节给你定位。
