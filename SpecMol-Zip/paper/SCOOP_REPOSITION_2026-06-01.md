# SpecMol 文献深调研:Scoop 风险 + 重定位（2026-06-01）

> 方法：11-agent workflow，对 5 篇疑似撞车的先验工作**联网抓取真实方法+结论**，
> 对每条"X 覆盖了本篇 Y"做**对抗交叉验证**（不轻信标题），再补 thread-2 注入新颖性、
> thread-1 谱图GNN、thread-3 负结果venue。配套产物：`specmol_litreview.ris`（导 Zotero）、
> `specmol_litreview_new.bib`（给 LaTeX）。

---

## 一句话结论

**论文没死，但它是一篇"有据可查的负结果 / matched-protocol 审计"，不是方法、也不是发现。**
- C2（指纹饱和）**已被 Xia2023 这篇更全面的工作抢先**（12 模型 × 14 数据集），你只能当"复现+增量"，必须引用、不能当头条。
- C1（注入机制都不分离）**精神被抢、但具体器械没人做过**——这是你真正的护城河。
- 必须**改名**（arXiv 上有真实同名 SpecMol）。
- **ICDE 基本不合适**，应转 **JCIM / TMLR / NeurIPS D&B**。

---

## 1. SCOOP 矩阵（对抗交叉验证后的最终裁决）

| 贡献 | 谁覆盖 | 强度（核验后）| 你**还独占**什么 |
|---|---|---|---|
| **C1** 注入机制都不与 2D-only V0 分离（matched 精确 split，n=3..9）| Xia2023（deep≤non-deep）、Praski2025（含 Uni-Mol 冻结嵌入≤ECFP）；**Hamakawa2025 反对**（正确构象+微调的 Uni-Mol 确实赢 2D）| **partial** —"deep/3D 不赢"的精神已被走烂；但"冻结 Uni-Mol pair → 每边 gate/bias/co-update 注入 ChebNetII 谱滤波、对 V0 在模型自身 fold split 上审计"这个**具体器械没人做过** | **注入能力阶梯**（V2-T5 静态门 / T7 pair-bias 注意力 / T8 co-update）作为受控负结果；matched 精确 split 对比；**n=3→n=9 崩塌**揭示"赢"是采样运气。这是 C1 的护城河。|
| **C2** Morgan+RDKit2D RF 碾压 BACE/ESOL/Lipo，连冻结 Uni-Mol 嵌入+RF 都够不到 | **Xia2023 finding(iii)"树+指纹最好" = 标准母本**；Praski2025（Uni-Mol 嵌入+RF≤ECFP，仅分类）；Deng2023（已引）；vanTilborg2022 | **substantially-overlaps**（核验确认；命题是 Xia(iii) 的更窄实例）**但非"完全抢光"** | (a)**回归那一半**（ESOL 0.45 / Lipo 0.24 RMSE，Praski 零回归）；(b)**matched 精确 split** 钉在模型自身 Uni-Mol fold；(c)**冻结 Uni-Mol 嵌入+RF 对照**；(d)饱和**穿透注入路径**仍成立。你 §RW 已诚实当复现——保持。|
| **C3** 可再生 matched-protocol harness（谱图GNN对比预训练+线性探针，6 任务）| Praski2025（scikit-fingerprints harness）、vanTilborg（MoleculeACE）、Hamakawa（自有repo）| **adjacent** —matched-protocol harness 多家有，但**没有这套代码/协议** | 绑定 ChebNetII 双路+NT-Xent+冻结探针+注入阶梯 的特定 harness。真实但**增量**——harness 当不了头条。|
| **机制**（aromatic-routing 自我证伪的"mechanism without metric" + 单向键 featurization bug 取证）| **无人**。Xia 的 activity-cliff "why" 只是主题相关；Hamakawa 是行为对照非门可解释性 | **no-overlap / 完全独占** | 全是你的。自我证伪的可解释性脆弱性 + featurization bug 审计**没人碰**——但被你框成 secondary/caveated 且是负面/警示，**扛不起头条**。|

**净读数**：C2 ≈ 实质被抢（诚实归功后剩部分增量）；C1 精神被抢但器械是新的负结果；C3 邻接/增量；机制独占但次要。

---

## 2. 改名（撞名已确认 🔴）

`shen2025specmol`（**arXiv:2509.21861**，后改名 **MolSpectLLM**）是一个真实、公开的 Qwen2.5-7B 光谱（NMR/IR/MS）基础模型——**纯撞名、零内容重叠**，但它是 arXiv 上活的 "SpecMol"。双盲送审 + 检索都会撞，**必须改名**。你的"spectral"是图拉普拉斯/ChebNetII,不是分析光谱，所以**去掉 "Spec"**。

候选（审计风、不暗示方法创新）：
1. **ChebInject** — *"Auditing Frozen 3D-Pair Injection into Spectral GNNs for Molecular Property Prediction"*（推荐）
2. **SpecProbe-Mol** — 强调 linear-probe + matched-protocol 审计
3. **GeoGate-Audit / InjectBench** — *"When Frozen 3D Geometry Doesn't Help: A Matched-Protocol Audit of Pair-Representation Injection into ChebNetII"*（标题即论点）

---

## 3. 重定位

**Venue（按契合度）：**
- **JCIM**（最契合——Xia/Hamakawa/vanTilborg 这类领域审计都发这；接受严谨负结果+released harness）
- **TMLR**（接受扎实的负结果/复现；无新颖性门槛，只看"claim 是否被支撑"——审计论文理想地）
- **NeurIPS Datasets & Benchmarks**（若能把 C3 harness + matched-protocol 当 artifact 卖）
- 地板：ML4Molecules / AI4Science workshop
- ❌ **不要投 ICDE**（cheminformatics 负结果审计，非数据系统贡献，desk-reject/scope 风险高）

**幸存的一句话 thesis：**
> *在六个 MoleculeNet 任务上做 matched-protocol 审计：把冻结的 Uni-Mol 3D pair 表示注入谱图（ChebNetII）GNN——无论作为每边 gate、注意力 bias、还是 atom↔pair co-update——都不能在 seed 方差内与 2D-only 基线可靠分离（n=3 的表观"赢"在 n=9 崩塌）；同时 matched 精确 split 的 Morgan+RDKit2D Random Forest 在回归和 BACE 端点上碾压，证实是指纹饱和、而非 3D 几何，构成这些 benchmark 上的天花板。*

---

## 4. 后续方向：「真正解冻 3D 源 + 几何任务」值不值？

**有条件值——而且是唯一能把论文从"负结果"升级成"正向边界发现"的下一步，但要换任务集、不只是解冻。**

- **Hamakawa2025 是决定性信号**：*微调的、正确构象的* Uni-Mol **确实**赢 2D，错误构象对照会让它崩——即几何**在 (a) 端到端可训练 AND (b) 性质对构象敏感（QM 偶极/HOMO-LUMO、对映选择性、熔点）时被利用**。你的 null 很可能是 **(i) 冻结 3D 源 + (ii) 用了指纹饱和的 MoleculeNet ADMET 任务** 这两个一起造成的——恰好是几何没法发挥的区间。
- **所以值得做的 = 解冻 + 换到对构象敏感的任务**（QM9 / QM 风格 / Hamakawa 型端点），**不是**在同一批饱和 MoleculeNet 上解冻（那大概率复现 null，RF 还是赢）。
- **新颖性检查**：这会把论文转向正向对比边界——*"冻结 vs 可训练的 3D 注入进谱图 GNN，从何时开始起作用？"*——这**没被直接抢**（Hamakawa 端到端微调 Uni-Mol，没有谱图 GNN、没有冻结-vs-可训练注入阶梯；thread-2 确认谱图注入这个组合未被报道）。还能把 Hamakawa 当"几何有用"的对照锚点引用,而不是当必须吸收的反对证据。
- **裁决**：**只有换成构象敏感任务才值得做**；否则跳过，直接把审计当 TMLR/JCIM 负结果论文发。考虑算力（本机 RTX 3050 4GB；MBZUAI 1-GPU 串行），**更省事、能完稿的做法是现在就发审计（C1 领衔），把"QM 上解冻"列为 future work**,而不是卡着不投。

---

## 5. thread 检索补充结论
- **thread-1（谱图GNN-on-molecule）**：EAGCN 之后这块**很 niche**——你的 ChebNetII 骨干**没被抢**。补充 KA-GNN(2025)、DG-GCN 作上下文即可。
- **thread-2（具体注入）**：精确组合（冻结 Uni-Mol pair → ChebNetII 谱滤波的可学每边权重，NT-Xent+冻结探针）**未被报道**；但新颖性**薄且可分解**（=EAGCN 边加权 + 唯一新料"冻结3D-pair权重源"）。→ **做零方法新颖性声明**，定位审计。
- **thread-3（venue）**：JCIM/TMLR/NeurIPS-D&B 有大量同类先例（Zhang2024、Fooladi2025、Mayr2018 等都在 JCIM/ChemSci）。

---

## 产物清单
- `paper/specmol_litreview.ris` — 13 条，**导入 Zotero**（File → Import；tag 已分 thread1/2/3 + SCOOP/NAME-COLLISION/followup-anchor）
- `paper/specmol_litreview_new.bib` — 同内容给 LaTeX（已标 scoop-critical / 与 refs.bib 重复者勿重复 key）
- 本报告
