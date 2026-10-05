# ChebInject (ex-"SpecMol") — AAAI-2027 Repositioning Plan

**Date:** 2026-06-02 · **Confirmed:** positioning = **P4 (hybrid boundary characterization)**; **QM9 positive experiment greenlit**; Zotero handled autonomously.
**Confidence tags:** [H] high / [M] medium / [L] low. Uncertainties listed explicitly per section.
**Honesty anchor:** do not claim anything current data refutes (3D≈random, RF fingerprint-saturation, frozen-probe artifact, n=9 collapse of BBBP/FreeSolv wins + aromatic mechanism). New experiments may *extend*, not contradict, these.

---

## ① Literature survey (verified — every entry's real method/finding was fetched; scoop verdicts adversarial)

**Structural finding that defines the whole repositioning [H]:** every pair-bias-attention 3D model (Transformer-M, Uni-Mol+, Uni-Mol2, EPT) evaluates **only on conformation-sensitive quantum/binding tasks** (QM9, PCQM4Mv2, PDBBind, OC20) — **none on MoleculeNet ADMET**; and Cremer2023 directly shows a 3D-equivariant model ≈ 2D on ADMET toxicity. The field has *implicitly* drawn the "3D helps on quantum, not ADMET" boundary but **nobody has explicitly characterized it with a controlled frozen-injection ladder + random-pair control spanning both regimes.** → that gap is the moat.

### Thread 2 — 3D / pair-bias-attention injection (mechanism precursors)
| Paper | Real method / finding | Verdict |
|---|---|---|
| **Graphormer** (Ying+, NeurIPS'21) | SPD + bond-feature **additive attention bias**; **zero 3D**, trained from scratch | adjacent-no-scoop · **must-cite T7 precursor** [H] |
| **Transformer-M** (Luo+, ICLR'23 Oral) | 2D-bias + 3D-Gaussian-distance-bias channels, **jointly trained**; SOTA QM9/PCQM4Mv2/PDBBind, **no ADMET** | actually-supports-P4 · **must-cite** [H] |
| **Uni-Mol+** (Lu+, NatComms'24) | two-track Transformer, pair→attention bias, RDKit→DFT refine, **end-to-end**; PCQM4Mv2+OC20 only | supports-P4 · **must-cite** [H] |
| **Uni-Mol2** (Ji+, NeurIPS'24) | 1.1B params, AlphaFold-style pair update, SelfAttentionPairBias; QM9/COMPAS only | supports-P4 [H] |
| **EPT** (Jiao+, '25) | E(3)-equivariant, distance+edge attention bias, end-to-end; binding+QM9 | supports-P4 [H] |
| Unified 2D&3D (Zhu+, KDD'22) | joint 2D/3D pretrain, message-passing fusion; modest ADMET gains | context [H] |
| Noisy Nodes (Godwin+, ICLR'22) | 3D coordinate-denoising aux loss; QM9/OGB | context [H] |

### Thread 1 — spectral / polynomial GNNs on molecules (backbone)
| Paper | Real method / finding | Verdict |
|---|---|---|
| **Fallani2025** (JCIM) | QM-property pretraining shifts attention to **low-frequency Laplacian eigenmodes**; 22 TDC ADMET | supports-P4 + **backs dual-path spectral story** [H] |
| Stable-ChebNet (Hariri+, '25) | ChebNet as stable dynamical system; competitive on long-range | backbone-legitimacy [H] |
| S²GNN (Geisler+, NeurIPS'24) | spatial+spectral filters, over-squashing; PCQM4Mv2(2D)/LRGB | context [H] |
| Specformer (Bo+, ICLR'23) | Transformer over full eigenspectrum; beats ChebNetII on node/graph cls (no molecular) | context — "why ChebNetII not Specformer?" [H] |
| Exploring Heterophily graph-level (Hou+, NeurIPS'25 WS) | graph-level tasks need **mixed-frequency** dynamics | supports dual low+high-pass [M] |
| DGCL (Jiang+, BiB'24) | dual GNN (GIN+GAT) contrastive + fingerprint fusion | context (closest "dual-path+FP fusion") [H] |

### Thread 3 — simple-baseline-wins / fingerprint saturation
| Paper | Real method / finding | Verdict |
|---|---|---|
| **Jiang2021** (J Cheminform) | descriptor models (SVM/XGB/RF) **outperform GNNs** on 11 MoleculeNet, random splits | supports-C2 · **must-cite** [H] |
| MOLTOP (Adamczyk+, ECAI'24) | topology-descriptor + RF, **no training/HPs**, ≈ pretrained GNNs | supports-C2 [H] |
| Adamczyk peptide-FP ('25) | count-FP + LightGBM **beat GNNs/transformers** on 132 peptide sets | supports-C2 [M] |

### Thread 4 — evaluation protocol (the P2 pillar; most contested)
| Paper | Real method / finding | Verdict |
|---|---|---|
| **MolGraphEval** (Wang+, NeurIPS'23 D&B) | frozen-vs-finetune **rank corr 0.77**; JOAO best frozen→4th finetuned; "probing fixed embeddings may not reflect downstream perf" | **partial-overlap, KEY P2 precursor** · must-cite [H] |
| **MolGPS** (Sypetkowski+, NeurIPS'24) | for **large supervised** foundation models, **probing ≥ finetuning** (Polaris 0.91 vs 0.85) | **opposes-direction · scope P2** · must-cite [H] |
| Sun2022 "Does GNN Pretraining Help?" (NeurIPS'22) | SSL gains "not always significant"; HPs/splits dominate; **only finetunes, never frozen-probe** | supports-P2/P3 · must-cite [H] |
| Pinto2025 (intermediate layers) | frozen final-layer probe under-measures by 5.4–28.6% on 22 ADMET | supports-P2 [M] |
| Kumar2022 LP-FT (ICLR'22 Oral) | finetune +ID/−OOD vs linear probe (vision/NLP) | context · scope to in-distribution [H] |
| Probing Graph Reps (Akhondzadeh+, AISTATS'23) | probing as diagnostic, not downstream gap | context [H] |
| BOOM (Antoniuk+, NeurIPS'25) | chemically-informed OOD: top model 3× ID error | context (split fragility) [H] |

### Thread 5 — where 3D actually helps (the P4 positive side)
| Paper | Real method / finding | Verdict |
|---|---|---|
| **Cremer2023** (Chem Res Tox) | E(3)-equivariant ≈ 2D GEM on Tox21/ClinTox; "3D no clear advantage" on ADMET | **supports-P4 · the most direct boundary point** · must-cite [H] |
| **QM9** (Ramakrishnan+, Sci Data'14) | DFT props for 134k mols; gold conformation-sensitive benchmark | **P4 target-dataset anchor** · must-cite [H] |
| Equiformer (Liao+, ICLR'23 Oral) | SE(3) attn; best 11/12 QM9 tasks | supports-P4 [H] |
| Pre-training via Denoising (Zaidi+, ICLR'23) | 3D denoising ≈ force-field; SOTA QM9 | supports-P4 [H] |
| GeoTMI (Kim+, NeurIPS'23) | cheap geometry usable for quantum **when trainable** | supports-P4 [H] |
| Axelrod2023 (MLST) | 3D conformer help is **data-regime dependent**; 2D wins on smallest sets | supports-P4 [H] |
| Adams&Coley2025 | conformer-quality sensitivity is **task-specific** (steric vs electronic) | supports-P4 [M] |
| SpaceFormer (Lu+, ICML'25) | grid 3D MAE pretrain; +20% HOMO/LUMO | supports-P4 [H] |

### AAAI-fit precedents (prove "AAAI accepts this")
GeomGCL (AAAI'22), MolKGNN (AAAI'23), Gode (AAAI'25), S-CGIB (AAAI'25), Association-Pattern (AAAI'25), Energy-Motivated Equivariant Pretraining (AAAI'23); + **Forest-vs-Tree (AAAI'26)** = pure eval-methodology precedent. [H] (some author lists not fully retrieved — flagged `metadata-partial` in the `.ris`/`.bib`.)

**Net scoop reading [H]:** nothing scoops P4. Only real pressure = **P2 is precedented by MolGraphEval** (frozen≠finetune *ranking*) and **bounded by MolGPS** (probing fine for big models). Our defensible P2 novelty = the **absolute-AUC correction** (BACE V0 0.758→0.82→0.886) **used to overturn a concrete 3D-injection conclusion**, scoped to the small-model / contrastive / linear-probe regime — not the bare existence of the gap.

---

## ② Design space → recommended positioning

| Positioning | Core claim | New experiments needed | AAAI strength | Cost/Risk | Main risk |
|---|---|---|---|---|---|
| P1 pure positive method | injection makes 3D work / spectral backbone wins | must manufacture a real win | **LOW (dishonest)** | — | no win exists on MoleculeNet (3D≈random, n=9 collapse) |
| P2 measurement/protocol | frozen probe systematically under-measures molecular pretraining | integrate inductive-sweep finetune cells | LOW–MED | low | MolGraphEval precedent; MolGPS opposite; **AAAI has no benchmark track** |
| P3 negative audit | frozen 3D injection inert on saturated ADMET | none (have it) | **LOW** | low | "narrow / single-pipeline / not surprising" to broad PC |
| **★P4 hybrid boundary** | **"when does cheap 3D injection help?"** inert on saturated ADMET (ladder≈random, inductive-robust; RF dominates) **but useful on conformation-sensitive QM9 under finetuning** | **QM9 positive cell** + finetune integration + split-alignment fix | **MED–HIGH** | med–high | QM9 cell could come back null even when trainable → fall back to P2/P3 |

**Recommendation: P4 [H].** It hits AAAI's explicitly most-valued mode ("explore new territory… address research questions… principled critical analysis" > "incremental SOTA"), has a positive cell (not pure negative), is fully honest, and converts every "scoop" into a supporting citation.
**Fallback ladder:** P4 → if QM9 null even when trainable, retreat to **P2-as-general-principle**; if AAAI fit weakens, **NeurIPS D&B / TMLR / JCIM** are the natural homes for P2/P3 (AAAI is not).

**Why not P1 standalone:** the only "win" is *proper training of a 2D model* (measurement artifact correction), not an architecture/injection win — claiming P1 would be the motivated-reasoning trap the project already flagged.

---

## ③ Positioning paragraph + Related-Work taxonomy

### Thesis paragraph (intro-facing, AAAI)
> Injecting three-dimensional structure into molecular GNNs is usually validated on quantum-chemistry targets, where geometry is end-to-end trainable and the property is conformation-determined. We ask a question the field has left implicit: **when a *frozen*, pretrained 3D pair representation is injected into a 2D/spectral backbone, does it help — and where?** Using a controlled injection ladder on a dual-path ChebNetII backbone — a static per-bond gate (V2-T5), a Graphormer-style pair-biased attention term (T7), and an atom↔pair co-update (T8) — together with a *random-pair control* and a matched 30-seed Random-Forest baseline, we show that on six fingerprint-saturated MoleculeNet ADMET endpoints the injected geometry is **statistically indistinguishable from a random pair tensor and from a 2D-only baseline** (robust under an inductive pretrain-on-train protocol; RF dominates BACE/ESOL/Lipo). We further show this null is *not* a backbone or mechanism failure: on a conformation-sensitive QM9 target, **the same injection becomes useful once the pair pathway is trained end-to-end** — localizing the benefit of cheap 3D injection to the (trainable × geometry-rich) corner. Along the way we correct a measurement artifact — a frozen linear probe under-measured every deep model in our pipeline (BACE V0 0.758→0.886 under finetuning, on par with RF) — that, uncorrected, would have manufactured a false "deep is weak" story.

### Related-Work taxonomy (5 subsections; keys = existing refs.bib + new `.bib`)
1. **Spectral & message-passing GNNs** — `defferrard2016chebnet, he2022chebnetii, shang2021eagcn, chen2024polygcl, nogueira2025spectra` + `bo2023specformer, hariri2025stablechebnet, geisler2024s2gnn, hou2025heterophily, fallani2025qmpretrain`. *Frame: backbone is established; novelty is the injection question, not the filter.*
2. **3D representations & pair-bias attention** — `zhou2023unimol, liu2022graphmvp, stark2022infomax, fang2022gem, feng2024frad` + `ying2021graphormer, luo2023transformerm, lu2024unimolplus, ji2024unimol2, jiao2025ept, zhu2022unified2d3d, godwin2022noisynodes`. *Frame: pair-bias-in-attention is NOT novel (Graphormer→Uni-Mol2); the frozen-source + spectral-target + null-finding is.*
3. **Simple baselines & fingerprint saturation** — `deng2023systematic, xia2023whydeep, praski2025embedding, vantilborg2022cliffs, wijaya2024twostage` + `jiang2021descriptorvsgraph, adamczyk2024moltop, adamczyk2025peptidefp`. *Frame: replicate, credit Xia/Deng/Jiang as parents.*
4. **Evaluation protocol** — `guo2024scaffold, huang2021tdc, tossou2024polaris, tdcaudit2026` + `wang2023molgrapheval, sypetkowski2024molgps, sun2022gnnpretrain, pinto2025intermediate, kumar2022lpft, akhondzadeh2023probing, antoniuk2025boom`. *Frame: corroborate MolGraphEval, scope vs MolGPS, add absolute-AUC correction.*
5. **Where 3D helps (the boundary)** — `schutt2017schnet, klicpera2020dimenet, satorras2021egnn, hamakawa2025conformation` + `ramakrishnan2014qm9, liao2023equiformer, zaidi2023denoising, kim2023geotmi, cremer2023egnntox, axelrod2023conformer, adams2025conformerquality, lu2025spaceformer`. *Frame: literature draws the boundary implicitly; we draw it explicitly + controlled.*

### Pre-empt the 3 obvious reviewer questions
- *"T7 = Transformer-M/Graphormer?"* → cite both; difference = **frozen external source + spectral target + null finding** (they train end-to-end, evaluate only quantum tasks, never run a random-pair control).
- *"Frozen-probe under-measurement is known (MolGraphEval)."* → we corroborate it and add an **absolute correction that flips a concrete 3D-injection conclusion**, scoped (per MolGPS) to small-model/contrastive/linear-probe.
- *"Isn't 3D-doesn't-help-ADMET old news (Cremer, Deng, Xia)?"* → yes for *unstructured* 3D; new = a **controlled frozen-injection ladder + random-pair control** that isolates *which corner* fails, plus the positive QM9 cell.

---

## ④ Prioritized plan + experiment schedule

**AAAI-27 CFP [H]:** abstract **2026-07-21**, full paper **07-28**, supp/code **07-31**; **7 pages** technical content + unlimited refs; 2-col AAAI LaTeX (authorkit27 confirmed live); **double-blind**; **no Datasets/Benchmarks track**. Phase-1 reject 09-24, rebuttal 10-19/25, decision 11-30. → **~7 weeks** to abstract; QM9 must start now.
*Uncertain [M]:* AAAI-27 detailed CFP page still "Coming Soon" — formatting/reproducibility-checklist/double-blind restatement inferred from authorkit27 + AAAI-26 precedent.

### Tasks (priority · owner · cost · risk)
- **P0 — correctness gate (HPC, low) [H]:** run `verify_split_alignment.py --task bace` on `down_task_bace_unifold` vs `down_task_bace_v2`. Local check found V0/V2 share only 58/152 test mols; if the canonical pipeline is also misaligned, the BACE matched-protocol claim has a crack. **Must resolve before submission.**
- **P0 — QM9 positive cell (the linchpin; see design below).**
- **P1 — frozen→finetune integration (low) [H]:** lift the inductive-sweep **finetune** numbers (V0 BACE 0.886≈RF; BBBP 0.864>RF) into the main tables → establishes the P2 measurement pillar. (Data exists; needs consolidation, n≥3 pretraining draws to tame frozen variance.)
- **P1 — real Uni-Mol finetune row (HPC, low–med) [H]:** run shipped `reproduce_unimol_finetune.py` + sbatch; replace the misleading frozen-emb+RF "Uni-Mol" row.
- **P2 — n=9 consolidation + delete refuted aromatic mechanism from `main.tex` (writing) [H]:** `main.tex` prose + `tables/classification.tex,regression.tex` still show stale n=3 (0.862/0.638) and the aromatic-routing claim that **B1 refuted at n=9**; `tables/main_results.tex` already holds the n=9 truth — reconcile to it.
- **P2 — confirm T8 status:** thesis claims a V2-T5/T7/**T8** ladder; verify T8 (atom↔pair co-update) has results, else drop "T8" from the ladder framing or run it.
- **P3 — ICDE→AAAI migration + 7pp cut (writing) [H]:** convert to AAAI template; cut ~30%. **First cuts:** the vestigial experimental-spectroscopy RW paragraph (+~25 mass-spec/NMR/IR refs — orthogonal, only there to disambiguate the old "Spec" name we're dropping) → one sentence; the 14-repo featurization-audit table → appendix/supp.
- **P3 — rename:** SpecMol → **ChebInject** (name collision w/ arXiv:2509.21861; double-blind anyway).

### QM9 experiment design (the make-or-break) [H]
**Key practical win:** QM9 ships **DFT-optimized 3D coordinates** — so the pair feature can be computed directly (Gaussian-basis expansion of pairwise distances from QM9 geometry), **no Uni-Mol pair extraction needed** (sidesteps the memory wall that blocked HIV/Tox21). Use the existing FreeSolv RDKit-GBF pair path, fed with QM9's *true* coords.

**2×2 boundary (4 cells, with existing ADMET cells as the other half):**
| | saturated ADMET (have) | conformation-sensitive QM9 (new) |
|---|---|---|
| **frozen injection** | inert ≈ random ✓ | expected weak/inert |
| **trainable injection** | inert ≈ random (inductive sweep) ✓ | **expected real-pair > random-pair = the positive cell** |

- **Targets:** dipole μ and/or HOMO–LUMO gap (most conformation-sensitive). Subsample QM9 to ~10–20k (stratified) to fit 1-GPU budget.
- **Variants:** V0 / V2-T5 / **random-pair control** (Gaussian tensor, identical shape) × {frozen-probe, end-to-end finetune} × 3 seeds.
- **Pre-registered decision rule:** P4's positive cell **holds** iff, under finetune on ≥1 QM9 target, **real-pair beats random-pair** with a paired test across seeds (and ideally frozen-real ≈ frozen-random, isolating the trainable axis). If real-pair ≈ random-pair even when trainable → **positive cell fails → retreat to P2/P3** and present the boundary as literature-supported, not self-demonstrated. *No motivated reasoning: report whichever way it lands.*
- **Compute estimate:** ~2 targets × 3 variants × 2 modes × 3 seeds ≈ 36 cells × ~4 GPU-h ≈ 6 days serial (1 GPU) / ~1–2 days on thk A6000 multi-GPU. Feasible within the 7-week window if started this week.
- **Risk [M]:** ChebNetII + scalar/bias injection is a weak 3D model vs Equiformer/Uni-Mol2; the QM9 effect may be small. We do **not** claim QM9 SOTA — only that the *same mechanism* crosses from indistinguishable-from-random (ADMET) to distinguishable (QM9, trained). The random-pair control is the load-bearing apparatus.

### Schedule (against 07-21 abstract / 07-28 full)
- **Wk 1 (Jun 2–8):** P0 split-alignment check; QM9 data build (coords→GBF pair, subsample); kick off P1 Uni-Mol finetune + frozen→finetune consolidation on HPC.
- **Wk 2–3 (Jun 9–22):** run QM9 ladder×mode×seed; n=9 consolidation; start AAAI template port + rename.
- **Wk 4 (Jun 23–29):** QM9 results in; **decision gate** (positive cell holds? → P4 final; else P2/P3 fallback). Lock contribution list.
- **Wk 5–6 (Jun 30–Jul 13):** write to 7pp; RW taxonomy; figures (boundary 2×2, gate/attention as supp); internal review.
- **Wk 7 (Jul 14–21):** polish, anonymize, abstract submit 07-21; full paper 07-28; supp/code 07-31.

---

## ⑤ Suggested new BibTeX

- **File:** `paper/specmol_litreview_aaai.bib` (39 entries; verified DOIs/arXiv ids). **Do NOT blind-merge into `refs.bib`** — add the **must-cite subset** first, others as the 7pp budget allows.
- **Must-cite for AAAI [H]:** `ying2021graphormer, luo2023transformerm, lu2024unimolplus, jiang2021descriptorvsgraph, sun2022gnnpretrain, wang2023molgrapheval, sypetkowski2024molgps, cremer2023egnntox, ramakrishnan2014qm9` (+ `liao2023equiformer` for the QM9 cell).
- **AAAI-fit precedents** (`*geomgcl, *molkgnn, *gode, *scgib, *assocpattern, *forestvstree*`) — cite 1–2 in intro to pre-empt "is molecular ML in scope for AAAI?".
- **Metadata-partial** (flagged in `.ris`/`.bib`): Gode / S-CGIB / Association-Pattern / Forest-vs-Tree author lists not fully retrieved — verify before camera-ready (OJS URLs are correct).
- **Already in `refs.bib`** (do not duplicate): all spectral basics, Uni-Mol/GraphMVP/3D-Infomax/GEM/Frad, Deng/Xia/Praski/vanTilborg/Hamakawa, MoleculeNet/Chemprop/TDC/Polaris/scaffold-split.

---

## Appendix — Zotero status
- **52 refs imported** to group "Agent-Master" (13 prior `specmol-litreview` + 39 new `specmol-litreview`+`chebinject-litreview`+role tags). Currently in collection **"Agent and Security"** (connector writes only to the UI-selected collection; REST API is read-only → cannot create/move programmatically).
- **Manual step (UI-only, ~2 clicks):** New Collection `ChebInject-litreview` → filter tag `specmol-litreview` → select 52 → drag in.
- Import sources: `paper/specmol_litreview.ris` (13) + `paper/specmol_litreview_aaai.ris` (39).
