# ChebInject — Nearest-Neighbor Differential & Grounded Plan (2026-06-02)

> Built from a from-source deep-read of 65 related works (real reported numbers) + a full read of the actual architecture/data code. Every "A did X but not Y" below is grounded in the paper's verified method/results, not its title.

---

## 0. The hard constraint the corpus forces (read this first)

**You cannot win on classification AUC. Stop trying; build the audit/boundary instead.** Grounded SOTA on *your* benchmarks (note: split-confounded across papers, but the ceiling is unambiguous):

| Benchmark | SOTA in corpus | Method | Your V0 / V2-T5 |
|---|---|---|---|
| BBBP | **0.937** (MoLFormer), 0.933 (Galformer) | end-to-end SMILES/2D+3D pretrain | 0.828 / 0.831 |
| BACE | **0.966** (SpaceFormer), 0.897 (Galformer), 0.890 (KA-GNN, **2D+5Å edges, no pretrain**) ≈ RF 0.894 | various, all end-to-end | 0.757 / 0.763 |
| ESOL/FreeSolv (RMSE) | 0.279/0.231 (MoLFormer), 0.285/0.263 (MAT) | end-to-end | ~0.82 / 0.64–0.67 |

Your model is **~0.10–0.17 AUC below SOTA on classification** and far below on ESOL. A reviewer who wants SOTA will reject on sight. So the paper's value is **not the model** — it is the **controlled audit** (random-pair null + frozen-probe correction + the frozen-vs-trainable × saturated-vs-QM9 boundary). The corpus *confirms* this is unoccupied territory.

---

## 1. Nearest-neighbor differential ("A did X, NOT Y; ChebInject does Z")

Ordered by closeness. Δ = the one sentence you put in Related Work.

### Closest competitors — 3D-into-2D / frozen-injection regime
- **SCAGE** (Nat. Commun. 2025) — *did:* 2D+3D multitask pretrain (incl. a **fingerprint-prediction objective** + bond-angle + distance-masked attention) on 5M mols, MMFF conformers, **end-to-end finetune**, BBBP/BACE + 30 activity-cliff sets. *Did NOT:* freeze the 3D source or inject a frozen pair at inference; no random-pair control; no frozen-probe-vs-finetune; not spectral. **Δ:** ChebInject studies the *frozen-injection* regime SCAGE trains through, with a random-pair null SCAGE never runs. (Closest competitor; independently confirms 3D-into-2D is bounded on ADMET.)
- **3D-PGT** (KDD 2023) — *did:* 3D generative pretrain (bond length/angle/dihedral) on a GPS backbone, then **2D-only inference**; BACE 0.809. *Did NOT:* frozen pair injection, random control, spectral backbone. **Δ:** distillation-into-weights (theirs) vs frozen-pair-injection-at-inference (ours); we add the null control.
- **Praski2025** (arXiv 2508.06199) — *did:* benchmark **25 frozen embeddings incl. Uni-Mol** on 25 ADMET sets → **Uni-Mol 76.85% < ECFP 79.89%**; only CLAMP (fingerprint-fused) beats ECFP. *Did NOT:* inject the frozen pair into a backbone, run a random-pair control, or compare frozen-vs-finetune. **Δ:** strongest *external validation of F2* (frozen Uni-Mol ≈ useless); we explain *why* (random-pair tie) and show finetune recovers it (C2). **Cite as evidence FOR us.**
- **Hamakawa2025** (JCIM, DOI 10.1021/acs.jcim.5c00018) — *did:* a **wrong-conformer control** on **finetuned** Uni-Mol; correct > wrong ≈ ECFP on QM/stereo tasks. *Did NOT:* a random-pair control as a formal protocol; frozen injection into a spectral GNN; matched MoleculeNet/inductive audit. **Δ:** our random-pair (shape-matched Gaussian) generalizes the wrong-conformer idea into a reusable null-protocol on a frozen spectral pipeline. **The single most-aligned prior — must-cite, frame as our protocol's ancestor.**

### Pair-bias attention mechanism (scopes T7 to ~zero novelty)
- **Graphormer** (NeurIPS 2021) — *did:* pair (SPD/edge) as additive attention bias, **2D-only, from scratch**. *Did NOT:* 3D, frozen source, MoleculeNet, control. **Δ:** T7's "pair-as-bias" is Graphormer's; ours is frozen-3D-source + spectral-target + null.
- **MAT** (2020) — *did:* **3D inter-atomic distance + adjacency as additive attention bias**, end-to-end; BBBP 0.728, ESOL 0.285, FreeSolv 0.263. *Did NOT:* frozen external 3D encoder; random control; spectral. **Δ:** raw trainable distance (theirs) vs frozen Uni-Mol pair (ours); we add the null.
- **EGT** (KDD 2022) — *did:* all-pairs edge channels + **sigmoid-gated** value aggregation, end-to-end, PCQM4Mv2 SOTA. *Did NOT:* frozen edge features, ADMET, control. **Δ:** the gated all-pairs pattern is EGT's; we freeze + audit on ADMET.
- **Transformer-M / Uni-Mol+ / Uni-Mol2** (ICLR23 / NatComms24 / NeurIPS24) — *did:* 3D-distance-as-bias / two-track pair, **end-to-end**, QM9/PCQM4Mv2 (HOMO 17.5 / 15.2 meV, etc.). *Did NOT:* MoleculeNet ADMET, frozen injection, random control. **Δ:** they are the upstream pair source we consume *frozen*; they prove geometry helps *when trained* (C3 support). Uni-Mol2 is 1.1B params — our frozen 47M Uni-Mol v1 is honestly weaker.

### Spectral GNN backbone (scopes V2-T5 gate to ~zero novelty)
- **EAGCN** (Neurocomputing 2021) — *did:* learned **edge-attention modulating the molecular graph Laplacian** (Tox21/HIV/FreeSolv/Lipo). *Did NOT:* frozen 3D-pair source, random control, frozen-probe. **Δ:** V2-T5 = EAGCN's edge-weighted Laplacian with the gate values **sourced from a frozen Uni-Mol 3D pair** instead of learned-from-scratch + the audit. **Must-cite; do not claim edge-gating novelty.**
- **FAGCN** (AAAI 2021) — *did:* per-edge **scalar gate in [−1,1] mixing low/high-pass**. *Did NOT:* molecules, 3D, frozen source. **Δ:** direct antecedent of the V2-T5 scalar gate and the dual low/high-pass design.
- **KA-GNN** (Nat. Mach. Intell. 2025) — *did:* 2D message-passing + Fourier-KAN + **5 Å through-space "contact" edges**; BACE **0.890** (≈ RF), BBBP 0.787. *Did NOT:* spectral backbone, frozen pair, random control, pretraining. **Δ (important caveat):** **KA-GNN already adds the through-space contact edges I floated as your "best positive method" (T8/`dist_to_edge_weight`)** — so that mechanism is *not novel*. Your favorable point: BBBP 0.831 (V2-T5) > KA-GNN 0.787 (split-confounded). Treat contact-edge as a completeness rung, not a headline.
- **Specformer / JacobiConv / OptBasis / Stable-ChebNet** (ICLR23/ICML22/ICML23/2025) — *did:* show ChebNetII is **not** the optimal spectral filter (on node-classification). *Did NOT:* MoleculeNet, 3D, frozen probe. **Δ:** justify ChebNetII as a *tractable, studied backbone for the audit*, not a SOTA filter claim (one sentence).

### Evaluation-protocol cluster (C2 precedented — credit, don't claim discovery)
- **MolGraphEval** (NeurIPS 2023 D&B) — *did:* frozen-vs-finetune **rank-corr 0.77** across 9 SSL methods; JOAO 1st frozen → 4th finetuned. *Did NOT:* absolute-AUC correction that flips a conclusion; 3D pair; random control. **Δ:** we give the **absolute delta on one pipeline (0.758→0.886) and use it to overturn a concrete 3D-injection claim**. C2's increment, not its discovery.
- **MolGPS** (NeurIPS 2024) — *did:* probing **≥** finetune for **large supervised** 2D foundation models (Polaris 0.91 vs 0.85). **Δ (constraint):** scope C2 strictly to **small / contrastive / linear-probe**; cite MolGPS as the opposite-regime boundary.
- **Pinto2025** — frozen final-layer under-measures encoders (incl. Uni-Mol) by 5–40%. Supports C2.
- **Sun2022** — SSL pretraining gains vanish with rich features/balanced splits (only finetunes). Supports C2; never isolates frozen probe.
- **probing_graph** (AISTATS 2023) — **random-init graph Transformers ≈ pretrained** on chemical-property probing. **Δ:** principled backing for *why* random-pair ≈ real-pair (C1).

### Fingerprint-saturation cluster (C2 ceiling — multi-study consensus, must credit)
- **xia2023** (BACE XGB 0.896 / RF 0.890; BBBP RF 0.923; ESOL/Lipo XGB best), **jiang2021** (RF 0.927 BBBP), **MOLTOP** (BACE 0.829, "no sig. diff. vs pretrained GIN"), **OGB** (MorganFP+RF 0.806 > GIN 0.77 on molhiv), **kamuntavicius2025** (RDKit-desc rank 1.91 > all DL embeddings on 25 ADMET), **vanTilborg2022** (SVM+ECFP wins on activity cliffs), **Lo-Hi** (ECFP-SVM wins lead-opt under hard splits). **Δ:** "RF beats deep on MoleculeNet" is **established** — you *replicate under a matched Uni-Mol-fold + frozen-Uni-Mol-embedding control*, never claim it new.

### The SOTA ceiling you do NOT compete with
- **MoLFormer** (BBBP 0.937), **Galformer** (BBBP 0.933 / BACE 0.897), **SpaceFormer** (BACE 0.966), **MoleculeSDE** (BACE 0.804), **GeomGCL** (ClinTox 0.919). All end-to-end, big pretrain. **Δ:** acknowledge as SOTA; your contribution is orthogonal (audit/protocol), not AUC.

---

## 2. The moat (confirmed unoccupied across all 65 papers)

The **intersection** no paper covers: **(a)** a frozen, pretrained 3D *pair* representation **(b)** injected into a *spectral* GNN **(c)** measured against a **shape-matched random-pair null** **(d)** under a **matched inductive split** **(e)** with a **frozen→finetune absolute-AUC correction** on the same pipeline. Each ingredient is individually precedented; the **5-way intersection is yours**. That is the paper.

---

## 3. Dataset choice (grounded in code + corpus)

- **Null corner (keep, done):** BBBP/BACE/ClinTox + FreeSolv/ESOL/Lipo, Uni-Mol scaffold-fold-k10-seed42, fold0=test. Pipeline (`make_bace_from_unimol.py`) needs per-mol Uni-Mol `encoder_pair_rep` + coords → the OOM wall that killed HIV/Tox21.
- **QM9 positive corner (C3, make-or-break):** the corpus is unanimous that 3D wins live *here* (Equiformer HOMO 15 meV, Transformer-M 17.5, SpaceFormer beats Uni-Mol 19–33%). **Grounded build fact:** the data pipeline already stores `data.pair_dist_edge` from coords — so the **contact-edge rung runs on QM9's native DFT coords with ZERO Uni-Mol extraction**; the V2-T5/T7 rungs need a Uni-Mol run on a ~10–20k subset. **Use true GDB-9 DFT coords, never RDKit-MMFF re-embedding** (Hamakawa/Adams&Coley show wrong conformers collapse the signal). Targets: dipole μ, HOMO–LUMO gap.
- **Add MoleculeACE (van Tilborg activity cliffs)** — it is *the* recognized benchmark where descriptors beat DL and where the random-pair/saturation story is sharpest; cheap, and answers "is your null just a weak split?" Optionally re-test the null on one **Lo-Hi / OGB-molhiv** split (RF 0.806 > GNN documented) to align with the protocol-critique literature.
- **Provenance landmine (from code):** never mix the deprecated `down_task_v2` (0.87) with post-B4 (0.76); the corpus shows cross-paper/split AUC comparison is invalid (OGB vs DeepChem vs Uni-Mol-fold differ by 5–6 AUC).

---

## 4. Method improvement (grounded — and honest about what's scooped)

1. **The random-pair control is your method contribution** (it's already in code: `--randomize_pair` → `randn_like` at the dynamic consumption site). Promote it to a *named null-protocol* for 3D-injection claims. This is the one genuinely-unscooped, reusable thing.
2. **Do NOT headline the contact-edge rung** — KA-GNN (2025) already adds 5 Å through-space edges to a molecular GNN (BACE 0.890). Run it for ladder-completeness; frame as "even an operator-changing rung is bounded," not as novelty.
3. **V2-T5's 64→1 scalar gate is the weakness *and* the honest finding:** under `normalization='sym'` (verified `LH_Direct_ChebnetII_prop_v2.py:161`), the gate's *mean* is spectrally invisible; only heterogeneity matters. The reviewer attack "your null is just a 64→1 bottleneck" is answered by **T8 multi-channel co-update** (`pair_atom_coupdate.py`, ReZero soft-start, **0 results — run one BACE cell or drop the T8 claim**).
4. **The positive QM9 cell must be end-to-end finetune** — the corpus is unanimous (Transformer-M/Uni-Mol+/GeomGCL/DG-GCN/3D-PGT all *train* geometry), and T7's gate froze precisely because NT-Xent never pressures it.
5. **Do not add novelty to T7/T8 attention** — Graphormer→EGT→Uni-Mol2 own it. Your delta is frozen-source + spectral-target + control, full stop.

---

## 5. Holistic best advice (grounded in the whole project)

- **Commit to the identity:** *"A controlled audit of when frozen 3D-pair injection helps a molecular GNN — with a random-pair null protocol."* Not a method paper. The architecture is the *instrument*, the negatives are the *measurements*, the QM9 cell is the *positive boundary*.
- **Three contributions, all corpus-verified unoccupied:** C1 random-pair null-protocol + inductive matched audit; C2 frozen-probe absolute-AUC correction (scoped per MolGPS); C3 frozen+saturated-vs-trainable+QM9 boundary.
- **Run order:** (i) free CPU spectral readout + n=9 reconcile + rename; (ii) QM9 DFT-coord cell (contact-edge first, cheap) end-to-end + DFT-vs-MMFF contrast; (iii) one T8 BACE cell to kill the bottleneck attack; (iv) MoleculeACE null. Pre-register the QM9 decision rule.
- **Venue (honest):** given you cannot beat SOTA AUC and the contribution is a controlled negative/protocol, the *best-fit* homes are **TMLR / JCIM / NeurIPS D&B**. **AAAI is a genuine stretch** — viable only if the QM9 positive cell lands and you sell it as "principled analysis + a reusable protocol" (CFP's preferred mode), not as a model. Do not over-index on AAAI.

---

## 6. Bibliographic errors to fix before citing (caught during deep-read)
- **vanTilborg2022** correct id = DOI **10.1021/acs.jcim.2c01073** (arXiv:2205.10889 is an unrelated hardware paper).
- **Cremer2023** correct = "Equivariant GNNs for Toxicity Prediction," Chem. Res. Toxicol., DOI **10.1021/acs.chemrestox.3c00032** (arXiv:2209.13218 is a physics paper). Could not verify numbers — fetch before citing.
- **MolKGNN** (AAAI'23) and **S-CGIB** (AAAI'25): the arXiv IDs I tried were math papers; use the OJS URLs in `specmol_litreview_aaai.bib`. **S-CGIB is a possible C1 scoop risk (spectral + contrastive + info-bottleneck) — re-fetch with the correct ID before submission.**
