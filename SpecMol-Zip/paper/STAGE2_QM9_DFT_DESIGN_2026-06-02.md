# Stage-2 QM9 DFT-coordinate cell — data-sourcing DESIGN (2026-06-02, no compute yet)

Status: **design only** for the make-or-break experiment in plan
`shiny-hugging-glacier.md`. Nothing run, nothing committed. The user picked
"design the取数方案 first, then decide whether to run." This doc nails the source,
the alignment hazard, the surgical change set, the validation gate, and the
compute/contrast plan.

---

## 0. Goal & why it's the deciding bet

The QM9-μ pilot fed Uni-Mol **RDKit-MMFF** conformers, so "does *real* 3D help
on a geometry-pure target" is untested. Stage-2 = swap MMFF → **true GDB-9 DFT**
coordinates, hold everything else fixed, re-run the 7 arms, and apply the
pre-registered venue gate. The RF ceiling is computed from **geometry-blind**
features (Morgan + 2D descriptors), so the swap does not give RF any new signal
— but RF is still **re-fit and re-evaluated on the embed_ok subset of whichever
manifest you feed it**, and the DFT manifest drops more molecules than MMFF, so
the RF *number* changes (see the correction box below). The entire question is
whether feeding Uni-Mol real DFT geometry pushes the best deep arm **below the
matched strong-RF on the same DFT subset** and amplifies the T8 geometry effect
beyond MMFF's −0.033.

> **CORRECTION (2026-06-03, post-run).** An earlier draft of this doc (and §6.4
> below) claimed RF "gives the same value by construction" on the DFT vs MMFF
> manifest because RF is geometry-blind. **That is wrong.** RF *features* are
> geometry-blind, but RF is re-fit on the manifest's `embed_ok` training rows
> and scored on its `embed_ok` test rows, and **DFT and MMFF have different
> `embed_ok` subsets** (DFT test = 1256 ⊂ MMFF test = 1473). On the easier 12.5k
> DFT subset the strong charge-aware RF is **0.845**; on the 14.8k MMFF set it is
> **0.925**. *That different subset is exactly why 0.925 and 0.845 differ* — they
> are NOT equal by construction. Only the **deep-vs-RF comparison ON THE SAME
> manifest subset** is valid; MMFF and DFT absolute RMSE (deep or RF) are not
> comparable. Consequently `qm9mu_rf_matched.py` now makes the manifest dir a
> **required** argument (no silent MMFF default) and writes a per-manifest
> results JSON.

---

## 1. DFT-coordinate source (decided)

**`torch_geometric.datasets.QM9`** (PyG) is the clean source (verified from
installed source + PyG docs):
- `data.pos` = **DFT B3LYP/6-31G(2df,p) coordinates** read from `gdb9.sdf`
  (`removeHs=False` → all atoms incl. H, native gdb9 order).
- `data.z` = atomic numbers (gdb9 order); `data.smiles`; `data.y` = 19 targets,
  **μ (dipole, Debye) = `data.y[:, 0]`**; `data.name` = gdb id; `data.idx`.
- First call downloads molnet `qm9.zip` (~80 MB, **not** multi-GB). **Use the
  Python312 interpreter (PyG 2.7.0)** — its modern reader preserves native atom
  order; the `.venv` (PyG 2.3.1) is older and riskier here.
- **Exclude** the 3,054 molecules in `uncharacterized.txt` (failed geometry
  consistency) — PyG QM9 already drops them in `process()`; if reading raw, honor
  it.

Nothing local or on HPC currently holds DFT coords (only the MMFF SDF) — must
download fresh.

---

## 2. THE alignment hazard (make-or-break)

The current pipeline's atom order is **RDKit canonical-from-SMILES**:
- conformer/SDF mol: `MolFromSmiles(smiles)` → `AddHs` (heavy first, then H)
  — `prepare_qm9mu_unimol_input.py:94,97`.
- graph mol: a **fresh** `MolFromSmiles(smiles)` (heavy only)
  — `create_data_DC.py:75-90`.
- Uni-Mol heavy-pair rows ↔ graph nodes align **because both come from the same
  canonical SMILES parse** (Uni-Mol filters H by symbol, order preserved;
  validated at `tools/extract_unimol_pair.py:202-237` and
  `make_task_from_unimol.py:66-81`).

PyG QM9's `pos`/`z` are in **gdb9 `.xyz` order, which differs from canonical
SMILES order.** Naively doing `conf.SetAtomPosition(i, data.pos[i])` on the
pipeline's `mol_h` attaches DFT coords to the **wrong atoms** → silently
corrupted geometry (worse than MMFF). This exact failure is documented in PyG
issues #697 / #10560. **Reconciling the two orderings is the whole design.**

---

## 3. Recommended approach — single source of truth (Solution A)

Build BOTH the Uni-Mol SDF and the graph from the **same gdb9-order RDKit mol**,
so order is aligned by construction (no fragile atom-mapping). The GNN is
node-order-invariant, so a different-but-internally-consistent node order does
**not** weaken the DFT-vs-MMFF contrast (the comparison is at the
molecule/prediction level, and the split is identical).

Per molecule:
1. From PyG QM9 get the gdb9 RDKit mol (with H + DFT `pos`), its `smiles`, μ.
2. Write the Uni-Mol SDF **from this mol** (gdb9 order, DFT coords).
3. Build the graph (node features, bonds) **from this mol's heavy atoms**
   (gdb9 order) — needs a `mol_to_graph(mol)` variant of
   `create_data_DC.smile_to_graph(smiles)` that takes a mol instead of
   re-parsing SMILES (the only non-trivial code change).
4. Uni-Mol heavy-pair order == graph heavy order == gdb9 order → aligned.

**Identity & split reuse (clean contrast):** reuse the EXISTING pilot manifest
`unimol_out_qm9mu/manifest.csv` for the molecule set + split. Match each manifest
SMILES to PyG QM9 by a stable key — **prefer `mol_id`/gdb-name** (deepchem
`qm9.csv` has a `mol_id` column; PyG QM9 has `data.name`) over SMILES-canonical
matching, because gdb-id is exact. The manifest currently lacks `mol_id`, so:
re-run the seed-42 subsample on `qm9.csv` to recover `mol_id`↔SMILES, OR add
`mol_id` while rebuilding. Restrict the final set to **MMFF-ok ∩ DFT-ok** so the
two cells run on the identical molecule list (log the count; expect ≈14,773
minus any not-found / uncharacterized).

### Fallback (Solution B, if we must keep byte-identical node order)
Keep the pipeline's canonical-SMILES order and **reorder** DFT coords onto
`mol_h` via an RDKit atom map (`GetSubstructMatch` between the gdb9 mol-with-H
and `mol_h`, or matched `CanonicalRankAtoms`). Symmetric atoms map arbitrarily
but to chemically-equivalent positions (benign for coordinates). Riskier
(mapping can fail / mis-handle symmetry), so only if Solution A's node-reorder is
unacceptable. **Either way the validation gate below is mandatory.**

---

## 4. Surgical change set

| File | Change |
|---|---|
| `prepare_qm9mu_unimol_input.py:93-105` (`embed_mol_3d`) | Replace `EmbedMolecule + MMFFOptimizeMolecule` with "fetch gdb9 mol+DFT pos for this molecule." New input: a `smiles→(mol_with_DFT_pos)` lookup built once from PyG QM9. Keep returning a mol whose SDF write is unchanged. |
| `create_data_DC.py` | Add `mol_to_graph(mol)` (factor out of `smile_to_graph`) so the graph builds from the gdb9-order mol (Solution A). |
| `make_qm9mu_from_unimol.py:57` | Call the mol-based graph build for the DFT set; write to a NEW dir. |
| NEW `build_qm9_dft_lookup.py` | One-time: load PyG QM9 (Python312), build `{mol_id→(rdkit_mol_with_pos, mu)}`, dump to a cache (pickle/parquet) so HPC extraction doesn't need PyG. |
| Output dirs | `unimol_out_qm9mu_dft/` (SDF+manifest+pair_rep) and `down_task_qm9mu_unimol_v2_dft/` — **never overwrite the MMFF versions** (keep for the contrast). |

`tools/extract_unimol_pair.py` and the model/training code need **no change**
(Uni-Mol just consumes the new SDF; arms unchanged).

---

## 5. Validation gate (MANDATORY before trusting any result)

A misaligned coordinate assignment is silent, so gate the rebuild on physics:
1. **Per-molecule bond-length check.** For every covalent bond in the mol,
   `||pos[i]-pos[j]||` must be in a physical range (≈0.9–1.8 Å for first/second-row
   single/double bonds; allow up to ~2.0). Any molecule with an out-of-range
   bonded distance → **alignment is broken** → drop + log (should be ~0 if
   aligned; a high failure rate means the ordering is wrong).
2. **Symbol consistency.** `data.z` element sequence (heavy, post-filter) must
   equal the graph's `atoms` symbol sequence (re-uses the existing
   `verify_alignment` / `infer_keep_atom_indices` asserts).
3. **DFT≠MMFF sanity.** Mean per-atom displacement DFT-vs-MMFF should be
   non-trivial (>~0.1 Å) — confirms we actually swapped geometry.
4. **Spot-check 5 molecules** by eye (RMSD DFT vs MMFF, dipole-vs-geometry
   plausibility) before launching the full 15k extraction.

---

## 6. Compute plan (HPC, after go-ahead)

1. Local (Python312): `build_qm9_dft_lookup.py` → DFT mol cache (~80 MB download).
2. HPC: `prepare_*_dft` → `unimol_out_qm9mu_dft/{sdf,manifest}` (CPU); run the
   validation gate; then Uni-Mol extraction (GPU — 15k fit in the MMFF pilot, so
   DFT fits too) → `pair_rep/`; then `make_*_dft` → `down_task_qm9mu_unimol_v2_dft/`.
3. HPC: re-run **7 arms** (V0/V2T5/random/T7/T7random/T8/T8random) end-to-end
   finetune, seeds 9/19/29 (same protocol as the MMFF pilot, jobs 138872-138894
   — that cost is the reference; manageable, 1-GPU serial).
4. RF ceiling: **MUST re-run on the DFT manifest** — do NOT reuse the MMFF
   0.925. (Earlier draft was wrong here.) The DFT `embed_ok` subset (1256-test) is
   smaller than MMFF's (1473-test), so RF, though geometry-blind in its features,
   is re-fit/re-scored on a different molecule set and lands at **0.845**, not
   0.925. Run `python qm9mu_rf_matched.py unimol_out_qm9mu_dft` (the manifest dir
   is required; it writes `unimol_out_qm9mu_dft_rf_matched.json`). The deep arms
   are evaluated on this same DFT subset, so deep-vs-RF on the DFT manifest is the
   valid gate; MMFF-vs-DFT absolute values are not comparable.

Note the single GPU is currently on the T8-BACE cell; Stage-2 GPU work queues
behind it (or after — it is the higher-priority use of the card once T8 lands).

---

## 7. Contrast + pre-registered decision rule (from the plan)

Report side-by-side **DFT vs MMFF** for all 7 arms (same molecules/split):
- `T8_geom(DFT) = T8 − T8random` vs MMFF's −0.033; architecture = random − V0.
- **AAAI positive boundary gate:** best deep arm RMSE **< the matched strong-RF
  on the SAME DFT subset** (NOT the MMFF 0.9249 — see correction; the DFT-subset
  RF is the apples-to-apples ceiling, which post-run was **0.845**) **AND**
  `T8_geom(DFT)` significant (>2σ at n=3 **and** |effect| > MMFF's 0.033). Both →
  AAAI positive cell.
- **Else** → audit paper to TMLR/NeurIPS D&B; QM9 becomes the "even true DFT
  geometry doesn't break the descriptor ceiling" boundary (still strengthens C1).
- Honest either way; do **not** escalate to QM9-134k without a separate decision.

---

## 8. Risks / open items

- **Identity matching coverage**: if mol_id recovery is messy, SMILES-canonical
  matching may drop some molecules → smaller matched set (log it; contrast still
  valid on the intersection).
- **PyG QM9 download** on HPC/VPN: do the download + lookup-cache **locally**
  (Python312), ship the cache to HPC, so HPC needs no PyG/internet.
- **Solution A node-order change** means DFT and MMFF `.pt` are not atom-aligned
  (can't diff `pos` directly) — fine for the molecule-level contrast; if a
  byte-clean per-atom diff is wanted, use Solution B + the validation gate.
- Effort: ~1 focused implementation pass (lookup builder + `mol_to_graph` +
  prepare/make DFT variants + validation), then the HPC run.
