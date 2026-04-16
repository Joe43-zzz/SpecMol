# PairLearn Spectrum Port Notes

Current status:
- `LH_Direct` no longer depends on Uni-Mol pair files to enable pair updates.
- If `use_pair_update=True` and `edge_weight_3d` / `pair_rep_flat` are absent, the model can initialize pair bias from a sparse edge attribute tensor.
- The input attribute name is configurable through `pair_input_attr_name`.

Recommended PyG data contract for a spectrum graph:
- `data.x`: peak/node features, shape `[N, Dx]`
- `data.edge_index`: sparse candidate relations between peaks, shape `[2, E]`
- `data.pair_attr`: spectrum pair features, shape `[E, Dp]`

How to enable it:
- Set `pair_input_attr_name="pair_attr"`
- Set `pair_edge_attr_dim=Dp`
- Keep `use_pair_update=True`

Suggested first version of `pair_attr` for spectra:
- `delta_mz_abs`
- `delta_mz_signed`
- `log_intensity_ratio`
- `same_isotope_rule`
- `same_adduct_rule`
- `neutral_loss_rule`
- `peak_cooccurrence_score`
- `same_fragment_family_score`

Practical recommendation:
- Start with a sparse graph instead of an all-pairs graph.
- Keep only chemically or spectrally plausible candidate edges.
- Let `pair_update` learn the final scalar edge weights from those sparse pair features.

Interpretation:
- This is closer to the Uni-Mol idea than directly copying Uni-Mol pair values.
- What is transferred is the update mechanism: `pair init -> dynamic pair update -> edge weight`.
