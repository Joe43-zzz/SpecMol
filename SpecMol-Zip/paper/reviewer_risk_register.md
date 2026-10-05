# ICDE reviewer risk register

This file is the project-level control panel for the submission. It is not part of
the paper build. The goal is to keep the paper defensible under the actual data:
modest predictive wins, strong classical baselines, and a mechanism finding that
is more reliable than the headline AUC story.

## Current positioning

Do not pitch the paper as a universal SOTA molecular predictor. Pitch it as a
strict matched-protocol study of frozen 3D pair injection into a spectral GNN:
V2-T5 is a minimal, reproducible probe; the learned gate gives a chemically
interpretable routing signal on BBBP; strong fingerprint/RF controls expose when
graph-side 3D mechanisms do not add test-time value.

## Risks and paper-side defenses

| # | Severity | Reviewer attack | Why it is dangerous | Required defense or patch |
|---|---|---|---|---|
| 1 | High | There is no broad performance win. | BBBP V2-T5 vs V0 collapses to a tie at n=9 and BACE is dominated by RF/fingerprint baselines, even though FreeSolv now shows a direct Uni-Mol V2-T5 RMSE reduction. | Keep the abstract/conclusion focused on conditional gains, mechanism, fair protocol, and baseline ceiling rather than SOTA. Avoid "improves prediction" as a general claim. |
| 2 | Resolved/Watch | FreeSolv does not test Uni-Mol pair injection. | Earlier FreeSolv V2-T5 numbers used RDKit 3D + GBF surrogate pair features. The final V2-T5 FreeSolv cell now uses Uni-Mol pair tensors at n=3, but T7 FreeSolv still uses the earlier GBF pipeline. | Keep the table caption and text explicit: V2-T5 FreeSolv is Uni-Mol; T7 FreeSolv is descriptive GBF and not directly comparable to that cell. |
| 3 | High | Mechanism claim is based on only three audited BBBP seeds. | The BBBP AUC win was expanded to n=9, but the per-bond gate audit remains on the original three BBBP seeds. | Explicitly narrow the claim to the audited cohort unless the six extra n=9 checkpoints are audited. Do not imply population-level mechanism proof. |
| 4 | High | Protocol/caption inconsistencies reduce trust. | Mixed n=3/n=9 reporting and FreeSolv split/pair-source asymmetry are easy reviewer targets. | Maintain one authoritative protocol paragraph and table captions that state split, seed unit, n, and Uni-Mol-vs-GBF source. |
| 5 | High | The paper is not ICDE-shaped enough. | A small molecular GNN paper with weak gains may look outside scope. | Make reproducible evaluation, split reconstruction, paired-seed reporting, leakage-safe 3D feature injection, and strong classical baselines first-class contributions. |
| 6 | Medium-High | Random forest beating the model undermines novelty. | RF reaches 0.894 on BACE, far above the graph variants; a reviewer may ask why the graph model matters. | Turn this into the methodological point: fingerprint saturation can invalidate graph-model claims unless classical controls are reported, while FreeSolv demonstrates the conditional case where Uni-Mol pair gating helps. |
| 7 | Medium | Statistical evidence is underpowered. | Several claims are n=3; sign tests and paired t-tests do not reach significance. | Use "matched-seed exploratory/descriptive evidence" language. Do not promote non-significant mean deltas as wins. |
| 8 | Medium | Post-B4 rebuild makes earlier results look broken. | The directed-edge bug can trigger reproducibility concern. | State that all reported BBBP/BACE numbers use the post-B4 rebuild and keep pre-B4 numbers out of the paper. Maintain script/source provenance. |
| 9 | Medium | T7 looks like a failed contribution. | T7 has no significant gain and only a small BACE mean advantage. | Keep T7 in results/limitations only. Do not add an abstract sentence, contribution bullet, or method subsection. |
| 10 | Medium | T6/T7 attention capacity story is speculative. | The claim that contrastive pretraining gives little pressure to attention outputs is mechanistic interpretation, not directly proven. | Phrase as "consistent with" the observed gate behavior. Point to auxiliary objectives or attention-mediated pooling as future work. |

## Delegation map

- Paper surgeon: keep T7 minimal, fix captions/protocol wording, preserve 14 pages.
- Reproducibility auditor: trace every table cell to source JSON and seed unit.
- Reviewer simulator: maintain this risk register as the paper evolves.
- Claim/citation auditor: mark any sentence whose strength exceeds evidence.

## Hard submission gates

- Main table includes T7 only as descriptive n=3.
- FreeSolv V2-T5 is described as direct Uni-Mol evidence; FreeSolv T7 is described as a GBF exploratory control.
- BBBP n=9 tie is stated plainly.
- BACE fingerprint/RF ceiling is stated plainly.
- PDF remains 14 pages.
- No undefined citation warnings.
- Every main table number has a source JSON and seed convention.
