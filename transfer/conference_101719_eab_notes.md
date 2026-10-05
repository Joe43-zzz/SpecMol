# EAB Draft Notes

This draft is an ICDE-oriented alternative to `conference_101719.tex`.
It keeps the original paper intact and reframes the story as a controlled
benchmark-and-analysis paper rather than a model-forward paper.

## What changed

- The title now uses the ICDE EAB tag and foregrounds split-consistent evaluation.
- The abstract now leads with protocol consistency, strong baselines, and
  benchmark-dependent findings instead of model novelty.
- The introduction and contributions now frame the paper as a controlled
  comparison across V0, V2-T5, T6, FP-only, Chemprop, and RF.
- The main-results narrative now prioritizes:
  BBBP tie at `n=9`, BACE strong-baseline dominance, and FreeSolv as an
  indirect 3D test via RDKit 3D + GBF surrogate.
- The discussion and conclusion now state explicitly that the paper's most
  reliable contribution is controlled analysis and mechanism audit, not a
  universal state-of-the-art claim.

## Why this fits ICDE better

- ICDE explicitly accepts `Experiment, Analysis, and Benchmark` papers.
- This version matches that category by emphasizing evaluation protocol,
  reproducibility risk, benchmark interpretation, and baseline completeness.
- The strongest results in the current project are methodological:
  split confound recovery, post-B4 rebuild discipline, strong classical
  baselines on BACE, and chemistry-aware gate analysis on BBBP.

## Claims tightened relative to the original draft

- BBBP is written as a tie within seed variance at `n=9`, not as a stable win.
- The BBBP aromatic-routing claim is limited to the audited cohort rather than
  generalized to the full `n=9` extension.
- BACE is described as a post-B4 corrected, fingerprint-saturated benchmark.
- FreeSolv is described as an indirect test because it uses an RDKit 3D + GBF
  surrogate instead of the direct Uni-Mol classification pipeline.
- T7 remains outside the main story and is not used as pending evidence.
