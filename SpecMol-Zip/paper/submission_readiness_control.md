# ICDE submission readiness control

This is the project brain checklist for getting the current SpecMol paper to a
credible ICDE submission. It is intentionally conservative: the paper should be
accepted, if at all, because it is honest, reproducible, and methodologically
useful, not because it oversells small benchmark gains.

## Submission thesis

V2-T5 is a minimal frozen-3D-pair-to-edge gating probe for spectral molecular
GNNs. Under matched protocols it shows conditional value: a direct FreeSolv
Uni-Mol RMSE reduction, an interpretable BBBP aromatic-routing mechanism within
the audited cohort, and no graph-side advantage on BACE once strong fingerprint
and RF baselines are present.

## Non-negotiable narrative rules

- Do not claim universal 3D improvement.
- Do not describe BBBP as a stable AUC win; it is an n=9 tie.
- Do not describe BACE as a graph-model success; it is a fingerprint/RF ceiling.
- Do not promote T7 as a contribution; it is a descriptive controlled exploration.
- Do not compare FreeSolv T7 directly against FreeSolv V2-T5 without noting that
  T7 used the earlier GBF pair-feature pipeline while V2-T5 uses Uni-Mol pairs.
- Do not use FreeSolv to claim broad small-benchmark superiority; keep it scoped
  to the regression endpoint and n=3 evidence.

## Current must-pass gates

| Gate | Evidence | Status |
|---|---|---|
| PDF page limit | `main.log` says `Output written on main.pdf (14 pages, ...)` | Passing |
| Citation health | `main.log` has no undefined citation warnings after BibTeX/pdflatex | Passing |
| Main table FreeSolv V2-T5 | `paper/tables/main_results.tex` shows `0.638 +/- 0.017` | Passing |
| FreeSolv provenance | `freesolv_unimol_v2t5_results.json` records seeds 9/19/29 | Passing, but replace with full HPC JSON if available |
| T7 scope | T7 appears in results/limitations, not abstract/contribution | Passing |
| BBBP scope | Main text says n=9 tie and n=3 upper-tail sampling | Passing |
| BACE scope | Main text says fingerprint/RF ceiling | Passing |

## Delegation plan

### Claude Code: paper surgeon

Owns only `paper/main.tex` and generated table text after approval.

Tasks:
- Remove any remaining overclaim that makes a weak result look significant.
- Keep the PDF at 14 pages.
- Preserve the current thesis and do not add a T7 contribution.

Acceptance check:
- `pdflatex` still reports 14 pages.
- No new abstract/contribution claims are added for T7.

### Codex window A: provenance auditor

Owns no edits unless asked; produces a report.

Tasks:
- Trace every main-table cell to its source JSON.
- Confirm sample std vs population std conventions.
- Confirm FreeSolv Uni-Mol per-seed JSONs match the original HPC outputs.

Acceptance check:
- Any mismatch is reported with exact file and key path.

### Codex window B: reviewer simulator

Owns `paper/reviewer_risk_register.md`.

Tasks:
- Update the risk register after every paper change.
- Keep the top rejection risks sorted by severity.
- Add one sentence of paper-side defense for each risk.

Acceptance check:
- No stale risk says FreeSolv is only a GBF surrogate.

### Codex window C: claim/citation auditor

Owns no edits unless asked; produces a report.

Tasks:
- Flag claims that use causal language without evidence.
- Check that ReZero/LayerScale citations support only the near-identity residual
  analogy, not a performance claim.
- Check that external baselines are described as protocol-matched only when the
  local JSON/protocol supports that.

Acceptance check:
- Report contains only actionable, high-impact issues.

## Exact local commands

Regenerate tables:

```powershell
cd D:\SpecMol-Zip\SpecMol-Zip
..\python38-embed\python.exe paper\make_tables.py
```

Compile without latexmk:

```powershell
cd D:\SpecMol-Zip\SpecMol-Zip\paper
& "C:\Users\zhoutianyang\AppData\Local\Programs\MiKTeX\miktex\bin\x64\bibtex.exe" main
& "C:\Users\zhoutianyang\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe" -interaction=nonstopmode main.tex
& "C:\Users\zhoutianyang\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe" -interaction=nonstopmode main.tex
```

Verification:

```powershell
Select-String -Path main.log -Pattern "Warning|undefined|Citation|Output written"
Select-String -Path paper\main.tex -Pattern "0\.638|0\.675|T7|paired|0\.93|SOTA|universal"
Select-String -Path paper\tables\main_results.tex -Pattern "V2-T5|T7|FreeSolv"
```

## Stop-doing list

- No new architecture rescue before submission.
- No new dataset unless the paper is already otherwise submission-ready.
- No extra T7 seed expansion before submission.
- No abstract/contribution rewrite around T7.
- No broad SOTA framing.
- No merge or push until the submission snapshot is reviewed.
