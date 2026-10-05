"""Build main_acm.tex (KDD 2027 D&B, ACM acmart sigconf) from main.tex.

Non-destructive: reads main.tex, swaps the IEEEtran preamble/front-matter for an
acmart preamble (sigconf, anonymous, review for double-blind), moves the abstract
before \\maketitle (acmart requirement), switches the bibliography style to
ACM-Reference-Format, and RELOCATES supporting-detail blocks into an \\appendix
(after the bibliography) so the main body fits KDD's 8 content-page limit. Nothing
is deleted -- moved content stays in the paper, just under the appendix.

Camera-ready TODO: drop nonacm/anonymous, add \\acmConference + CCS concepts +
keywords; re-balance main vs appendix per author preference.

Usage:  python paper/build_acm.py
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = (HERE / "main.tex").read_text(encoding="utf-8")

BEG = "\\begin{abstract}"
END = "\\end{abstract}"
a0 = src.index(BEG) + len(BEG)
a1 = src.index(END)
abstract = src[a0:a1].strip()

# Body: from Introduction to end of document; swap bib style to ACM.
i = src.index("\\section{Introduction}")
body = src[i:].replace("\\bibliographystyle{IEEEtranN}",
                       "\\bibliographystyle{ACM-Reference-Format}")

# Caveat-sentence fixup: the V0 frozen->finetune detail moves to the appendix
# (the real Uni-Mol finetune, kept in main, is now the primary C2 evidence).
body = body.replace(
    "in the first part of this subsection are the",
    "(detailed in the appendix) are the")

# --- Relocate supporting-detail blocks into the appendix (contiguous, by markers) ---
# Each spec: (name, start_marker, end_marker, end_inclusive). end_inclusive=True
# means the matched end_marker text is part of the block; False means cut up to
# (but not including) the end_marker (which stays in the main body).
specs = [
    ("Experimental Spectra: Nomenclature Disambiguation",
     "\\paragraph{Experimental spectra as a coupled molecular modality.}",
     "\\section{Method}", False),
    ("Featurization-Control Audit (unidirectional-bond regression; 14-repository check)",
     "\\subsection{Featurization Control}",
     "\\subsection{Training Objective and Downstream Evaluation}", False),
    ("Variant V2-T5: Pair-Representation Source and Tensor-Gate Extension",
     "\\paragraph{Pair-representation source.}",
     "\\subsection{Variant T7", False),
    ("Additional Ablations (bias init, gate distribution, chemistry-aware routing)",
     "\\paragraph{(a) Bias initialization",
     "\\paragraph{(b) Uni-Mol pair vs.\\ random pair", False),
    ("DFT Gate-Movement Probe",
     "\\subsection{Gate-movement probe", "\\subsection{Takeaway", False),
    ("Frozen-Probe-to-Finetune Correction: 2D-only V0 Detail",
     "\\paragraph{The $0.758$ reading does not reproduce.}",
     "\\paragraph{The gain is the encoder, not the injected geometry.}", False),
    ("Spectral and Message-Passing GNN Background",
     "\\paragraph{Spectral and message-passing GNNs.}",
     "\\paragraph{Three-dimensional representations", False),
    ("Three-Dimensional Representations and Self-Supervised Pretraining Background",
     "\\paragraph{Three-dimensional representations and self-supervised pretraining.}",
     "\\paragraph{Downstream injection of frozen, pretrained 3D geometry.}", False),
    ("Geometry-Pure Stress Test: Motivation",
     "\\subsection{Motivation: is the negative just weak targets or coarse 3D?}",
     "\\subsection{QM7", False),
]

appendix_sections = []
for name, smark, emark, einc in specs:
    try:
        s = body.index(smark)
        e = body.index(emark, s)
        if einc:
            e += len(emark)
        # Trim trailing whitespace/comment-only lines from the cut for tidiness.
        block = body[s:e].rstrip()
        body = body[:s] + body[e:]
        appendix_sections.append((name, block))
        print("relocated -> appendix:", name, "(%d chars)" % len(block))
    except ValueError:
        print("WARNING: block not found, skipped:", name)

# Assemble the appendix and splice it in just before \end{document}
appendix = "\n\\appendix\n\n"
for name, block in appendix_sections:
    appendix += "\\section{%s}\n%s\n\n" % (name, block)
body = body.replace("\\end{document}", appendix + "\\end{document}")

TITLE = ("[Experiment, Analysis, and Benchmark] When Does Geometry-Aware "
         "Pretraining Help? A Matched-Protocol Map Separating Finetuned-Encoder "
         "Gains from Inert Frozen 3D Injection in Molecular Property Prediction")

preamble = (
    "% SpecMol -- KDD 2027 D&B ACM (acmart) version. AUTO-built from main.tex by build_acm.py.\n"
    "% Camera-ready TODO: remove nonacm/anonymous, add \\acmConference + CCS + keywords;\n"
    "% re-balance main vs \\appendix per author preference.\n"
    "\\documentclass[sigconf,nonacm,anonymous,review]{acmart}\n"
    "\\settopmatter{printccs=false,printacmref=false}\n"
    "\\let\\Bbbk\\relax  % avoid clash between amssymb and acmart's newtxmath\n"
    "\\usepackage{amssymb}\n"
    "\\usepackage{pifont}\n"
    "\\newcommand{\\cmark}{\\textcolor{green!55!black}{\\checkmark}}\n"
    "\\newcommand{\\xmark}{\\textcolor{red!70!black}{\\ding{55}}}\n"
    "\\setcopyright{none}\n"
    "\\renewcommand\\footnotetextcopyrightpermission[1]{}\n\n"
    "\\begin{document}\n"
    "\\title{" + TITLE + "}\n"
    "\\author{Anonymous Author(s)}\n"
    "\\affiliation{\\institution{Anonymous Institution}\\country{}}\n\n"
    "\\begin{abstract}\n" + abstract + "\n\\end{abstract}\n\n"
    "\\maketitle\n\n"
)

out = HERE / "main_acm.tex"
out.write_text(preamble + body, encoding="utf-8")
print("wrote", out, "(%d chars)" % len(preamble + body))
