"""Verify _names() routing and parse_regression_matrix file lookup."""

# Replicate _names() logic from aggregate_regression_seeds.py
def _names(task, variant):
    if task == "freesolv" and variant == "v2t5":
        return "freesolv_unimol_v2t5", "freesolv_unimol_v2t5_results.json"
    out = {
        "v0":   f"{task}_v0_results.json",
        "v2t5": f"{task}_v2t5_results.json",
        "t6":   f"{task}_t6_results.json",
        "t7":   f"{task}_t7_bare_results.json",
    }[variant]
    return f"{task}_{variant}", out

print("=== _names() routing for all (task, variant) pairs ===")
for task in ("esol", "lipo", "freesolv"):
    for variant in ("v0", "v2t5", "t6", "t7"):
        prefix, out = _names(task, variant)
        print(f"  ({task:9s}, {variant:5s}) -> prefix={prefix:35s}  out={out}")

print()

# Check: parse_regression_matrix uses freesolv_unimol_v2t5_results.json for FreeSolv V2-T5
print("=== parse_regression_matrix sources for freesolv ===")
task = "freesolv"
v2_name = "freesolv_unimol_v2t5_results.json" if task == "freesolv" else f"{task}_v2t5_results.json"
sources = [
    ("V0",    f"{task}_v0_results.json"),
    ("V2-T5", v2_name),
    ("T6",    f"{task}_t6_results.json"),
    ("T7",    f"{task}_t7_bare_results.json"),
]
for label, fname in sources:
    print(f"  {label:6s} -> {fname}")

print()

# KEY CONCERN: aggregate_regression_seeds.py writes freesolv_unimol_v2t5_results.json
# for (freesolv, v2t5). parse_regression_matrix reads freesolv_unimol_v2t5_results.json.
# These MATCH.

# CONCERN: aggregate writes to REPO root (parent of SpecMol-Zip)
# REPO = Path(__file__).resolve().parents[1]  -> from paper/ go up 2 levels
# paper/ is under SpecMol-Zip/, so REPO = D:\specmol-zip\SpecMol-Zip
# (REPO / out_name) -> D:\specmol-zip\SpecMol-Zip\freesolv_unimol_v2t5_results.json
print("=== REPO path resolution ===")
import pathlib
script = pathlib.Path("D:/specmol-zip/SpecMol-Zip/paper/aggregate_regression_seeds.py")
REPO = script.resolve().parents[1]
print(f"  REPO = {REPO}")
print(f"  freesolv_unimol_v2t5_results.json -> {REPO / 'freesolv_unimol_v2t5_results.json'}")
print()

# And make_tables.py load() uses:
# REPO = Path(__file__).resolve().parents[1]  -> same logic from paper/make_tables.py
script2 = pathlib.Path("D:/specmol-zip/SpecMol-Zip/paper/make_tables.py")
REPO2 = script2.resolve().parents[1]
print(f"  make_tables REPO = {REPO2}")
print(f"  Both REPO paths same: {REPO == REPO2}")
