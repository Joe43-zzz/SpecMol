"""Build the authoritative archive for the 2026-06-01 inductive-protocol sweep.

Embeds the raw per-seed results pulled from thk:~/specmol/logs/sweep_*.log
(1000-epoch sweep) plus the earlier 300-epoch preliminary probes, computes
mean/std, and writes:
  paper/audits/inductive_sweep_2026-06-01.json   (full structured data)
  paper/audits/inductive_sweep_2026-06-01.md     (human-readable + decision gates)

Re-run: python paper/audits/build_inductive_sweep_archive.py
"""
import json
import statistics
from pathlib import Path

OUT = Path(__file__).resolve().parent

# --- raw 1000ep sweep dump (JOB|name|rc|nmol|metric|vals|eps) ---
RAW_SWEEP = """
bace_all_random_finetune|0|1513|Auc|0.8685031185031185 0.8808038808038808 0.8683298683298682|35 31 33
bace_all_T7_finetune|0|1513|Auc|0.8724878724878724 0.8911988911988913 0.8932778932778933|10 18 22
bace_all_V0_finetune|0|1513|Auc|0.8856548856548856 0.883922383922384 0.8880803880803881|29 33 32
bace_all_V0_frozen|0|1513|Auc|0.8236313236313236 0.8225918225918225 0.8208593208593208|1110 1217 1008
bace_all_V2T5_finetune|0|1513|Auc|0.8879071379071379 0.8886001386001386 0.8636521136521136|20 14 13
bace_all_V2T5_frozen|0|1513|Auc|0.8737006237006237 0.8776853776853776 0.8735273735273735|1864 1973 1909
bace_scaffold_all_V0_frozen|0|1513|Auc|0.8182663690476191 0.8459939531368104 0.8058007566204287|1888 1961 1948
bace_train_random_finetune|0|1209|Auc|0.893970893970894 0.8768191268191268 0.8875606375606375|20 31 22
bace_train_T7_finetune|0|1209|Auc|0.8892931392931392 0.8756063756063756 0.8858281358281358|20 11 12
bace_train_V0_finetune|0|1209|Auc|0.879071379071379 0.8811503811503811 0.8646916146916147|35 14 13
bace_train_V0_frozen|0|1209|Auc|0.8707553707553708 0.8762993762993763 0.8752598752598753|1612 1784 1848
bace_train_V2T5_finetune|0|1209|Auc|0.8901593901593902 0.8771656271656272 0.8721413721413722|20 32 12
bace_train_V2T5_frozen|0|1209|Auc|0.852044352044352 0.8529106029106029 0.853083853083853|1864 1580 1635
bbbp_all_random_finetune|0|2002|Auc|0.8248580278781621 0.8506711409395973 0.865771812080537|39 10 8
bbbp_all_T7_finetune|0|2002|Auc|0.8631905007743934 0.8553175012906558 0.8718378936499742|5 10 8
bbbp_all_V0_finetune|0|2002|Auc|0.8718378936499741 0.8683531233866805 0.8409912235415591|10 8 39
bbbp_all_V0_frozen|0|2002|Auc|0.768327310273619 0.7601316468766134 0.7594217862674237|1661 1115 1343
bbbp_all_V2T5_finetune|0|2002|Auc|0.8642230252968508 0.8377645844088797 0.8262777490965411|11 38 33
bbbp_train_random_finetune|0|1598|Auc|0.8655782137325762 0.8588022715539494 0.8302142488384099|11 9 38
bbbp_train_T7_finetune|0|1598|Auc|0.8668043366029943 0.8515745998967477 0.8678368611254517|10 10 8
bbbp_train_V0_finetune|0|1598|Auc|0.8642230252968508 0.8635776974703149 0.8649974186886938|7 9 8
bbbp_train_V0_frozen|0|1598|Auc|0.7863964894166237 0.785234899328859 0.7820727929788333|1992 1974 1970
bbbp_train_V2T5_finetune|0|1598|Auc|0.8660299432111513 0.8598347960764068 0.8646102219927723|11 10 8
freesolv_all_random_finetune|0|642|RMSE|0.6242268085479736 0.6285333633422852 0.638107419013977|35 33 21
freesolv_all_T7_finetune|0|642|RMSE|0.6577895879745483 0.6263079047203064 0.6014498472213745|36 32 37
freesolv_all_V0_finetune|0|642|RMSE|0.5699775218963623 0.5812037587165833 0.6552509665489197|34 32 21
freesolv_all_V0_frozen|0|642|RMSE|0.6488516926765442 0.6495245099067688 0.6693400144577026|1613 1988 1993
freesolv_all_V2T5_finetune|0|642|RMSE|0.6596173048019409 0.6201779842376709 0.7097961902618408|35 33 21
freesolv_train_random_finetune|0|513|RMSE|0.6324694156646729 0.6315145492553711 0.6281081438064575|35 39 22
freesolv_train_T7_finetune|0|513|RMSE|0.6774695515632629 0.6525283455848694 0.6635855436325073|35 11 18
freesolv_train_V0_finetune|0|513|RMSE|0.6429344415664673 0.6357541680335999 0.7181466817855835|35 35 21
freesolv_train_V0_frozen|0|513|RMSE|0.6738933324813843 0.6563527584075928 0.6429677605628967|1196 1672 1286
freesolv_train_V2T5_finetune|0|513|RMSE|0.677944540977478 0.6100972890853882 0.6527363657951355|35 39 22
""".strip()

# --- earlier 300ep preliminary probes (superseded by the 1000ep sweep) ---
# name|metric|vals  (finetune unless *_frozen)
RAW_PRELIM = """
bace_ft_v0|Auc|0.8768191268191268 0.8731808731808731 0.8671171171171171
bace_ft_v2t5|Auc|0.8721413721413721 0.8766458766458767 0.8624393624393624
bace_ft_t7|Auc|0.8778586278586278 0.8830561330561331 0.8816701316701316
bace_ft_random|Auc|0.875952875952876 0.8927581427581427 0.8735273735273735
freesolv_ft_v0|RMSE|0.5699793100357056 0.5813742876052856 0.6572732329368591
freesolv_ft_v2t5|RMSE|0.7448824644088745 0.5949231386184692 0.7060078978538513
freesolv_ft_t7|RMSE|0.6545538306236267 0.6488881707191467 0.616861879825592
freesolv_ft_random|RMSE|0.6437385082244873 0.578412652015686 0.6585830450057983
bbbp_ft_v0|Auc|0.8275684047496128 0.8480898296334538 0.8420237480640165
bbbp_ft_v2t5|Auc|0.849767681982447 0.8495095508518329 0.8517036654620548
bbbp_ft_t7|Auc|0.856350025813113 0.8544140423335055 0.8558337635518843
bbbp_ft_random|Auc|0.8489932885906041 0.8625451729478575 0.8542849767681983
bace_frozen_ctrl_v0|Auc|0.843901593901594 0.8322938322938322 0.8298683298683299
""".strip()


def stats(vals):
    return {
        "seeds": vals,
        "mean": round(statistics.mean(vals), 4),
        "std": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
        "n": len(vals),
    }


def parse_sweep():
    jobs = {}
    for line in RAW_SWEEP.splitlines():
        name, rc, nmol, metric, vals, eps = line.split("|")
        ds, split, arm, mode = name.rsplit("_", 3)
        seeds = [float(x) for x in vals.split()]
        epochs = [int(x) for x in eps.split()]
        jobs[name] = {
            "dataset": ds, "split": split, "arm": arm, "mode": mode,
            "n_pretrain_mol": int(nmol), "metric": metric, "rc": int(rc),
            "best_epochs": epochs, **stats(seeds),
        }
    return jobs


def parse_prelim():
    out = {}
    for line in RAW_PRELIM.splitlines():
        name, metric, vals = line.split("|")
        out[name] = {"metric": metric, **stats([float(x) for x in vals.split()])}
    return out


def m(jobs, name):
    return jobs[name]["mean"] if name in jobs else None


def build():
    jobs = parse_sweep()
    prelim = parse_prelim()

    # frozen-vs-finetune V0 table
    fvf = []
    for ds, met in [("bace", "Auc"), ("bbbp", "Auc"), ("freesolv", "RMSE")]:
        for split in ("all", "train"):
            fr = m(jobs, f"{ds}_{split}_V0_frozen")
            ft = m(jobs, f"{ds}_{split}_V0_finetune")
            delta = round(ft - fr, 4) if (fr is not None and ft is not None) else None
            fvf.append({"dataset": ds, "metric": met, "split": split,
                        "frozen": fr, "finetune": ft, "finetune_minus_frozen": delta})

    # 3D arms finetune table
    arms3d = []
    for ds in ("bace", "bbbp", "freesolv"):
        row = {"dataset": ds, "metric": jobs[f"{ds}_all_V0_finetune"]["metric"]}
        for split in ("all", "train"):
            for arm in ("V0", "V2T5", "T7", "random"):
                row[f"{arm}_{split}"] = m(jobs, f"{ds}_{split}_{arm}_finetune")
        arms3d.append(row)

    archive = {
        "experiment": "Inductive-protocol sweep: pretrain_split={all,train} x arm x mode @ 1000 epochs",
        "date": "2026-06-01",
        "machine": "thk A6000 (GPU 0-3), conda env specmol",
        "protocol": {
            "pretrain_epochs": 1000, "eval_seeds": [9, 19, 29], "batch_size": 256,
            "K": 10, "hid_dim": 512, "frozen_eval_epochs": 2000,
            "finetune": "--ft_max_epochs 40 --ft_patience 15 --ft_gate_lr 1e-2 "
                        "--ft_encoder_lr 1e-4 --ft_head_lr 1e-3",
            "pretrain_split_flag": "main_pretrain.py --pretrain_split (added 2026-06-01); "
                                   "'all'=transductive (train+valid+test structures), "
                                   "'train'=inductive (encoder never sees valid/test)",
            "n_seeds": "3 init seeds on a single fixed split (variance is init-only, not split)",
        },
        "provenance": {
            "runner": "run_overnight_sweep.sh (slot scheduler, 4 GPUs)",
            "logs": "thk:~/specmol/logs/sweep_*.log",
            "data": {
                "bace": "down_task_bace_v2 (single Uni-Mol fold-0)",
                "freesolv": "down_task_freesolv_unimol_v2 (513/64/65)",
                "bbbp": "down_task_bbbp_v2 (1598/203/201; byte-identical to canonical)",
                "bace_scaffold": "P6: per-seed DeepChem ScaffoldSplitter from dataset/bace/raw/smiles.csv",
            },
        },
        "jobs": jobs,
        "preliminary_300ep": {
            "note": "earlier same-night probes at 300 epochs; SUPERSEDED by the 1000ep sweep, "
                    "kept for cross-check. bace_frozen_ctrl was the dedicated frozen control.",
            "results": prelim,
        },
        "tables": {"frozen_vs_finetune_V0": fvf, "finetune_3D_arms": arms3d},
        "decision_gates": {
            "gate1_inductive_preserves_conclusions": {
                "verdict": "YES",
                "note": "3D=random holds under both all and train (all 6 finetune cells tie within "
                        "seed noise); finetune numbers barely move all->train for classification.",
            },
            "gate2_real_finetune_vs_frozen_delta": {
                "verdict": "+0.00 to +0.10, dataset-dependent (the '+11' was a cross-protocol artifact vs paper 0.758)",
                "bbbp": "finetune beats frozen by +0.079..+0.097 (both protocols)",
                "bace": "all: +0.064 ; train: +0.001 (tie)",
                "freesolv": "all: finetune better by 0.054 RMSE ; train: finetune worse by 0.008 (tie)",
            },
            "gate3_leakage_magnitude_train_vs_all_finetune": {
                "verdict": "classification ~0, FreeSolv regression moderate",
                "bace_V0": round(m(jobs, "bace_train_V0_finetune") - m(jobs, "bace_all_V0_finetune"), 4),
                "bbbp_V0": round(m(jobs, "bbbp_train_V0_finetune") - m(jobs, "bbbp_all_V0_finetune"), 4),
                "freesolv_V0_rmse": round(m(jobs, "freesolv_train_V0_finetune") - m(jobs, "freesolv_all_V0_finetune"), 4),
            },
            "gate4_0758_reconciliation_P6": {
                "verdict": "0.758 does NOT reproduce; it is an under-trained-probe artifact",
                "P6_scaffold_3split_frozen_V0_mean": m(jobs, "bace_scaffold_all_V0_frozen"),
                "single_fold_frozen_V0_all_mean": m(jobs, "bace_all_V0_frozen"),
                "paper_canonical": 0.758,
                "note": "P6 reproduces the canonical protocol (3 scaffold splits, all-pretrain, frozen V0, "
                        "1000ep) -> 0.823, not 0.758. Properly-trained frozen V0 ~0.82 regardless of split.",
            },
        },
        "headline": (
            "The paper's 'deep model is weak (BACE V0 0.758)' is a measurement artifact of an "
            "under-trained frozen probe -- 0.758 does not reproduce even under its own protocol "
            "(->0.823). Properly measured, deep V0 reaches BACE finetune 0.886 (~RF 0.894, tie) and "
            "BBBP finetune 0.864 (>RF 0.81). 3D injection (V2-T5/T7) = random = V0 robustly, including "
            "under the clean inductive (train-only) pretraining protocol. Transductive pretraining "
            "('all') gives ~0 benefit to classification finetune and only a moderate FreeSolv-regression edge."
        ),
    }

    json_path = OUT / "inductive_sweep_2026-06-01.json"
    json_path.write_text(json.dumps(archive, indent=2), encoding="utf-8")

    # markdown
    md = []
    md.append("# Inductive-protocol sweep archive (2026-06-01)\n")
    md.append(archive["headline"] + "\n")
    md.append("**Protocol**: 1000ep pretrain, n=3 init seeds, single fixed split; "
              "frozen=2000ep LogReg, finetune=encoder unfrozen (40ep, patience 15). "
              "`--pretrain_split all|train` controls transductive vs inductive pretraining.\n")
    md.append("## Frozen vs Finetune (V0)\n")
    md.append("| dataset | metric | split | frozen | finetune | Δ(ft−fr) |")
    md.append("|---|---|---|---|---|---|")
    for r in fvf:
        md.append(f"| {r['dataset']} | {r['metric']} | {r['split']} | {r['frozen']} | "
                  f"{r['finetune']} | {r['finetune_minus_frozen']:+} |")
    md.append("\n## Finetune 3D arms (mean)\n")
    md.append("| dataset | metric | V0 all/train | V2T5 all/train | T7 all/train | random all/train |")
    md.append("|---|---|---|---|---|---|")
    for r in arms3d:
        def cell(a):
            return f"{r[a+'_all']} / {r[a+'_train']}"
        md.append(f"| {r['dataset']} | {r['metric']} | {cell('V0')} | {cell('V2T5')} | "
                  f"{cell('T7')} | {cell('random')} |")
    md.append("\n## Decision gates\n")
    for k, v in archive["decision_gates"].items():
        md.append(f"### {k}")
        md.append(f"- **verdict**: {v['verdict']}")
        for kk, vv in v.items():
            if kk != "verdict":
                md.append(f"- {kk}: {vv}")
        md.append("")
    md.append("## Provenance\n")
    md.append(f"- runner: `{archive['provenance']['runner']}`")
    md.append(f"- logs: `{archive['provenance']['logs']}`")
    md.append(f"- full per-seed data: `inductive_sweep_2026-06-01.json` ({len(jobs)} jobs, all rc=0)")
    (OUT / "inductive_sweep_2026-06-01.md").write_text("\n".join(md), encoding="utf-8")

    print(f"[ok] wrote {json_path.name} ({len(jobs)} jobs) + .md")
    print(f"     gate3 leakage (train-all, finetune V0): "
          f"bace={archive['decision_gates']['gate3_leakage_magnitude_train_vs_all_finetune']['bace_V0']} "
          f"bbbp={archive['decision_gates']['gate3_leakage_magnitude_train_vs_all_finetune']['bbbp_V0']} "
          f"freesolv_rmse={archive['decision_gates']['gate3_leakage_magnitude_train_vs_all_finetune']['freesolv_V0_rmse']}")


if __name__ == "__main__":
    build()
