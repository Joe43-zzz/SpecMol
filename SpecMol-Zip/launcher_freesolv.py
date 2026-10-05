"""Launcher for FreeSolv experiments. Runs baseline + V2-T5 serially via subprocesses."""

import json
import os
import subprocess
import sys
import time
from datetime import datetime

PYTHON = r"D:\SpecMol-Zip\.venv\Scripts\python.exe"
WORKDIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(WORKDIR, "launcher_freesolv.log")
ERR_FILE = os.path.join(WORKDIR, "launcher_freesolv_err.log")

TIMEOUT = 90 * 60  # 90 minutes per cell

SEEDS = [9, 19, 29]

CELLS = []
for seed in SEEDS:
    CELLS.append({
        "name": f"baseline_seed_{seed}",
        "cmd": [PYTHON, "run_freesolv_baseline.py", "--seed", str(seed), "--gpu", "0"],
    })
for seed in SEEDS:
    CELLS.append({
        "name": f"v2t5_seed_{seed}",
        "cmd": [PYTHON, "run_freesolv_v2t5.py", "--seed", str(seed), "--gpu", "0"],
    })


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def main():
    log(f"FreeSolv launcher started. {len(CELLS)} cells to run.")
    log(f"Python: {PYTHON}")
    log(f"Workdir: {WORKDIR}")
    log(f"Timeout per cell: {TIMEOUT}s")

    status = {}

    for i, cell in enumerate(CELLS):
        name = cell["name"]
        cmd = cell["cmd"]
        log(f"\n--- Cell {i+1}/{len(CELLS)}: {name} ---")
        log(f"Command: {' '.join(cmd)}")

        t0 = time.time()
        try:
            with open(ERR_FILE, "a") as ef:
                result = subprocess.run(
                    cmd,
                    cwd=WORKDIR,
                    timeout=TIMEOUT,
                    stdout=subprocess.PIPE,
                    stderr=ef,
                    text=True,
                )
            elapsed = time.time() - t0
            # Write stdout to log
            with open(LOG_FILE, "a") as f:
                f.write(result.stdout + "\n")

            if result.returncode == 0:
                log(f"  DONE: {name} in {elapsed:.0f}s")
                status[name] = "done"
            else:
                log(f"  FAILED: {name} (exit code {result.returncode}) in {elapsed:.0f}s")
                status[name] = f"failed_rc{result.returncode}"

        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            log(f"  TIMED_OUT: {name} after {elapsed:.0f}s")
            status[name] = "timed_out"

        except Exception as e:
            elapsed = time.time() - t0
            log(f"  ERROR: {name} - {e}")
            status[name] = f"error: {e}"

    log(f"\n{'='*60}")
    log("LAUNCHER COMPLETE")
    log(f"Status: {json.dumps(status, indent=2)}")

    # Write status file
    with open(os.path.join(WORKDIR, "launcher_freesolv_status.json"), "w") as f:
        json.dump(status, f, indent=2)


if __name__ == "__main__":
    main()
