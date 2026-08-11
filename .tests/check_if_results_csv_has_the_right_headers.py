#!/usr/bin/env python3
"""Start an OmniOpt2 run and see if the results.csv has the proper headers."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _framework.helpers import red_text


REPO_ROOT = THIS_DIR.parent
RUN_DIR = REPO_ROOT / "runs" / "__main__tests__BOTORCH_MODULAR___nogridsearch" / "0"
CSV_FILE = RUN_DIR / "results.csv"

REQUIRED_COLUMNS = [
    "trial_index", "arm_name", "trial_status", "generation_node",
    "RESULT", "int_param",
]


def main(argv=None) -> int:
    if RUN_DIR.exists():
        shutil.rmtree(RUN_DIR)

    num_gpus = 1 if shutil.which("nvidia-smi") else 0

    cmd = (
        f"{REPO_ROOT}/.tests/start_simple_optimization_run.py "
        f"--num_parallel_jobs=1 --gpus={num_gpus} --num_random_steps=1 "
        f"--max_eval=1 --mem_gb=4 --generate_all_jobs_at_once --random_sem "
        "--nr_results=1 --follow"
    )
    proc = subprocess.run(cmd, shell=True, cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        return proc.returncode

    if not CSV_FILE.exists():
        red_text(f"{CSV_FILE} not found")
        return 1

    with open(CSV_FILE, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")

    missing = [c for c in REQUIRED_COLUMNS if c not in header]
    if missing:
        red_text(f"Missing columns in {CSV_FILE}: {missing}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
