#!/usr/bin/env python3
"""Test if adding jobs from existing jobs to new jobs works."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _framework.helpers import (
    green_text,
    red_text,
    yellow_text,
)


REPO_ROOT = THIS_DIR.parent


def main(argv=None) -> int:
    omniopt_call = os.environ.get("OMNIOPT_CALL", "./omniopt")
    os.environ["OMNIOPT_CALL"] = omniopt_call
    num_gpus = 1 if shutil.which("nvidia-smi") else 0

    this_test_name = "adding_old_jobs_to_new_jobs"
    old_runs_dir = REPO_ROOT / "runs" / this_test_name
    if old_runs_dir.exists():
        yellow_text(f"Deleting {old_runs_dir}...")
        shutil.rmtree(old_runs_dir)
        yellow_text(f"Deleted {old_runs_dir}")

    base_cmd = [
        omniopt_call,
        "--live_share", "--send_anonymized_usage_stats",
        "--partition", "alpha",
        f"--experiment_name={this_test_name}",
        "--mem_gb=4", "--time=60", "--worker_timeout=5",
        "--max_eval", "2", "--num_parallel_jobs", "1",
        f"--gpus={num_gpus}",
        "--run_program",
        "Li8udGVzdHMvb3B0aW1pemF0aW9uX2V4YW1wbGUgLS1pbnRfcGFyYW09JyUoaW50X3BhcmFtKScgLS1mbG9hdF9wYXJhbT0nJShmbG9hdF9wYXJhbSknIC0tY2hvaWNlX3BhcmFtPSclKGNob2ljZV9wYXJhbSknICAtLWludF9wYXJhbV90d289JyUoaW50X3BhcmFtX3R3byknIC0tbnJfcmVzdWx0cz0x",
        "--parameter", "int_param range -100 10 int",
        "--parameter", "float_param range -100 10 float",
        "--parameter", "choice_param choice 1,2,4,8,16,hallo",
        "--parameter", "int_param_two range -100 10 int",
        "--follow", "--num_random_steps", "1",
        "--model", "BOTORCH_MODULAR", "--auto_exclude_defective_hosts",
    ]

    start = time.time()
    for _ in range(2):
        proc = subprocess.run(base_cmd, cwd=str(REPO_ROOT))
        if proc.returncode != 0:
            red_text(f"run_first_job(s) failed with exit_code {proc.returncode}, wanted 0")
            return 1

    cmd2 = list(base_cmd) + [
        "--load_data_from_existing_jobs",
        f"{old_runs_dir}/0", f"{old_runs_dir}/1",
    ]
    proc2 = subprocess.run(cmd2, cwd=str(REPO_ROOT))
    if proc2.returncode != 0:
        red_text(f"run_first_job(s) failed with exit_code {proc2.returncode}, wanted 0")
        return 2

    last_csv = old_runs_dir / "2" / "results.csv"
    if not last_csv.exists():
        red_text(f"{last_csv} not found")
        return 3

    line_count = sum(1 for _ in open(last_csv))
    wanted = 7
    if line_count != wanted:
        red_text(f"{last_csv} must have {wanted} lines, but has {line_count}")
        return 4

    elapsed = int(time.time() - start)
    green_text(f"Test ok. Took {elapsed} seconds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
