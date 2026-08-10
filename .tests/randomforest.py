#!/usr/bin/env python3
"""Test if random forest works properly."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


REPO_ROOT = THIS_DIR.parent


def main(argv=None) -> int:
    rundir = REPO_ROOT / "runs" / "randomforest"
    if rundir.exists():
        shutil.rmtree(rundir)
    num_gpus = 1 if shutil.which("nvidia-smi") else 0

    cmd = [
        "./omniopt",
        "--partition=alpha",
        "--experiment_name=randomforest",
        "--mem_gb=1", "--time=60", "--worker_timeout=60",
        "--max_eval=2", "--num_parallel_jobs=1", f"--gpus={num_gpus}",
        "--num_random_steps=1", "--follow",
        "--live_share", "--send_anonymized_usage_stats",
        "--result_names", "RESULT=min",
        "--run_program=ZWNobyAiUkVTVUxUOiAlYSUoeCklKHkpJXoi",
        "--cpus_per_task=1", "--nodes_per_job=1",
        "--generate_all_jobs_at_once",
        "--revert_to_random_when_seemingly_exhausted",
        "--model=RANDOMFOREST", "--run_mode=local",
        "--occ_type=euclid", "--main_process_gb=8",
        "--max_nr_of_zero_results=1", "--slurm_signal_delay_s=0",
        "--n_estimators_randomforest=100",
        "--parameter", "x fixed 123",
        "--parameter", "y range 5431 1234 int false",
        "--parameter", "z range 0 1 float false",
        "--parameter", "a choice 1,2,3",
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        print(f"Test randomforest failed: OmniOpt2 exited with {proc.returncode} instead of 0")
        return 1

    csv = rundir / "0" / "results.csv"
    if not csv.exists():
        print(f"Test randomforest failed: {csv} could not be found")
        return 2

    with open(csv, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    if len(lines) != 3:
        print(f"The file {csv} does not contain 3 lines, but {len(lines)}:")
        print("\n".join(lines))
        return 3

    keywords = ["trial_index", "SOBOL", "RANDOMFOREST"]
    if len(lines) != len(keywords):
        print(f"The file '{csv}' contains {len(lines)} lines, but {len(keywords)} were expected.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
