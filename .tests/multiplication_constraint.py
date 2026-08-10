#!/usr/bin/env python3
"""Test non-standard constraints (multiplication)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _framework.helpers import green_text, red_text


REPO_ROOT = THIS_DIR.parent


def main(argv=None) -> int:
    num_gpus = 1 if shutil.which("nvidia-smi") else 0
    cmd = [
        "./omniopt",
        "--partition=alpha",
        "--experiment_name=multiplication_constraint",
        "--mem_gb=10",
        "--time=60",
        "--worker_timeout=60",
        "--max_eval=4",
        "--num_parallel_jobs=4",
        f"--gpus={num_gpus}",
        "--num_random_steps=2",
        "--follow",
        "--live_share",
        "--send_anonymized_usage_stats",
        "--result_names", "RESULT=min",
        "--run_program=cGVybCAtZSAncHJpbnQgIlJFU1VMVDogIiAuICglYSArICViKSc=",
        "--cpus_per_task=1",
        "--nodes_per_job=1",
        "--generate_all_jobs_at_once",
        "--revert_to_random_when_seemingly_exhausted",
        "--model=BOTORCH_MODULAR",
        "--n_estimators_randomforest=100",
        "--run_mode=local",
        "--occ_type=euclid",
        "--main_process_gb=8",
        "--max_nr_of_zero_results=50",
        "--slurm_signal_delay_s=0",
        "--max_failed_jobs=0",
        "--parameter", "a range 5 100 float false",
        "--parameter", "b range 5 100 float false",
        "--experiment_constraints", "YSAqIGIgPj0gMTA=",
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if proc.returncode == 0:
        green_text("Test multiplication_constraint OK")
        return 0
    red_text(f"Test multiplication_constraint failed. Exit-Code: {proc.returncode}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
