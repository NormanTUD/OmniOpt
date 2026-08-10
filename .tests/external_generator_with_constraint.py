#!/usr/bin/env python3
"""Testing external generator with constraint."""

from __future__ import annotations

import base64
import shutil
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


REPO_ROOT = THIS_DIR.parent


def _b64(s: str) -> str:
    return base64.b64encode(s.encode("utf-8")).decode("ascii")


def main(argv=None) -> int:
    num_gpus = 1 if shutil.which("nvidia-smi") else 0
    cmd = [
        "./omniopt",
        "--partition=alpha",
        "--experiment_name=EXTERNAL_GENERATOR_with_constraints_test",
        "--mem_gb=1",
        "--time=60",
        "--worker_timeout=60",
        "--max_eval=2",
        "--num_parallel_jobs=1",
        f"--gpus={num_gpus}",
        "--num_random_steps=1",
        "--follow",
        "--live_share",
        "--send_anonymized_usage_stats",
        "--result_names", "RESULT=max",
        "--run_program=ZWNobyAiUkVTVUxUOiAlKHgpJSh5KSIgJiYgZWNobyAiUkVTVUxUMjogJXoi",
        "--cpus_per_task=1",
        "--nodes_per_job=1",
        "--generate_all_jobs_at_once",
        "--revert_to_random_when_seemingly_exhausted",
        "--model=EXTERNAL_GENERATOR",
        "--run_mode=local",
        "--occ_type=euclid",
        "--main_process_gb=8",
        "--max_nr_of_zero_results=1",
        "--slurm_signal_delay_s=0",
        "--n_estimators_randomforest=100",
        "--parameter", "x range 123 1000 int false",
        "--parameter", "y range 1234 4321",
        "--parameter", "z range 111 222 int",
        f"--external_generator={_b64(f'python3 {REPO_ROOT}/.tests/example_external.py')}",
        f"--experiment_constraint={_b64('x >= y')}",
        "--seed", "1234",
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
