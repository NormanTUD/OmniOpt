#!/usr/bin/env python3
"""Test external generator."""

from __future__ import annotations

import argparse
import base64
import os
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
    parser = argparse.ArgumentParser(description="External generator test", add_help=False)
    parser.add_argument("--mem_gb", type=int, default=1)
    parser.add_argument("--time", type=int, default=60)
    parser.add_argument("--worker_timeout", type=int, default=60)
    parser.add_argument("--max_eval", type=int, default=2)
    parser.add_argument("--num_parallel_jobs", type=int, default=1)
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--num_random_steps", type=int, default=1)
    parser.add_argument("--external_generator", type=str, default=None)
    parser.add_argument("--help", "-h", action="store_true")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if args.help:
        parser.print_help()
        return 0

    num_gpus = args.gpus if args.gpus is not None else (
        1 if shutil.which("nvidia-smi") else 0
    )

    if args.external_generator:
        ext_gen = _b64(args.external_generator)
    else:
        ext_gen = _b64(f"python3 {REPO_ROOT}/.tests/example_external.py")

    cmd = [
        "./omniopt",
        "--partition=alpha",
        "--experiment_name=EXTERNAL_GENERATOR_test",
        f"--mem_gb={args.mem_gb}",
        f"--time={args.time}",
        f"--worker_timeout={args.worker_timeout}",
        f"--max_eval={args.max_eval}",
        f"--num_parallel_jobs={args.num_parallel_jobs}",
        f"--gpus={num_gpus}",
        f"--num_random_steps={args.num_random_steps}",
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
        "--parameter", "y choice 5431,1234",
        "--parameter", "z fixed 111",
        f"--external_generator={ext_gen}",
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
