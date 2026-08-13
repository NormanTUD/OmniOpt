#!/usr/bin/env python3
"""Test the Tree Parzen Estimator model (TPE)."""

from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="Run TPE test", add_help=False)
    parser.add_argument("--max_eval", type=int, default=2)
    parser.add_argument("--num_random_steps", type=int, default=1)
    parser.add_argument("--num_parallel_jobs", type=int, default=1)
    parser.add_argument("--mem_gb", type=int, default=4)
    parser.add_argument("--testname", type=str, default="TPE")
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--additional", type=str, default="")
    parser.add_argument("--help", "-h", action="store_true")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if args.help:
        parser.print_help()
        return 0

    num_gpus = args.gpus if args.gpus is not None else (
        1 if shutil.which("nvidia-smi") else 0
    )

    rundir = REPO_ROOT / "runs" / args.testname
    csv_path = rundir / "0" / "results.csv"

    # First call: MOO test (TPE does not support nr_results>1, should exit 108).
    if rundir.exists():
        shutil.rmtree(rundir)

    cmd1 = [
        f"{REPO_ROOT}/.tests/start_simple_optimization_run.py",
        "--max_eval=2",
        "--num_parallel_jobs=1",
        "--num_random_steps=1",
        f"--mem_gb={args.mem_gb}",
        "--generate_all_jobs_at_once",
        "--follow",
        f"--additional_parameter=--model=TPE {args.additional}",
        f"--testname={args.testname}",
        f"--gpus={num_gpus}",
        "--nr_results=2",
    ]
    proc = subprocess.run(cmd1, cwd=str(REPO_ROOT))
    if proc.returncode != 108:
        red_text(
            f"Exit code for MOO TPE test should be 108, since it is not "
            f"supported, but is {proc.returncode}"
        )
        return 5

    # Second call: actual TPE run.
    if rundir.exists():
        shutil.rmtree(rundir)

    cmd2 = [
        f"{REPO_ROOT}/.tests/start_simple_optimization_run.py",
        f"--max_eval={args.max_eval}",
        f"--num_parallel_jobs={args.num_parallel_jobs}",
        f"--num_random_steps={args.num_random_steps}",
        f"--mem_gb={args.mem_gb}",
        "--generate_all_jobs_at_once",
        "--follow",
        f"--additional_parameter=--model=TPE {args.additional}",
        f"--testname={args.testname}",
        f"--gpus={num_gpus}",
    ]
    proc2 = subprocess.run(cmd2, cwd=str(REPO_ROOT))
    if proc2.returncode != 0:
        red_text(f"TPE run failed with exit code {proc2.returncode}. Test failed.")
        return 1

    if not csv_path.exists():
        red_text(f"{csv_path} could not be found.")
        return 2

    expected = args.num_random_steps + args.max_eval
    actual = sum(1 for _ in open(csv_path))
    if actual != expected:
        red_text(f"{csv_path} does not contain {expected} lines of results")
        return 3

    content = csv_path.read_text(encoding="utf-8", errors="ignore")
    if "EXTERNAL_GENERATOR" not in content:
        red_text(f"{csv_path}: does not contain EXTERNAL_GENERATOR")
        return 4

    green_text("TPE test OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
