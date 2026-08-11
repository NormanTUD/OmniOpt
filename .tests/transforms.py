#!/usr/bin/env python3
"""Test transforms."""

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
    parser = argparse.ArgumentParser(description="Run transforms tests", add_help=False)
    parser.add_argument("--max_eval", type=int, default=2)
    parser.add_argument("--num_random_steps", type=int, default=1)
    parser.add_argument("--num_parallel_jobs", type=int, default=2)
    parser.add_argument("--mem_gb", type=int, default=4)
    parser.add_argument("--testname", type=str, default="TRANSFORMS")
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

    cases = [
        ("no_transforms", "", 0),
        ("Cont_X_trans", "--transforms=Cont_X_trans", 3),
        ("Cont_X_trans_Y_trans", "--transforms=Cont_X_trans_Y_trans", 9),
    ]

    for suffix, transform_args, exit_base in cases:
        runname = f"{args.testname}_{suffix}"
        csv_path = REPO_ROOT / "runs" / runname / "0" / "results.csv"
        rundir = REPO_ROOT / "runs" / runname
        if rundir.exists():
            shutil.rmtree(rundir)

        cmd = [
            f"{REPO_ROOT}/.tests/start_simple_optimization_run.py",
            f"--max_eval={args.max_eval}",
            "--num_parallel_jobs=1",
            f"--num_random_steps={args.num_random_steps}",
            f"--mem_gb={args.mem_gb}",
            "--generate_all_jobs_at_once",
            "--follow",
            f"--additional_parameter={args.additional}",
            f"--testname={runname} {transform_args}",
            f"--gpus={num_gpus}",
            "--nr_results=2",
        ]
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
        if proc.returncode != 0:
            red_text(
                f"[{runname}] Exit code should have been 0, but is {proc.returncode}"
            )
            return exit_base

        if not csv_path.exists():
            red_text(f"[{runname}] {csv_path} could not be found.")
            return exit_base + 1

        expected = args.num_random_steps + args.max_eval
        actual = sum(1 for _ in open(csv_path))
        if actual != expected:
            red_text(
                f"[{runname}] Expected {expected} lines, got {actual}"
            )
            return exit_base + 2

    green_text("Transforms-Test OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
