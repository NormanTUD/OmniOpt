#!/usr/bin/env python3
"""Tests that multiple evaluations per arm work correctly."""

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
    testname = "multiple_evals_per_arm"
    rundir = REPO_ROOT / "runs" / testname

    if rundir.exists():
        shutil.rmtree(rundir)

    cmd = (
        f"{REPO_ROOT}/.tests/start_simple_optimization_run.py --max_eval=2 "
        "--num_parallel_jobs=1 --revert_to_random_when_seemingly_exhausted "
        "--generate_all_jobs_at_once --alternate_min_max --num_random_steps=1 "
        "--time=1200 --additional_parameter='--nr_evals_per_arm=2 --beartype' "
        f"--nr_results=2 --testname={testname}"
    )
    proc = subprocess.run(cmd, shell=True, cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        print("Running OmniOpt itself failed")
        return 1

    run = rundir / "0"
    if not run.exists():
        print(f"{run} did not exist")
        return 2
    if not (run / "arm_evals").exists():
        print(f"{run}/arm_evals did not exist")
        return 3
    arm_evals_results = run / "arm_evals_results.csv"
    if not arm_evals_results.exists():
        print(f"{arm_evals_results} did not exist")
        return 4

    line_count = sum(1 for _ in open(arm_evals_results))
    wanted = 5
    if line_count != wanted:
        print(f"{arm_evals_results} should have {wanted} lines but has {line_count}")
        return 5
    return 0


if __name__ == "__main__":
    sys.exit(main())
