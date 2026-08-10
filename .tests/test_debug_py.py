#!/usr/bin/env python3
"""Tests that the generated debug.py contains alternating minimize lines."""

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
    num_gpus = 1 if shutil.which("nvidia-smi") else 0
    testname = "test_if_debug_py_has_alternating"
    rundir = REPO_ROOT / "runs" / testname

    if rundir.exists():
        shutil.rmtree(rundir)

    cmd = (
        f"{REPO_ROOT}/.tests/start_simple_optimization_run.py "
        f"--num_parallel_jobs=1 --gpus={num_gpus} --num_random_steps=1 "
        "--max_eval=1 --mem_gb=4 --generate_all_jobs_at_once --random_sem "
        f"--nr_results=1 --follow --testname={testname} --nr_results=3 "
        "--alternate_min_max"
    )
    proc = subprocess.run(cmd, shell=True, cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        print(f"Command failed with exit code {proc.returncode}")
        return 1

    debug_file = rundir / "0" / "debug.py"
    if not debug_file.exists():
        print(f"{debug_file} not found")
        return 1

    content = debug_file.read_text(encoding="utf-8", errors="ignore")
    if content.count("minimize=True") != 2:
        print("debug.py has not 2 'minimize=True' lines")
        return 2
    if content.count("minimize=False") != 1:
        print("debug.py has not one 'minimize=False' lines")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
