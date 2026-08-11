#!/usr/bin/env python3
"""Check that the progressbar log contains SOBOL, BOTORCH_MODULAR and 2 results."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


REPO_ROOT = THIS_DIR.parent
LOGS_DIR = REPO_ROOT / "logs"


def main(argv=None) -> int:
    num_gpus = 1 if shutil.which("nvidia-smi") else 0

    cmd = (
        f"{REPO_ROOT}/.tests/start_simple_optimization_run.py "
        f"--num_parallel_jobs=2 --gpus=0 --num_random_steps=1 --max_eval=2 "
        f"--mem_gb=1 --generate_all_jobs_at_once --follow --gpus={num_gpus}"
    )
    proc = subprocess.run(cmd, shell=True, cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        print(f"start_simple_optimization_run failed with exit_code {proc.returncode}")
        return 1

    if not LOGS_DIR.is_dir():
        print("logs directory not found")
        return 1

    progress_files = sorted(
        LOGS_DIR.glob("*progressbar*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not progress_files:
        print("no progressbar log found")
        return 1

    last_progressbar = progress_files[0]
    content = last_progressbar.read_text(encoding="utf-8", errors="ignore")

    if "SOBOL" not in content:
        print(f"{last_progressbar} does not contain SOBOL")
        return 2
    if "BOTORCH_MODULAR" not in content:
        print(f"{last_progressbar} does not contain BOTORCH_MODULAR")
        return 3
    if content.count("new result") != 2:
        print(f"{last_progressbar} does not contain 2 results")
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
