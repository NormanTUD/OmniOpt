#!/usr/bin/env python3
"""Run an OmniOpt2 job with 2 result objectives, cancel it, and test if the
Pareto-front is calculated correctly afterward."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _framework.helpers import (
    command_exists,
    green_text,
    red_text,
)


REPO_ROOT = THIS_DIR.parent


def main(argv=None) -> int:
    num_gpus = 1 if shutil.which("nvidia-smi") else 0

    run_dir = REPO_ROOT / "runs" / "__main__tests__BOTORCH_MODULAR___nogridsearch_nr_results_2"
    if run_dir.exists():
        shutil.rmtree(run_dir)

    os.environ["DIE_AFTER_THIS_NR_OF_DONE_JOBS"] = "1"

    cmd = (
        f"{REPO_ROOT}/.tests/start_simple_optimization_run.py "
        f"--num_parallel_jobs=1 --gpus={num_gpus} --num_random_steps=1 "
        "--max_eval=200 --mem_gb=4 --generate_all_jobs_at_once --random_sem "
        "--nr_results=2 --follow"
    )
    proc = subprocess.run(cmd, shell=True, cwd=str(REPO_ROOT))
    if proc.returncode != 34:
        red_text(
            f"post_job_pareto_front_calculation: The OmniOpt2 job failed. "
            f"It should have exited with exit code 34, but had {proc.returncode}."
        )
        return 1

    proc2 = subprocess.run(
        ["./omniopt", "--calculate_pareto_front_of_job", str(run_dir / "0")],
        cwd=str(REPO_ROOT),
    )
    if proc2.returncode != 0:
        red_text("post_job_pareto_front_calculation: The OmniOpt2 job failed.")
        return 2

    json_file = run_dir / "0" / "pareto_idxs.json"
    if not json_file.exists():
        red_text(f"post_job_pareto_front_calculation: {json_file} does not exist")
        return 3

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        red_text(
            f"post_job_pareto_front_calculation: The file '{json_file}' is "
            f"not valid JSON: {exc}"
        )
        return 4

    green_text("post_job_pareto_front_calculation: Test OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
