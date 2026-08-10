#!/usr/bin/env python3
"""Testing if continuing a constrained run works."""

from __future__ import annotations

import base64
import shutil
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _framework.helpers import green_text, red_text


REPO_ROOT = THIS_DIR.parent


def _b64(s: str) -> str:
    return base64.b64encode(s.encode("utf-8")).decode("ascii")


def main(argv=None) -> int:
    num_gpus = 1 if shutil.which("nvidia-smi") else 0
    name = "test_continued_constraint"
    rundir = REPO_ROOT / "runs" / name
    if rundir.exists():
        shutil.rmtree(rundir)

    cmd1 = [
        "./omniopt",
        "--live_share",
        "--send_anonymized_usage_stats",
        "--partition", "alpha",
        "--experiment_name", name,
        "--mem_gb=4",
        "--time", "60",
        "--worker_timeout=5",
        "--max_eval", "1",
        "--num_parallel_jobs", "1",
        "--gpus", str(num_gpus),
        "--run_program", _b64(
            "./.tests/optimization_example --random_sem "
            "--int_param='%(int_param)' --float_param='%(float_param)' "
            "--choice_param='%(choice_param)' "
            "--int_param_two='%(int_param_two)' --nr_results=1"
        ),
        "--parameter", "int_param range -100 10 int",
        "--parameter", "float_param range -100 10 float",
        "--parameter", "choice_param choice 1,2,4,8,16,hallo",
        "--parameter", "int_param_two range -100 10 int",
        "--follow",
        "--num_random_steps", "1",
        "--model", "BOTORCH_MODULAR",
        "--auto_exclude_defective_hosts",
        "--generate_all_jobs_at_once",
        "--experiment_constraints",
        _b64("int_param + 2*int_param_two >= 0"),
        _b64("2*int_param_two >= 0"),
    ]
    proc = subprocess.run(cmd1, cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        red_text(f"First call failed with exit-code {proc.returncode}. Exiting.")
        return 1

    proc2 = subprocess.run(
        ["./omniopt", "--continue", f"runs/{name}/0"],
        cwd=str(REPO_ROOT),
    )
    if proc2.returncode != 0:
        red_text(f"Second call failed with exit-code {proc2.returncode}. Exiting.")
        return 1

    green_text("continued_constraints test OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
