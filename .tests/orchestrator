#!/usr/bin/env python3
"""Tests if the orchestrator runs properly."""

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
    parser = argparse.ArgumentParser(description="Orchestrator test", add_help=False)
    parser.add_argument("--num_random_steps", type=int, default=20)
    parser.add_argument("--nosuccess", action="store_true")
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--help", "-h", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if args.help:
        parser.print_help()
        return 0

    num_gpus = args.gpus if args.gpus is not None else (
        1 if shutil.which("nvidia-smi") else 0
    )
    omniopt_call = os.environ.get("OMNIOPT_CALL", "./omniopt")

    which_programs = ["simple_ok", "storage_error", "timeout_failure", "gpu_disconnected"]
    which_programs_string = ",".join(which_programs)
    number_of_evals = len(which_programs)

    cmd = [
        omniopt_call,
        "--live_share", "--send_anonymized_usage_stats",
        "--partition=alpha",
        "--experiment_name=test_orchestrator",
        "--mem_gb=5", "--time=60", "--worker_timeout=5",
        f"--max_eval={number_of_evals}",
        f"--num_parallel_jobs={number_of_evals}",
        f"--gpus={num_gpus}",
        "--run_program", _b64("./.tests/orchestrator_tests.bin/%(name)"),
        "--parameter", "name", "choice", which_programs_string,
        "--num_random_steps=1",
        "--model=BOTORCH_MODULAR",
        "--auto_exclude_defective_hosts",
        "--orchestrator_file", str(REPO_ROOT / ".tests" / "example_orchestrator_config.yaml"),
        "--seed", "1234",
        "--follow",
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
