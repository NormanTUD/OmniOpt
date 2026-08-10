#!/usr/bin/env python3
"""Test if OmniOpt2 reacts properly if stuff goes wrong."""

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
    parser = argparse.ArgumentParser(description="Wronggoing stuff test", add_help=False)
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

    which_programs = ["divide_by_0"]
    if not args.nosuccess:
        which_programs += [
            "simple_ok", "perl", "exit_code_137", "result_but_exit_code_stdout_stderr",
            "exit_code_stdout", "signal_but_has_output", "exit_code_no_output",
            "exit_code_stdout_stderr", "module_not_found", "signal", "empty",
            "file_does_not_exist", "sleep_10_and_result_exit_252", "no_chmod_x",
            "no_shebang", "perl_module_fail", "segfault", "wrong_arch",
            "python_name_error", "syntax_error", "tensorflow_syntax_error",
            "force_keyboard_interrupt",
        ]

    which_programs_string = ",".join(which_programs)
    number_of_evals = len(which_programs)
    experiment_name = "test_wronggoing_stuff"
    if args.nosuccess:
        experiment_name += "_nosuccess"

    omniopt_call = os.environ.get("OMNIOPT_CALL", "./omniopt")
    cmd = [
        omniopt_call,
        "--max_nr_of_zero_results", "3",
        "--live_share",
        "--send_anonymized_usage_stats",
        "--partition=alpha",
        f"--gpus={num_gpus}",
        f"--experiment_name={experiment_name}",
        "--mem_gb=1",
        "--time=120",
        "--worker_timeout=1",
        f"--max_eval={number_of_evals}",
        "--num_parallel_jobs=10",
        "--run_program", _b64("./.tests/test_wronggoing_stuff.bin/bin/%(program)"),
        "--parameter", "program", "choice", which_programs_string,
        f"--num_random_steps={args.num_random_steps}",
        "--follow",
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
