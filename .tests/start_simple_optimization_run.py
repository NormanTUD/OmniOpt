#!/usr/bin/env python3
"""Start a simple optimization based on the shekel-function.

This is the Python equivalent of the bash .tests/start_simple_optimization_run
script. It accepts the same CLI flags and builds the same omniopt command.
"""

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

from _framework.helpers import (
    green_text,
    red_text,
)


REPO_ROOT = THIS_DIR.parent


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def _parse_int(value: str, name: str) -> int:
    try:
        return int(value)
    except ValueError:
        red_text(f"Error: --{name} must be an integer: {value}")
        sys.exit(100)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Start a simple optimization run.", add_help=False,
    )
    parser.add_argument("--max_eval", type=int, default=None)
    parser.add_argument("--mem_gb", type=int, default=None)
    parser.add_argument("--num_parallel_jobs", type=int, default=None)
    parser.add_argument("--num_random_steps", type=int, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--gridsearch", action="store_true")
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--time", type=int, default=None)
    parser.add_argument("--allow_failure", action="store_true")
    parser.add_argument("--force_local_execution", action="store_true")
    parser.add_argument("--all_float", action="store_true")
    parser.add_argument("--flame_graph", action="store_true")
    parser.add_argument("--one_param", action="store_true")
    parser.add_argument("--two_params", action="store_true")
    parser.add_argument("--nr_results", type=int, default=None)
    parser.add_argument("--seed", type=str, default=None)
    parser.add_argument("--additional_parameter", type=str, default="")
    parser.add_argument("--alternate_min_max", action="store_true")
    parser.add_argument("--force_choice_for_ranges", action="store_true")
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--generate_all_jobs_at_once", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--revert_to_random_when_seemingly_exhausted", action="store_true",
    )
    parser.add_argument("--testname", type=str, default=None)
    parser.add_argument("--show_ram_every_n_seconds", type=int, default=None)
    parser.add_argument("--random_sem", action="store_true")
    parser.add_argument("--skip_search", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--help", "-h", action="store_true")

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if args.help:
        parser.print_help()
        return 0

    # Defaults.
    num_gpus = args.gpus if args.gpus is not None else (
        1 if shutil.which("nvidia-smi") else 0
    )
    has_sbatch = shutil.which("sbatch") is not None
    max_eval = args.max_eval if args.max_eval is not None else (2 if not has_sbatch else 30)
    num_parallel_jobs = (
        args.num_parallel_jobs if args.num_parallel_jobs is not None else (1 if not has_sbatch else 20)
    )
    num_random_steps = (
        args.num_random_steps if args.num_random_steps is not None else num_parallel_jobs
    )
    mem_gb = args.mem_gb if args.mem_gb is not None else 4
    model = args.model if args.model else "BOTORCH_MODULAR"
    time_limit = args.time if args.time is not None else 60
    nr_results = args.nr_results if args.nr_results is not None else 1
    random_sem_str = "--random_sem" if args.random_sem else ""
    show_ram = args.show_ram_every_n_seconds if args.show_ram_every_n_seconds is not None else 0

    # Validation.
    if args.one_param and args.two_params:
        red_text("--one_param and --two_params cannot be combined.")
        return 1
    if args.one_param and args.all_float:
        red_text("--one_param cannot be used with --all_float.")
        return 1
    if args.two_params and args.all_float:
        red_text("--two_params cannot be used with --all_float.")
        return 1
    if num_parallel_jobs < 1:
        red_text(f"num_parallel_jobs must be larger than 1, is {num_parallel_jobs}")
        return 1
    if max_eval < 1:
        red_text(f"max_eval must be larger than 1, is {max_eval}")
        return 1

    omniopt_call = os.environ.get("OMNIOPT_CALL", "./omniopt")
    partition = "alpha"
    testname = f"__main__tests__{model}__"
    if args.force_local_execution:
        testname += "_local"
    testname += "_gridsearch" if args.gridsearch else "_nogridsearch"
    if args.one_param:
        testname = f"__main__tests__{model}__one_param"
        if args.force_local_execution:
            testname += "_local"
        testname += "_gridsearch" if args.gridsearch else "_nogridsearch"
    if nr_results != 1:
        testname = f"{testname}_nr_results_{nr_results}"
    if args.testname:
        testname = args.testname

    # Build the run_program and base cmd based on the variant.
    if args.one_param:
        run_program_text = (
            f"./.tests/optimization_example.py {random_sem_str} "
            "--int_param='%(int_param)' --float_param='1' "
            "--choice_param='1' --int_param_two='1'"
        )
        if args.allow_failure:
            run_program_text += " --fail_or_not=%(fail_or_not)"
        run_program = _b64(run_program_text)
        cmd = [
            omniopt_call, "--live_share", "--send_anonymized_usage_stats",
            "--partition", partition, "--experiment_name", testname,
            f"--mem_gb={mem_gb}", "--time", str(time_limit),
            "--worker_timeout=5", "--max_eval", str(max_eval),
            "--num_parallel_jobs", str(num_parallel_jobs),
            "--gpus", str(num_gpus),
            "--run_program", run_program,
            "--parameter", "int_param", "range", "-100", "10", "int",
            "--num_random_steps", str(num_random_steps),
            "--model", model,
        ]
    elif args.all_float:
        run_program_text = (
            f".tests/optimization_example_all_float.py {random_sem_str} "
            "--x=%(x) --y=%(y) --z=%(z) --a=%(a)"
        )
        if args.allow_failure:
            run_program_text += " --fail_or_not=%(fail_or_not)"
        run_program = _b64(run_program_text)
        cmd = [
            omniopt_call, "--partition", partition,
            "--experiment_name=example_all_float",
            f"--mem_gb={mem_gb}", "--time", str(time_limit),
            "--worker_timeout=5", "--max_eval", str(max_eval),
            "--num_parallel_jobs", str(num_parallel_jobs),
            f"--gpus={num_gpus}",
            "--num_random_steps", str(num_random_steps),
            "--send_anonymized_usage_stats",
            "--run_program", run_program,
            "--cpus_per_task=1", "--nodes_per_job=1",
            "--model=BOTORCH_MODULAR", "--run_mode=local",
            "--parameter", "x", "range", "-1000", "1000", "float",
            "--parameter", "y", "range", "-1000", "1000", "float",
            "--parameter", "z", "range", "-1000", "1000", "float",
            "--parameter", "a", "range", "-1000", "1000", "float",
            "--live_share",
        ]
    elif args.two_params:
        run_program_text = (
            f"./.tests/optimization_example.py {random_sem_str} "
            "--int_param='%(int_param)' --float_param='1' "
            "--choice_param='1' --int_param_two='1'"
        )
        if args.allow_failure:
            run_program_text = (
                f"./.tests/optimization_example.py {random_sem_str} "
                "--int_param='%(int_param)' --float_param='%(float_param)' "
                "--choice_param='1' --int_param_two='1' "
                "--fail_or_not=%(fail_or_not)"
            )
        run_program = _b64(run_program_text)
        cmd = [
            omniopt_call, "--live_share", "--send_anonymized_usage_stats",
            "--partition", partition, "--experiment_name", testname,
            f"--mem_gb={mem_gb}", "--time", str(time_limit),
            "--worker_timeout=5", "--max_eval", str(max_eval),
            "--num_parallel_jobs", str(num_parallel_jobs),
            "--gpus", str(num_gpus),
            "--run_program", run_program,
            "--parameter", "int_param", "range", "-100", "10", "int",
            "--parameter", "float_param", "range", "-100", "10", "float",
            "--num_random_steps", str(num_random_steps),
            "--model", model,
        ]
    else:
        run_program_text = (
            f"./.tests/optimization_example.py {random_sem_str} "
            "--int_param='%(int_param)' --float_param='%(float_param)' "
            f"--choice_param='%(choice_param)' --int_param_two='%(int_param_two)' "
            f"--nr_results={nr_results}"
        )
        if args.allow_failure:
            run_program_text += " --fail_or_not=%(fail_or_not)"
        run_program = _b64(run_program_text)
        cmd = [
            omniopt_call, "--live_share", "--send_anonymized_usage_stats",
            "--partition", partition, "--experiment_name", testname,
            f"--mem_gb={mem_gb}", "--time", str(time_limit),
            "--worker_timeout=5", "--max_eval", str(max_eval),
            "--num_parallel_jobs", str(num_parallel_jobs),
            "--gpus", str(num_gpus),
            "--run_program", run_program,
            "--parameter", "int_param", "range", "-100", "10", "int",
            "--parameter", "float_param", "range", "-100", "10", "float",
            "--parameter", "choice_param", "choice", "1,2,4,8,16,hallo",
            "--parameter", "int_param_two", "range", "-100", "10", "int",
            "--num_random_steps", str(num_random_steps),
            "--model", model,
            "--auto_exclude_defective_hosts",
        ]

    if args.gridsearch:
        cmd.append("--gridsearch")
    if args.allow_failure:
        cmd.extend(["--parameter", "fail_or_not", "choice", "0,1"])
    if args.force_local_execution:
        cmd.append("--force_local_execution")

    if nr_results != 1:
        cmd.append("--result_names")
        for i in range(1, nr_results + 1):
            if args.alternate_min_max and i % 2 == 0:
                cmd.append(f"RESULT{i}=max")
            else:
                cmd.append(f"RESULT{i}=min")

    if args.flame_graph:
        cmd.append("--flame_graph")
    if args.debug:
        cmd.append("--debug")
    if show_ram:
        cmd.append(f"--show_ram_every_n_seconds={show_ram}")
    if args.generate_all_jobs_at_once:
        cmd.append("--generate_all_jobs_at_once")
    if args.force_choice_for_ranges:
        cmd.append("--force_choice_for_ranges")
    if args.follow:
        cmd.append("--follow")
    if args.verbose:
        cmd.append("--verbose")
    if args.revert_to_random_when_seemingly_exhausted:
        cmd.append("--revert_to_random_when_seemingly_exhausted")
    if args.skip_search:
        cmd.append("--skip_search")
    if args.seed:
        cmd.extend(["--seed", args.seed])
    if args.additional_parameter:
        cmd.append(args.additional_parameter)

    cmd.append("--show_generate_time_table")

    if not os.environ.get("DONT_SHOW_STARTUP_COMMAND"):
        green_text(" ".join(cmd))

    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
