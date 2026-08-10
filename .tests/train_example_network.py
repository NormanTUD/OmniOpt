#!/usr/bin/env python3
"""Starts an OmniOpt2 run that trains an example neural network."""

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
EXAMPLE_NETWORK = REPO_ROOT / ".tests" / "example_network"


def _b64(s: str) -> str:
    return base64.b64encode(s.encode("utf-8")).decode("ascii")


def _generate_parameter(name: str, min_value, max_value, type_: str) -> str:
    if min_value == max_value:
        return f"--parameter {name} fixed {min_value}"
    return f"--parameter {name} range {min_value} {max_value} {type_}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Train example network via OmniOpt", add_help=False,
    )
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--max_eval", type=int, default=None)
    parser.add_argument("--num_random_steps", type=int, default=None)
    parser.add_argument("--num_parallel_jobs", type=int, default=None)
    parser.add_argument("--mem_gb", type=int, default=None)
    parser.add_argument("--runtime", type=int, default=None)
    parser.add_argument("--worker_timeout", type=int, default=None)
    parser.add_argument("--nr_evals_per_arm", type=int, default=1)
    parser.add_argument("--validation_split", type=float, default=0.2)
    parser.add_argument("--min_width", type=int, default=8)
    parser.add_argument("--max_width", type=int, default=64)
    parser.add_argument("--min_height", type=int, default=8)
    parser.add_argument("--max_height", type=int, default=64)
    parser.add_argument("--min_dense", type=int, default=1)
    parser.add_argument("--max_dense", type=int, default=3)
    parser.add_argument("--min_dense_units", type=int, default=16)
    parser.add_argument("--max_dense_units", type=int, default=128)
    parser.add_argument("--min_epochs", type=int, default=1)
    parser.add_argument("--max_epochs", dest="max_epochs_legacy", type=int, default=5)
    parser.add_argument("--min_conv", type=int, default=0)
    parser.add_argument("--max_conv", type=int, default=2)
    parser.add_argument("--min_conv_filters", type=int, default=8)
    parser.add_argument("--max_conv_filters", type=int, default=32)
    parser.add_argument("--min_learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_learning_rate", type=float, default=1e-2)
    parser.add_argument("--data", type=str, default="mnist")
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--gridsearch", action="store_true")
    parser.add_argument("--generate_all_jobs_at_once", action="store_true")
    parser.add_argument("--revert_to_random_when_seemingly_exhausted",
                        action="store_true")
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--result_names", type=str, nargs="+", default=None)
    parser.add_argument("--help", "-h", action="store_true")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if args.help:
        parser.print_help()
        return 0

    num_gpus = args.gpus if args.gpus is not None else (
        1 if shutil.which("nvidia-smi") else 0
    )

    omniopt_call = os.environ.get("OMNIOPT_CALL", "./omniopt")
    result_names = args.result_names or ["RESULT=min"]

    run_program = (
        f"bash {EXAMPLE_NETWORK}/run.sh --learning_rate=%(learning_rate) "
        "--epochs=%(epochs) --validation_split={args.validation_split} "
        "--width=%(width) --height=%(height) --dense=%(dense) "
        "--dense_units=%(dense_units) --conv=%(conv) "
        "--conv_filters=%(conv_filters) --data={args.data} "
        "--activation=%(activation)"
    )
    run_program_b64 = _b64(run_program)

    cmd = [
        omniopt_call,
        "--live_share", "--send_anonymized_usage_stats",
        "--partition=alpha",
        "--experiment_name=example_network",
        f"--mem_gb={args.mem_gb or 4}",
        f"--time={args.runtime or 60}",
        f"--worker_timeout={args.worker_timeout or 60}",
        f"--max_eval={args.max_eval or 2}",
        f"--num_parallel_jobs={args.num_parallel_jobs or 1}",
        f"--nr_evals_per_arm={args.nr_evals_per_arm}",
        f"--gpus={num_gpus}",
        "--run_program", run_program_b64,
        _generate_parameter("width", args.min_width, args.max_width, "int"),
        _generate_parameter("height", args.min_height, args.max_height, "int"),
        _generate_parameter("dense", args.min_dense, args.max_dense, "int"),
        _generate_parameter("dense_units", args.min_dense_units,
                            args.max_dense_units, "int"),
        _generate_parameter("epochs", args.min_epochs,
                            args.max_epochs_legacy, "int"),
        _generate_parameter("conv", args.min_conv, args.max_conv, "int"),
        _generate_parameter("conv_filters", args.min_conv_filters,
                            args.max_conv_filters, "int"),
        _generate_parameter("learning_rate", args.min_learning_rate,
                            args.max_learning_rate, "float"),
        "--parameter", "activation", "choice",
        "relu,sigmoid,swish,leaky_relu,tanh,gelu",
        "--result_names", *result_names,
        f"--num_random_steps={args.num_random_steps or 1}",
    ]

    experiment_name = "example_network_gridsearch" if args.gridsearch else "example_network"
    # Replace experiment_name in cmd
    for i, tok in enumerate(cmd):
        if tok == "--experiment_name=example_network":
            cmd[i] = f"--experiment_name={experiment_name}"

    if args.gridsearch:
        cmd.append("--gridsearch")
    if args.generate_all_jobs_at_once:
        cmd.append("--generate_all_jobs_at_once")
    if args.revert_to_random_when_seemingly_exhausted:
        cmd.append("--revert_to_random_when_seemingly_exhausted")
    if args.follow:
        cmd.append("--follow")

    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
