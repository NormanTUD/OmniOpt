#!/usr/bin/env python3
"""Run the example_network (Python replacement for ``run.sh``).

Dispatches to ``train.py`` or ``predict.py`` based on the ``--train`` and
``--predict`` flags, prints a runtime summary and forwards any remaining
arguments to the underlying script.

The original bash version also wired up a debug ``trap`` and HPC ``lmod``
modules.  These are not portable to Python and were specific to the
development workstation of the original author; they are intentionally
dropped here.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = THIS_DIR / "train.py"
PREDICT_SCRIPT = THIS_DIR / "predict.py"

GREEN = "\033[0;32m"
RED = "\033[0;31m"
RESET = "\033[0m"


def _color(text: str, color: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{color}{text}{RESET}"


def green(text: str) -> None:
    print(_color(text, GREEN), flush=True)


def red(text: str) -> None:
    print(_color(text, RED), file=sys.stderr, flush=True)


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Run the example_network train/predict scripts.",
        add_help=True,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--train", action="store_true",
                       help="Run train.py (default).")
    group.add_argument("--predict", action="store_true",
                       help="Run predict.py.")
    parser.add_argument("--debug", action="store_true",
                        help="Reserved for compatibility with the old run.sh.")
    parser.add_argument("rest", nargs=argparse.REMAINDER,
                        help="Arguments forwarded to train.py / predict.py.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    os.environ.setdefault("install_tests", "1")

    if argv is None:
        argv = sys.argv[1:]

    args = _parse_args(argv)

    if args.predict:
        mode = "predict"
        target = PREDICT_SCRIPT
    else:
        mode = "train"
        target = TRAIN_SCRIPT

    if not target.exists():
        red(f"{target.name} not found in {THIS_DIR}")
        return 1

    forwarded = args.rest
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    green(f"Running {mode}: {target.name} {' '.join(forwarded)}")
    start = time.time()
    try:
        result = subprocess.run([sys.executable, str(target), *forwarded])
    except KeyboardInterrupt:
        red("Interrupted by user")
        return 130
    elapsed = int(time.time() - start)
    green(f"RUNTIME: {elapsed} seconds ({mode})")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
