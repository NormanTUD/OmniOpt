#!/usr/bin/env python3
"""Linter orchestrator (replaces .tests/linter bash script).

Runs multiple linters as defined in the config. Each linter is itself a
test in .tests/ (e.g. .tests/pylint, .tests/flake8, etc.).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _framework.helpers import (
    Colors,
    green_text,
    human_readable_time,
    red_text,
    yellow_text,
)


REPO_ROOT = THIS_DIR.parent
DEFAULT_LINTERS = ["lizard", "pylint", "bandit", "deadcode", "flake8"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run linters.", add_help=False)
    parser.add_argument("--quick", action="store_true", help="Skip slow linters.")
    parser.add_argument("--dont_fail_on_error", action="store_true",
                        help="Don't fail when an error is encountered.")
    parser.add_argument("--check_only_changed_since_last_success",
                        action="store_true",
                        help="Check only files changed since last successful run.")
    parser.add_argument("--help", "-h", action="store_true")
    parser.add_argument("linter", nargs="?", default=None,
                        help="Run only this linter.")
    parser.add_argument("files", nargs="*", help="Run linter only on these files.")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if args.help:
        print(".tests/linter.py")
        print("  --quick             Disable slow tests")
        print("  --dont_fail_on_error")
        print("  --check_only_changed_since_last_success")
        print("  [files]             Run linters only on specified files")
        print("  [linter]            Run a specific linter")
        return 0

    os.environ.setdefault("install_tests", "1")
    os.environ.setdefault("DONT_SHOW_DONT_INSTALL_MESSAGE", "1")
    if args.check_only_changed_since_last_success:
        os.environ["ONLY_CHECK_CHANGED_SINCE_LAST_COMMIT"] = "1"

    errors: List[str] = []
    linters_to_run = []
    if args.linter:
        linters_to_run = [args.linter]
    else:
        linters_to_run = list(DEFAULT_LINTERS)
        if args.quick:
            linters_to_run = [l for l in linters_to_run if l in ("lizard", "flake8", "pyflakes")]

    for linter in linters_to_run:
        if errors and not args.dont_fail_on_error:
            yellow_text(f"Skipping linter {linter} because there were previous errors...")
            continue
        yellow_text(f"Running {linter}...")
        script = REPO_ROOT / ".tests" / f"{linter}.py"
        if not script.exists():
            script = REPO_ROOT / ".tests" / linter
        cmd = [str(script)] + list(args.files)
        try:
            proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
        except FileNotFoundError:
            red_text(f"Linter {linter} not found at {script}")
            errors.append(f"{linter} not found")
            continue
        if proc.returncode != 0:
            red_text(f"{linter} failed\n")
            errors.append(f"{linter} failed")

    if not errors:
        green_text("No linter errors")
        return 0
    red_text("=> LINTERS-ERRORS =>")
    for e in errors:
        red_text(e)
    return len(errors)


if __name__ == "__main__":
    sys.exit(main())
