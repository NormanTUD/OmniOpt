#!/usr/bin/env python3
"""Linter orchestrator (replaces .tests/linter bash script).

Runs multiple linters as defined in the config. Each linter is itself a
test in .tests/ (e.g. .tests/pylint, .tests/flake8, etc.).

By default linters run in parallel (one worker per CPU, capped at the
number of linters). Pass ``--no-parallel`` to fall back to sequential
execution, or ``--parallel=N`` to pin a specific worker count.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

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


def _resolve_linter_cmd(linter: str, extra_files: List[str]) -> Tuple[List[str], Path]:
    """Build the command for one linter. Raises ``FileNotFoundError``."""
    script = REPO_ROOT / ".tests" / f"{linter}.py"
    if not script.exists():
        script = REPO_ROOT / ".tests" / linter
    return [str(script)] + list(extra_files), script


def _run_linter(linter: str, extra_files: List[str]) -> Tuple[str, int, str]:
    """Run a single linter. Returns ``(name, returncode, error_or_empty)``."""
    try:
        cmd, _ = _resolve_linter_cmd(linter, extra_files)
    except Exception as exc:  # pragma: no cover - defensive
        return linter, 0, f"{linter} not found ({exc})"

    start = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    except FileNotFoundError as exc:
        return linter, 0, f"{linter} not found at {exc.filename}"
    except Exception as exc:  # pragma: no cover - defensive
        return linter, 1, f"{linter} crashed: {exc}"
    runtime = time.time() - start

    if proc.returncode != 0:
        yellow_text(
            f"  {Colors.RED}FAIL{Colors.RESET}  {linter} "
            f"(exit={proc.returncode}, {human_readable_time(runtime)})"
        )
        return linter, proc.returncode, f"{linter} failed"
    yellow_text(
        f"  {Colors.GREEN}OK  {Colors.RESET} {linter} "
        f"({human_readable_time(runtime)})"
    )
    return linter, 0, ""


def _resolve_workers(arg: Optional[str], n_linters: int) -> int:
    """``--parallel`` value -> worker count. 0/auto/None means cpu-bounded."""
    if arg is None:
        return max(1, min(n_linters, os.cpu_count() or 1))
    s = str(arg).strip().lower()
    if s in ("", "auto", "max"):
        return max(1, min(n_linters, os.cpu_count() or 1))
    if s in ("no", "off", "false", "0"):
        return 1
    try:
        n = int(s)
    except ValueError:
        raise SystemExit(f"Invalid --parallel value: {arg!r}")
    return max(1, n)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run linters.", add_help=False)
    parser.add_argument("--quick", action="store_true", help="Skip slow linters.")
    parser.add_argument("--dont_fail_on_error", action="store_true",
                        help="Don't fail when an error is encountered.")
    parser.add_argument("--check_only_changed_since_last_success",
                        action="store_true",
                        help="Check only files changed since last successful run.")
    parser.add_argument("--parallel", dest="parallel", default=None,
                        help="Run linters in parallel. Use an int for the "
                             "worker count, 'auto' for one worker per CPU, "
                             "or 'no'/'0' for sequential. Defaults to auto.")
    parser.add_argument("--no-parallel", dest="parallel", action="store_const",
                        const="no", default=None,
                        help="Force sequential execution.")
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
        print("  --parallel[=N]      Run linters in parallel (default: auto)")
        print("  --no-parallel       Run linters sequentially")
        print("  [files]             Run linters only on specified files")
        print("  [linter]            Run a specific linter")
        return 0

    os.environ.setdefault("install_tests", "1")
    os.environ.setdefault("DONT_SHOW_DONT_INSTALL_MESSAGE", "1")
    if args.check_only_changed_since_last_success:
        os.environ["ONLY_CHECK_CHANGED_SINCE_LAST_COMMIT"] = "1"

    if args.linter:
        linters_to_run = [args.linter]
    else:
        linters_to_run = list(DEFAULT_LINTERS)
        if args.quick:
            linters_to_run = [l for l in linters_to_run if l in ("lizard", "flake8", "pyflakes")]

    workers = _resolve_workers(args.parallel, len(linters_to_run))
    sequential = workers <= 1 or len(linters_to_run) <= 1

    if sequential:
        yellow_text(f"Running {len(linters_to_run)} linter(s) sequentially...")
    else:
        yellow_text(
            f"Running {len(linters_to_run)} linter(s) in parallel "
            f"({workers} workers)..."
        )

    errors: List[Tuple[str, str]] = []  # (linter, error_message)
    start = time.time()

    if sequential:
        for linter in linters_to_run:
            name, returncode, err = _run_linter(linter, args.files)
            if err:
                errors.append((name, err))
                if not args.dont_fail_on_error:
                    yellow_text(
                        f"Skipping remaining linters because {name} failed "
                        "(--dont_fail_on_error to keep going)..."
                    )
                    break
    else:
        with ThreadPoolExecutor(max_workers=workers,
                                thread_name_prefix="linter") as pool:
            future_to_linter = {
                pool.submit(_run_linter, linter, args.files): linter
                for linter in linters_to_run
            }
            try:
                for future in as_completed(future_to_linter):
                    name, returncode, err = future.result()
                    if err:
                        errors.append((name, err))
            except KeyboardInterrupt:
                red_text("\nInterrupted by Ctrl+C.")
                return 130

    total_runtime = time.time() - start
    print()

    if not errors:
        green_text(
            f"No linter errors ({len(linters_to_run)} linter(s) in "
            f"{human_readable_time(total_runtime)})"
        )
        return 0

    red_text("=> LINTERS-ERRORS =>")
    for name, err in errors:
        red_text(f"  {name}: {err}")
    return len(errors)


if __name__ == "__main__":
    sys.exit(main())
