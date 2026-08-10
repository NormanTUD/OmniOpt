#!/usr/bin/env python3
"""Smoke tests (replaces .tests/smoke_tests bash script).

Runs each test script in the `smoke_tests` list from config.yaml.
A test passes if the script exits with code 0.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _framework import helpers
from _framework.config import load_config, substitute
from _framework.installer import ensure_dependencies
from _framework.helpers import (
    Colors,
    green_text,
    human_readable_time,
    red_text,
)


def _run_test(name: str) -> int:
    script = THIS_DIR / name
    py_script = THIS_DIR / f"{name}.py"

    if py_script.exists():
        cmd = [sys.executable, str(py_script)]
    elif script.exists():
        cmd = [str(script)]
    else:
        print(f"  {Colors.RED}MISS{Colors.RESET}  {name} (no script found)")
        return 99

    proc = helpers.run(" ".join(f'"{c}"' for c in cmd))
    return proc.returncode


def main(argv=None) -> int:
    ensure_dependencies(include_tests=True)
    config = load_config()
    repo_root = config.repo_root
    tests = config.smoke_tests

    total_start = time.time()
    runtimes: dict[str, int] = {}

    print(f"Running {len(tests)} smoke tests...")
    print()

    next_exit = 1
    for test_name in tests:
        start = time.time()
        exit_code = _run_test(test_name)
        end = time.time()
        runtime = int(end - start)
        runtimes[test_name] = runtime

        ok = exit_code == 0
        if ok:
            prefix = f"{Colors.GREEN}OK  {Colors.RESET}"
        else:
            prefix = f"{Colors.RED}FAIL{Colors.RESET}"
        suffix = f" ({human_readable_time(runtime)})"
        print(f"  {prefix}  {test_name}{suffix}")

        if not ok:
            red_text(f"Smoke test {test_name} failed with exit code {exit_code}\n")
            return exit_code if exit_code else next_exit

        next_exit += 1

    total_runtime = int(time.time() - total_start)
    print()
    green_text(f"Smoke Tests OK in {human_readable_time(total_runtime)}")
    print()

    if runtimes:
        maxlen = max(len(t) for t in runtimes)
        print(f"| {('Test').ljust(maxlen)} | {'Runtime':<12} |")
        print(f"| {'-' * maxlen} | {'-' * 12} |")
        for t, rt in runtimes.items():
            print(f"| {t.ljust(maxlen)} | {human_readable_time(rt):<12} |")

    return 0


if __name__ == "__main__":
    sys.exit(main())
