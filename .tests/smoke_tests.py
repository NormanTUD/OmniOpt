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
from _framework.helpers import (
    Colors,
    green_text,
    human_readable_time,
    red_text,
    yellow_text,
)


def main(argv=None) -> int:
    config = load_config()
    repo_root = config.repo_root
    params = config.resolve_parameters({})
    tests = config.smoke_tests

    total_start = time.time()
    runtimes = {}

    print(f"Running {len(tests)} smoke tests...")
    print()

    next_exit_code = 1
    for test_name in tests:
        script = repo_root / ".tests" / test_name
        cmd_variants = [
            f"python3 {script}",
            f"bash {script}",
            f"{script}",
        ]

        start = time.time()
        exit_code = None
        for cmd in cmd_variants:
            proc = helpers.run(cmd)
            exit_code = proc.returncode
            break

        end = time.time()
        runtime = int(end - start)
        runtimes[test_name] = runtime

        ok = exit_code == 0
        prefix = Colors.GREEN + "OK  " + Colors.RESET if ok else Colors.RED + "FAIL" + Colors.RESET
        suffix = Colors.GRAY + f" ({human_readable_time(runtime)})" + Colors.RESET if not helpers._NO_COLOR else f" ({human_readable_time(runtime)})"
        print(f"  {prefix}  {test_name}{suffix}")

        if not ok:
            red_text(f"Smoke test {test_name} failed with exit code {exit_code}\n")
            return exit_code if exit_code and exit_code != 0 else next_exit_code

        next_exit_code += 1

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
