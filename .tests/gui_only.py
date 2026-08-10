#!/usr/bin/env python3
"""GUI-only test suite: runs only tests relevant to the .gui/ directory."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


REPO_ROOT = THIS_DIR.parent
TESTS = [
    "php_files", "php_unit_tests", "js_syntax", "js_gui_unit_tests",
    "php_search", "share", "share_tests", "find_typos_in_gui",
    "find_typos_in_js", "find_typos_in_php",
    "all_oo_options_are_in_gui_and_vice_versa", "find_double_function_names",
    "find_unused_share_functions", "doc_has_outdated_params",
    "find_missing_toc_in_md_files", "find_undocumented_exit_codes",
    "missing_plot_metadata",
]


def main(argv=None) -> int:
    runtimes: dict[str, int] = {}
    total_start = time.time()
    next_exit = 1
    for t in TESTS:
        start = time.time()
        print(f"Running {t} ...")
        cmd_variants = [
            ["python3", str(THIS_DIR / f"{t}.py")],
            ["bash", str(THIS_DIR / t)],
        ]
        ok = False
        for cmd in cmd_variants:
            if not Path(cmd[1]).exists():
                continue
            proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
            if proc.returncode == 0:
                ok = True
                break
        end = time.time()
        runtime = int(end - start)
        runtimes[t] = runtime
        if not ok:
            print(f"FAIL {t} ({runtime}s)")
            return next_exit
        next_exit += 1

    total_runtime = int(time.time() - total_start)
    print(f"\nGUI-only tests OK in {total_runtime}s\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
