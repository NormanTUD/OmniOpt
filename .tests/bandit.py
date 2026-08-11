#!/usr/bin/env python3
"""Tests for security-related python bugs and improvement suggestions."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _framework.helpers import red_text


REPO_ROOT = THIS_DIR.parent


def main(argv=None) -> int:
    os.environ.setdefault("install_tests", "1")
    if not shutil.which("bandit"):
        red_text("bandit not found")
        return 1

    errors: list[str] = []
    for py_file in sorted(REPO_ROOT.glob(".*.py")):
        if py_file.name == ".helpers.py":
            continue
        cmd = ["bandit", "-lll", "-q", "-s", "B602", str(py_file)]
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
        if proc.returncode != 0:
            errstr = (
                f"Failed linting {py_file.name}: Run "
                f"'bandit -lll -q -s B602 {py_file.name}' to see details."
            )
            red_text(errstr)
            errors.append(errstr)

    if errors:
        return len(errors)
    return 0


if __name__ == "__main__":
    sys.exit(main())
