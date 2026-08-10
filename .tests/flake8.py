#!/usr/bin/env python3
"""Runs flake8 on Python files in the repo."""

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
    if not shutil.which("flake8"):
        red_text("flake8 not found")
        return 1

    errors: list[str] = []
    ignore = os.environ.get("IGNOREME", "")
    ignore_args = [f"--ignore={ignore}"] if ignore else []

    for py_file in sorted(REPO_ROOT.glob(".*.py")):
        if py_file.name == ".helpers.py":
            continue
        cmd = ["flake8", *ignore_args, str(py_file)]
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
        if proc.returncode != 0:
            errstr = (
                f"Failed linting {py_file.name}: Run "
                f"'flake8 {' '.join(ignore_args)} {py_file.name}' "
                "to see details."
            )
            red_text(errstr)
            errors.append(errstr)

    if errors:
        red_text("=> FLAKE8-ERRORS =>")
        for e in errors:
            red_text(e)
        return len(errors)
    return 0


if __name__ == "__main__":
    sys.exit(main())
