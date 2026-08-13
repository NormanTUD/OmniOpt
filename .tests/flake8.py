#!/usr/bin/env python3
"""Runs flake8 on Python files in the repo."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _framework.helpers import (
    file_has_changed_since_last_tagged_version,
    green_text,
    human_readable_time,
    red_text,
)
from _framework.installer import ensure_dependencies


REPO_ROOT = THIS_DIR.parent


def _lint_one(target: str, ignore_args: list[str]) -> subprocess.CompletedProcess:
    flake8_bin = shutil.which("flake8")
    if flake8_bin:
        cmd = [flake8_bin, *ignore_args, target]
    else:
        # Fall back to `python3 -m flake8` so this works when flake8 is
        # only installed inside a venv that has no bin entry on PATH.
        cmd = [sys.executable, "-m", "flake8", *ignore_args, target]
    return subprocess.run(cmd, cwd=str(REPO_ROOT))


def main(argv=None) -> int:
    ensure_dependencies()
    os.environ.setdefault("install_tests", "1")

    errors: list[str] = []
    ignore = os.environ.get("IGNOREME", "E501,E302,E265,E128,E305,E261,E126,E124,F824")
    ignore_args = [f"--ignore={ignore}"] if ignore else []

    flake8_bin = shutil.which("flake8") or f"{sys.executable} -m flake8"
    cmd_hint = (
        f"'{flake8_bin}'" if shutil.which("flake8")
        else f"'source activate && {sys.executable} -m flake8'"
    )

    start = time.time()
    for py_file in sorted(REPO_ROOT.glob(".*.py")):
        if py_file.name == ".helpers.py":
            continue
        if not file_has_changed_since_last_tagged_version(
            py_file.name, cwd=str(REPO_ROOT)
        ):
            continue
        proc = _lint_one(str(py_file), ignore_args)
        if proc.returncode != 0:
            errstr = (
                f"Failed linting {py_file.name}: Run "
                f"'flake8 {' '.join(ignore_args)} {py_file.name}' "
                "to see details."
            )
            red_text(f"{errstr}\n")
            errors.append(errstr)

    elapsed = int(time.time() - start)
    print(f"Flake8 test took: {human_readable_time(elapsed)}")

    if not errors:
        green_text("No flake8 errors")
        return 0

    red_text("=> FLAKE8-ERRORS => FLAKE8-ERRORS => FLAKE8-ERRORS =>\n")
    for e in errors:
        red_text(f"{e}\n")
    return len(errors)


if __name__ == "__main__":
    sys.exit(main())
