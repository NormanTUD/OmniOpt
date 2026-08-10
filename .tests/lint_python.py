#!/usr/bin/env python3
"""Python linter runner (replaces .tests/lint_python bash script).

Runs ruff on each Python file in the repo root.
"""

from __future__ import annotations

import glob
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
    Colors,
    command_exists,
    green_text,
    human_readable_time,
    red_text,
)


REPO_ROOT = THIS_DIR.parent


def main(argv=None) -> int:
    os.environ.setdefault("install_tests", "1")

    if not command_exists("ruff"):
        red_text("ruff not found")
        return 1

    files = sorted(glob.glob(str(REPO_ROOT / ".*.py")) + glob.glob(str(REPO_ROOT / "*.py")))
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        green_text("No python files to lint.")
        return 0

    errors: list[str] = []
    start = time.time()
    for path in files:
        proc = subprocess.run(["ruff", "check", path], cwd=str(REPO_ROOT))
        if proc.returncode != 0:
            errstr = f"Failed linting {path}: Run 'ruff check {path}' to see details."
            red_text(f"\n{errstr}")
            errors.append(errstr)

    elapsed = int(time.time() - start)
    print(f"Lint test took: {human_readable_time(elapsed)}")

    if not errors:
        green_text("No lint-python errors")
        return 0
    red_text("=> LINT-ERRORS =>")
    for e in errors:
        red_text(e)
    return len(errors)


if __name__ == "__main__":
    sys.exit(main())
