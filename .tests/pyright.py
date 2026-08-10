#!/usr/bin/env python3
"""Runs pyright linter on python."""

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

from _framework.helpers import red_text, green_text


REPO_ROOT = THIS_DIR.parent


def main(argv=None) -> int:
    os.environ.setdefault("install_tests", "1")
    if not shutil.which("pyright"):
        red_text("pyright not found")
        return 1

    errors: list[str] = []
    start = time.time()

    for py_file in sorted(REPO_ROOT.glob(".*.py")):
        if py_file.name == ".helpers.py":
            continue
        print(f"Pyright {py_file.name}:")
        proc = subprocess.run(["pyright", str(py_file)], cwd=str(REPO_ROOT))
        if proc.returncode != 0:
            errstr = (
                f"Failed linting {py_file.name}: Run 'pyright {py_file.name}' "
                "to see details."
            )
            red_text(errstr)
            errors.append(errstr)

    elapsed = int(time.time() - start)
    hrs = elapsed // 3600
    mins = (elapsed % 3600) // 60
    secs = elapsed % 60
    print(f"pyright test took: {hrs:02d}:{mins:02d}:{secs:02d}")

    if not errors:
        green_text("No pyright errors")
        return 0
    red_text("=> PYRIGHT-ERRORS =>")
    for e in errors:
        red_text(e)
    return len(errors)


if __name__ == "__main__":
    sys.exit(main())
