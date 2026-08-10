#!/usr/bin/env python3
"""Run pyright on changed Python files only (faster than full run)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


REPO_ROOT = THIS_DIR.parent


def main(argv=None) -> int:
    os.environ.setdefault("install_tests", "1")
    os.environ["ONLY_CHECK_CHANGED_SINCE_LAST_COMMIT"] = "1"

    pyright = THIS_DIR / "pyright.py"
    if pyright.exists():
        proc = subprocess.run([sys.executable, str(pyright)], cwd=str(REPO_ROOT))
        return proc.returncode

    if not shutil.which("pyright"):
        print("pyright not found")
        return 1
    proc = subprocess.run(["pyright", "."], cwd=str(REPO_ROOT))
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
