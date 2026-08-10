#!/usr/bin/env python3
"""Find unreachable and unused code."""

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

    current = subprocess.run(
        [sys.executable, "--version"], capture_output=True, text=True
    ).stdout.strip()
    if current == "Python 3.8.10":
        red_text(f"deadcode cannot be run with {current}")
        return 0

    if not shutil.which("deadcode"):
        red_text("deadcode not found")
        return 1

    proc = subprocess.run(["deadcode", str(REPO_ROOT)], cwd=str(REPO_ROOT))
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
