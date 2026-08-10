#!/usr/bin/env python3
"""Run mypy to find variables that are not typed properly."""

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
    if not shutil.which("mypy"):
        print("Warning: mypy is not installed. Cannot run mypy")
        return 0

    args = sys.argv[1:]
    if not args:
        targets = sorted(str(p) for p in REPO_ROOT.glob(".*.py"))
    else:
        targets = list(args)

    cmd = [
        "mypy",
        "--check-untyped-defs",
        "--ignore-missing-imports",
        "--disallow-untyped-defs",
        *targets,
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
