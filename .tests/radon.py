#!/usr/bin/env python3
"""Find functions with too large cyclomatic complexity."""

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

from _framework.helpers import yellow_text


REPO_ROOT = THIS_DIR.parent


def main(argv=None) -> int:
    os.environ.setdefault("install_tests", "1")
    if not shutil.which("radon"):
        print("radon not found")
        return 1

    start = time.time()
    for py_file in sorted(REPO_ROOT.glob(".*.py")):
        if py_file.name == ".helpers.py":
            continue
        try:
            proc = subprocess.run(
                ["radon", "cc", str(py_file)],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
            )
        except Exception:
            continue
        output = proc.stdout
        relevant = [line for line in output.splitlines() if line.rstrip().endswith("[DEF]")]
        if relevant:
            yellow_text(str(py_file) + ":")
            yellow_text("\n".join(relevant))

    elapsed = int(time.time() - start)
    hrs = elapsed // 3600
    mins = (elapsed % 3600) // 60
    secs = elapsed % 60
    print(f"Radon test took: {hrs:02d}:{mins:02d}:{secs:02d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
