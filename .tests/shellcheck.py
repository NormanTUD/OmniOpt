#!/usr/bin/env python3
"""Lint bash files via shellcheck."""

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
    if not shutil.which("shellcheck"):
        red_text("shellcheck not found")
        return 1

    bash_files: list[str] = []
    for pattern in ("*.sh", ".shellscript_functions", ".general.sh",
                    ".colorfunctions.sh"):
        bash_files.extend(str(p) for p in REPO_ROOT.glob(pattern))
    for p in REPO_ROOT.iterdir():
        if p.is_file() and not p.suffix and os.access(p, os.X_OK):
            try:
                with open(p, "rb") as f:
                    if f.read(2) == b"#!":
                        bash_files.append(str(p))
            except OSError:
                continue

    if not bash_files:
        return 0

    proc = subprocess.run(["shellcheck", *bash_files], cwd=str(REPO_ROOT))
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
