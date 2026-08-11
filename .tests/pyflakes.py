#!/usr/bin/env python3
"""Run pyflakes linter on .py files."""

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
    if not shutil.which("pyflakes"):
        red_text("pyflakes not found")
        return 1

    targets = [str(p) for p in REPO_ROOT.glob(".*.py")]
    targets = [t for t in targets if not t.endswith("/.helpers.py")]
    targets += [str(REPO_ROOT / "omniopt")]
    proc = subprocess.run(["pyflakes", *targets], cwd=str(REPO_ROOT))
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
