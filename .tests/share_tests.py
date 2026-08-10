#!/usr/bin/env python3
"""Run different tests for the share functions."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _framework.helpers import red_text, yellow_text


REPO_ROOT = THIS_DIR.parent


def main(argv=None) -> int:
    if not shutil.which("php"):
        yellow_text("Cannot run share_tests when PHP is not installed!")
        return 255

    proc = subprocess.run(
        ["php", str(REPO_ROOT / ".gui" / "test_share_functions.php")],
        cwd=str(REPO_ROOT),
    )
    if proc.returncode != 0:
        red_text(f"php .gui/test_share_functions.php failed with exit-code {proc.returncode}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
