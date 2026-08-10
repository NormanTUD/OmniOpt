#!/usr/bin/env python3
"""Run PHP unit tests."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _framework.helpers import green_text, red_text


REPO_ROOT = THIS_DIR.parent
GUI_DIR = REPO_ROOT / ".gui"


def main(argv=None) -> int:
    if not shutil.which("php"):
        print("php not installed! Will skip testing.")
        return 0

    env = os.environ.copy()
    env["IS_RUNNING_UNIT_TESTS"] = "1"

    p1 = subprocess.run(["php", "tests.php"], cwd=str(GUI_DIR), env=env)
    p2 = subprocess.run(["php", "tests_extended.php"], cwd=str(GUI_DIR), env=env)

    if p1.returncode != 0 or p2.returncode != 0:
        red_text("At least one PHP-unit test failed")
        return 1
    if os.environ.get("SHOW_SUCCESS"):
        green_text("PHP-Unit-Tests OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
