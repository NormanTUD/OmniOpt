#!/usr/bin/env python3
"""Run JS GUI unit tests."""

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
    if not shutil.which("node"):
        print("node not installed! Will skip JS testing.")
        return 0

    test_file = GUI_DIR / "gui_js_tests.js"
    if not test_file.exists():
        print(f"{test_file} not found")
        return 1

    proc = subprocess.run(["node", str(test_file)], cwd=str(GUI_DIR))
    if proc.returncode != 0:
        red_text("At least one JS-unit test failed")
        return 1

    if os.environ.get("SHOW_SUCCESS"):
        green_text("JS-Unit-Tests OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
