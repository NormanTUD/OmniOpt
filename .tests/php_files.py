#!/usr/bin/env python3
"""Syntax check all PHP files."""

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
GUI_DIR = REPO_ROOT / ".gui"


def main(argv=None) -> int:
    os.environ.setdefault("NO_WHIPTAIL", "1")

    if not shutil.which("php"):
        print("PHP not installed; skipping php_files")
        return 0

    errors = 0
    for php_file in sorted(GUI_DIR.rglob("*.php")):
        proc = subprocess.run(
            ["php", "-l", str(php_file)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            red_text(f"PHP syntax error in {php_file}: {proc.stderr.strip()}")
            errors += 1

    if errors:
        return errors
    return 0


if __name__ == "__main__":
    sys.exit(main())
