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

    php_files = sorted(GUI_DIR.rglob("*.php"))
    errors = 0

    # PHP -l only accepts one file per invocation.  Running it in a
    # subprocess per file costs ~200 ms of startup per file.  Instead,
    # we shell out to a single batch helper that loops over all files
    # inside one PHP process.
    helper_path = THIS_DIR / "php_files_helper.php"
    proc = subprocess.run(
        ["php", str(helper_path), str(GUI_DIR)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        for line in proc.stderr.strip().splitlines():
            red_text(f"PHP syntax error: {line}")
        # Each failure line is the bad-file path + 1 status line.
        errors = proc.stderr.count("Errors parsing") or 1
    if errors:
        return errors
    return 0


if __name__ == "__main__":
    sys.exit(main())
