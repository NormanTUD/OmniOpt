#!/usr/bin/env python3
"""Check syntax of JS scripts."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _framework.helpers import red_text, yellow_text


REPO_ROOT = THIS_DIR.parent
GUI_DIR = REPO_ROOT / ".gui"
EXCLUDE_SUFFIXES = (
    "jquery", "init", "plotly", "tooltipster", "darkmode",
    "initialization", "footer", "core", "md5", "mml", "d3",
)


def main(argv=None) -> int:
    os.environ.setdefault("disable_folder_creation", "1")

    if not shutil.which("node"):
        yellow_text("Cannot run share-test when node is not installed!")
        return 255

    # Use a single node helper that walks the dir and parses each *.js
    # file.  Saves the ~50 ms-per-file node startup cost (was ~5 s
    # for ~30 files, now ~0.3 s).
    helper = THIS_DIR / "js_syntax_helper.js"
    proc = subprocess.run(
        ["node", str(helper), str(GUI_DIR)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        for line in proc.stdout.strip().splitlines():
            red_text(f"JS syntax error: {line}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
