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

    errors: list[str] = []
    for js_file in sorted(GUI_DIR.glob("*.js")):
        if any(s in js_file.name for s in EXCLUDE_SUFFIXES):
            continue
        proc = subprocess.run(
            ["node", "--check", str(js_file)],
            cwd=str(GUI_DIR),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            err = proc.stderr.strip() or proc.stdout.strip()
            errors.append(f"{js_file.name}: {err}")
            red_text(f"{js_file.name}: {err}")

    if errors:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
