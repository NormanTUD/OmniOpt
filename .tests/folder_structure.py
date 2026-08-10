#!/usr/bin/env python3
"""Tests if all files that are in the run subfolders are documented properly."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _framework.helpers import (
    green_text,
    human_readable_time,
    red_text,
)


REPO_ROOT = THIS_DIR.parent
MD_FILE = REPO_ROOT / ".gui" / "_tutorials" / "folder_structure.md"


def main(argv=None) -> int:
    start = time.time()
    errors: list[str] = []

    proc = subprocess.run(
        "ls runs/*/*/state_files 2>/dev/null | grep -v '^runs' | sort | uniq | grep -v '^\\s*$'",
        shell=True,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )

    if not MD_FILE.exists():
        errmsg = f"{MD_FILE} not found"
        red_text(errmsg)
        errors.append(errmsg)
    else:
        md_content = MD_FILE.read_text(encoding="utf-8")
        for state_file in proc.stdout.splitlines():
            state_file = state_file.strip()
            if not state_file:
                continue
            if state_file not in md_content:
                errmsg = f"State file >>{state_file}<< does not appear in {MD_FILE}"
                red_text(errmsg)
                errors.append(errmsg)

    elapsed = int(time.time() - start)
    hrs = elapsed // 3600
    mins = (elapsed % 3600) // 60
    secs = elapsed % 60
    print(f"folder_structure has all the items took: {hrs:02d}:{mins:02d}:{secs:02d}")

    if not errors:
        green_text("No folder_structure errors")
        return 0

    red_text("=> FOLDER-STRUCTURE-ERRORS =>")
    for e in errors:
        red_text(e)
    return len(errors)


if __name__ == "__main__":
    sys.exit(main())
