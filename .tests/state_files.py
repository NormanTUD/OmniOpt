#!/usr/bin/env python3
"""Test if all state_files are properly documented."""

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
    yellow_text,
)


REPO_ROOT = THIS_DIR.parent
MD_FILE = REPO_ROOT / ".gui" / "_tutorials" / "folder_structure.md"


def main(argv=None) -> int:
    os.environ.setdefault("NO_WHIPTAIL", "1")
    start = time.time()

    errors: list[str] = []

    if not MD_FILE.exists():
        errmsg = f"{MD_FILE} not found"
        red_text(errmsg)
        errors.append(errmsg)
    else:
        md_content = MD_FILE.read_text(encoding="utf-8")
        # Find all state files in runs/*/*/state_files
        proc = subprocess.run(
            "ls runs/*/*/state_files 2>/dev/null | grep -v '^runs' | sort | uniq | grep -v '^\\s*$'",
            shell=True,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        for state_file in proc.stdout.splitlines():
            state_file = state_file.strip()
            if not state_file:
                continue
            if f"`{state_file}`" not in md_content:
                errmsg = f"State file type {state_file} does not appear in {MD_FILE}"
                red_text(errmsg)
                errors.append(errmsg)

    elapsed = int(time.time() - start)
    print(f"state_files test took: {human_readable_time(elapsed)}")

    if not errors:
        green_text("No state_files errors")
        return 0

    red_text("=> STATE_FILES-ERRORS =>")
    for e in errors:
        red_text(e)
    return len(errors)


if __name__ == "__main__":
    sys.exit(main())
