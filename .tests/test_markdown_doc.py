#!/usr/bin/env python3
"""Using pymarkdown to scan markdown files for problems."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


REPO_ROOT = THIS_DIR.parent


def main(argv=None) -> int:
    os.environ.setdefault("install_tests", "1")
    if not shutil.which("pymarkdown"):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pymarkdownlnt"], check=False)

    proc = subprocess.run(
        ["pymarkdown", "scan"] + [str(p) for p in (REPO_ROOT / ".gui" / "_tutorials").glob("*.md")],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    skip_phrases = (
        "Line length", "Hard tabs", "Inline HTML",
        "Fenced code blocks should have a language specified",
    )
    lines = []
    for line in (proc.stdout or "").splitlines() + (proc.stderr or "").splitlines():
        if any(p in line for p in skip_phrases):
            continue
        if not line.strip():
            continue
        lines.append(line)

    for line in lines:
        print(line)
    return len(lines)


if __name__ == "__main__":
    sys.exit(main())
