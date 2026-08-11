#!/usr/bin/env python3
"""Find typos in the GUI (PHP)."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


REPO_ROOT = THIS_DIR.parent


def main(argv=None) -> int:
    os.environ.setdefault("install_tests", "1")

    if not shutil.which("php"):
        print("Warning: PHP is not installed. Cannot run php_spellchecker")
        return 0

    tool = REPO_ROOT / ".tools" / "php_spellchecker.py"
    if not tool.exists():
        print(f"php_spellchecker not found at {tool}")
        return 1

    # Import and run in-process to avoid the extra ~300 ms python
    # startup cost of a subprocess.
    import runpy
    sys.argv = [str(tool), str(REPO_ROOT / ".gui")]
    try:
        runpy.run_path(str(tool), run_name="__main__")
        return 0
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Cancelled by user", file=sys.stderr)
        sys.exit(0)
