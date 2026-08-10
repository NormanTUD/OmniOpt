#!/usr/bin/env python3
"""Find functions that are defined in any .py and also .helpers.py."""

from __future__ import annotations

import re
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


REPO_ROOT = THIS_DIR.parent


def main(argv=None) -> int:
    helpers = REPO_ROOT / ".helpers.py"
    if not helpers.exists():
        print(".helpers.py not found")
        return 1

    helper_funcs = []
    for line in helpers.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.match(r"^def\s+(\w+)", line)
        if m:
            helper_funcs.append(m.group(1))

    count = 0
    for py_file in REPO_ROOT.glob(".*.py"):
        if py_file.name == ".helpers.py":
            continue
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for func in helper_funcs:
            pattern = re.compile(rf"def\s+{re.escape(func)}\s*\(")
            if pattern.search(content):
                print(f"{func} exists in .helpers.py and {py_file.name}")
                count += 1

    return count


if __name__ == "__main__":
    sys.exit(main())
