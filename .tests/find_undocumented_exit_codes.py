#!/usr/bin/env python3
"""Check if all used exit codes are documented."""

from __future__ import annotations

import re
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


REPO_ROOT = THIS_DIR.parent


def main(argv=None) -> int:
    omniopt = REPO_ROOT / "omniopt"
    exit_table = REPO_ROOT / ".gui" / "exit_code_table.php"
    if not omniopt.exists() or not exit_table.exists():
        print("omniopt or exit_code_table.php not found")
        return 1

    content = omniopt.read_text(encoding="utf-8", errors="ignore")
    codes: set[int] = set()
    for m in re.finditer(r"my_exit\s*\(\s*(\d+)", content):
        codes.add(int(m.group(1)))
    for m in re.finditer(r"_fatal_error\s*\([^,]+,\s*(\d+)", content):
        codes.add(int(m.group(1)))
    codes.discard(0)

    table_content = exit_table.read_text(encoding="utf-8", errors="ignore")
    errors = 0
    for code in sorted(codes):
        pattern = re.compile(rf"^\s*{code}\s*=>", re.MULTILINE)
        if not pattern.search(table_content):
            print(f"Missing exit code: {code}")
            errors = 1
    return errors


if __name__ == "__main__":
    sys.exit(main())
