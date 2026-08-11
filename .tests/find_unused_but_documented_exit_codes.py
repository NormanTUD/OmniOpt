#!/usr/bin/env python3
"""Find exit codes in documentation that are not used in OmniOpt2."""

from __future__ import annotations

import re
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _framework.helpers import (
    green_text,
    red_text,
    yellow_text,
)


REPO_ROOT = THIS_DIR.parent
IGNORED = {255, 245, 143}


def main(argv=None) -> int:
    yellow_text("Warning: This test is not fully finished yet.")

    exit_table = REPO_ROOT / ".gui" / "exit_code_table.php"
    omniopt = REPO_ROOT / "omniopt"
    if not exit_table.exists() or not omniopt.exists():
        print("exit_code_table.php or omniopt not found")
        return 1

    table_content = exit_table.read_text(encoding="utf-8", errors="ignore")
    omniopt_content = omniopt.read_text(encoding="utf-8", errors="ignore")

    codes: set[int] = set()
    for m in re.finditer(r"^\s*(\d+)\s*=>", table_content, re.MULTILINE):
        try:
            codes.add(int(m.group(1)))
        except ValueError:
            continue

    errors = 0
    for code in sorted(codes):
        if code in IGNORED:
            continue
        pattern = re.compile(rf"(exit|_fatal_error).*{code}")
        if not pattern.search(omniopt_content):
            if code not in (22, 137):
                print(f"Exit code {code} not found in omniopt")
                errors += 1

    if errors == 0:
        green_text("No unused exit-codes found")
        return 0
    red_text(f"Found {errors} unused exit codes")
    return 1


if __name__ == "__main__":
    sys.exit(main())
