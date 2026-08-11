#!/usr/bin/env python3
"""Find arguments added multiple times in omniopt."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from collections import Counter

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _framework.helpers import green_text, red_text


REPO_ROOT = THIS_DIR.parent


def main(argv=None) -> int:
    omniopt = REPO_ROOT / "omniopt"
    if not omniopt.exists():
        red_text("omniopt not found")
        return 1

    content = omniopt.read_text(encoding="utf-8", errors="ignore")
    pattern = re.compile(r"\.add_argument\(([^)]*help=)?[^)]*\)")
    helps = []
    for m in pattern.finditer(content):
        args_str = m.group(0)
        help_match = re.search(r"help=[\"']([^\"']+)[\"']", args_str)
        if help_match:
            helps.append(help_match.group(1))

    counter = Counter(helps)
    duplicates = [h for h, c in counter.items() if c > 1]

    if not duplicates:
        green_text("find_double_add_arguments_help: OK")
        return 0

    errs = len(duplicates)
    for h in duplicates:
        red_text(f"Found double help: {h}")
    print(f"Failed find_double_add_arguments_help: Found {errs} errors")
    return 1


if __name__ == "__main__":
    sys.exit(main())
