#!/usr/bin/env python3
"""Find function names that are defined twice or more in .gui."""

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
GUI_DIR = REPO_ROOT / ".gui"


def main(argv=None) -> int:
    if not GUI_DIR.is_dir():
        red_text(f"{GUI_DIR} not found")
        return 1

    all_funcs = []
    for js_file in GUI_DIR.glob("*.js"):
        if js_file.name in ("gui_js_tests.js",) or "jquery" in js_file.name:
            continue
        try:
            content = js_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for line in content.splitlines():
            if line.startswith("function "):
                m = re.match(r"function\s+(\w+)", line)
                if m:
                    all_funcs.append(m.group(1))

    counter = Counter(all_funcs)
    duplicates = [name for name, c in counter.items() if c > 1]

    if not duplicates:
        green_text("Found no double defined functions")
        return 0

    red_text(f"Found {len(duplicates)} double defined functions:")
    for name in duplicates:
        red_text(name)
    return 1


if __name__ == "__main__":
    sys.exit(main())
