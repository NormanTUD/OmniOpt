#!/usr/bin/env python3
"""Test if all state files that are written are documented."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _framework.helpers import red_text


REPO_ROOT = THIS_DIR.parent
MD_FILE = REPO_ROOT / ".gui" / "_tutorials" / "folder_structure.md"


def main(argv=None) -> int:
    if not (REPO_ROOT / "omniopt").exists():
        print("omniopt not found")
        return 1

    omniopt_content = (REPO_ROOT / "omniopt").read_text(encoding="utf-8", errors="ignore")

    pattern = re.compile(r"^\s*write_state_file\(\s*[\"']([^\"']+)[\"']", re.MULTILINE)
    from_codes = set(pattern.findall(omniopt_content))

    if not MD_FILE.exists():
        print(f"{MD_FILE} not found")
        return 1

    md_content = MD_FILE.read_text(encoding="utf-8")

    missing = 0
    for code in sorted(from_codes):
        # Match: "- `<code>`" (allow whitespace)
        item_pattern = re.compile(rf"^\s*-\s+`{re.escape(code)}`", re.MULTILINE)
        if len(item_pattern.findall(md_content)) != 1:
            red_text(f"'{code}' missing in {MD_FILE}")
            missing += 1

    return missing


if __name__ == "__main__":
    sys.exit(main())
