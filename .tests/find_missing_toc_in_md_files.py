#!/usr/bin/env python3
"""Find help documentation files missing the Table-of-Contents."""

from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _framework.helpers import green_text, red_text


REPO_ROOT = THIS_DIR.parent
TUTORIALS = REPO_ROOT / ".gui" / "_tutorials"


def main(argv=None) -> int:
    if not TUTORIALS.is_dir():
        print(f"{TUTORIALS} not found")
        return 1

    errors = 0
    for md_file in TUTORIALS.rglob("*.md"):
        try:
            content = md_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if 'id="toc"' not in content and content.lstrip().startswith("## "):
            red_text(f"Missing TOC in {md_file}")
            errors += 1

    if errors == 0:
        green_text("find_missing_toc_in_md_files: No found")
        return 0
    red_text(f"find_missing_toc_in_md_files: {errors} found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
