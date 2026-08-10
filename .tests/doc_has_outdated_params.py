#!/usr/bin/env python3
"""Test if any documentation page has outdated options for omniopt."""

from __future__ import annotations

import re
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _framework.helpers import green_text, red_text


REPO_ROOT = THIS_DIR.parent


def main(argv=None) -> int:
    omniopt = REPO_ROOT / "omniopt"
    if not omniopt.exists():
        print("omniopt not found")
        return 1

    omniopt_content = omniopt.read_text(encoding="utf-8", errors="ignore")

    tutorials_dir = REPO_ROOT / ".gui" / "_tutorials"
    if not tutorials_dir.is_dir():
        green_text("doc_has_outdated_params: No tutorials directory, OK")
        return 0

    errors = 0
    for f in sorted(tutorials_dir.iterdir()):
        if not f.is_file():
            continue
        try:
            content = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for line in content.splitlines():
            if "--" not in line or "<!--" in line:
                continue
            stripped = line.lstrip()
            if not stripped.startswith("--"):
                continue
            param = stripped.split("=", 1)[0].split()[0]
            if not re.match(r"^--[A-Za-z0-9_-]+$", param):
                continue
            if f"add_argument('{param}" not in omniopt_content:
                red_text(f"Parameter {param} in {f} was not found")
                errors += 1

    if errors > 0:
        red_text(f"{errors} errors")
        return 1
    green_text("doc_has_outdated_params: No errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
