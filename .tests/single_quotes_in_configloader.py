#!/usr/bin/env python3
"""Tests if the config-loader in omniopt has only single quotes."""

from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _framework.helpers import red_text


REPO_ROOT = THIS_DIR.parent


def main(argv=None) -> int:
    omniopt = REPO_ROOT / "omniopt"
    if not omniopt.exists():
        print("omniopt not found")
        return 1

    content = omniopt.read_text(encoding="utf-8", errors="ignore")
    lines = content.splitlines()
    errors = 0

    for line in lines:
        if 'action="' in line:
            red_text(f"Found double quotes action {line}")
            errors += 1
        if 'nargs="' in line:
            red_text(f"Found double quoted nargs {line}")
            errors += 1
        if "add_argument" in line and '"' in line:
            red_text(f"Found double quote {line}")
            errors += 1

    if errors == 0:
        print("single_quotes_in_configloader OK")
        return 0
    red_text(f"single_quotes_in_configloader failed: {errors} double quotes found")
    return 1


if __name__ == "__main__":
    sys.exit(main())
