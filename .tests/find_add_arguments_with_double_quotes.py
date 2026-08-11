#!/usr/bin/env python3
"""Find arguments in omniopt that have double quotes.

All add_argument calls should use single quotes because other parsers
rely on that.
"""

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
        red_text("omniopt not found")
        return 1

    content = omniopt.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"\.add_argument\(\"[^\"]*\"", content)

    if not matches:
        green_text("find_add_arguments_with_double_quotes OK")
        return 0

    red_text(f"Error: found {len(matches)} double-quoted add_argument's:")
    for m in matches:
        red_text(m)
    return 1


if __name__ == "__main__":
    sys.exit(main())
