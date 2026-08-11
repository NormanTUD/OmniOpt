#!/usr/bin/env python3
"""Find CLI arguments that are not typed properly for mypy and beartype."""

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
    add_args = re.findall(r"\.add_argument\([\"'](--[^\"']+)[\"']", content)
    cli_args = [a[2:] for a in add_args]

    errs = 0
    for cli_arg in cli_args:
        if cli_arg == "debug_stack_regex":
            continue
        type_pattern = re.compile(rf"^\s{{4}}{re.escape(cli_arg)}:", re.MULTILINE)
        if not type_pattern.search(content):
            red_text(f"{cli_arg} has no type in ConfigLoader")
            errs += 1

    if errs == 0:
        green_text("find_untyped_cli_args: OK")
        return 0
    return errs


if __name__ == "__main__":
    sys.exit(main())
