#!/usr/bin/env python3
"""Find argparse parameter for plots that go unused."""

from __future__ import annotations

import re
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _framework.helpers import red_text, yellow_text


REPO_ROOT = THIS_DIR.parent


def main(argv=None) -> int:
    helpers_content = ""
    helpers_py = REPO_ROOT / ".helpers.py"
    if helpers_py.exists():
        helpers_content = helpers_py.read_text(encoding="utf-8", errors="ignore")

    errors: list[str] = []

    for plot_file in sorted(REPO_ROOT.glob(".omniopt_plot*.py")):
        try:
            content = plot_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        k = 0
        for m in re.finditer(r"add\.argument\(\s*['\"]--([A-Za-z0-9_-]+)", content):
            arg_name = m.group(1)
            j = f"args.{arg_name}"
            count = content.count(j) + helpers_content.count(j)
            if count == 0:
                if k == 0:
                    yellow_text(str(plot_file))
                red_text(f"Unused arg: {j}")
                errors.append(f"Unused arg: {j} in {plot_file}")
                k += 1

    if not errors:
        return 0

    print()
    red_text("=> FIND_UNUSED_PARAMS_IN_PLOTS-ERRORS =>")
    for e in errors:
        print()
        red_text(e)
    return 1


if __name__ == "__main__":
    sys.exit(main())
