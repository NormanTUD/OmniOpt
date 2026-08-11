#!/usr/bin/env python3
"""Tests if all plot scripts have the metadata for the help pages."""

from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


REPO_ROOT = THIS_DIR.parent


def main(argv=None) -> int:
    cnt = 0
    for plot_file in sorted(REPO_ROOT.glob(".omniopt_plot_*.py")):
        try:
            content = plot_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "# EXPECTED FILES" not in content:
            print(f"Missing '# EXPECTED FILES' in {plot_file}")
            cnt += 1
        if "# DESCRIPTION" not in content:
            print(f"Missing '# DESCRIPTION' in {plot_file}")
            cnt += 1
    return cnt


if __name__ == "__main__":
    sys.exit(main())
