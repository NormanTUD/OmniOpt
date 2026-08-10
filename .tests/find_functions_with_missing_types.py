#!/usr/bin/env python3
"""Find python-scripts with missing function return types."""

from __future__ import annotations

import re
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


REPO_ROOT = THIS_DIR.parent


def main(argv=None) -> int:
    exit_code = 0
    # bash checks .*.py excluding .random_generator.py
    pattern_func = re.compile(r"^def\s+\w+\([^)]*\)(?!\s*->)", re.MULTILINE)

    for py_file in sorted(REPO_ROOT.glob(".*.py")):
        if py_file.name == ".random_generator.py":
            continue
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        matches = pattern_func.findall(content)
        if len(matches) > 1:
            print(f"====> {py_file.name} =====>")
            for m in matches:
                print(m)
            exit_code += 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
