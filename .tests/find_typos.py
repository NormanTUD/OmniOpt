#!/usr/bin/env python3
"""Find all kinds of typos (delegates to find_typos_in_* scripts)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


STEPS = [
    ("bash", "find_typos_in_bash", 1),
    ("gui", "find_typos_in_gui", 2),
    ("js", "find_typos_in_js", 3),
    ("md", "find_typos_in_md", 4),
    ("php", "find_typos_in_php", 5),
    ("python", "find_typos_in_python", 6),
]


def main(argv=None) -> int:
    for label, name, exit_code in STEPS:
        script = THIS_DIR / name
        if not script.exists():
            print(f"{name}: not found", file=sys.stderr)
            return exit_code
        # Prefer the .py variant if it exists.
        candidates = []
        if (THIS_DIR / f"{name}.py").exists():
            candidates.append(f"python3 {THIS_DIR / f'{name}.py'}")
        candidates.append(f"bash {script}")
        last_code = 0
        for cmd in candidates:
            proc = subprocess.run(cmd, shell=True, cwd=str(THIS_DIR.parent))
            last_code = proc.returncode
            if proc.returncode == 0:
                break
        if last_code != 0:
            return exit_code
    return 0


if __name__ == "__main__":
    sys.exit(main())
