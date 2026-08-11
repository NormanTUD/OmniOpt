#!/usr/bin/env python3
"""See if all bash scripts have --help."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _framework.helpers import red_text, yellow_text


REPO_ROOT = THIS_DIR.parent
EXCLUDE = ("plot_documentation_has_all_plot_types", "help_page")


def main(argv=None) -> int:
    proc = subprocess.run(
        "grep -rIl '#!/usr/bin/env bash' .",
        shell=True,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    candidates = [
        line.strip() for line in proc.stdout.splitlines()
        if line.strip()
        and not any(x in line for x in EXCLUDE)
        and line.strip() != "./omniopt"
    ]

    errors = 0
    for script in candidates:
        path = Path(script)
        if not path.exists():
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "--help" not in content:
            continue
        yellow_text(f"bash {script} --help")
        out = subprocess.run(
            ["bash", script, "--help"], capture_output=True, text=True,
        )
        print(out.stdout)
        if not out.stdout.strip():
            red_text(f"bash {script} --help had no output ({out.returncode})")
            errors += 1
            continue
        if out.returncode != 0:
            red_text(f"bash {script} --help failed with exit-code {out.returncode}")
            errors += 1

    return errors


if __name__ == "__main__":
    sys.exit(main())
