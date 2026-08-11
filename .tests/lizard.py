#!/usr/bin/env python3
"""Run lizard (cyclomatic complexity) on Python files."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _framework.helpers import red_text
from _framework.installer import ensure_dependencies


REPO_ROOT = THIS_DIR.parent

LIZARD_FLAGS = ["--CCN", "15", "--arguments", "6", "--length", "100"]


def main(argv=None) -> int:
    # Installs missing tools (e.g. lizard) from test_requirements.txt into
    # the framework venv when they are not available yet.
    ensure_dependencies()
    os.environ.setdefault("install_tests", "1")

    targets = [str(p) for p in REPO_ROOT.glob(".*.py")]
    targets += [str(p) for p in REPO_ROOT.glob(".omniopt_plot_*.py")]
    targets += [str(REPO_ROOT / "omniopt")]

    lizard_bin = shutil.which("lizard")
    if lizard_bin is not None:
        cmd = [lizard_bin, *LIZARD_FLAGS, *targets]
    else:
        # Fall back to running lizard as a Python module (works when it's
        # installed inside the framework venv but has no bin entry on PATH).
        # We cannot detect this with find_spec/import, because this very
        # script (.tests/lizard.py) shadows the "lizard" module on sys.path.
        cmd = [sys.executable, "-m", "lizard", *LIZARD_FLAGS, *targets]

    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)

    if proc.returncode != 0 and "No module named" in (proc.stderr or ""):
        red_text("lizard not found and could not be installed")
        return 1

    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    # lizard exits non-zero when it finds warnings; that's expected for this
    # test — it should report but not fail.
    return 0


if __name__ == "__main__":
    sys.exit(main())
