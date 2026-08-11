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


def main(argv=None) -> int:
    ensure_dependencies()
    os.environ.setdefault("install_tests", "1")
    lizard_bin = shutil.which("lizard")
    if lizard_bin is None:
        # Fall back to running lizard as a Python module (works when we're
        # already inside a venv that has lizard installed but no bin entry
        # on PATH).
        proc = subprocess.run(
            [sys.executable, "-m", "lizard", "--CCN", "15", "--arguments", "6",
             "--length", "100"],
            cwd=str(REPO_ROOT),
        )
        if proc.returncode != 0 and "No module named lizard" in (
            proc.stderr or ""
        ):
            red_text("lizard not found")
            return 1
        # lizard exits non-zero when it finds warnings; that's expected for
        # this test — it should report but not fail.
        return 0

    targets = [str(p) for p in REPO_ROOT.glob(".*.py")]
    targets += [str(p) for p in REPO_ROOT.glob(".omniopt_plot_*.py")]
    targets += [str(REPO_ROOT / "omniopt")]

    # The original bash lizard test only printed warnings and never exited
    # non-zero, so we capture and surface them but always return 0.
    proc = subprocess.run(
        [lizard_bin, "--CCN", "15", "--arguments", "6", "--length", "100",
         *targets],
        cwd=str(REPO_ROOT),
    )
    # lizard exits non-zero when it finds warnings; that's expected for this
    # test — it should report but not fail.
    return 0


if __name__ == "__main__":
    sys.exit(main())
