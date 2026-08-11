#!/usr/bin/env python3
"""Find unreachable and unused code."""

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

    current = subprocess.run(
        [sys.executable, "--version"], capture_output=True, text=True
    ).stdout.strip()
    if current == "Python 3.8.10":
        red_text(f"deadcode cannot be run with {current}")
        return 0

    deadcode_bin = shutil.which("deadcode")
    if deadcode_bin is None:
        # deadcode is a package with a console_script entry point that
        # may not be on PATH even though it's installed in the venv.
        # Resolve it via the venv's bin directory instead.
        venv_bin = Path(sys.prefix) / "bin" / "deadcode"
        if venv_bin.exists():
            deadcode_bin = str(venv_bin)
        else:
            proc = subprocess.run(
                [sys.executable, "-c",
                 "from deadcode.cli import main; main()",
                 str(REPO_ROOT)],
                cwd=str(REPO_ROOT),
            )
            return proc.returncode

    proc = subprocess.run([deadcode_bin, str(REPO_ROOT)], cwd=str(REPO_ROOT))
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
