#!/usr/bin/env python3
"""Run pylint linter on python files."""

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
RC_FILE = REPO_ROOT / ".tests" / "pylint.rc"


def _pylint_bin() -> str | None:
    """Locate the pylint binary (on PATH or inside the framework venv)."""
    bin_ = shutil.which("pylint")
    if bin_:
        return bin_
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        candidate = Path(venv) / "bin" / "pylint"
        if candidate.exists():
            return str(candidate)
    return None


def main(argv=None) -> int:
    # Installs missing tools (e.g. pylint) from test_requirements.txt into
    # the framework venv when they are not available yet.
    ensure_dependencies()
    os.environ.setdefault("install_tests", "1")

    targets = [str(p) for p in REPO_ROOT.glob(".*.py")]
    targets = [t for t in targets if not t.endswith("/.helpers.py")]
    targets += [str(REPO_ROOT / "omniopt")]

    flags: list[str] = []
    if RC_FILE.exists():
        flags += [f"--rcfile={RC_FILE}"]
    flags += targets

    pylint_bin = _pylint_bin()
    if pylint_bin is None:
        # Fall back to `python -m pylint` (works when pylint is only
        # installed inside the framework venv that is not on PATH).
        proc = subprocess.run(
            [sys.executable, "-m", "pylint", *flags], cwd=str(REPO_ROOT)
        )
        return proc.returncode

    proc = subprocess.run([pylint_bin, *flags], cwd=str(REPO_ROOT))
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
