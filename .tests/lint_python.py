#!/usr/bin/env python3
"""Python linter runner (replaces .tests/lint_python bash script).

Runs ruff on each Python file in the repo root.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _framework.helpers import (
    Colors,
    command_exists,
    green_text,
    human_readable_time,
    red_text,
)
from _framework.installer import ensure_dependencies, install_packages


REPO_ROOT = THIS_DIR.parent


def _ruff_bin() -> str | None:
    """Locate the ruff binary (on PATH or inside the framework venv)."""
    bin_ = shutil.which("ruff")
    if bin_:
        return bin_
    candidates: list[Path] = []
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        candidates.append(Path(venv) / "bin" / "ruff")
    from _framework.installer import _resolve_venv_dir
    try:
        venv_dir = _resolve_venv_dir()
        candidates.append(venv_dir / "bin" / "ruff")
    except Exception:
        pass
    for p in sys.path:
        if not p.endswith("site-packages"):
            continue
        candidates.append(Path(p.rsplit("/lib/", 1)[0]) / "bin" / "ruff")
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def main(argv=None) -> int:
    os.environ.setdefault("install_tests", "1")

    # Ensure dependencies are installed (this should install ruff from test_requirements.txt)
    ensure_dependencies(include_tests=True)

    # If ruff is not available, try to install it using the framework's installer
    ruff_bin = _ruff_bin()
    if ruff_bin is None:
        print("ruff not found, attempting to install...")
        try:
            venv_dir = install_packages(["ruff"], quiet=False)
            if venv_dir is None:
                red_text("Failed to install ruff")
                return 1
            # Re-check for ruff binary after installation
            ruff_bin = _ruff_bin()
            if ruff_bin is None:
                red_text("ruff was installed but could not be located")
                return 1
        except Exception as e:
            red_text(f"Failed to install ruff: {e}")
            return 1

    files = sorted(glob.glob(str(REPO_ROOT / ".*.py")))
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        green_text("No python files to lint.")
        return 0

    errors: list[str] = []
    start = time.time()
    for path in files:
        proc = subprocess.run([ruff_bin, "check", path], cwd=str(REPO_ROOT))
        if proc.returncode != 0:
            errstr = f"Failed linting {path}: Run 'ruff check {path}' to see details."
            red_text(f"\n{errstr}")
            errors.append(errstr)

    elapsed = int(time.time() - start)
    print(f"Lint test took: {human_readable_time(elapsed)}")

    if not errors:
        green_text("No lint-python errors")
        return 0
    red_text("=> LINT-ERRORS =>")
    for e in errors:
        red_text(e)
    return len(errors)


if __name__ == "__main__":
    sys.exit(main())
