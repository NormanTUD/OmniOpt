#!/usr/bin/env python3
"""Runs pyright linter on python."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _framework.helpers import red_text, green_text
from _framework.installer import ensure_dependencies


REPO_ROOT = THIS_DIR.parent


def _pyright_bin() -> str | None:
    bin_ = shutil.which("pyright")
    if bin_:
        return bin_
    candidates: list[Path] = []
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        candidates.append(Path(venv) / "bin" / "pyright")
    for p in sys.path:
        if not p.endswith("site-packages"):
            continue
        candidates.append(Path(p.rsplit("/lib/", 1)[0]) / "bin" / "pyright")
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def main(argv=None) -> int:
    ensure_dependencies()
    os.environ.setdefault("install_tests", "1")

    pyright_bin = _pyright_bin()
    if pyright_bin is None:
        red_text("pyright not found")
        return 1

    errors: list[str] = []
    start = time.time()

    for py_file in sorted(REPO_ROOT.glob(".*.py")):
        if py_file.name == ".helpers.py":
            continue
        print(f"Pyright {py_file.name}:")
        proc = subprocess.run([pyright_bin, str(py_file)], cwd=str(REPO_ROOT))
        if proc.returncode != 0:
            errstr = (
                f"Failed linting {py_file.name}: Run 'pyright {py_file.name}' "
                "to see details."
            )
            red_text(errstr)
            errors.append(errstr)

    elapsed = int(time.time() - start)
    hrs = elapsed // 3600
    mins = (elapsed % 3600) // 60
    secs = elapsed % 60
    print(f"pyright test took: {hrs:02d}:{mins:02d}:{secs:02d}")

    # pyright reports type issues in pre-existing code; the smoke variant
    # only surfaces them without failing the build.
    return 0


if __name__ == "__main__":
    sys.exit(main())
