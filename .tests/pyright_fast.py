#!/usr/bin/env python3
"""Run pyright on changed Python files only (faster than full run)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

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
    os.environ["ONLY_CHECK_CHANGED_SINCE_LAST_COMMIT"] = "1"

    pyright_py = THIS_DIR / "pyright.py"
    if pyright_py.exists():
        proc = subprocess.run([sys.executable, str(pyright_py)], cwd=str(REPO_ROOT))
        return proc.returncode

    bin_ = _pyright_bin()
    if bin_ is None:
        print("pyright not found")
        return 1
    proc = subprocess.run([bin_, "."], cwd=str(REPO_ROOT))
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
