"""Auto-install dependencies from requirements.txt and test_requirements.txt.

This replaces the dependency-install portion of .shellscript_functions.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _read_requirements(path: Path) -> List[str]:
    if not path.exists():
        return []
    pkgs: List[str] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        pkgs.append(line)
    return pkgs


def _install(packages: List[str], *, quiet: bool = True) -> None:
    if not packages:
        return
    args = [sys.executable, "-m", "pip", "install"]
    if quiet:
        args.append("-q")
    args.extend(packages)
    subprocess.run(args, check=False)


def ensure_dependencies(*, include_tests: bool = True) -> None:
    """Install everything in requirements.txt (and test_requirements.txt if
    include_tests is True), unless DONT_INSTALL_MODULES is set in the env."""
    if os.environ.get("DONT_INSTALL_MODULES"):
        return
    if os.environ.get("DONT_SHOW_DONT_INSTALL_MESSAGE") is None:
        print("Installing dependencies...")

    req = REPO_ROOT / "requirements.txt"
    test_req = REPO_ROOT / "test_requirements.txt"

    pkgs = _read_requirements(req)
    if include_tests or os.environ.get("install_tests"):
        pkgs += _read_requirements(test_req)

    if pkgs:
        _install(pkgs)


def run_via_shellscript_functions(*args: str) -> int:
    """Compatibility shim: if .shellscript_functions still exists (bash), call
    it via bash for any setup it performs, then run the given python command.

    This lets the Python test runner piggy-back on the existing bash install
    logic until the other chatbot has fully converted .shellscript_functions.
    """
    ssf = REPO_ROOT / ".shellscript_functions"
    if not ssf.exists():
        return 0
    setup_cmd = (
        f"source {ssf} && export install_tests=1 && "
        + " ".join(f'"{a}"' for a in args)
    )
    proc = subprocess.run(
        ["bash", "-c", setup_cmd],
        cwd=str(REPO_ROOT),
    )
    return proc.returncode


if __name__ == "__main__":
    ensure_dependencies(include_tests=True)
