#!/usr/bin/env python3
"""Install the dependencies of the example_network (Python replacement for
``install.sh``).  Used by the Docker build and can be invoked manually from a
checkout.

The original bash version sourced a local ``.shellscript_functions`` library
that handled venv creation, HPC ``lmod`` modules and pip installation.  This
script keeps the useful parts (a few progress-friendly pip installs) and
drops the HPC integration which is irrelevant inside the Docker image where
the script is currently invoked.

The venv-resolution logic mirrors the one in ``.shellscript_functions`` /
``.tests/_framework/installer.py``: prefer ``$VIRTUAL_ENV``, then
``$root_venv_dir``, then ``$HOME``.  When no venv exists (e.g. on a fresh
checkout on a PEP 668 protected Python), fall back to ``pip install
--break-system-packages`` so installation still succeeds.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REQUIREMENTS_FILE = THIS_DIR / "requirements.txt"

GREEN = "\033[0;32m"
RED = "\033[0;31m"
RESET = "\033[0m"


def _color(text: str, color: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{color}{text}{RESET}"


def green(text: str) -> None:
    print(_color(text, GREEN), flush=True)


def red(text: str) -> None:
    print(_color(text, RED), file=sys.stderr, flush=True)


def _venv_dir_name() -> str:
    py_version = (
        f"{sys.version_info.major}."
        f"{sys.version_info.minor}."
        f"{sys.version_info.micro}"
    )
    return f".omniax_venvs/Python_{py_version}/{platform.machine()}/"


def _resolve_venv_dir() -> Path | None:
    """Return the venv directory used by ``.shellscript_functions`` or None."""
    existing = os.environ.get("VIRTUAL_ENV")
    if existing and Path(existing, "bin", "pip").exists():
        return Path(existing)

    root = os.environ.get("root_venv_dir")
    if root and Path(root).is_dir():
        candidate = Path(root) / _venv_dir_name()
        if (candidate / "bin" / "pip").exists():
            return candidate

    home_candidate = Path.home() / _venv_dir_name()
    if (home_candidate / "bin" / "pip").exists():
        return home_candidate

    return None


def _pip_cmd(requirement: str) -> list[str]:
    """Build the pip invocation, preferring a project venv if one exists."""
    venv = _resolve_venv_dir()
    if venv is not None:
        pip = venv / "bin" / "pip"
        return [str(pip), "--disable-pip-version-check", "install", "-q", requirement]

    cmd = [
        sys.executable,
        "-m",
        "pip",
        "--disable-pip-version-check",
        "install",
        "-q",
        requirement,
    ]
    if shutil.which("apt") and Path("/etc/debian_version").exists():
        cmd.append("--break-system-packages")
    return cmd


def main(argv=None) -> int:
    os.environ.setdefault("install_tests", "1")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    if not REQUIREMENTS_FILE.exists():
        red(f"Requirements file not found: {REQUIREMENTS_FILE}")
        return 1

    requirements = [
        line.strip()
        for line in REQUIREMENTS_FILE.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    if not requirements:
        red("No requirements found.")
        return 1

    venv = _resolve_venv_dir()
    if venv is not None:
        green(f"Using venv at {venv}")
    else:
        green("No project venv found; installing into system Python.")

    green(f"Installing {len(requirements)} requirements from {REQUIREMENTS_FILE.name} ...")
    for requirement in requirements:
        result = subprocess.run(_pip_cmd(requirement), check=False)
        if result.returncode != 0:
            red(f"Failed to install {requirement}")
            return result.returncode
        green(f"  {requirement} installed")

    green("All requirements installed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
