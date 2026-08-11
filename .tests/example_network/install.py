#!/usr/bin/env python3
"""Install the dependencies of the example_network (Python replacement for
``install.sh``).  Used by the Docker build and can be invoked manually from a
checkout.

The original bash version sourced a local ``.shellscript_functions`` library
that handled venv creation, HPC ``lmod`` modules and pip installation.  This
script keeps the useful parts (a few progress-friendly pip installs) and
drops the HPC integration which is irrelevant inside the Docker image where
the script is currently invoked.
"""

from __future__ import annotations

import os
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

    green(f"Installing {len(requirements)} requirements from {REQUIREMENTS_FILE.name} ...")
    for requirement in requirements:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "--disable-pip-version-check",
                "install",
                "-q",
                requirement,
            ],
            check=False,
        )
        if result.returncode != 0:
            red(f"Failed to install {requirement}")
            return result.returncode
        green(f"  {requirement} installed")

    green("All requirements installed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
