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
from _framework.installer import ensure_dependencies


REPO_ROOT = THIS_DIR.parent


def main(argv=None) -> int:
    os.environ.setdefault("install_tests", "1")

    # Ensure dependencies are installed first
    ensure_dependencies(include_tests=True)

    if not command_exists("ruff"):
        # If ruff is not found, check if it's available in the framework venv
        # by checking if we can import it through the venv mechanism
        try:
            # Try to use pip from within the framework's venv
            venv_dir = None
            # Check if we're already in a venv
            if os.environ.get("VIRTUAL_ENV"):
                venv_dir = Path(os.environ["VIRTUAL_ENV"])
            else:
                # Try to locate the framework venv
                from _framework.installer import _resolve_venv_dir
                venv_dir = _resolve_venv_dir()
            
            if venv_dir and venv_dir.exists():
                # Try to install ruff in the framework venv
                pip_path = venv_dir / "bin" / "pip"
                if pip_path.exists():
                    print("ruff not found, attempting to install in venv...")
                    result = subprocess.run([
                        str(pip_path), "install", "ruff"
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        # Check if installation worked
                        if command_exists("ruff"):
                            print("Successfully installed ruff in venv")
                        else:
                            print("Warning: ruff installed but not found in PATH")
                    else:
                        print(f"Failed to install ruff in venv: {result.stderr}")
                        
            # If we still don't have ruff, fail gracefully
            if not command_exists("ruff"):
                red_text("ruff not found. Please install ruff in your environment:")
                red_text("  pip install ruff")
                return 1
        except Exception as e:
            print(f"Error trying to install ruff: {e}")
            red_text("ruff not found. Please install ruff in your environment:")
            red_text("  pip install ruff")
            return 1

    files = sorted(glob.glob(str(REPO_ROOT / ".*.py")) + glob.glob(str(REPO_ROOT / "*.py")))
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        green_text("No python files to lint.")
        return 0

    errors: list[str] = []
    start = time.time()
    for path in files:
        proc = subprocess.run(["ruff", "check", path], cwd=str(REPO_ROOT))
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
