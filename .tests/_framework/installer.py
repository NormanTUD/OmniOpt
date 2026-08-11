"""Auto-install dependencies inside a project-local venv.

Mirrors the logic in .shellscript_functions:
  VENV_DIR_NAME=".omniax_venvs/$(python3 --version | sed -e 's# #_#g')/$(uname -m)/"
  ROOT_VENV_DIR=$HOME  (overridable via $root_venv_dir)
  If VIRTUAL_ENV is already set, use that.
  Create venv if missing, then `pip install -r requirements.txt`
  and (when install_tests=1) -r test_requirements.txt.
  Hash-cache the two requirements files inside the venv to skip re-install
  when nothing has changed.
"""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
import sys
import venv
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _venv_dir_name() -> str:
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    return f".omniax_venvs/Python_{py_version}/{platform.machine()}/"


def _resolve_venv_dir() -> Path:
    """Match .shellscript_functions: $VIRTUAL_ENV wins, then $root_venv_dir,
    then $HOME."""
    existing = os.environ.get("VIRTUAL_ENV")
    if existing:
        return Path(existing)
    root = os.environ.get("root_venv_dir")
    if root and Path(root).is_dir():
        return Path(root) / _venv_dir_name()
    return Path.home() / _venv_dir_name()


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


def _hash_file(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.md5(path.read_bytes()).hexdigest()


def _pip(venv_dir: Path, *args: str, quiet: bool = True) -> int:
    pip = venv_dir / "bin" / "pip"
    cmd = [str(pip), "--disable-pip-version-check"]
    if quiet:
        cmd.append("-q")
    cmd.extend(args)
    try:
        return subprocess.run(cmd).returncode
    except KeyboardInterrupt:
        print("pip cancelled by user", file=sys.stderr)
        return 130


def _create_venv(venv_dir: Path) -> bool:
    if venv_dir.exists():
        return True
    print(f"➤ Environment {venv_dir} was not found. Creating it...")
    try:
        venv.create(str(venv_dir), with_pip=True)
    except Exception as exc:
        print(f"❌ Failed to create venv in {venv_dir}: {exc}")
        return False
    print(f"✅ Virtual Environment {venv_dir} created.")
    return True


def venv_site_packages(venv_dir: Path) -> Path | None:
    """Locate the site-packages directory of a venv.

    Different Python versions use different lib paths
    (lib/, lib/python3.X/site-packages, ...), so probe the common ones.
    """
    candidates = [
        venv_dir / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages",
        venv_dir / "lib" / "site-packages",
        venv_dir / "lib" / "python3" / "site-packages",
    ]
    return next((p for p in candidates if p.is_dir()), None)


def add_venv_to_path(venv_dir: Path) -> None:
    """Make a venv's site-packages importable from the current process."""
    site_pkg = venv_site_packages(venv_dir)
    if site_pkg and str(site_pkg) not in sys.path:
        sys.path.insert(0, str(site_pkg))
        os.environ["VIRTUAL_ENV"] = str(venv_dir)
        os.environ["PYTHONPATH"] = (
            str(site_pkg) + os.pathsep + os.environ.get("PYTHONPATH", "")
        )


def install_packages(packages: List[str], *, quiet: bool = True) -> Path | None:
    """Install the given packages into the framework venv (creating it if
    needed) and return the venv directory, or None on failure."""
    if os.environ.get("DONT_INSTALL_MODULES"):
        return None
    venv_dir = _resolve_venv_dir()
    if not _create_venv(venv_dir):
        return None
    if _pip(venv_dir, "install", *packages, quiet=quiet) != 0:
        return None
    return venv_dir


def _run_in_venv(venv_dir: Path, args: List[str]) -> int:
    """Re-exec the current script inside the venv python if we're not
    already running there. Returns the new exit code (or 0 if no re-exec
    happened)."""
    py = venv_dir / "bin" / "python"
    if not py.exists():
        print(f"❌ venv python not found at {py}")
        return 20
    if Path(sys.executable).resolve() == py.resolve():
        return 0  # already inside the venv
    print(f"➡️  Re-executing inside venv: {py}")
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(venv_dir)
    proc = subprocess.run([str(py), *args], env=env)
    return proc.returncode


def ensure_venv() -> Path:
    """Make sure the venv exists and we are running inside it.

    Returns the venv directory. If we are not yet inside, the script
    re-execs itself in the venv and the *new* process exits; the caller
    in the *old* process will see the re-execed exit code via
    ``run_in_venv_or_exit``.
    """
    venv_dir = _resolve_venv_dir()
    if not _create_venv(venv_dir):
        print(f"❌ Failed to create venv at {venv_dir}")
        sys.exit(20)
    return venv_dir


def ensure_dependencies(*, include_tests: bool = True,
                         venv_dir: Path | None = None) -> Path:
    """Create venv if needed, install requirements, and ensure the venv's
    site-packages is on sys.path.

    The venv on this machine has its `python` binary as a symlink to the
    system python (`/usr/bin/python3.13`), so a naive ``sys.executable``
    comparison always reports "already inside". Instead we detect by
    checking whether the venv's ``site-packages`` is already importable
    for a sentinel package, and if not we prepend it to ``sys.path``.
    """
    if os.environ.get("DONT_INSTALL_MODULES"):
        return Path(sys.prefix)

    venv_dir = venv_dir or _resolve_venv_dir()
    if not _create_venv(venv_dir):
        sys.exit(20)

    # Make the venv's site-packages importable.
    add_venv_to_path(venv_dir)

    # Decide if we need to install.
    req_main = REPO_ROOT / "requirements.txt"
    req_test = REPO_ROOT / "test_requirements.txt"

    want_test = include_tests or os.environ.get("install_tests")
    hash_main = _hash_file(req_main)
    hash_test = _hash_file(req_test) if want_test else ""

    hash_file_main = venv_dir / "hash_main"
    hash_file_test = venv_dir / "hash_test"

    need_main = hash_file_main.exists() and hash_file_main.read_text().strip() != hash_main
    need_test = want_test and (
        hash_file_test.exists() and hash_file_test.read_text().strip() != hash_test
    )
    no_main_hash = not hash_file_main.exists() and req_main.exists()
    no_test_hash = want_test and not hash_file_test.exists() and req_test.exists()

    if need_main or need_test or no_main_hash or no_test_hash:
        print("Installing dependencies (this may take a while)...")
        if req_main.exists():
            _pip(venv_dir, "install", "-r", str(req_main), quiet=False)
        if want_test and req_test.exists():
            _pip(venv_dir, "install", "-r", str(req_test), quiet=False)

        if req_main.exists():
            hash_file_main.write_text(hash_main)
        if want_test and req_test.exists():
            hash_file_test.write_text(hash_test)
        print("✅ Dependencies installed.")
    return venv_dir


if __name__ == "__main__":
    ensure_dependencies()
