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
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# HPC / environment-modules (lmod) handling.
#
# On HPC login nodes the *system* python that launched this process is often
# old (e.g. Python 3.9.25) and NOT the intended compute environment.  We must
# load the newest available Python via `ml` (lmod) and build the venv *on top
# of that* interpreter -- otherwise the venv path is wrong ("Python_3.9.25")
# and pip is missing, which crashes every run.
# ---------------------------------------------------------------------------

_MODULES_LOADED_VAR = "OMNIOPT_HPC_MODULES_LOADED"
_BASE_PY_CACHE: list = []  # single slot: resolved base python path


def _arch() -> str:
    return platform.machine()


def _has_lmod() -> bool:
    """Whether an lmod-style environment-module system (``ml``/``module``) is
    available on this machine (typical for HPC login nodes)."""
    for _c in ("ml", "module", "modulecmd"):
        if shutil.which(_c):
            return True
    if os.environ.get("LMOD_CMD") or os.environ.get("MODULESHOME"):
        return True
    return (
        os.path.exists("/usr/share/lmod/lmod/init/bash")
        or os.path.exists("/etc/profile.d/modules.sh")
    )


def _probe_python(py: str) -> bool:
    """Return whether ``py`` actually runs on THIS node.  Rejects
    interpreters that die with a signal (e.g. SIGILL / -4) because they were
    built for a different CPU/ISA than the current node."""
    try:
        r = subprocess.run(
            [py, "-c",
             "import sys, ssl, hashlib, ctypes, struct, json, decimal, "
             "_ssl, _ctypes, _decimal; print('OK')"],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0 and "OK" in (r.stdout or "")


def _lmod_script(cmd: str) -> str:
    """Build a bash -lc body that initialises lmod if needed, runs `cmd`,
    and (for python resolution) prints the resolved python path last."""
    return (
        'if ! command -v ml >/dev/null 2>&1 && [ -n "${LMOD_CMD:-}" ]; then '
        'eval "$(${LMOD_CMD} bash)"; fi; '
        + cmd
    )


def _newest_python_module() -> str:
    """Ask ``ml spider Python`` and return the newest ``Python/x.y.z`` module
    name (empty string if none found). ``sort -V`` picks the highest version;
    spurious ``Python/x.y.z-bare`` builds are excluded by the grep regex."""
    if not _has_lmod():
        return ""
    script = _lmod_script(
        "{ ml spider Python 2>/dev/null; } | "
        r"grep -oE 'Python/[0-9]+(\.[0-9]+)*' | sort -V | tail -n 1"
    )
    try:
        r = subprocess.run(
            ["bash", "-lc", script], capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return (r.stdout or "").strip()


def _ml_python(modules: str) -> str | None:
    """Load ``modules`` via lmod and return the resolved ``python3`` path, or
    None if the load produced nothing usable.  The returned interpreter is
    probe-validated (rejects SIGILL/broken builds)."""
    if not _has_lmod() or not modules:
        return None
    script = _lmod_script(
        f"ml --quiet {modules} >/dev/null 2>&1; command -v python3 || true"
    )
    try:
        r = subprocess.run(
            ["bash", "-lc", script], capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = (r.stdout or "").strip().splitlines()
    if not lines:
        return None
    cand = lines[-1].strip()
    if not cand or not os.path.exists(cand):
        return None
    if not _probe_python(cand):
        return None
    return cand


def _resolve_base_python() -> str:
    """Return the interpreter to build the venv from.

    Preference on HPC (lmod present): the newest available Python module,
    else the known-good toolchain's python.  Fallback: whatever launched us.
    The result is probe-validated and cached.
    """
    if _BASE_PY_CACHE:
        return _BASE_PY_CACHE[0]

    base: str | None = None
    if _has_lmod():
        newest = _newest_python_module()
        if newest:
            base = _ml_python(newest)
        if not base:
            # Fall back to the known-good toolchain python.
            base = _ml_python("release/24.04 GCC/12.3.0 OpenMPI/4.1.5 PyTorch/2.1.2")

    if not base or not _probe_python(base):
        base = sys.executable

    _BASE_PY_CACHE.append(base)
    return base


def _venv_dir_name(base_python: str | None = None) -> str:
    """Version-keyed venv dir, derived from the *base python* used to create
    the venv (NOT the launching process -- that can be a stale login-node
    system python).  Mirrors the old ``Python_X.Y.Z/$(uname -m)/`` layout."""
    base = base_python or _resolve_base_python()
    py_version = ""
    try:
        r = subprocess.run(
            [base, "-c",
             "import sys; print(f'{sys.version_info.major}."
             "{sys.version_info.minor}.{sys.version_info.micro}')"],
            capture_output=True, text=True, timeout=30,
        )
        py_version = (r.stdout or "").strip()
    except (OSError, subprocess.TimeoutExpired):
        py_version = ""
    if not py_version:
        py_version = "unknown_python"
    return f".omniax_venvs/Python_{py_version}/{_arch()}/"


def _resolve_venv_dir() -> Path:
    """Match .shellscript_functions: $VIRTUAL_ENV wins, then $root_venv_dir,
    then $HOME.  The version part reflects the *resolved base python*, so the
    venv dir matches the interpreter the venv is actually built from."""
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


def _ensure_pip_in_venv(venv_dir: Path) -> Path | None:
    """Make sure the venv has a working ``pip``.  Some python distributions
    (and old/broken venvs) create a venv without pip.  Returns the pip path
    or None if we could not obtain one."""
    pip = venv_dir / "bin" / "pip"
    if pip.exists():
        return pip
    py = venv_dir / "bin" / "python"
    if not py.exists():
        return None
    try:
        subprocess.run(
            [str(py), "-m", "ensurepip", "--upgrade"],
            check=False,
        )
    except OSError:
        return None
    return pip if pip.exists() else None


def _pip(venv_dir: Path, *args: str, quiet: bool = True) -> int:
    pip = _ensure_pip_in_venv(venv_dir)
    if pip is None:
        print(
            f"❌ No working pip in {venv_dir} -- the venv is unusable. "
            "Delete it and re-run.",
            file=sys.stderr,
        )
        return 20
    cmd = [str(pip), "--disable-pip-version-check"]
    if quiet:
        cmd.append("-q")
    cmd.extend(args)
    _pip._cancelled = False  # type: ignore[attr-defined]
    try:
        return subprocess.run(cmd).returncode
    except KeyboardInterrupt:
        _pip._cancelled = True  # type: ignore[attr-defined]
        print("pip cancelled by user", file=sys.stderr)
        return 130


def _create_venv(venv_dir: Path) -> bool:
    return _create_venv_from(venv_dir, None)


def venv_site_packages(venv_dir: Path) -> Path | None:
    """Locate the site-packages directory of a venv.

    The venv may be built from a *module* Python whose version differs from
    the launching process, so we probe the real ``lib/*/site-packages`` that
    actually exists instead of guessing from ``sys.version_info``.
    """
    lib = next(
        (d for d in (venv_dir / "lib").glob("python*/site-packages") if d.is_dir()),
        None,
    )
    if lib is None:
        candidates = [
            venv_dir / "lib" / "site-packages",
            venv_dir / "lib" / "python3" / "site-packages",
        ]
        return next((p for p in candidates if p.is_dir()), None)
    return lib


def _create_venv_from(venv_dir: Path, base_python: str | None = None) -> bool:
    if venv_dir.exists():
        return True
    base = base_python or _resolve_base_python()
    if not _probe_python(base):
        print(
            f"❌ Base python {base} cannot run on this node; "
            "refusing to build a venv on a broken interpreter.",
            file=sys.stderr,
        )
        return False
    print(f"➤ Environment {venv_dir} was not found. Creating it...")
    _vdir = str(venv_dir)
    # Build from the resolved base python (usually the newest HPC module
    # python), NOT from sys.executable of the launching (possibly stale)
    # process.
    try:
        subprocess.run(
            [base, "-m", "venv", _vdir],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"❌ Failed to create venv in {venv_dir}: {exc}")
        return False
    py = venv_dir / "bin" / "python"
    if not py.exists() or not _probe_python(str(py)):
        print(
            f"❌ venv python missing/broken at {py}; delete {venv_dir} and re-run.",
            file=sys.stderr,
        )
        return False
    if not _ensure_pip_in_venv(venv_dir):
        print(
            f"❌ Could not obtain pip in {venv_dir}; "
            f"delete {venv_dir} and re-run.",
            file=sys.stderr,
        )
        return False
    print(f"✅ Virtual Environment {venv_dir} created (python {base}).")
    return True


def add_venv_to_path(venv_dir: Path) -> None:
    """Make a venv's site-packages importable from the current process."""
    site_pkg = venv_site_packages(venv_dir)
    if site_pkg and str(site_pkg) not in sys.path:
        sys.path.insert(0, str(site_pkg))
        os.environ["VIRTUAL_ENV"] = str(venv_dir)
        os.environ["PYTHONPATH"] = (
            str(site_pkg) + os.pathsep + os.environ.get("PYTHONPATH", "")
        )


def _venv_packages(venv_dir: Path) -> Dict[str, str]:
    """Return ``{normalized_name: version_str}`` for every installed
    package in the framework venv.  Used to skip reinstalling packages
    that are already present (saves ~3 s of pip-startup per script)."""
    site_pkg = venv_site_packages(venv_dir)
    if not site_pkg or not site_pkg.is_dir():
        return {}
    out: Dict[str, str] = {}
    for dist_info in site_pkg.glob("*.dist-info"):
        metadata = dist_info / "METADATA"
        if not metadata.is_file():
            continue
        try:
            text = metadata.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        name = ""
        version = ""
        for line in text.splitlines():
            low = line.lower()
            if not name and low.startswith("name:"):
                name = line.split(":", 1)[1].strip().lower()
            elif low.startswith("version:"):
                version = line.split(":", 1)[1].strip()
        if name:
            out[name] = version
    return out


def install_packages(packages: List[str], *, quiet: bool = True) -> Path | None:
    """Install the given packages into the framework venv (creating it if
    needed) and return the venv directory, or None on failure."""
    if os.environ.get("DONT_INSTALL_MODULES"):
        return None
    venv_dir = _resolve_venv_dir()
    if not _create_venv(venv_dir):
        return None
    try:
        rc = _pip(venv_dir, "install", *packages, quiet=quiet)
    except KeyboardInterrupt:
        print("install_packages cancelled by user", file=sys.stderr)
        return None
    if rc != 0:
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
