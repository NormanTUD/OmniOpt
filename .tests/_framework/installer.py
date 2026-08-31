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
    """Build a bash -lc body that makes `ml` (lmod) available even in a
    NON-interactive shell, then runs `cmd`.

    On HPC login nodes `ml` is usually defined only in `.bashrc` (interactive
    login), so a `bash -lc` does NOT see it and LMOD_CMD may not be exported.
    We therefore explicitly source the common lmod init files -- this is the
    step that makes the whole "load newest Python module" path actually work
    here -- before running `cmd`.
    """
    return (
        'if ! command -v ml >/dev/null 2>&1 && [ -n "${LMOD_CMD:-}" ]; then '
        'eval "$(${LMOD_CMD} bash)"; fi; '
        'if ! command -v ml >/dev/null 2>&1 && ! command -v module >/dev/null 2>&1; then '
        'for _lmi in /usr/share/lmod/lmod/init/bash /opt/lmod/lmod/init/bash '
        '/etc/profile.d/modules.sh /etc/profile.d/lmod.sh; do '
        '[ -f "$_lmi" ] && . "$_lmi" 2>/dev/null; done; fi; unset _lmi; '
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
    probe-validated (rejects SIGILL/broken builds).

    A module load that silently fails must return None (NOT the unchanged
    system python): we compare the resolved python before vs after the load
    and only trust a path the module actually substituted.  Otherwise a
    failed toolchain/Python load would make us build on the stale login-node
    system python (e.g. Python 3.9.25) -- the exact bug we're fixing.
    """
    if not _has_lmod() or not modules:
        return None
    script = _lmod_script(
        '_before="$(command -v python3 || true)"; '
        f"ml --quiet {modules} >/dev/null 2>&1; "
        '_after="$(command -v python3 || true)"; '
        'if [ -n "$_after" ] && [ "$_after" != "$_before" ]; then '
        'printf \'%s\' "$_after"; fi'
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
    # suppress pip's own raw `━━━` progress bars (noise in TTY, log spam when
    # non-TTY/--follow).  `--progress-bar off` is accepted on modern pip;
    # it keeps clean `Collecting/Downloading/Installing` lines.
    if "--progress-bar" not in args:
        cmd.append("--progress-bar off")
    cmd.extend(args)
    _pip._cancelled = False  # type: ignore[attr-defined]
    try:
        return subprocess.run(cmd).returncode
    except KeyboardInterrupt:
        _pip._cancelled = True  # type: ignore[attr-defined]
        print("pip cancelled by user", file=sys.stderr)
        return 130


# ---------------------------------------------------------------------------
# Rich progress-bar install.
#
# The `.tests/main` bootstrap runs *before* Rich is installed (Rich itself is
# a requirement), so pip was previously invoked raw -- which dumped pip's own
# `━━━` progress bars into the terminal / HPC log.  To make sure the user
# ALWAYS sees a clean, single, transient Rich progress bar while the install
# is working (and raw pip output ONLY when something actually fails), we run
# the install in a dedicated child Python that:
#
#   guardrail A) ensures Rich is importable (installs it quietly if missing),
#   guardrail B) always passes `--progress-bar off` to pip inside the child,
#   guardrail C) on a TTY renders ONE transient Rich progress bar (with
#                elapsed/ETA) that disappears on success,
#   guardrail D) on a non-TTY (HPC log / `--follow`) prints a single clean
#                result line instead of `\r`-redraw soup,
#   guardrail E) shows raw pip output ONLY on failure (de-ANSI'd tail).
# ---------------------------------------------------------------------------


def _install_requirements_rich(
    venv_dir: Path, req_file: Path, label: str = "requirements"
) -> bool:
    """Install ``req_file`` inside ``venv_dir`` using a transient Rich progress
    bar in a child process (so the parent bootstrap stays stdlib-only).

    Raw pip output is never shown while the install is healthy -- it is only
    surfaced (as a tail) when the install actually fails.
    """
    py = venv_dir / "bin" / "python"

    # Guardrail A: Rich must be present inside the venv, otherwise the child
    # can't render a bar.  Because Rich might itself be one of the packages
    # being installed, install it first (quietly, bars off, no raw output).
    try:
        _has_rich = (
            subprocess.run(
                [str(py), "-c", "import rich" ],
                capture_output=True, timeout=60,
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        _has_rich = False
    if not _has_rich:
        _try_quiet_pip(venv_dir, "install", "--progress-bar", "off", "-q", "rich")

    child = r'''
import subprocess, sys, time, re, signal, os

_label = sys.argv[1]
_reqfile = sys.argv[2]
_TTY = bool(sys.stdout.isatty())


def _deansi(s):
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def _tail(err, n=6):
    lines = [_deansi(l) for l in err.splitlines() if l.strip()]
    return lines[-n:] if lines else ["(no output captured)"]


def _quiet(_path):
    # Guardrail D: non-interactive fallback -- one clean line, raw pip on fail.
    _t0 = time.time()
    _p = subprocess.run(
        [sys.executable, "-u", "-m", "pip", "install",
         "--default-timeout=300", "--disable-pip-version-check",
         "--quiet", "--progress-bar", "off", "-r", _path],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    _el = time.time() - _t0
    if _p.returncode == 0:
        sys.stdout.write(f"[omniopt] installed {_label} ({_el:.1f}s)\n")
        sys.stdout.flush()
        return 0
    sys.stdout.write(f"[omniopt] pip install {_label} failed (exit {_p.returncode}, {_el:.1f}s)\n")
    for _l in _tail(_p.stderr or ""):
        sys.stdout.write("  " + _l + "\n")
    sys.stdout.flush()
    return 1


if not _TTY:
    sys.exit(_quiet(_reqfile))

try:
    from rich.console import Console
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn, BarColumn,
        TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn,
    )
except ImportError:
    # Guardrail A (defensive): even if the probe above raced, degrade to a
    # clean one-line install rather than dumping raw pip output.
    sys.exit(_quiet(_reqfile))

console = Console(force_terminal=True, soft_wrap=False)
_interrupted = {"v": False}


def _on_sigint(signum, frame):
    _interrupted["v"] = True


signal.signal(signal.SIGINT, _on_sigint)


def _count_reqs(path):
    n = 0
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                if ln.startswith("-"):
                    if (ln.startswith("-r") or ln.startswith("--requirement")):
                        rest = ln.split(maxsplit=1)
                        if len(rest) > 1:
                            try:
                                n += _count_reqs(os.path.join(
                                    os.path.dirname(path) or ".", rest[1]))
                            except Exception:
                                pass
                    continue
                if ln.startswith("--"):
                    continue
                n += 1
    except OSError:
        pass
    return n


try:
    with Progress(
        SpinnerColumn("dots"),
        TextColumn("{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TextColumn("elapsed"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
        refresh_per_second=10,
    ) as progress:
        task = progress.add_task(
            f"[cyan]preparing {_label} ...".ljust(80),
            total=100, completed=0,
        )

        # Flatten the requirements file into one installable spec per package,
        # recursing into `-r` includes, skipping comments / `--flags` / blank
        # lines.  Each spec gets its OWN `pip install <spec>` so we get REAL
        # per-package progress ("installing <name> -- N remaining") instead
        # of pip's single un-timed "Installing collected packages: ..." line.
        def _flatten_reqs(path, _seen=None):
            _seen = _seen if _seen is not None else set()
            _real = os.path.realpath(path)
            if _real in _seen:
                return []
            _seen.add(_real)
            out = []
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    for ln in f:
                        ln = ln.strip()
                        if not ln or ln.startswith("#"):
                            continue
                        if (ln.startswith("-r") or ln.startswith("--requirement")):
                            rest = ln.split(maxsplit=1)
                            if len(rest) > 1:
                                out += _flatten_reqs(
                                    os.path.join(os.path.dirname(path) or ".", rest[1]),
                                    _seen,
                                )
                            continue
                        if ln.startswith("-") or ln.startswith("--"):
                            continue
                        out.append(ln)
            except OSError:
                pass
            return out

        _specs = _flatten_reqs(_reqfile)
        if not _specs:
            _specs = [_reqfile]
        _total = len(_specs)
        _rc = 0
        captured = []

        # Guardrail: heartbeat keeps the spinner alive during slow/dep
        # resolution; without it the bar freezes for seconds at a time.
        import threading
        _stop = {"v": False}

        def _hb():
            while not _stop["v"]:
                try:
                    progress.refresh()
                except Exception:
                    pass
                time.sleep(0.1)
        _t = threading.Thread(target=_hb, daemon=True)
        _t.start()

        _t0 = time.time()
        try:
            for _i, _spec in enumerate(_specs, start=1):
                if _interrupted["v"]:
                    _rc = 130
                    break
                _name = _spec.split()[0]
                _remaining = _total - _i
                progress.update(
                    task,
                    total=_total,
                    completed=_i - 1,
                    description=(
                        f"[cyan]installing[/cyan] [bold]{_name}[/bold]  "
                        f"[dim]({_i}/{_total}) -- {_remaining} remaining[/dim]".ljust(80)
                    ),
                )
                try:
                    _p = subprocess.Popen(
                        [sys.executable, "-u", "-m", "pip", "install",
                         "--default-timeout=300", "--disable-pip-version-check",
                         "--progress-bar", "off", _spec],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=0,
                    )
                except Exception as e:
                    captured.append(str(e))
                    _rc = 1
                    break
                for _l in _p.stdout:
                    if _interrupted["v"]:
                        break
                    _l = _l.rstrip()
                    if _l:
                        captured.append(_l)
                _p.wait()
                if _interrupted["v"]:
                    _rc = 130
                    break
                if _p.returncode != 0:
                    _rc = _p.returncode or 1
                    break
                progress.update(task, completed=_i, total=_total)
        finally:
            _stop["v"] = True
except KeyboardInterrupt:
    _rc = 130

_el = time.time() - _t0
if _rc == 0:
    # transient bar already cleared itself; emit one final clean line.
    sys.stdout.write(f"done installing {_label} ({_el:.1f}s)\n")
    sys.stdout.flush()
    sys.exit(0)

# Guardrail E: only on failure do we show pip's raw output (tail, de-ANSI'd).
sys.stdout.write(f"pip install {_label} failed (exit {_rc}, {_el:.1f}s)\n")
for _l in _tail("\n".join(captured)):
    sys.stdout.write("  " + _l + "\n")
sys.stdout.flush()
sys.exit(1 if _rc != 130 else 130)
'''
    try:
        r = subprocess.run([str(py), "-u", "-c", child, label, str(req_file)])
    except KeyboardInterrupt:
        return False
    return r.returncode == 0


def _try_quiet_pip(venv_dir: Path, *args: str) -> int:
    """Run a pip command quietly (bars off, stderr piped) without ever
    printing raw pip output on success; returns the exit code."""
    pip = _ensure_pip_in_venv(venv_dir)
    if pip is None:
        return 20
    try:
        return subprocess.run(
            [str(pip), "--disable-pip-version-check",
             "--progress-bar", "off", *args],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        ).returncode
    except KeyboardInterrupt:
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
    base = base_python or _resolve_base_python()
    if not _probe_python(base):
        print(
            f"❌ Base python {base} cannot run on this node; "
            "refusing to build a venv on a broken interpreter.",
            file=sys.stderr,
        )
        return False

    py = venv_dir / "bin" / "python"
    if venv_dir.exists():
        # Validate an *existing* venv instead of blindly trusting it.  A
        # stale/broken venv (e.g. an old "Python_3.9.25" with no pip) must be
        # rebuilt on the correct base python, or every later pip call crashes.
        if (
            py.exists()
            and _probe_python(str(py))
            and (venv_dir / "bin" / "pip").exists()
        ):
            return True
        print(
            f"⚠️ Existing environment {venv_dir} is broken or incomplete; "
            "recreating it...",
            file=sys.stderr,
        )
        shutil.rmtree(str(venv_dir), ignore_errors=True)

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
            # Guardrail: ALWAYS route through the Rich progress-bar child so
            # pip's raw `━━━` output is never shown while things are healthy.
            if not _install_requirements_rich(venv_dir, req_main, "requirements"):
                print("❌ Failed to install main requirements.", file=sys.stderr)
                sys.exit(20)
        if want_test and req_test.exists():
            if not _install_requirements_rich(venv_dir, req_test, "test requirements"):
                print("❌ Failed to install test requirements.", file=sys.stderr)
                sys.exit(20)

        if req_main.exists():
            hash_file_main.write_text(hash_main)
        if want_test and req_test.exists():
            hash_file_test.write_text(hash_test)
        print("✅ Dependencies installed.")
    return venv_dir


if __name__ == "__main__":
    ensure_dependencies()
