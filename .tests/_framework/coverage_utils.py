"""Helpers for measuring coverage across the many Python subprocesses that
the OmniOpt test-suite spawns.

The test-suite starts Python in a lot of places: the test scripts themselves,
``./omniopt``, worker processes, plot scripts and so on.  Wrapping every single
call in ``coverage run -p`` is fragile (and misses shebang scripts), so instead
we use coverage.py's documented subprocess mechanism:

* ``COVERAGE_PROCESS_START`` points at the repository's ``.coveragerc``.
* A tiny ``sitecustomize.py`` is put on ``PYTHONPATH``; it calls
  ``coverage.process_startup()`` in every child interpreter.
* The venv's ``site-packages`` is put on ``PYTHONPATH`` as well so child
  processes started with the *system* Python can still import ``coverage``.

All data files are written in parallel mode and combined later by
``.tests/coverage_report``.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

BOOTSTRAP_DIR = Path(__file__).resolve().parent / "coverage_bootstrap"
PHP_PREPEND_FILE = Path(__file__).resolve().parent.parent / "php_coverage_prepend.php"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _prepend_env_path(name: str, entry: str) -> None:
    """Prepend *entry* to the path-like environment variable *name* once."""
    entry = str(entry)
    parts = [p for p in os.environ.get(name, "").split(os.pathsep) if p]
    if entry in parts:
        return
    os.environ[name] = os.pathsep.join([entry, *parts])


def site_packages_dirs(venv_dir: Path) -> list[Path]:
    """Return the ``site-packages`` directories of a virtualenv."""
    return sorted(Path(venv_dir).glob("lib/python*/site-packages"))


def _php_has_xdebug(php: str) -> bool:
    try:
        proc = subprocess.run(
            [php, "-r", "echo extension_loaded('xdebug') ? '1' : '0';"],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return (proc.stdout or "").strip() == "1"


def enable_php_coverage(repo_root: Optional[Path] = None) -> Optional[Path]:
    """Enable Xdebug line coverage for every PHP process in this test run.

    A small ``php`` wrapper is put on ``PATH`` so we do not have to modify
    every individual test invocation.  The wrapper sets Xdebug's coverage
    mode and an ``auto_prepend_file`` that records line coverage.  The
    built-in PHP server used by the share tests therefore gets per-request
    coverage as well.
    """
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    active_dir = os.environ.get("OMNIOPT_PHP_COVERAGE_DIR")
    if os.environ.get("OMNIOPT_PHP_COVERAGE_ACTIVE") and active_dir:
        return Path(active_dir)

    real_php = os.environ.get("OMNIOPT_PHP_COVERAGE_REAL_PHP") or shutil.which("php")
    if not real_php or not _php_has_xdebug(real_php):
        return None

    data_dir = Path(active_dir or (root / ".php-coverage"))
    data_dir.mkdir(parents=True, exist_ok=True)
    os.environ["OMNIOPT_PHP_COVERAGE_DIR"] = str(data_dir)
    os.environ["XDEBUG_MODE"] = "coverage"

    wrapper_dir = Path(tempfile.mkdtemp(prefix="omniopt_php_coverage_"))
    wrapper = wrapper_dir / "php"
    prepend_arg = shlex.quote(f"auto_prepend_file={PHP_PREPEND_FILE}")
    wrapper.write_text(
        "#!/bin/sh\n"
        "export XDEBUG_MODE=coverage\n"
        f"exec {shlex.quote(real_php)} -d {prepend_arg} \"$@\"\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    _prepend_env_path("PATH", str(wrapper_dir))
    os.environ["OMNIOPT_PHP_COVERAGE_ACTIVE"] = "1"
    os.environ["OMNIOPT_PHP_COVERAGE_REAL_PHP"] = real_php
    return data_dir


def enable_js_coverage(repo_root: Optional[Path] = None) -> Optional[Path]:
    """Enable V8 coverage for every Node process in this test run."""
    if not shutil.which("node"):
        return None
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    data_dir = Path(os.environ.get("NODE_V8_COVERAGE") or (root / ".js-coverage"))
    data_dir.mkdir(parents=True, exist_ok=True)
    os.environ["NODE_V8_COVERAGE"] = str(data_dir)
    return data_dir


def enable_subprocess_coverage(
    *,
    repo_root: Optional[Path] = None,
    venv_dir: Optional[Path] = None,
) -> Path:
    """Enable coverage for Python, PHP and JavaScript child processes.

    Returns the ``.coveragerc`` that was activated.  Calling the function
    multiple times is safe.
    """
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    coveragerc = root / ".coveragerc"

    os.environ["RUN_WITH_COVERAGE"] = "1"
    os.environ["COVERAGE_PROCESS_START"] = str(coveragerc)
    _prepend_env_path("PYTHONPATH", str(BOOTSTRAP_DIR))

    if venv_dir is not None:
        venv_dir = Path(venv_dir)
        for site_packages in site_packages_dirs(venv_dir):
            _prepend_env_path("PYTHONPATH", str(site_packages))
        _prepend_env_path("PATH", str(venv_dir / "bin"))

    enable_php_coverage(root)
    enable_js_coverage(root)
    return coveragerc
