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
from pathlib import Path
from typing import Optional

BOOTSTRAP_DIR = Path(__file__).resolve().parent / "coverage_bootstrap"
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


def enable_subprocess_coverage(
    *,
    repo_root: Optional[Path] = None,
    venv_dir: Optional[Path] = None,
) -> Path:
    """Enable coverage for this process and all Python children.

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

    return coveragerc
