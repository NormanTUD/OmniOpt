"""Auto-install missing Python packages for the typo-check tools.

The find_typos_in_* scripts and .tools/php_spellchecker.py rely on
third-party packages (pyspellchecker, emoji, rich, beautifulsoup4). If a
package is not importable this module installs it so the tools work in a
bare environment.

Packages are always installed into the framework venv (see
``_framework.installer``) and that venv's site-packages is then added to
``sys.path``. This avoids ever touching an externally-managed system
Python (PEP 668).

Set DONT_INSTALL_MODULES=1 to forbid installation (used by the main test
runner, where the venv already provides everything).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import List, Tuple


def _importable(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def ensure_imports(
    requirements: Tuple[Tuple[str, str], ...],
    *,
    quiet: bool = True,
) -> bool:
    """Ensure every ``module_name, package_name`` is importable.

    Missing packages are installed automatically into the framework venv.
    Returns True when all modules are available afterwards. When
    ``DONT_INSTALL_MODULES`` is set in the environment, installation is
    skipped and False is returned for any missing module.

    A ``KeyboardInterrupt`` raised inside the pip install is swallowed
    and recorded in ``ensure_imports._cancelled`` so callers (typically
    ``ensure_imports_or_exit``) can treat the cancel as exit 0 instead
    of a real failure.
    """
    ensure_imports._cancelled = False  # type: ignore[attr-defined]
    missing = [
        (module, package)
        for module, package in requirements
        if not _importable(module)
    ]

    if not missing:
        return True

    # If the framework venv already has every requested package, just
    # add it to sys.path instead of re-installing.  This saves ~3 s of
    # pip-startup per script invocation in the smoke-test suite.
    from .installer import (
        _resolve_venv_dir,
        _venv_packages,
        add_venv_to_path,
        install_packages,
    )
    venv_dir = _resolve_venv_dir()
    if venv_dir.exists() and all(
        _venv_packages(venv_dir).get(p) for _, p in missing
    ):
        add_venv_to_path(venv_dir)
        # Re-check now that the venv is on sys.path.
        if all(_importable(m) for m, _ in missing):
            return True

    if os.environ.get("DONT_INSTALL_MODULES"):
        print("DONT_INSTALL_MODULES is set - refusing to install missing modules.")
        for module, package in missing:
            print(f"  missing module '{module}' (package '{package}')")
        return False

    packages: List[str] = []
    for module, package in missing:
        if package not in packages:
            packages.append(package)

    try:
        venv_dir = install_packages(packages, quiet=quiet)
    except KeyboardInterrupt:
        ensure_imports._cancelled = True  # type: ignore[attr-defined]
        print("Dependency install cancelled by user", file=sys.stderr)
        return False
    if venv_dir is None and _pip_was_cancelled():
        ensure_imports._cancelled = True  # type: ignore[attr-defined]
    if venv_dir:
        add_venv_to_path(venv_dir)
        if all(_importable(module) for module, _ in missing):
            return True

    return False


def _pip_was_cancelled() -> bool:
    """Did the last ``installer._pip(...)`` call observe a KeyboardInterrupt?"""
    from .installer import _pip
    return getattr(_pip, "_cancelled", False)


def ensure_imports_or_exit(
    requirements: Tuple[Tuple[str, str], ...],
    *,
    quiet: bool = True,
    exit_code: int = 1,
) -> None:
    """Call :func:`ensure_imports` and exit the script when it fails.

    When ``DONT_INSTALL_MODULES`` is set the script exits 0 (graceful
    skip), otherwise it exits with ``exit_code`` so the failure is
    visible to the caller.

    A ``KeyboardInterrupt`` during dependency installation is treated as
    "user cancelled" and also exits 0 so a stray Ctrl-C in CI never
    fails the whole test suite.
    """
    try:
        ok = ensure_imports(requirements, quiet=quiet)
    except KeyboardInterrupt:
        print("Cancelled by user", file=sys.stderr)
        sys.exit(0)
    if ok:
        return
    if os.environ.get("DONT_INSTALL_MODULES"):
        print("Required modules unavailable - skipping because DONT_INSTALL_MODULES is set.")
        sys.exit(0)
    if _was_install_cancelled():
        print("Cancelled by user (deps install was interrupted)", file=sys.stderr)
        sys.exit(0)
    print("Required modules could not be loaded. Cannot continue.")
    sys.exit(exit_code)


def _was_install_cancelled() -> bool:
    """Heuristic: was the most recent install_packages() call aborted by SIGINT?"""
    return getattr(ensure_imports, "_cancelled", False)
