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
    """Ensure every ``(module_name, package_name)`` is importable.

    Missing packages are installed automatically into the framework venv.
    Returns True when all modules are available afterwards. When
    ``DONT_INSTALL_MODULES`` is set in the environment, installation is
    skipped and False is returned for any missing module.
    """
    missing = [
        (module, package)
        for module, package in requirements
        if not _importable(module)
    ]

    if not missing:
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

    from .installer import add_venv_to_path, install_packages

    venv_dir = install_packages(packages, quiet=quiet)
    if venv_dir:
        add_venv_to_path(venv_dir)
        if all(_importable(module) for module, _ in missing):
            return True

    return False


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
    """
    if ensure_imports(requirements, quiet=quiet):
        return
    if os.environ.get("DONT_INSTALL_MODULES"):
        print("Required modules unavailable - skipping because DONT_INSTALL_MODULES is set.")
        sys.exit(0)
    print("Required modules could not be loaded. Cannot continue.")
    sys.exit(exit_code)
