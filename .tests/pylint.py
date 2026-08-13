#!/usr/bin/env python3
"""Run pylint linter on python files."""

from __future__ import annotations

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
    file_has_changed_since_last_tagged_version,
    green_text,
    human_readable_time,
    red_text,
    yellow_text,
)
from _framework.installer import ensure_dependencies


REPO_ROOT = THIS_DIR.parent
RC_FILE = REPO_ROOT / ".tests" / "pylint.rc"
DEST_RC = Path.home() / ".pylintrc"


def _pylint_bin() -> str | None:
    """Locate the pylint binary (on PATH or inside the framework venv)."""
    bin_ = shutil.which("pylint")
    if bin_:
        return bin_
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        candidate = Path(venv) / "bin" / "pylint"
        if candidate.exists():
            return str(candidate)
    return None


def _stage_pylintrc() -> None:
    """Mirror the bash pylintrc backup dance.

    If ``~/.pylintrc`` exists and differs from the repo's
    ``.tests/pylint.rc``, the user file is moved to the next free
    ``~/.pylintrc_N`` slot and the repo's pylintrc is copied into place.
    If no user file exists, the repo's pylintrc is simply copied.
    """
    if not RC_FILE.exists():
        return

    try:
        same = DEST_RC.exists() and RC_FILE.read_bytes() == DEST_RC.read_bytes()
    except OSError:
        same = False

    if DEST_RC.exists() and not same:
        i = 1
        while (Path.home() / f".pylintrc_{i}").exists():
            i += 1
        backup = Path.home() / f".pylintrc_{i}"
        try:
            DEST_RC.rename(backup)
        except OSError as exc:
            red_text(f"Could not back up {DEST_RC}: {exc}")
            return
    elif not DEST_RC.exists():
        pass
    else:
        return  # identical, nothing to do

    try:
        shutil.copy2(RC_FILE, DEST_RC)
    except OSError as exc:
        red_text(f"Could not install {RC_FILE} as {DEST_RC}: {exc}")


def _pylint_targets() -> list[str]:
    """Discover ``.*.py`` and ``.*/*.py`` files (mirrors the bash glob)."""
    targets = [str(p) for p in REPO_ROOT.glob(".*.py")]
    targets += [str(p) for p in REPO_ROOT.glob(".*/*.py") if p.suffix == ".py"]
    targets = [t for t in targets if not t.endswith("/.helpers.py")]
    return targets


def main(argv=None) -> int:
    ensure_dependencies()
    os.environ.setdefault("install_tests", "1")

    _stage_pylintrc()

    pylint_bin = _pylint_bin()
    rcflag = [f"--rcfile={RC_FILE}"] if RC_FILE.exists() else []
    errors: list[str] = []
    start = time.time()
    try:
        for target in _pylint_targets():
            relpath = str(Path(target).relative_to(REPO_ROOT))
            if not file_has_changed_since_last_tagged_version(
                relpath, cwd=str(REPO_ROOT)
            ):
                continue
            yellow_text(f"pylint {relpath}")
            if pylint_bin is not None:
                cmd = [pylint_bin, *rcflag, target]
            else:
                cmd = [sys.executable, "-m", "pylint", *rcflag, target]
            proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
            if proc.returncode != 0:
                errstr = (
                    f"Failed linting {relpath}: Run "
                    f"'pylint {relpath}' to see details."
                )
                red_text(f"{errstr}\n")
                errors.append(errstr)
    except KeyboardInterrupt:
        return 130

    elapsed = int(time.time() - start)
    print(f"pylint test took: {human_readable_time(elapsed)}")

    if not errors:
        green_text("No pylint errors")
        return 0

    red_text("=> PYLINT-ERRORS => PYLINT-ERRORS => PYLINT-ERRORS =>\n")
    for e in errors:
        red_text(f"{e}\n")
    return len(errors)


if __name__ == "__main__":
    sys.exit(main())
