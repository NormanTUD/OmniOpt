#!/usr/bin/env python3
"""Share tests (replaces .tests/share bash script).

Tests the share page for PHP-syntax errors and other simple-to-test stuff.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _framework.helpers import (
    command_exists,
    green_text,
    red_text,
    run,
    yellow_text,
)


REPO_ROOT = THIS_DIR.parent


def normalize_php_output(text: str) -> str:
    text = re.sub(r"<!--\s*[^-]*\s*-->", "", text)
    text = re.sub(r"^\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n", "", text)
    text = re.sub(r"\s\s*", " ", text)
    text = re.sub(r"\s*$", "", text)
    text = re.sub(r"^\s", "", text)
    return text.strip()


def main(argv=None) -> int:
    os.environ["disable_folder_creation"] = "1"

    if not command_exists("php"):
        yellow_text("Cannot run share-test when PHP is not installed!")
        return 255

    gui_dir = REPO_ROOT / ".gui"
    if not (gui_dir / "share_internal.php").exists():
        red_text(f"share_internal.php not found in {gui_dir}")
        return 1

    errors: list[str] = []

    os.environ["share_path"] = "_share_test_case"
    os.environ["user_id"] = "test_user"
    os.environ["experiment_name"] = "ClusteredStatisticalTestDriftDetectionMethod_NOAAWeather"
    os.environ["run_nr"] = "0"

    try:
        proc = subprocess.run(
            ["php", "share_internal.php"],
            cwd=str(gui_dir),
            capture_output=True,
            text=True,
            env=os.environ,
        )
    except Exception as exc:
        red_text(f"Failed to run share_internal.php: {exc}")
        return 1

    got = normalize_php_output((proc.stdout or "") + (proc.stderr or ""))
    if proc.returncode != 0:
        errors.append(f"php share_internal.php exited with {proc.returncode}")
        errors.append(f"  stdout: {proc.stdout[:200]!r}")
        errors.append(f"  stderr: {proc.stderr[:200]!r}")
    elif "PHP Parse error" in got or "PHP Fatal error" in got:
        errors.append(
            "share_internal.php raised a PHP parse/fatal error.\n"
            f"  Got: {got[:200]!r}\n"
        )
    else:
        green_text("share_internal.php ran cleanly (exit 0, no PHP errors)")

    if not errors:
        green_text("No errors")
        return 0

    red_text("=> SHARE-ERRORS =>")
    for err in errors:
        red_text(err)
    return len(errors)


if __name__ == "__main__":
    sys.exit(main())
