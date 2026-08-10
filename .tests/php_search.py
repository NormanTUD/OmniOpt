#!/usr/bin/env python3
"""Test if the search.php compiles and delivers proper results."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _framework.helpers import (
    green_text,
    in_container,
    red_text,
    yellow_text,
)


REPO_ROOT = THIS_DIR.parent
GUI_DIR = REPO_ROOT / ".gui"


def main(argv=None) -> int:
    if in_container():
        green_text("Not running this test in Docker")
        return 0

    if not shutil.which("php"):
        yellow_text("Cannot run php_search-test when PHP is not installed!")
        return 255

    errors: list[str] = []

    env = os.environ.copy()
    env.pop("regex", None)
    proc = subprocess.run(
        ["php", "search.php"],
        cwd=str(GUI_DIR),
        capture_output=True,
        text=True,
        env=env,
    )
    php_no_regex = (proc.stdout or "").strip()
    expected_no_regex = '{"error":"No \'regex\' parameter given for search"}'

    if php_no_regex == expected_no_regex:
        green_text("php .gui/search.php without regex succeeded")
    else:
        msg = (
            "php .gui/search 'No regex given' failed. "
            f"Got: >{php_no_regex}<, expected: {expected_no_regex}"
        )
        red_text(msg)
        errors.append(msg)

    env2 = os.environ.copy()
    env2["regex"] = "IWILLNEVERFINDANYTHINGHOPEFULLY"
    proc2 = subprocess.run(
        ["php", "search.php"],
        cwd=str(GUI_DIR),
        capture_output=True,
        text=True,
        env=env2,
    )
    php_no_result = (proc2.stdout or "").strip()
    expected_no_result = "[]"

    if php_no_result == expected_no_result:
        green_text("php .gui/search.php no-result succeeded")
    else:
        msg = (
            "php .gui/search 'no result' failed. "
            f"Got: >{php_no_result}<, expected: {expected_no_result}"
        )
        red_text(msg)
        errors.append(msg)

    # Simple regex test with jq.
    jq = shutil.which("jq")
    if not jq:
        jq_path = REPO_ROOT / ".tools" / f"jq_{os.uname().machine}"
        if jq_path.exists():
            jq = str(jq_path)
        else:
            red_text(f"Neither jq installed nor {jq_path} found; skipping simple regex test")

    if jq:
        env3 = os.environ.copy()
        env3["regex"] = "a"
        proc3 = subprocess.run(
            ["php", "search.php"],
            cwd=str(GUI_DIR),
            capture_output=True,
            text=True,
            env=env3,
        )
        jq_proc = subprocess.run(
            [jq],
            input=proc3.stdout or "",
            capture_output=True,
            text=True,
        )
        if jq_proc.returncode == 0:
            green_text("php .gui/search.php simple regex succeeded")
        else:
            msg = "php .gui/search 'simple regex' failed."
            red_text(msg)
            errors.append(msg)

    if errors:
        red_text("=> ERRORS =>")
        for e in errors:
            red_text(e)
            print()
        return len(errors)
    green_text("No errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
