#!/usr/bin/env python3
"""Find functions in share_functions.php that are not used anywhere."""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _framework.helpers import (
    green_text,
    human_readable_time,
    red_text,
)


REPO_ROOT = THIS_DIR.parent
GUI_DIR = REPO_ROOT / ".gui"


def main(argv=None) -> int:
    if not (GUI_DIR / "share_functions.php").exists():
        red_text("share_functions.php not found")
        return 1

    start = time.time()
    share_content = (GUI_DIR / "share_functions.php").read_text(encoding="utf-8", errors="ignore")

    funcnames = []
    for line in share_content.splitlines():
        m = re.search(r"function\s+(\w+)", line)
        if m:
            funcnames.append(m.group(1))

    # Concatenate all PHP files (excluding the function definitions themselves).
    php_files = list(GUI_DIR.glob("*.php")) + list(GUI_DIR.glob("**/*.php"))
    all_php = ""
    for php in php_files:
        if php.name == "share_functions.php":
            continue
        try:
            all_php += php.read_text(encoding="utf-8", errors="ignore") + "\n"
        except OSError:
            continue

    errors: list[str] = []
    for fname in funcnames:
        # Search for the function name (excluding function declarations).
        pattern = re.compile(rf"\b{re.escape(fname)}\b")
        count = len(pattern.findall(all_php))
        if count == 0:
            errstr = f"Function {fname} in share_functions.php is never used anywhere."
            red_text(errstr)
            errors.append(errstr)

    elapsed = int(time.time() - start)
    hrs = elapsed // 3600
    mins = (elapsed % 3600) // 60
    secs = elapsed % 60
    print(f"find_unused_share_functions test took: {hrs:02d}:{mins:02d}:{secs:02d}")

    if not errors:
        green_text("No find_unused_share_functions errors")
        return 0
    red_text("=> FIND_UNUSED_SHARE_FUNCTIONS-ERRORS =>")
    for e in errors:
        red_text(e)
    return len(errors)


if __name__ == "__main__":
    sys.exit(main())
