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
    share_file = GUI_DIR / "share_functions.php"
    if not share_file.exists():
        red_text("share_functions.php not found")
        return 1

    start = time.time()
    content = share_file.read_text(encoding="utf-8", errors="ignore")

    funcnames: list[str] = []
    for line in content.splitlines():
        m = re.search(r"function\s+(\w+)", line)
        if m:
            funcnames.append(m.group(1))

    # Concatenate all PHP files (matching the bash `cat *.php **/*.php`),
    # but skip lines that contain the word "function" (matching
    # `grep -v function`).
    php_files = list(GUI_DIR.glob("*.php")) + list(GUI_DIR.glob("**/*.php"))
    all_php_lines: list[str] = []
    for php in php_files:
        if not php.is_file():
            continue
        try:
            for line in php.read_text(encoding="utf-8", errors="ignore").splitlines():
                if "function" in line:
                    continue
                all_php_lines.append(line)
        except OSError:
            continue

    haystack = "\n".join(all_php_lines)

    # Build ONE combined regex of all function names instead of compiling
    # a new pattern per function.  This is the difference between
    # ~10 s and ~50 ms on the real share_functions.php.
    if funcnames:
        combined = re.compile(
            r"\b(?:" + "|".join(re.escape(n) for n in funcnames) + r")\b"
        )
        found_names = set(combined.findall(haystack))
    else:
        found_names = set()

    errors: list[str] = []
    for fname in funcnames:
        if fname not in found_names:
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
