#!/usr/bin/env python3
"""Find functions in omniopt that are unused."""

from __future__ import annotations

import re
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


REPO_ROOT = THIS_DIR.parent

EXCEPTIONS = {
    "_on_sigint", "receive_signal_cont", "convert_to_serializable",
    "custom_warning_handler", "_scan_packages", "receive_usr_signal",
    "live_share_background", "receive_usr_signal_int_or_term",
    "receive_usr_signal_int", "receive_usr_signal_term",
    "execute_nvidia_smi", "parse_choice_param", "parse_range_param",
    "parse_fixed_param", "_finish_previous_jobs_helper_wrapper",
    "force_exit", "_cancel_all_tasks_at_exit",
    "_finish_previous_jobs_helper_check_and_process",
}


def main(argv=None) -> int:
    omniopt = REPO_ROOT / "omniopt"
    if not omniopt.exists():
        print("omniopt not found")
        return 1

    content = omniopt.read_text(encoding="utf-8", errors="ignore")

    funcnames: list[str] = []
    for m in re.finditer(r"^def\s+(\w+)", content, re.MULTILINE):
        fname = m.group(1)
        if fname not in EXCEPTIONS:
            funcnames.append(fname)

    # Build a content view that excludes def lines.
    content_without_defs = re.sub(r"^def\s.*$", "", content, flags=re.MULTILINE)

    unused_count = 0
    for funcname in funcnames:
        pattern = re.compile(rf"{re.escape(funcname)}\(")
        matches = pattern.findall(content_without_defs)
        # Filter out matches that are part of "is_equal" pattern.
        filtered = []
        for line in content_without_defs.splitlines():
            if funcname + "(" in line and "is_equal" not in line:
                filtered.append(line)
        cnt = len(filtered)
        if cnt == 0:
            print(f"{funcname}: {cnt}")
            unused_count += 1

    return unused_count


if __name__ == "__main__":
    sys.exit(main())
