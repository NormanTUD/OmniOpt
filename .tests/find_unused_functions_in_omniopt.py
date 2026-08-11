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

    # Count matches per function name with a single combined regex.
    # The original code scanned the full file once per function (~544
    # passes); this does it in one pass.
    if funcnames:
        combined = re.compile(
            r"\b(" + "|".join(re.escape(n) for n in funcnames) + r")\("
        )
        hits: dict[str, int] = {n: 0 for n in funcnames}
        for m in combined.finditer(content_without_defs):
            hits[m.group(1)] += 1
    else:
        hits = {}

    unused_count = 0
    for fname in funcnames:
        cnt = hits.get(fname, 0)
        if cnt == 0:
            print(f"{fname}: {cnt}")
            unused_count += 1

    return unused_count


if __name__ == "__main__":
    sys.exit(main())
