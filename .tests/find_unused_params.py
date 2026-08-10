#!/usr/bin/env python3
"""Find parameters that are defined in OmniOpt but never used."""

from __future__ import annotations

import re
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


REPO_ROOT = THIS_DIR.parent
ALLOWED = {
    "version", "num_cpus_main_job", "workdir", "show_ram_every_n_seconds",
    "send_anonymized_usage_stats", "run_mode", "root_venv_dir", "partition",
    "flame_graph", "ui_url", "debug", "dependency",
    "checkout_to_latest_tested_version",
}


def main(argv=None) -> int:
    omniopt = REPO_ROOT / "omniopt"
    if not omniopt.exists():
        print("omniopt not found")
        return 1

    content = omniopt.read_text(encoding="utf-8", errors="ignore")

    args: list[str] = []
    for m in re.finditer(r"\.add_argument\([\"'](--[^\"']+)[\"']", content):
        arg = m.group(1)[2:]
        if re.match(r"^\w*$", arg):
            args.append(arg)

    unused_params = 0
    for arg in args:
        cnt = len(re.findall(rf"args\.{re.escape(arg)}\b", content))
        if cnt == 0 and arg not in ALLOWED:
            print(
                f"Parameter {arg} found in omniopt, but never used anywhere "
                "nor in the special $allowed-array",
                file=sys.stderr,
            )
            unused_params += 1

    print(f"Unused params: {unused_params}")
    return unused_params


if __name__ == "__main__":
    sys.exit(main())
