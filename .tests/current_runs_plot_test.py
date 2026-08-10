#!/usr/bin/env python3
"""Plots all current projects in runs/ and tests if the plot script succeeded."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _framework.helpers import red_text, yellow_text


REPO_ROOT = THIS_DIR.parent


def main(argv=None) -> int:
    os.environ.setdefault("NO_WHIPTAIL", "1")
    os.environ.setdefault("NO_RUNTIME", "1")

    runs_dir = REPO_ROOT / "runs"
    if not runs_dir.is_dir():
        red_text("runs is not a directory. Cannot continue")
        return 1

    plot_types: list[str] = []
    for p in REPO_ROOT.glob(".omniopt_plot_*.py"):
        name = p.name[: -len(".py")]
        suffix = name[len(".omniopt_plot_"):]
        if suffix.endswith("3d"):
            continue
        plot_types.append(suffix)

    errors = 0
    for plot_type in plot_types:
        for project in sorted(runs_dir.iterdir()):
            if not project.is_dir():
                continue
            for project_nr in sorted(project.iterdir()):
                if not project_nr.is_dir():
                    continue
                yellow_text(
                    f"./omniopt_plot --run_dir {project_nr} --plot_type={plot_type} --no_plt_show"
                )
                proc = subprocess.run(
                    ["./omniopt_plot", "--run_dir", str(project_nr),
                     "--plot_type", plot_type, "--no_plt_show"],
                    cwd=str(REPO_ROOT),
                )
                if proc.returncode != 0:
                    errors += 1
    return errors


if __name__ == "__main__":
    sys.exit(main())
