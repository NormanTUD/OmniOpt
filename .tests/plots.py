#!/usr/bin/env python3
"""Plot tests (replaces .tests/plots bash script).

Tests all plot scripts for different jobs by invoking them against
the run directories in .tests/_plot_example_runs.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _framework import helpers
from _framework.helpers import (
    Colors,
    green_text,
    human_readable_time,
    red_text,
    yellow_text,
)


REPO_ROOT = THIS_DIR.parent
PLOT_PREFIX = ".omniopt_plot_"


def _print_progress(msg: str) -> None:
    sys.stdout.write(f"\r\033[K{msg}")
    sys.stdout.flush()


def _list_plot_types() -> List[str]:
    types = []
    for p in REPO_ROOT.glob(f"{PLOT_PREFIX}*.py"):
        name = p.name[: -len(".py")]
        if name.endswith("3d"):
            continue
        # extract type: ".omniopt_plot_<type>.py" -> "<type>"
        suffix = name[len(PLOT_PREFIX):]
        types.append(suffix)
    return types


def _build_commands_for_run(
    run_dir: Path,
    plot_types: List[str],
    quick: bool,
) -> List[str]:
    commands: List[str] = []
    for plot_type in plot_types:
        filename = REPO_ROOT / f"{PLOT_PREFIX}{plot_type}.py"
        if not filename.exists():
            continue
        content = filename.read_text(encoding="utf-8")
        if "add_argument" not in content or "save_to_file" not in content or "useless" in content:
            continue

        # Look for expected files comment: "# EXPECTED FILES: a,b,c"
        expected_match = re.search(r"#\s*EXPECTED FILES:\s*(.+)", content)
        if expected_match:
            expected_files = [f.strip() for f in expected_match.group(1).split(",") if f.strip()]
            missing = [f for f in expected_files if not (run_dir / f).exists()]
            if missing:
                continue  # skip this plot for this run

        this_img = f"{run_dir.parent.name}_{run_dir.name.split('/')[-1]}_image_{plot_type}.svg"
        cmd = (
            f"bash omniopt_plot --run_dir={run_dir} --save_to_file={this_img} "
            f"--plot_type={plot_type}"
        )
        commands.append(cmd)
        if quick:
            continue
        has_min = bool(re.search(r"add\.argument\(.{0,40}--min", content))
        has_max = bool(re.search(r"add\.argument\(.{0,40}--max", content))
        huge = "99999999999999999999999999999999999999999999999999999999999"
        neg_huge = f"-{huge}"
        if has_min and has_max:
            commands.append(f"{cmd} --min={neg_huge} --max={huge}")
        elif has_min:
            commands.append(f"{cmd} --min={neg_huge}")
        elif has_max:
            commands.append(f"{cmd} --max={huge}")
    return commands


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Plot tests", add_help=False)
    parser.add_argument("--quick", action="store_true", help="Only run a single variant per plot.")
    parser.add_argument("--keep_tmp", action="store_true", help="Keep temp files.")
    parser.add_argument("--test_types", dest="test_types", type=str, default="",
                        help="Regex for plot types to test.")
    parser.add_argument("--exit_on_first_error", action="store_true", help="Stop on first error.")
    parser.add_argument("--check_only_changed_since_last_success",
                        action="store_true",
                        help="Skip plots for unchanged files since last tag.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--help", "-h", action="store_true")
    parser.add_argument("--run_with_coverage", action="store_true")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if args.help:
        parser.print_help()
        return 0

    os.environ["install_tests"] = "1"
    os.environ["NO_RUNTIME"] = "1"
    os.environ["NO_NO_RESULT_ERROR"] = "1"
    os.environ["PLOT_TESTS"] = "1"
    os.environ["DONT_SHOW_DONT_INSTALL_MESSAGE"] = "1"

    plot_types = _list_plot_types()
    if args.test_types:
        pattern = re.compile(args.test_types)
        plot_types = [t for t in plot_types if pattern.search(t)]

    if not plot_types:
        red_text(f"No plot scripts found with --test_types={args.test_types}")
        return 1

    projectdir = REPO_ROOT / ".tests" / "_plot_example_runs"
    if not projectdir.is_dir():
        red_text(f"Plot example runs directory not found: {projectdir}")
        return 1

    projects = sorted(p.name for p in projectdir.iterdir() if p.is_dir())
    if not projects:
        red_text("No plot example runs found")
        return 1

    commands: List[str] = []
    for projectname in projects:
        run_dir = projectdir / projectname / "0"
        if not run_dir.is_dir():
            continue
        commands.extend(_build_commands_for_run(run_dir, plot_types, quick=args.quick))

    if not commands:
        yellow_text("No plot commands to run")
        return 0

    errors: List[str] = []
    total = len(commands)
    total_start = time.time()
    for idx, cmd in enumerate(commands, start=1):
        _print_progress(f"[{idx}/{total}] Running: {cmd}")
        start = time.time()
        proc = subprocess.run(cmd, shell=True, cwd=str(REPO_ROOT))
        runtime = time.time() - start
        _print_progress(
            f"[{idx}/{total}] Running: {cmd}  -> exit {proc.returncode}  "
            f"({human_readable_time(int(runtime))})"
        )
        print()
        if proc.returncode != 0:
            errors.append(f"{cmd} exited with {proc.returncode}")
            if args.exit_on_first_error:
                red_text("\n".join(errors))
                return len(errors)

    print()
    total_runtime = int(time.time() - total_start)
    print(f"Plots done in {human_readable_time(total_runtime)}")
    if errors:
        red_text("=> PLOT-ERRORS =>")
        for e in errors:
            red_text(e)
        return len(errors)
    green_text("All plot tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
