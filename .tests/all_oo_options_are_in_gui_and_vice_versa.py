#!/usr/bin/env python3
"""Find GUI options that are not in OmniOpt2 or vice versa."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _framework.helpers import command_exists, green_text, red_text


REPO_ROOT = THIS_DIR.parent

GUI_EXCEPTIONS = {
    "constraints", "installation_method", "PSEUDORANDOM", "BOTORCH_MODULAR",
    "SAASBO", "UNIFORM", "version", "SOBOL", "FACTORIAL",
    "EXTERNAL_GENERATOR", "RANDOMFOREST", "BO_MIXED", "TPE",
    "worker_generator_path",
}

HELP_EXCEPTIONS = {
    "raise_in_eval", "memray", "debug_stack_trace_regex", "config_yaml",
    "dump_config", "config_json", "skip_search", "range_max_difference",
    "version", "db_url", "config_toml", "show_generation_and_submission_sixel",
    "run_tests_that_fail_on_taurus", "calculate_pareto_front_of_job", "tests",
    "prettyprint", "just_return_defaults", "share_password", "parameter",
    "continue_previous_job", "beartype", "transforms", "debug_stack_regex",
    "num_cpus_main_job", "orchestrator_file", "disable_previous_job_constraint",
    "ui_url", "load_data_from_existing_jobs", "run_dir",
    "signed_weighted_euclidean_weights", "experiment_constraints", "minkowski_p",
    "max_parallelism", "show_ram_every_n_seconds", "runtime_debug",
    "show_func_name", "worker_generator_path", "help", "install"
}


def main(argv=None) -> int:
    if not command_exists("php"):
        green_text("PHP not installed. Will skip all_oo_options_are_in_gui_and_vice_versa.")
        return 0

    gui_data = REPO_ROOT / ".gui" / "gui_data.js"
    if not gui_data.exists():
        red_text(f"{gui_data} not found")
        return 1

    content = gui_data.read_text(encoding="utf-8", errors="ignore")
    gui_options: set[str] = set()
    for m in re.finditer(r"id:\s*['\"]([A-Za-z0-9_]+)['\"]", content):
        gui_options.add(m.group(1))

    help_php = REPO_ROOT / ".gui" / "_tutorials" / "help.php"
    if not help_php.exists():
        red_text(f"{help_php} not found")
        return 1

    try:
        proc = subprocess.run(
            ["php", str(help_php)],
            cwd=str(REPO_ROOT / ".gui"),
            capture_output=True,
            text=True,
            timeout=30,
        )
        help_output = proc.stdout
    except Exception as exc:
        red_text(f"Failed to run help.php: {exc}")
        return 1

    help_options: set[str] = set()
    for m in re.finditer(r">--([A-Za-z0-9_]+)<", help_output):
        help_options.add(m.group(1))

    errors = 0
    for opt in sorted(gui_options):
        opt = opt.strip("',")
        if opt in help_options or opt in GUI_EXCEPTIONS:
            continue
        red_text(f"GUI option {opt} is not in the --help")
        errors += 1

    for opt in sorted(help_options):
        if opt in gui_options or opt in HELP_EXCEPTIONS:
            continue
        red_text(f"--help option {opt} is not in the GUI")
        errors += 1

    if errors == 0:
        green_text("No errors in .tests/all_oo_options_are_in_gui_and_vice_versa")
        return 0
    red_text(f"{errors} errors")
    return errors


if __name__ == "__main__":
    sys.exit(main())
