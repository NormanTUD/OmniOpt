#!/usr/bin/env python3
"""Main test orchestrator (replaces .tests/main bash script).

Usage:
    python3 .tests/main.py [options]

All options from the original bash script remain supported, plus an abstract
tag/filter system for combining test suites:

    --only=suite:basic,quick:false   run tests with ALL of these tags
    --exclude=suite:docker            skip tests with ANY of these tags
    --tags=suite:linter               run tests with at least ONE of these tags
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _framework import helpers
from _framework.installer import ensure_dependencies
from _framework.cli import (
    adjust_quick_flags,
    build_base_parser,
    collect_cli_overrides,
    cmd_omniopt_call,
    env_skip,
    num_gpus,
    parse_args,
    quick_skip,
    render_command,
)
from _framework.config import load_config
from _framework.runner import (
    get_state,
    print_table,
    print_table_markdown,
    reset_state,
    run_test_objects,
)


HELP_EPILOG = [
    "Examples:",
    "  python3 .tests/main.py                            # run everything",
    "  python3 .tests/main.py --quick                    # only quick tests",
    "  python3 .tests/main.py --no_plots --no_linter     # skip plot & linter tests",
    "  python3 .tests/main.py --only=suite:basic         # only the basic tests",
    "  python3 .tests/main.py --exclude=suite:docker     # skip docker tests",
    "  python3 .tests/main.py --tags=suite:linter        # only linter tests",
    "  python3 .tests/main.py --list                     # list all matching tests",
]


def _select_tests(config, args):
    only = [t.strip() for t in (args.only or "").split(",") if t.strip()] or None
    exclude = [t.strip() for t in (args.exclude or "").split(",") if t.strip()] or None
    any_tag = [t.strip() for t in (args.tags or "").split(",") if t.strip()] or None

    selected = config.filter(only=only, exclude=exclude, any_tag=any_tag)

    if args.no_plots:
        selected = [t for t in selected if "suite:plot" not in t.tags]
    if args.no_linter:
        selected = [t for t in selected if "suite:linter" not in t.tags]
    if args.no_linkchecker:
        selected = [t for t in selected if "suite:link" not in t.tags]
    return selected


def _print_test_list(tests) -> None:
    if not tests:
        print("(no tests selected)")
        return
    longest_id = max(len(t.id) for t in tests)
    for t in tests:
        wanted = [t.wanted_exit_code]
        if t.alternative_exit_code is not None:
            wanted.append(t.alternative_exit_code)
        wanted_str = "/".join(str(c) for c in wanted)
        tags_str = ",".join(t.tags)
        print(f"  {t.id.ljust(longest_id)}  tags=[{tags_str}]  wanted={wanted_str}")
    print(f"\n{len(tests)} test(s) selected.")


def main(argv=None) -> int:
    os.environ.setdefault("ENABLE_BEARTYPE", "1")
    os.environ.setdefault("install_tests", "1")
    os.environ.setdefault("OO_MAIN_TESTS", "1")
    os.environ.setdefault("DONT_SHOW_DONT_INSTALL_MESSAGE", "1")
    os.environ.setdefault("DISABLE_SIXEL_GRAPHICS", "1")

    helpers.ensure_install_tests_env()
    ensure_dependencies(include_tests=True)
    config = load_config()

    parser = build_base_parser(
        config,
        description="Main test orchestrator for OmniOpt.",
    )
    args = parse_args(parser, argv if argv is not None else sys.argv[1:])
    args = adjust_quick_flags(args)

    # Environment-specific toggles.
    in_container = helpers.in_container()
    is_ci = helpers.is_ci()
    has_nvidia = helpers.command_exists("nvidia-smi")
    has_sbatch = helpers.command_exists("sbatch")
    has_docker = helpers.command_exists("docker")

    NUM_GPUS = num_gpus(args, has_nvidia)
    if not args.NUM_GPUS:
        args.NUM_GPUS = str(NUM_GPUS)
    if not has_sbatch and not args.max_eval:
        args.max_eval = "2"
        args.num_parallel_jobs = "1"
        args.num_random_steps = "1"
    if not has_sbatch and not args.num_parallel_jobs:
        args.num_parallel_jobs = "1"

    selected = _select_tests(config, args)

    if args.list_only:
        _print_test_list(selected)
        return 0

    if not selected:
        print("No tests selected.")
        return 0

    print(f"Python-Version:")
    import subprocess
    print(subprocess.check_output([sys.executable, "--version"], text=True).strip())
    print(f"Number of tests selected: {len(selected)}")

    reset_state()
    state = get_state()
    state.exit_on_first_error = bool(args.exit_on_first_error)
    state.skip_first_n = int(args.skip_first_n_tests or 0)

    start = time.time()

    def render(test):
        return render_command(test.cmd or "", config, args)

    run_test_objects(
        selected,
        quick_pred=lambda t: quick_skip(t, args),
        env_pred=lambda t: env_skip(
            t, args,
            in_container=in_container, is_ci=is_ci,
            has_nvidia=has_nvidia, has_sbatch=has_sbatch, has_docker=has_docker,
        ),
        render_command=render,
        dry_run=bool(args.dry_run),
    )

    elapsed = int(time.time() - start)
    print(f"\nTest took {helpers.displaytime(elapsed)}")

    print_table()
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(print_table_markdown() + "\n")

    return len(state.errors)


if __name__ == "__main__":
    for line in HELP_EPILOG:
        print(line)
    sys.exit(main())
