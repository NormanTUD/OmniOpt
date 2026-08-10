"""Common CLI parser for all entry points (main, plot, share, smoke, linter).

Each entry point constructs a parser from the YAML config's parameters +
the script-local options. All existing bash flags remain supported.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from . import helpers
from .config import (
    Parameter,
    TestConfig,
    list_cli_options,
    load_config,
    substitute,
)


def build_base_parser(
    config: TestConfig,
    *,
    with_int_options: bool = True,
    with_quick: bool = True,
    with_run_options: bool = True,
    description: str = "",
) -> argparse.ArgumentParser:
    """Build the common parser shared by every test entry point."""
    parser = argparse.ArgumentParser(description=description, add_help=False)

    _TYPE_MAP = {"int": int, "float": float, "str": str, "bool": bool}
    g_int = parser.add_argument_group("integer options")
    for p in list_cli_options(config):
        if p.type == "int":
            g_int.add_argument(p.cli, dest=p.name, type=int, default=None,
                               help=p.description)
    # model_name is used by some tests but is not a parameter in the YAML
    g_int.add_argument("--model_name", dest="model_name", type=str, default="BOTORCH_MODULAR",
                       help="Model name (used by all_float test).")

    g_basic = parser.add_argument_group("basic options")
    g_basic.add_argument("--help", "-h", action="store_true", help="Show this help.")
    g_basic.add_argument("--debug", action="store_true", help="Enable debug mode.")
    g_basic.add_argument("--run_with_coverage", action="store_true",
                         help="Use coverage run -p instead of python3.")
    g_basic.add_argument("--exit_on_first_error", action="store_true",
                         help="Exit on first error.")
    if with_run_options:
        g_basic.add_argument("--skip_search", action="store_true", help="Skip search.")
        g_basic.add_argument("--skip_worker_check", action="store_true",
                             help="Skip worker check.")
        g_basic.add_argument("--skip_test_job_nr", action="store_true",
                             help="Skip job-nr tests.")
        g_basic.add_argument("--skip_first_n_tests", dest="skip_first_n_tests", type=int,
                             default=0, help="Skip the first N tests.")
        g_basic.add_argument("--no_plots", action="store_true", help="Disable plot tests.")
        g_basic.add_argument("--no_linter", action="store_true", help="Disable linter.")
        g_basic.add_argument("--no_linkchecker", action="store_true",
                             help="Disable linkchecker.")
    if with_quick:
        g_basic.add_argument("--quick", action="store_true", help="Only run quick tests.")
        g_basic.add_argument("--reallyquick", action="store_true",
                             help="Only run really-quick tests.")
        g_basic.add_argument("--superquick", action="store_true",
                             help="Only run super-quick tests.")

    g_abs = parser.add_argument_group("abstract test selection")
    g_abs.add_argument("--only", dest="only", type=str, default=None,
                       help="Only run tests with ALL of these tags (comma-separated, e.g. suite:basic).")
    g_abs.add_argument("--exclude", dest="exclude", type=str, default=None,
                       help="Exclude tests that have ANY of these tags.")
    g_abs.add_argument("--tags", dest="tags", type=str, default=None,
                       help="Only run tests with at least ONE of these tags (OR).")
    g_abs.add_argument("--list", dest="list_only", action="store_true",
                       help="Print all matching tests and exit.")
    g_abs.add_argument("--dry-run", dest="dry_run", action="store_true",
                       help="Print commands without executing.")

    g_env = parser.add_argument_group("environment overrides")
    g_env.add_argument("--OMNIOPT_CALL", dest="OMNIOPT_CALL", default=None,
                       help="Override the omniopt command path.")
    g_env.add_argument("--TESTNAME", dest="TESTNAME", default=None,
                       help="Override the test name.")
    g_env.add_argument("--CONFIG_VARIANT", dest="CONFIG_VARIANT", default=None,
                       help="Override config variant (cpu/gpu).")

    return parser


def parse_args(parser: argparse.ArgumentParser, argv: Sequence[str]) -> argparse.Namespace:
    """Parse argv but handle --help specially (print + sys.exit(0))."""
    if "--help" in argv or "-h" in argv:
        parser.print_help()
        sys.exit(0)
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        # Surface unknown flags explicitly - the bash version exits with 100.
        sys.stderr.write(f"Unknown parameter(s): {' '.join(unknown)}\n")
        sys.exit(100)
    return args


def collect_cli_overrides(args: argparse.Namespace) -> dict:
    """Convert the parsed Namespace into a {cli_name: raw_value} dict."""
    overrides: dict = {}
    for k, v in vars(args).items():
        if v is None:
            continue
        overrides[k] = str(v)
    return overrides


def adjust_quick_flags(args: argparse.Namespace) -> argparse.Namespace:
    """Implement the --quick / --reallyquick / --superquick cascade."""
    if args.superquick:
        args.reallyquick = True
        args.quick = True
    if args.reallyquick:
        args.quick = True
    return args


def quick_skip(test, args) -> bool:
    """Decide whether a test should be skipped due to quick/reallyquick/superquick."""
    tags = set(test.tags)
    if args.superquick:
        if "quick:no" in tags or "quick:false" in tags or "quick:never" in tags:
            return True
    if args.reallyquick and not args.superquick:
        if "quick:false" in tags or "quick:never" in tags:
            return True
    if args.quick and not args.reallyquick:
        if "quick:never" in tags:
            return True
    return False


def env_skip(test, args, *, in_container: bool, is_ci: bool, has_nvidia: bool,
             has_sbatch: bool, has_docker: bool) -> Optional[str]:
    """Decide whether to skip a test based on env tags. Returns skip reason or None."""
    tags = set(test.tags)
    if "env:ci" in tags and not is_ci:
        return "env:ci (not in CI)"
    if "env:local" in tags and (is_ci or in_container):
        return "env:local (CI or container)"
    if "env:gpu" in tags and not has_nvidia:
        return "env:gpu (no GPU)"
    if "env:slurm" in tags and not has_sbatch:
        return "env:slurm (no sbatch)"
    if "env:docker" in tags and (is_ci or in_container or not has_docker):
        return "env:docker (no docker / in container / CI)"
    return None


def render_command(cmd: str, config: TestConfig, args: argparse.Namespace) -> str:
    """Substitute placeholders in a test command using current parameter values."""
    overrides = collect_cli_overrides(args)
    params = config.resolve_parameters(overrides)
    return substitute(cmd, params)


def print_help_extra(extra: List[str]) -> None:
    if extra:
        print("\n".join(extra))


def cmd_omniopt_call(args: argparse.Namespace, config: TestConfig) -> str:
    return args.OMNIOPT_CALL or config.parameters["OMNIOPT_CALL"].default


def num_gpus(args: argparse.Namespace, has_nvidia: bool) -> int:
    """The bash script: NUM_GPUS=1 if nvidia-smi is available."""
    if args.NUM_GPUS is not None:
        return int(args.NUM_GPUS)
    return 1 if has_nvidia else 0
