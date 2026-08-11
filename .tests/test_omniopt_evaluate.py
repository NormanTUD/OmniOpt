#!/usr/bin/env python3
"""Tests for omniopt_evaluate (Python rewrite).

Tightly coupled tests for the *data* portions of the script (wallclock
time, failed-job counter, experiment-name extraction, argument parsing,
textual-prompt emulation). The actual interactive menu is driven by a
``prompt`` callable that can be replaced in tests, so the menu logic
itself can be exercised without a real TTY.
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent

sys.path.insert(0, str(REPO_ROOT))
from importlib.machinery import SourceFileLoader  # noqa: E402

oe = SourceFileLoader(
    "omniopt_evaluate", str(REPO_ROOT / "omniopt_evaluate")
).load_module()

from _framework.helpers import red_text  # noqa: E402


def _check(condition: bool, message: str) -> bool:
    if not condition:
        red_text(f"FAIL: {message}")
        return False
    return True


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Pure data-function tests
# ---------------------------------------------------------------------------


def test_extract_experiment_name_basic() -> bool:
    return _check(
        oe.extract_experiment_name("runs/my_experiment/0") == "my_experiment",
        f"got {oe.extract_experiment_name('runs/my_experiment/0')!r}",
    )


def test_extract_experiment_name_with_trailing_slash() -> bool:
    return _check(
        oe.extract_experiment_name("runs/my_experiment/0/") == "my_experiment",
        f"got {oe.extract_experiment_name('runs/my_experiment/0/')!r}",
    )


def test_extract_experiment_name_absolute() -> bool:
    return _check(
        oe.extract_experiment_name("/tmp/runs/foo/3") == "foo",
        f"got {oe.extract_experiment_name('/tmp/runs/foo/3')!r}",
    )


def test_calculate_wallclock_time_simple() -> bool:
    """Reads start_time,end_time from 0.csv and returns seconds elapsed."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "0.csv"
        _write_csv(
            p,
            ["start_time", "end_time", "exit_code"],
            [
                ["1000.0", "1005.5", "0"],
                ["1010.0", "1020.0", "0"],
            ],
        )
        secs = oe.calculate_wallclock_time(str(p))
    return _check(secs == 20, f"expected 20 seconds, got {secs}")


def test_calculate_wallclock_time_zero_when_no_rows() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "0.csv"
        _write_csv(p, ["start_time", "end_time", "exit_code"], [])
        secs = oe.calculate_wallclock_time(str(p))
    return _check(secs == 0, f"expected 0, got {secs}")


def test_count_failed_jobs_none_failed() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "0.csv"
        _write_csv(
            p,
            ["start_time", "end_time", "exit_code"],
            [["1", "2", "0"], ["1", "2", "0"]],
        )
        n = oe.count_failed_jobs(str(p))
    return _check(n == 0, f"expected 0, got {n}")


def test_count_failed_jobs_some_failed() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "0.csv"
        _write_csv(
            p,
            ["start_time", "end_time", "exit_code"],
            [
                ["1", "2", "0"],
                ["1", "2", "1"],
                ["1", "2", "0"],
                ["1", "2", "5"],
            ],
        )
        n = oe.count_failed_jobs(str(p))
    return _check(n == 2, f"expected 2, got {n}")


def test_count_failed_jobs_missing_file_returns_zero() -> bool:
    return _check(
        oe.count_failed_jobs("/nonexistent/0.csv") == 0,
        "missing file should return 0",
    )


def test_format_duration_human_readable() -> bool:
    s = oe.format_duration(3725)
    return _check(
        "1" in s and "2" in s and "5" in s,
        f"expected hours/minutes/seconds in {s!r}",
    )


def test_format_duration_zero() -> bool:
    s = oe.format_duration(0)
    return _check("0" in s, f"expected '0' in {s!r}")


# ---------------------------------------------------------------------------
# Argument parsing tests
# ---------------------------------------------------------------------------


def test_parse_args_defaults() -> bool:
    args = oe.parse_args([])
    return _check(
        args.projectdir == "runs" and args.debug is False,
        f"defaults wrong: {args}",
    )


def test_parse_args_help() -> bool:
    args = oe.parse_args(["--help"])
    return _check(args.help is True, f"expected help=True, got {args!r}")


def test_parse_args_debug() -> bool:
    args = oe.parse_args(["--debug"])
    return _check(args.debug is True, f"expected debug=True, got {args!r}")


def test_parse_args_projectdir() -> bool:
    args = oe.parse_args(["--projectdir=/data/runs"])
    return _check(
        args.projectdir == "/data/runs",
        f"got {args.projectdir!r}",
    )


def test_parse_args_unknown_flag() -> bool:
    rc = oe.parse_args_with_exit_code(["--does-not-exist"])
    return _check(
        rc in (1, 2),
        f"unknown flag must exit non-zero, got {rc}",
    )


# ---------------------------------------------------------------------------
# Textual-prompt tests (no TTY required)
# ---------------------------------------------------------------------------


def test_textual_menu_returns_keyword() -> bool:
    """When the user types a valid key, the menu returns it."""
    inp = iter(["a"])
    out: list[str] = []
    choice = oe.textual_menu(
        title="Pick",
        options=[("a", "option a"), ("b", "option b")],
        input_fn=lambda prompt: next(inp),
        output_fn=out.append,
    )
    ok = _check(choice == "a", f"expected 'a', got {choice!r}")
    ok &= _check(any("option a" in s for s in out), f"menu should list option a: {out}")
    return ok


def test_textual_menu_returns_number() -> bool:
    inp = iter(["2"])
    out: list[str] = []
    choice = oe.textual_menu(
        title="Pick",
        options=[("a", "option a"), ("b", "option b")],
        input_fn=lambda prompt: next(inp),
        output_fn=out.append,
    )
    return _check(choice == "b", f"expected 'b', got {choice!r}")


def test_textual_menu_reprompts_on_invalid() -> bool:
    inp = iter(["99", "x", "1"])
    out: list[str] = []
    choice = oe.textual_menu(
        title="Pick",
        options=[("a", "option a")],
        input_fn=lambda prompt: next(inp),
        output_fn=out.append,
    )
    return _check(choice == "a", f"expected 'a' after re-prompting, got {choice!r}")


def test_textual_input_returns_user_value() -> bool:
    inp = iter(["hello"])
    out: list[str] = []
    val = oe.textual_input(
        "Enter name", default="world",
        input_fn=lambda prompt: next(inp),
        output_fn=out.append,
    )
    return _check(val == "hello", f"expected 'hello', got {val!r}")


def test_textual_input_returns_default_on_empty() -> bool:
    inp = iter([""])
    val = oe.textual_input(
        "Enter name", default="default-value",
        input_fn=lambda prompt: next(inp),
        output_fn=lambda s: None,
    )
    return _check(val == "default-value", f"expected default, got {val!r}")


# ---------------------------------------------------------------------------
# Project listing tests
# ---------------------------------------------------------------------------


def test_find_projects_lists_valid_ones() -> bool:
    """A project is a directory whose name contains a subdirectory
    with a results.csv file (matches the bash script's logic)."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "proj1" / "0").mkdir(parents=True)
        (root / "proj1" / "0" / "results.csv").write_text("a,b\n1,2\n")
        (root / "proj2" / "0").mkdir(parents=True)
        (root / "proj2" / "0" / "results.csv").write_text("a,b\n3,4\n")
        (root / "proj_without_csv" / "0").mkdir(parents=True)
        projs = oe.find_projects(str(root))
    ok = _check("proj1" in projs, f"proj1 missing: {projs}")
    ok &= _check("proj2" in projs, f"proj2 missing: {projs}")
    ok &= _check("proj_without_csv" not in projs, f"proj_without_csv should not be listed: {projs}")
    return ok


def test_find_projects_handles_missing_dir() -> bool:
    projs = oe.find_projects("/nonexistent/that/does/not/exist")
    return _check(projs == [], f"expected empty list, got {projs!r}")


def test_find_run_numbers_for_project() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "proj" / "0"
        root.mkdir(parents=True)
        (root / "results.csv").write_text("a\n1\n")
        # Add another run number
        (Path(tmp) / "proj" / "1").mkdir()
        (Path(tmp) / "proj" / "1" / "results.csv").write_text("a\n2\n")
        # And one without results.csv - should not be listed
        (Path(tmp) / "proj" / "2").mkdir()
        nums = oe.find_run_numbers(str(Path(tmp) / "proj"))
    ok = _check("0" in nums, f"run 0 missing: {nums}")
    ok &= _check("1" in nums, f"run 1 missing: {nums}")
    ok &= _check("2" not in nums, f"run 2 should not be listed: {nums}")
    return ok


TESTS = [
    test_extract_experiment_name_basic,
    test_extract_experiment_name_with_trailing_slash,
    test_extract_experiment_name_absolute,
    test_calculate_wallclock_time_simple,
    test_calculate_wallclock_time_zero_when_no_rows,
    test_count_failed_jobs_none_failed,
    test_count_failed_jobs_some_failed,
    test_count_failed_jobs_missing_file_returns_zero,
    test_format_duration_human_readable,
    test_format_duration_zero,
    test_parse_args_defaults,
    test_parse_args_help,
    test_parse_args_debug,
    test_parse_args_projectdir,
    test_parse_args_unknown_flag,
    test_textual_menu_returns_keyword,
    test_textual_menu_returns_number,
    test_textual_menu_reprompts_on_invalid,
    test_textual_input_returns_user_value,
    test_textual_input_returns_default_on_empty,
    test_find_projects_lists_valid_ones,
    test_find_projects_handles_missing_dir,
    test_find_run_numbers_for_project,
]


def main(argv=None) -> int:
    failures = 0
    for t in TESTS:
        print(f"running {t.__name__} ...", end=" ", flush=True)
        try:
            ok = t()
        except Exception as e:
            ok = False
            red_text(f"\n  EXCEPTION: {e!r}")
        if ok:
            print("ok")
        else:
            failures += 1
            print("FAIL")
    if failures:
        red_text(f"\n{failures} test(s) failed")
        return 1
    print("\nAll omniopt_evaluate tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
