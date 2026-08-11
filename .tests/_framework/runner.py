"""Test runner: executes individual test definitions and tracks results."""

from __future__ import annotations

import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from . import helpers
from .helpers import (
    Colors,
    green_bold_underline,
    green_text,
    human_readable_time,
    red_text,
    run,
    timestamp,
    yellow_text,
)


@dataclass
class TestResult:
    name: str
    command: str
    exit_code: int
    wanted_exit_codes: List[int]
    runtime: float
    failed: bool
    success_mark: str
    skipped: bool = False
    skip_reason: str = ""

    @property
    def wanted_str(self) -> str:
        return "/".join(str(c) for c in self.wanted_exit_codes)


@dataclass
class RunnerState:
    results: List[TestResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    skip_first_n: int = 0
    test_counter: int = 0
    exit_on_first_error: bool = False
    skip_search: bool = False
    interrupted: bool = False


_RUNNER_STATE = RunnerState()


def get_state() -> RunnerState:
    return _RUNNER_STATE


def reset_state() -> None:
    global _RUNNER_STATE
    _RUNNER_STATE = RunnerState()


def runtimes_summary(runtimes: List[float]) -> Optional[int]:
    if not runtimes:
        return None
    sorted_runtimes = sorted(runtimes)
    return sorted_runtimes[len(sorted_runtimes) // 2]


def _format_wanted(wanted: List[int]) -> str:
    return "/".join(str(c) for c in wanted)


def run_command(
    name: str,
    command: str,
    wanted_exit_code: int,
    alternative_exit_code: Optional[int] = None,
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
    timeout: Optional[float] = None,
) -> TestResult:
    """Execute a single shell command and return the result."""
    state = get_state()

    if state.skip_first_n and state.test_counter < state.skip_first_n:
        yellow_text(
            f"Skipping test {state.test_counter}, will start at {state.skip_first_n}"
        )
        state.test_counter += 1
        wanted = [wanted_exit_code]
        if alternative_exit_code is not None:
            wanted.append(alternative_exit_code)
        return TestResult(
            name=name,
            command=command,
            exit_code=wanted_exit_code,
            wanted_exit_codes=wanted,
            runtime=0.0,
            failed=False,
            success_mark="SKIP",
            skipped=True,
            skip_reason=f"skip_first_n_tests={state.skip_first_n}",
        )

    if state.exit_on_first_error and state.errors:
        state.test_counter += 1
        wanted = [wanted_exit_code]
        if alternative_exit_code is not None:
            wanted.append(alternative_exit_code)
        return TestResult(
            name=name,
            command=command,
            exit_code=0,
            wanted_exit_codes=wanted,
            runtime=0.0,
            failed=False,
            success_mark="SKIP",
            skipped=True,
            skip_reason="exit_on_first_error",
        )

    if "," in name:
        red_text(
            f"The name '{name}' contains a comma, which will confuse the table generation. "
            "Please remove the comma from the name."
        )
        sys.exit(255)

    start = time.time()
    exit_code = 0
    interrupted = False
    try:
        proc = run(command, cwd=cwd, env=env, timeout=timeout, live=True)
        exit_code = proc.returncode
    except KeyboardInterrupt:
        red_text("\nInterrupted by Ctrl+C. Stopping test run.")
        state.interrupted = True
        exit_code = 130
        interrupted = True
    except Exception as exc:  # pragma: no cover - defensive
        red_text(f"Test command crashed: {exc}")
        traceback.print_exc()
        exit_code = 1
    end = time.time()
    runtime = end - start
    state.test_counter += 1

    wanted = [wanted_exit_code]
    if alternative_exit_code is not None:
        wanted.append(alternative_exit_code)

    if interrupted:
        result = TestResult(
            name=name,
            command=command,
            exit_code=exit_code,
            wanted_exit_codes=wanted,
            runtime=runtime,
            failed=True,
            success_mark="\u2717",
        )
        state.results.append(result)
        print(f"Test took {human_readable_time(runtime)}")
        return result

    failed = exit_code not in wanted
    success_mark = "\u2717" if failed else "\u2713"

    if failed:
        if alternative_exit_code is None:
            err = (
                f"{name} exited with {exit_code} (wanted {wanted_exit_code}). "
                f"Command: {command}"
            )
        else:
            err = (
                f"{name} exited with {exit_code} (wanted {wanted_exit_code} or "
                f"{alternative_exit_code}). Command: {command}"
            )
        red_text(err + "\n")
        state.errors.append(err)

    print(f"Test took {human_readable_time(runtime)}")

    result = TestResult(
        name=name,
        command=command,
        exit_code=exit_code,
        wanted_exit_codes=wanted,
        runtime=runtime,
        failed=failed,
        success_mark=success_mark,
    )
    state.results.append(result)
    return result


def run_python_function(
    name: str,
    func: Callable[[], int],
    wanted_exit_code: int,
    alternative_exit_code: Optional[int] = None,
) -> TestResult:
    """Execute a Python test function directly (no shell)."""
    state = get_state()

    if state.skip_first_n and state.test_counter < state.skip_first_n:
        yellow_text(
            f"Skipping test {state.test_counter}, will start at {state.skip_first_n}"
        )
        state.test_counter += 1
        wanted = [wanted_exit_code]
        if alternative_exit_code is not None:
            wanted.append(alternative_exit_code)
        return TestResult(
            name=name,
            command=getattr(func, "__name__", repr(func)),
            exit_code=wanted_exit_code,
            wanted_exit_codes=wanted,
            runtime=0.0,
            failed=False,
            success_mark="SKIP",
            skipped=True,
            skip_reason=f"skip_first_n_tests={state.skip_first_n}",
        )

    if state.exit_on_first_error and state.errors:
        state.test_counter += 1
        wanted = [wanted_exit_code]
        if alternative_exit_code is not None:
            wanted.append(alternative_exit_code)
        return TestResult(
            name=name,
            command=getattr(func, "__name__", repr(func)),
            exit_code=0,
            wanted_exit_codes=wanted,
            runtime=0.0,
            failed=False,
            success_mark="SKIP",
            skipped=True,
            skip_reason="exit_on_first_error",
        )

    start = time.time()
    exit_code = 0
    try:
        exit_code = int(func() or 0)
    except KeyboardInterrupt:
        red_text("\nInterrupted by Ctrl+C. Stopping test run.")
        state.interrupted = True
        exit_code = 130
    except SystemExit as exc:
        code = exc.code
        exit_code = 0 if code is None else int(code)
    except Exception:
        traceback.print_exc()
        exit_code = 1
    end = time.time()
    runtime = end - start
    state.test_counter += 1

    wanted = [wanted_exit_code]
    if alternative_exit_code is not None:
        wanted.append(alternative_exit_code)

    failed = exit_code not in wanted
    success_mark = "\u2717" if failed else "\u2713"

    if failed:
        if alternative_exit_code is None:
            err = f"{name} exited with {exit_code} (wanted {wanted_exit_code})."
        else:
            err = (
                f"{name} exited with {exit_code} (wanted {wanted_exit_code} or "
                f"{alternative_exit_code})."
            )
        red_text(err + "\n")
        state.errors.append(err)

    print(f"Test took {human_readable_time(runtime)}")

    result = TestResult(
        name=name,
        command=getattr(func, "__name__", repr(func)),
        exit_code=exit_code,
        wanted_exit_codes=wanted,
        runtime=runtime,
        failed=failed,
        success_mark=success_mark,
    )
    state.results.append(result)
    return result


def print_table_markdown(results: Optional[List[TestResult]] = None) -> str:
    if results is None:
        results = get_state().results
    headers = ["Failed", "Name", "TestRunTime", "ExitCode", "WantedExitCodes", "Success"]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in results:
        row = [
            "1" if r.failed else "0",
            r.name,
            human_readable_time(r.runtime),
            str(r.exit_code),
            r.wanted_str,
            r.success_mark,
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def print_table(results: Optional[List[TestResult]] = None) -> None:
    if results is None:
        results = get_state().results
    headers = ["Failed", "Name", "TestRunTime", "ExitCode", "WantedExitCodes", "Success"]
    if not results:
        print("(no tests)")
        return

    max_lengths = {h: len(h) for h in headers}
    for r in results:
        values = [
            "1" if r.failed else "0",
            r.name,
            human_readable_time(r.runtime),
            str(r.exit_code),
            r.wanted_str,
            r.success_mark,
        ]
        for h, v in zip(headers, values):
            max_lengths[h] = max(max_lengths[h], len(v))

    def make_row(values: List[str], color: Optional[str] = None) -> str:
        line = ""
        for h, v in zip(headers, values):
            pad = max_lengths[h] - len(v) + 1
            line += f"| {v}{' ' * pad}"
        line += "|"
        if color and not helpers._NO_COLOR:
            return f"{color}{line}{Colors.RESET}"
        return line

    sep = "+" + "+".join("-" * (max_lengths[h] + 3) for h in headers) + "+"
    print(sep)
    print(make_row(headers))
    print(sep)
    for r in results:
        values = [
            "1" if r.failed else "0",
            r.name,
            human_readable_time(r.runtime),
            str(r.exit_code),
            r.wanted_str,
            r.success_mark,
        ]
        color = Colors.RED if r.failed else None
        print(make_row(values, color=color))
    print(sep)


def progress_line(current: int, total: int, msg: str) -> None:
    pct = int(current * 100 / total) if total else 0
    line = f"[{current}/{total}] ({pct}%) Running: {msg}".rstrip()
    green_bold_underline(line)


def _record_skip(test, reason: str) -> TestResult:
    state = get_state()
    state.test_counter += 1
    return TestResult(
        name=test.name,
        command=test.cmd or test.python_check or "",
        exit_code=0,
        wanted_exit_codes=[test.wanted_exit_code]
        + ([test.alternative_exit_code] if test.alternative_exit_code is not None else []),
        runtime=0.0,
        failed=False,
        success_mark="SKIP",
        skipped=True,
        skip_reason=reason,
    )


def run_test_objects(
    tests,
    *,
    quick_pred=None,
    env_pred=None,
    render_command=None,
    python_resolver=None,
    dry_run: bool = False,
) -> None:
    """Execute a list of Test objects sequentially with progress reporting."""
    state = get_state()
    total = len(tests)
    runtimes_so_far: List[float] = []

    for idx, test in enumerate(tests, start=1):
        if state.exit_on_first_error and state.errors:
            result = _record_skip(test, "exit_on_first_error")
            state.results.append(result)
            continue

        if quick_pred is not None and quick_pred(test):
            result = _record_skip(test, "quick mode")
            state.results.append(result)
            continue
        if env_pred is not None:
            reason = env_pred(test)
            if reason is not None:
                result = _record_skip(test, reason)
                state.results.append(result)
                continue

        cmd = render_command(test) if render_command else (test.cmd or "")
        progress_line(idx, total, cmd or test.id)

        if dry_run:
            state.test_counter += 1
            print(f"[dry-run] would run: {cmd}")
            continue

        start = time.time()
        if test.python_check and python_resolver is not None:
            func = python_resolver(test.python_check)
            result = run_python_function(test.name, func, test.wanted_exit_code,
                                         test.alternative_exit_code)
        else:
            result = run_command(test.name, cmd, test.wanted_exit_code,
                                 test.alternative_exit_code)
        end = time.time()
        duration = end - start
        if duration > 10:
            runtimes_so_far.append(duration)

        if state.interrupted:
            break

        median = runtimes_summary(runtimes_so_far)
        if median is not None:
            remaining = max(0, int(median * (total - idx)))
            if remaining > 0:
                if remaining >= 3600:
                    readable = f"{remaining // 3600}h {(remaining % 3600) // 60}m"
                elif remaining >= 60:
                    readable = f"{remaining // 60}m {remaining % 60}s"
                else:
                    readable = f"{remaining}s"
                green_bold_underline(f"-> Estimated time remaining: {readable}")
