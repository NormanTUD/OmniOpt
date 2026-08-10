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


_RUNNER_STATE = RunnerState()


def get_state() -> RunnerState:
    return _RUNNER_STATE


def reset_state() -> None:
    global _RUNNER_STATE
    _RUNNER_STATE = RunnerState()


def runtimes_summary(runtimes: List[float]) -> Optional[int]:
    """Return the median of runtimes if there are any."""
    if not runtimes:
        return None
    sorted_runtimes = sorted(runtimes)
    return sorted_runtimes[len(sorted_runtimes) // 2]


def run_test_case(
    name: str,
    command: str,
    wanted_exit_code: int,
    alternative_exit_code: Optional[int] = None,
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
    timeout: Optional[float] = None,
) -> TestResult:
    """Execute a single test command and return the result."""
    state = get_state()

    if state.skip_first_n and state.test_counter < state.skip_first_n:
        yellow_text(
            f"Skipping test {state.test_counter}, will start at {state.skip_first_n}"
        )
        state.test_counter += 1
        return TestResult(
            name=name,
            command=command,
            exit_code=wanted_exit_code,
            wanted_exit_codes=[wanted_exit_code]
            + ([alternative_exit_code] if alternative_exit_code is not None else []),
            runtime=0.0,
            failed=False,
            success_mark="SKIP",
            skipped=True,
            skip_reason=f"skip_first_n_tests={state.skip_first_n}",
        )

    if state.exit_on_first_error and state.errors:
        state.test_counter += 1
        return TestResult(
            name=name,
            command=command,
            exit_code=0,
            wanted_exit_codes=[wanted_exit_code]
            + ([alternative_exit_code] if alternative_exit_code is not None else []),
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
    try:
        proc = run(command, cwd=cwd, env=env, timeout=timeout)
        exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        exit_code = 124
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

    wanted_str = "/".join(str(c) for c in wanted)
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


def run_python_test(
    name: str,
    func: Callable[[], int],
    wanted_exit_code: int,
    alternative_exit_code: Optional[int] = None,
) -> TestResult:
    """Execute a Python test function directly. Mirrors run_test_case but
    catches Python exceptions cleanly and skips the shell process."""
    state = get_state()

    if state.skip_first_n and state.test_counter < state.skip_first_n:
        yellow_text(
            f"Skipping test {state.test_counter}, will start at {state.skip_first_n}"
        )
        state.test_counter += 1
        return TestResult(
            name=name,
            command=getattr(func, "__name__", repr(func)),
            exit_code=wanted_exit_code,
            wanted_exit_codes=[wanted_exit_code]
            + ([alternative_exit_code] if alternative_exit_code is not None else []),
            runtime=0.0,
            failed=False,
            success_mark="SKIP",
            skipped=True,
            skip_reason=f"skip_first_n_tests={state.skip_first_n}",
        )

    if state.exit_on_first_error and state.errors:
        state.test_counter += 1
        return TestResult(
            name=name,
            command=getattr(func, "__name__", repr(func)),
            exit_code=0,
            wanted_exit_codes=[wanted_exit_code]
            + ([alternative_exit_code] if alternative_exit_code is not None else []),
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
            err = (
                f"{name} exited with {exit_code} (wanted {wanted_exit_code})."
            )
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


def run_test_cases(
    cases: List[Dict],
    total: Optional[int] = None,
    runtimes_so_far: Optional[List[float]] = None,
) -> None:
    """Run a list of test case dicts in sequence with progress reporting."""
    state = get_state()
    if total is None:
        total = len(cases)
    runtimes_so_far = runtimes_so_far if runtimes_so_far is not None else []

    for idx, case in enumerate(cases, start=1):
        if state.exit_on_first_error and state.errors:
            continue

        name = case.get("name", f"test_{idx}")
        cmd = case["command"]
        wanted = case.get("wanted_exit_code", 0)
        alternative = case.get("alternative_exit_code")
        cwd = case.get("cwd")
        env = case.get("env")
        timeout = case.get("timeout")
        func = case.get("func")

        progress_line(idx, total, cmd if not func else name)
        start = time.time()
        if func is not None:
            run_python_test(name, func, wanted, alternative)
        else:
            run_test_case(name, cmd, wanted, alternative, cwd=cwd, env=env, timeout=timeout)
        end = time.time()
        duration = end - start

        if duration > 10:
            runtimes_so_far.append(duration)

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
