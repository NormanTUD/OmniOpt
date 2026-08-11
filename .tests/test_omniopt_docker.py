#!/usr/bin/env python3
"""Tests for omniopt_docker (Python rewrite).

These tests focus on the pieces that can be exercised without a real
docker daemon:

  * argument parsing & --help
  * choosing docker vs sudo docker (based on group membership)
  * prefix validation for the inner command (omniopt*, .tests/*, python3*)
  * construction of the ``docker run`` command (volumes, env, user, ...)
  * exit codes for known error paths

The tests are deliberately *tightly coupled* to the implementation: they
import internal helpers from :mod:`omniopt_docker` so that any silent
behaviour change is flagged here.

The expensive end-to-end docker tests live in the ``suite:docker`` config
suite and are skipped automatically when docker is unavailable.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent

sys.path.insert(0, str(REPO_ROOT))
from importlib.machinery import SourceFileLoader  # noqa: E402

od = SourceFileLoader(
    "omniopt_docker", str(REPO_ROOT / "omniopt_docker")
).load_module()

from _framework.helpers import red_text  # noqa: E402


def _check(condition: bool, message: str) -> bool:
    if not condition:
        red_text(f"FAIL: {message}")
        return False
    return True


def _all_passed(*mocks: str) -> dict:
    """Return a dict of mock callables whose names are mapped to ``lambda c: 0``."""
    return {name: (lambda *a, **kw: 0) for name in mocks}


# ---------------------------------------------------------------------------
# Pure-function tests (no I/O at all)
# ---------------------------------------------------------------------------


def test_docker_command_with_group() -> bool:
    compose, run = od.determine_docker_cmd(in_docker_group=True)
    return _check(
        compose == "docker compose" and run == "docker",
        f"got ({compose!r}, {run!r})",
    )


def test_docker_command_without_group() -> bool:
    compose, run = od.determine_docker_cmd(in_docker_group=False)
    return _check(
        compose == "sudo docker compose" and run == "sudo docker",
        f"got ({compose!r}, {run!r})",
    )


def test_validate_inner_command_valid_prefixes() -> bool:
    ok = True
    for inner in (
        "./omniopt --tests",
        "omniopt --tests",
        "./.tests/main.py",
        ".tests/main.py",
        "python3 .tests/main.py",
    ):
        try:
            od.validate_inner_command(inner)
        except ValueError as e:
            ok &= _check(False, f"unexpected rejection of {inner!r}: {e}")
    return ok


def test_validate_inner_command_invalid_prefix() -> bool:
    try:
        od.validate_inner_command("rm -rf /")
    except ValueError:
        return True
    return _check(False, "validate_inner_command accepted a dangerous input")


def test_build_run_command_no_display() -> bool:
    cmd = od.build_run_command(
        inner="./omniopt --tests",
        docker_name="omniopt_omniopt2",
        docker_cmd="docker",
        pwd="/work",
        home="/home/u",
        has_display=False,
    )
    s = " ".join(cmd)
    ok = _check("--rm" in cmd, f"--rm missing: {s}")
    ok &= _check("omniopt_omniopt2" in cmd, f"image missing: {s}")
    ok &= _check("/work/runs:/var/opt/omniopt/runs:rw" in s, f"runs mount missing: {s}")
    ok &= _check("/work/logs:/var/opt/omniopt/logs:rw" in s, f"logs mount missing: {s}")
    ok &= _check("--user=" not in s, f"--user must NOT be passed without DISPLAY: {s}")
    ok &= _check(
        "/etc/shadow" not in s,
        f"/etc/shadow must NOT be mounted without DISPLAY: {s}",
    )
    ok &= _check("/var/opt/omniopt/./omniopt" in s, f"target path wrong: {s}")
    return ok


def test_build_run_command_with_display() -> bool:
    cmd = od.build_run_command(
        inner="./omniopt --tests",
        docker_name="omniopt_omniopt2",
        docker_cmd="docker",
        pwd="/work",
        home="/home/u",
        has_display=True,
    )
    s = " ".join(cmd)
    ok = _check("--user=" in s, f"--user= must be set with DISPLAY: {s}")
    ok &= _check("/etc/shadow:/etc/shadow:ro" in s, f"/etc/shadow mount missing: {s}")
    ok &= _check("/tmp/.X11-unix" in s, f"X11 socket mount missing: {s}")
    ok &= _check("--env=DISPLAY" in s, f"DISPLAY env missing: {s}")
    return ok


def test_build_run_command_python3_inner() -> bool:
    cmd = od.build_run_command(
        inner="python3 ./.tests/main.py --quick",
        docker_name="omniopt_omniopt2",
        docker_cmd="docker",
        pwd="/work",
        home="/home/u",
        has_display=False,
    )
    s = " ".join(cmd)
    ok = _check("python3" in cmd, f"python3 must be the interpreter: {s}")
    ok &= _check("/var/opt/omniopt/.tests/main.py" in s, f"target path wrong: {s}")
    return ok


def test_build_run_command_uses_sudo_docker() -> bool:
    cmd = od.build_run_command(
        inner="./omniopt --tests",
        docker_name="omniopt_omniopt2",
        docker_cmd="sudo docker",
        pwd="/work",
        home="/home/u",
        has_display=False,
    )
    return _check(cmd[0] == "sudo", f"expected 'sudo' as first token: {cmd[:2]}")


# ---------------------------------------------------------------------------
# main() tests (with all side-effects mocked)
# ---------------------------------------------------------------------------


def _mk_runner(captured: dict) -> dict:
    """Return a kwargs dict with all I/O mocked for main()."""

    def fake_check(_cmd: str) -> bool:
        return True

    def fake_mkdir(paths) -> None:
        captured.setdefault("mkdir", []).extend(list(paths))

    def fake_build(compose_cmd: str) -> int:
        captured["built"] = compose_cmd
        return 0

    def fake_up(compose_cmd: str) -> int:
        captured["up"] = compose_cmd
        return 0

    def fake_run(cmd) -> int:
        captured["ran"] = cmd
        return 0

    return dict(
        check_cmd=fake_check,
        mkdir=fake_mkdir,
        docker_build=fake_build,
        docker_up=fake_up,
        docker_run=fake_run,
        docker_cmd="docker",
        in_docker_group=True,
        has_display=False,
    )


def test_help_flag_returns_zero() -> bool:
    rc = od.main(["--help"])
    return _check(rc == 0, f"--help should exit 0, got {rc}")


def test_no_args_just_builds_and_exits_zero() -> bool:
    captured: dict = {}
    rc = od.main([], **_mk_runner(captured))
    ok = _check(rc == 0, f"no-args exit should be 0, got {rc}")
    ok &= _check("built" in captured, "docker compose build was not invoked")
    ok &= _check("up" in captured, "docker compose up was not invoked")
    ok &= _check("ran" not in captured, "docker run must NOT be invoked without a command")
    return ok


def test_main_invalid_inner_command() -> bool:
    """If the inner command has a bad prefix, exit code must be 1 and no docker
    call is made (validation happens first)."""
    captured: dict = {}
    rc = od.main(["rm -rf /"], **_mk_runner(captured))
    ok = _check(rc == 1, f"invalid prefix must exit 1, got {rc}")
    ok &= _check("built" not in captured, "must not invoke docker build on bad input")
    ok &= _check("ran" not in captured, "must not invoke docker run on bad input")
    return ok


def test_main_runs_inner_command() -> bool:
    captured: dict = {}
    rc = od.main(["./omniopt", "--tests"], **_mk_runner(captured))
    ok = _check(rc == 0, f"expected exit 0, got {rc}")
    ok &= _check("ran" in captured, "docker_run was not called")
    ok &= _check(captured["ran"][0] == "docker", f"wrong docker binary: {captured['ran']}")
    return ok


def test_build_failure_exits_one() -> bool:
    captured: dict = {}

    def fail_build(_compose_cmd: str) -> int:
        return 7

    runner = _mk_runner(captured)
    runner["docker_build"] = fail_build
    rc = od.main([], **runner)
    return _check(rc == 1, f"build failure must exit 1, got {rc}")


def test_up_failure_exits_one() -> bool:
    captured: dict = {}

    def fail_up(_compose_cmd: str) -> int:
        return 5

    runner = _mk_runner(captured)
    runner["docker_up"] = fail_up
    rc = od.main([], **runner)
    return _check(rc == 1, f"up failure must exit 1, got {rc}")


def test_docker_run_failure_returns_exit_code() -> bool:
    captured: dict = {}

    def fail_run(_cmd) -> int:
        return 42

    runner = _mk_runner(captured)
    runner["docker_run"] = fail_run
    rc = od.main(["./omniopt", "--tests"], **runner)
    return _check(rc == 42, f"docker run failure must propagate exit code, got {rc}")


TESTS = [
    # pure-function tests
    test_docker_command_with_group,
    test_docker_command_without_group,
    test_validate_inner_command_valid_prefixes,
    test_validate_inner_command_invalid_prefix,
    test_build_run_command_no_display,
    test_build_run_command_with_display,
    test_build_run_command_python3_inner,
    test_build_run_command_uses_sudo_docker,
    # main() tests
    test_help_flag_returns_zero,
    test_no_args_just_builds_and_exits_zero,
    test_main_invalid_inner_command,
    test_main_runs_inner_command,
    test_build_failure_exits_one,
    test_up_failure_exits_one,
    test_docker_run_failure_returns_exit_code,
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
    print("\nAll omniopt_docker tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
