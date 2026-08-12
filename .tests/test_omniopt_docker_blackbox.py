#!/usr/bin/env python3
"""Black-box (decoupled) tests for omniopt_docker.

Unlike ``test_omniopt_docker.py`` (which imports the module's internal
helpers), these tests treat ``omniopt_docker`` as an opaque executable and
drive it through a subprocess against *fake* ``docker`` / ``sudo`` /
``groups`` binaries on PATH.  They observe only observable behaviour:

  * exit codes and help output
  * which docker commands get executed (logged argv)
  * how the inner command + its arguments are forwarded to ``docker run``
  * the exact argv passed to the interpreter (catches args being baked
    into the script path as ONE token)
  * sudo usage when the user is not in the docker group

Nothing here imports or inspects the implementation, so a refactor that
keeps the observable behaviour will keep these green, and a behaviour
regression will turn them red.

Regression covered by test_python_args_forwarded_as_separate_tokens:
the old code built ``/var/opt/omniopt/.tests/main --num_random_steps=1 ...``
as a SINGLE argv token, so ``python3`` tried to open a file whose name
contained all the flags.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
OMNIOPT_DOCKER = REPO_ROOT / "omniopt_docker"
IMAGE_NAME = "omniopt-omniopt2"

sys.path.insert(0, str(THIS_DIR))
from _framework.helpers import red_text  # noqa: E402

DOCKER_FAKE = """\
#!/usr/bin/env bash
echo "== INVOCATION ==" >> "$DOCKER_LOG"
printf '%s\\n' "$(basename -- "$0")" "$@" >> "$DOCKER_LOG"
if [ -n "$FAKE_FAIL_RUN" ] && [ "$1" = "run" ]; then
  exit "$FAKE_FAIL_RUN"
fi
exit 0
"""

SUDO_FAKE = """\
#!/usr/bin/env bash
echo "== INVOCATION ==" >> "$SUDO_LOG"
printf '%s\\n' "$(basename -- "$0")" "$@" >> "$SUDO_LOG"
exec "$@"
"""

GROUPS_FAKE = """\
#!/usr/bin/env bash
echo "${GROUPS_OUTPUT:-fakeuser users}"
"""


class Sandbox:
    """Temp dir + fake binaries + the captured docker/sudo logs."""

    def __init__(self, tmpdir_obj, tmp: Path, docker_log: Path, sudo_log: Path):
        self._tmpdir_obj = tmpdir_obj
        self.tmp = tmp
        self.home = tmp / "home"
        self.docker_log = docker_log
        self.sudo_log = sudo_log

    def docker_invocations(self) -> List[List[str]]:
        return _parse_invocations(self.docker_log)

    def sudo_invocations(self) -> List[List[str]]:
        return _parse_invocations(self.sudo_log)


def _make_fake_bin(tmp: Path) -> Path:
    bin_dir = tmp / "bin"
    bin_dir.mkdir()
    for name, body in (("docker", DOCKER_FAKE), ("sudo", SUDO_FAKE), ("groups", GROUPS_FAKE)):
        p = bin_dir / name
        p.write_text(body)
        p.chmod(0o755)
    return bin_dir


def _parse_invocations(log_path: Path) -> List[List[str]]:
    invocations: List[List[str]] = []
    cur: Optional[List[str]] = None
    for line in log_path.read_text().splitlines():
        if line == "== INVOCATION ==":
            cur = []
            invocations.append(cur)
        elif cur is not None:
            cur.append(line)
    return invocations


def _run_script(
    args: List[str],
    *,
    display: bool = False,
    fail_run: Optional[int] = None,
    in_docker_group: bool = True,
    extra_env: Optional[dict] = None,
    files: Optional[dict] = None,
) -> tuple:
    """Run omniopt_docker in a sandbox; returns ``(proc, Sandbox)``.

    ``files`` maps relative paths to file contents written into the temp
    working directory *before* the script runs (e.g. to test shebang
    detection).
    """
    tmpdir_obj = tempfile.TemporaryDirectory(prefix="oo_docker_blackbox_")
    tmp = Path(tmpdir_obj.name)
    (tmp / "home").mkdir()
    for rel, content in (files or {}).items():
        p = tmp / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    bin_dir = _make_fake_bin(tmp)
    docker_log = tmp / "docker.log"
    sudo_log = tmp / "sudo.log"
    docker_log.touch()
    sudo_log.touch()

    env = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "HOME": str(tmp / "home"),
        "USER": "fakeuser",
        "DOCKER_LOG": str(docker_log),
        "SUDO_LOG": str(sudo_log),
        "GROUPS_OUTPUT": "fakeuser users docker" if in_docker_group else "fakeuser users",
    }
    if extra_env:
        env.update(extra_env)
    if display:
        env["DISPLAY"] = ":0"
    if fail_run is not None:
        env["FAKE_FAIL_RUN"] = str(fail_run)

    proc = subprocess.run(
        [sys.executable, str(OMNIOPT_DOCKER), *args],
        cwd=str(tmp),
        env=env,
        capture_output=True,
        text=True,
    )
    return proc, Sandbox(tmpdir_obj, tmp, docker_log, sudo_log)


def _run_invocation(invocations: List[List[str]]) -> Optional[List[str]]:
    for inv in invocations:
        if len(inv) >= 2 and inv[0] == "docker" and inv[1] == "run":
            return inv
    return None


def _after_image(invocation: List[str]) -> List[str]:
    try:
        idx = invocation.index(IMAGE_NAME)
    except ValueError:
        return []
    return invocation[idx + 1:]


def _check(condition: bool, message: str) -> bool:
    if not condition:
        red_text(f"FAIL: {message}")
        return False
    return True


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_help_exits_zero_and_prints_usage() -> bool:
    proc, box = _run_script(["--help"])
    ok = _check(proc.returncode == 0, f"--help must exit 0, got {proc.returncode}")
    ok &= _check("Usage:" in proc.stdout, f"--help must print usage, got {proc.stdout!r}")
    ok &= _check(box.docker_invocations() == [], "--help must not invoke docker")
    return ok


def test_python_args_forwarded_as_separate_tokens() -> bool:
    proc, box = _run_script(
        ["python3", ".tests/main",
         "--num_random_steps=1", "--max_eval=2", "--exit_on_first_error",
         "--superquick", "--run_with_coverage"]
    )
    run = _run_invocation(box.docker_invocations())
    ok = _check(proc.returncode == 0, f"expected exit 0, got {proc.returncode}")
    ok &= _check(run is not None, "no `docker run` invocation logged")
    if run is None:
        return False
    tail = _after_image(run)
    ok &= _check(
        tail == ["python3", "/var/opt/omniopt/.tests/main",
                 "--num_random_steps=1", "--max_eval=2", "--exit_on_first_error",
                 "--superquick", "--run_with_coverage"],
        f"unexpected argv after image: {tail}",
    )
    ok &= _check(
        all(" " not in tok for tok in tail),
        "an argv token contains spaces: flags were baked into the script path",
    )
    return ok


def test_bash_inner_command_forwarded() -> bool:
    proc, box = _run_script(["./omniopt", "--tests", "--foo", "bar"])
    run = _run_invocation(box.docker_invocations())
    ok = _check(proc.returncode == 0, f"expected exit 0, got {proc.returncode}")
    ok &= _check(run is not None, "no `docker run` invocation logged")
    if run is None:
        return False
    tail = _after_image(run)
    ok &= _check(
        tail == ["bash", "/var/opt/omniopt/./omniopt", "--tests", "--foo", "bar"],
        f"unexpected argv after image: {tail}",
    )
    return ok


def test_argument_with_space_stays_single_token() -> bool:
    proc, box = _run_script(["python3", ".tests/main", "a b c"])
    run = _run_invocation(box.docker_invocations())
    ok = _check(proc.returncode == 0, f"expected exit 0, got {proc.returncode}")
    ok &= _check(run is not None, "no `docker run` invocation logged")
    if run is None:
        return False
    tail = _after_image(run)
    ok &= _check(
        tail == ["python3", "/var/opt/omniopt/.tests/main", "a b c"],
        f"unexpected argv after image: {tail}",
    )
    return ok


def test_python_shebang_script_detected() -> bool:
    proc, box = _run_script(
        ["./.tests/my_tool"],
        files={".tests/my_tool": "#!/usr/bin/env python3\nprint('hi')\n"},
    )
    run = _run_invocation(box.docker_invocations())
    ok = _check(proc.returncode == 0, f"expected exit 0, got {proc.returncode}")
    ok &= _check(run is not None, "no `docker run` invocation logged")
    if run is None:
        return False
    tail = _after_image(run)
    ok &= _check(
        tail == ["python3", "/var/opt/omniopt/.tests/my_tool"],
        f"shebang script should be run via python3, got: {tail}",
    )
    return ok


def test_non_python_script_dispatched_to_bash() -> bool:
    proc, box = _run_script(
        ["./.tests/my_tool"],
        files={".tests/my_tool": "#!/usr/bin/env bash\necho hi\n"},
    )
    run = _run_invocation(box.docker_invocations())
    ok = _check(proc.returncode == 0, f"expected exit 0, got {proc.returncode}")
    ok &= _check(run is not None, "no `docker run` invocation logged")
    if run is None:
        return False
    tail = _after_image(run)
    ok &= _check(
        tail == ["bash", "/var/opt/omniopt/./.tests/my_tool"],
        f"non-python script should be run via bash, got: {tail}",
    )
    return ok


def test_invalid_prefix_rejected_without_docker() -> bool:
    proc, box = _run_script(["rm", "-rf", "/"])
    ok = _check(proc.returncode == 1, f"invalid prefix must exit 1, got {proc.returncode}")
    ok &= _check(
        _run_invocation(box.docker_invocations()) is None,
        "must not run docker for a rejected inner command",
    )
    return ok


def test_no_display_mounts_and_created_dirs() -> bool:
    proc, box = _run_script(["python3", ".tests/main", "--quick"])
    run = _run_invocation(box.docker_invocations())
    ok = _check(proc.returncode == 0, f"expected exit 0, got {proc.returncode}")
    ok &= _check(run is not None, "no `docker run` invocation logged")
    if run is None:
        return False
    s = " ".join(run)
    ok &= _check(
        f"{box.tmp}/runs:/var/opt/omniopt/runs:rw" in s, f"runs volume missing: {s}"
    )
    ok &= _check(
        f"{box.tmp}/logs:/var/opt/omniopt/logs:rw" in s, f"logs volume missing: {s}"
    )
    ok &= _check("--user=" not in s, f"--user= must NOT be passed without DISPLAY: {s}")
    ok &= _check(
        (box.tmp / "runs").is_dir() and (box.tmp / "logs").is_dir(),
        "runs/ and logs/ must be created in the working directory",
    )
    ok &= _check(
        (box.home / ".config" / "matplotlib_docker_omniopt").is_dir(),
        "matplotlib docker config dir must be created in $HOME",
    )
    return ok


def test_display_adds_user_and_x11_mounts() -> bool:
    proc, box = _run_script(["python3", ".tests/main"], display=True)
    run = _run_invocation(box.docker_invocations())
    ok = _check(proc.returncode == 0, f"expected exit 0, got {proc.returncode}")
    ok &= _check(run is not None, "no `docker run` invocation logged")
    if run is None:
        return False
    s = " ".join(run)
    ok &= _check("--user=" in s, f"--user= must be set with DISPLAY: {s}")
    ok &= _check("--env=DISPLAY" in s, f"DISPLAY env missing: {s}")
    ok &= _check("/tmp/.X11-unix" in s, f"X11 socket mount missing: {s}")
    return ok


def test_run_failure_exit_code_propagates() -> bool:
    proc, box = _run_script(["./omniopt", "--tests"], fail_run=42)
    invocations = box.docker_invocations()
    ok = _check(proc.returncode == 42, f"docker run failure must propagate, got {proc.returncode}")
    ok &= _check(
        any(inv == ["docker", "images"] for inv in invocations),
        "docker images must be listed on run failure",
    )
    return ok


def test_sudo_used_when_not_in_docker_group() -> bool:
    proc, box = _run_script(["./omniopt", "--tests"], in_docker_group=False)
    sudo_run = None
    for inv in box.sudo_invocations():
        if len(inv) >= 3 and inv[:2] == ["sudo", "docker"] and inv[2] == "run":
            sudo_run = inv
            break
    ok = _check(proc.returncode == 0, f"expected exit 0, got {proc.returncode}")
    ok &= _check(
        sudo_run is not None,
        "expected `sudo docker run` when user is not in the docker group",
    )
    if sudo_run is not None:
        tail = _after_image(sudo_run)
        ok &= _check(
            tail == ["bash", "/var/opt/omniopt/./omniopt", "--tests"],
            f"unexpected argv after image via sudo: {tail}",
        )
    return ok


TESTS = [
    test_help_exits_zero_and_prints_usage,
    test_python_args_forwarded_as_separate_tokens,
    test_bash_inner_command_forwarded,
    test_argument_with_space_stays_single_token,
    test_python_shebang_script_detected,
    test_non_python_script_dispatched_to_bash,
    test_invalid_prefix_rejected_without_docker,
    test_no_display_mounts_and_created_dirs,
    test_display_adds_user_and_x11_mounts,
    test_run_failure_exit_code_propagates,
    test_sudo_used_when_not_in_docker_group,
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
    print("\nAll omniopt_docker black-box tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
