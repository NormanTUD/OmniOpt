"""Color/text/time helpers for the test framework.

This module replaces the functions normally provided by
.shellscript_functions and .colorfunctions.sh.
"""

from __future__ import annotations

import os
import sys
import time
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


class Colors:
    """ANSI color codes."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[0;33m"
    BLUE = "\033[0;34m"
    CYAN = "\033[0;36m"
    MAGENTA = "\033[0;35m"
    RED = "\033[0;31m"
    WHITE = "\033[1;37m"
    GRAY = "\033[0;90m"


_NO_COLOR = os.environ.get("NO_COLOR") is not None
if not sys.stdout.isatty():
    _NO_COLOR = True


def _color(text: str, color: str) -> str:
    if _NO_COLOR:
        return text
    return f"{color}{text}{Colors.RESET}"


def echoerr(*args: object, end: str = "\n") -> None:
    print(*args, file=sys.stderr, end=end)


def red_text(text: str, end: str = "\n") -> None:
    echoerr(_color(text.rstrip("\n"), Colors.RED), end=end)


def yellow_text(text: str, end: str = "\n") -> None:
    echoerr(_color(text.rstrip("\n"), Colors.YELLOW), end=end)


def green_text(text: str, end: str = "\n") -> None:
    echoerr(_color(text.rstrip("\n"), Colors.GREEN), end=end)


def blue_text(text: str, end: str = "\n") -> None:
    echoerr(_color(text.rstrip("\n"), Colors.BLUE), end=end)


def cyan_text(text: str, end: str = "\n") -> None:
    echoerr(_color(text.rstrip("\n"), Colors.CYAN), end=end)


def magenta_text(text: str, end: str = "\n") -> None:
    echoerr(_color(text.rstrip("\n"), Colors.MAGENTA), end=end)


def green_text_no_newline(text: str) -> None:
    green_text(text, end="")


def yellow_text_no_newline(text: str) -> None:
    yellow_text(text, end="")


def _green_text_bold_underline(text: str) -> None:
    if _NO_COLOR:
        echoerr(text)
    else:
        echoerr(f"{Colors.GREEN}{Colors.BOLD}{Colors.UNDERLINE}{text}{Colors.RESET}")


def green_bold_underline(text: str) -> None:
    _green_text_bold_underline(text)


def displaytime(seconds: int) -> str:
    """Format a number of seconds as a human-readable string."""
    seconds = int(seconds)
    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    parts = []
    if days > 0:
        parts.append(f"{days} days")
    if hours > 0:
        parts.append(f"{hours} hours")
    if minutes > 0:
        parts.append(f"{minutes} minutes")
    if days or hours or minutes:
        parts.append("and ")
    parts.append(f"{secs} seconds")
    return "".join(parts)


def human_readable_time(seconds: int) -> str:
    """Compact form (e.g. '5s', '2m 3s', '1h 5m 3s')."""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60}m {seconds % 60}s"


def join_by(delimiter: str, first: str, items: Sequence[str]) -> str:
    if not items:
        return first
    return first + "".join(delimiter + s for s in items)


def in_container() -> bool:
    if os.path.exists("/.dockerenv"):
        return True
    if os.path.exists("/run/.containerenv"):
        return True
    try:
        with open("/proc/1/cgroup", "r") as f:
            content = f.read()
            if any(token in content for token in ("/docker/", "/lxc/", "/kubepods/", "/containerd/")):
                return True
    except OSError:
        pass
    if os.environ.get("container") or os.environ.get("KUBERNETES_SERVICE_HOST"):
        return True
    try:
        with open("/proc/1/sched", "r") as f:
            first = f.read(1024).split(maxsplit=1)[0]
            if first != "1":
                return True
    except OSError:
        pass
    try:
        with open("/proc/self/mountinfo", "r") as f:
            if "overlay" in f.read():
                return True
    except OSError:
        pass
    return False


def is_ci() -> bool:
    return bool(os.environ.get("CI"))


def command_exists(cmd: str) -> bool:
    return shutil.which(cmd) is not None


DEFAULT_EXCLUDE_DIRS = {
    ".git", "runs", "logs", "node_modules", "__pycache__", ".mypy_cache",
    ".ruff_cache", ".pytest_cache", ".tox", "site-packages", "dist-packages",
    "build", "dist", ".cache", "libs", "conceptdrift",
}


def find_files(
    root: str | os.PathLike,
    suffixes: Sequence[str],
    exclude_dirs: Optional[Iterable[str]] = None,
) -> List[Path]:
    """Recursively find source files under ``root`` matching any suffix.

    Directories whose name matches an entry in ``exclude_dirs`` (merged
    with :data:`DEFAULT_EXCLUDE_DIRS`) or that contains "venv" are pruned
    during traversal.
    """
    root = Path(root)
    extra = set(exclude_dirs or ())
    suffixes = tuple(suffixes)

    def keep_dir(name: str) -> bool:
        low = name.lower()
        return low not in DEFAULT_EXCLUDE_DIRS and low not in extra and "venv" not in low

    files: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if keep_dir(d))
        for fname in sorted(filenames):
            if fname.endswith(suffixes):
                files.append(Path(dirpath) / fname)
    return files


def find_shell_scripts(
    root: str | os.PathLike,
    exclude_dirs: Optional[Iterable[str]] = None,
) -> List[Path]:
    """Find shell scripts under ``root``.

    Matches ``*.sh`` files and extensionless files whose shebang names
    ``bash`` or ``sh`` (zsh and other shells are skipped).
    """
    root = Path(root)
    extra = set(exclude_dirs or ())

    def keep_dir(name: str) -> bool:
        low = name.lower()
        return low not in DEFAULT_EXCLUDE_DIRS and low not in extra and "venv" not in low

    scripts: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if keep_dir(d))
        for fname in sorted(filenames):
            path = Path(dirpath) / fname
            if fname.endswith(".sh"):
                scripts.append(path)
                continue
            if "." in fname:
                continue
            try:
                with open(path, "rb") as fh:
                    first_line = fh.readline(128)
            except OSError:
                continue
            if first_line.startswith(b"#!") and (
                b"bash" in first_line or b"/sh" in first_line
            ):
                scripts.append(path)
    return scripts


def get_nvidia_smi_gpus() -> int:
    if command_exists("nvidia-smi"):
        return 1
    return 0


def run(
    cmd: str,
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
    timeout: Optional[float] = None,
    check: bool = False,
    live: bool = False,
) -> subprocess.CompletedProcess:
    """Run a shell command and return the completed process.

    When ``live`` is True, the child's stdout/stderr stream straight to
    the terminal instead of being captured. The child is killed when the
    user presses Ctrl+C so no orphan process keeps running.
    """
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    if live:
        proc = subprocess.Popen(cmd, shell=True, cwd=cwd, env=full_env)
    else:
        proc = subprocess.Popen(
            cmd,
            shell=True,
            cwd=cwd,
            env=full_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except KeyboardInterrupt:
        proc.kill()
        proc.wait()
        raise
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode, proc.args, output=stdout, stderr=stderr
        )
    return subprocess.CompletedProcess(
        proc.args, proc.returncode, stdout=stdout, stderr=stderr
    )


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_file_lines(path: str) -> List[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.readlines()
    except FileNotFoundError:
        return []


def ensure_install_tests_env() -> None:
    """Mimic `export install_tests=1` from .shellscript_functions."""
    os.environ.setdefault("install_tests", "1")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    os.environ.setdefault("DONT_SHOW_DONT_INSTALL_MESSAGE", "1")
    os.environ.setdefault("DISABLE_SIXEL_GRAPHICS", "1")
    os.environ.setdefault("ENABLE_BEARTYPE", "1")
    os.environ.setdefault("OO_MAIN_TESTS", "1")


def ensure_omniopt_call() -> str:
    """Returns the omniopt command (either ./omniopt or $OMNIOPT_CALL)."""
    return os.environ.get("OMNIOPT_CALL", "./omniopt")


def now_ts() -> int:
    return int(time.time())
