"""
Environment setup and helper functions for OmniOpt2 scripts.

This is the Python equivalent of the former ``.shellscript_functions``
bash library.  It provides the venv bootstrap that prepares the Python
environment (virtualenv creation, dependency installation) as well as
small helpers for colored output, progress bars and error dialogs.

Importing this module performs no side effects.  Call
``setup_environment()`` explicitly to run the bootstrap, which mirrors
what ``source .shellscript_functions`` did in bash.

Pure helpers (``displaytime``, ``generate_progress_bar``) can also be
invoked from the command line::

    python3 .shellscript_functions.py displaytime 3661
"""

from __future__ import annotations

import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

install_those: list[str] = []

Green = "\033[0;32m"
Color_Off = "\033[0m"
Red = "\033[0;31m"


def echoerr(*args: object) -> None:
    """Print the arguments to stderr, separated by spaces, with a newline."""
    print(*args, file=sys.stderr)


def red_text(msg: str) -> None:
    """Print ``msg`` in red on stdout without a trailing newline."""
    sys.stdout.write(f"{Red}{msg}{Color_Off}")


def yellow_text(msg: str) -> None:
    """Print ``msg`` in yellow on stderr with a trailing newline."""
    sys.stderr.write(f"\033[0;33m{msg}\033[0m\n")


def green_text_no_newline(msg: str) -> None:
    """Print ``msg`` in green on stderr without a trailing newline."""
    sys.stderr.write(f"\033[0;32m{msg}\033[0m")


def green_text(msg: str) -> None:
    """Print ``msg`` in green on stderr with a trailing newline."""
    sys.stderr.write(f"\033[0;32m{msg}\033[0m\n")


def green(msg: str) -> None:
    """Print ``msg`` in green on stdout without a trailing newline."""
    sys.stdout.write(f"{Green}{msg}{Color_Off}")


def _tput(character: str) -> int:
    """Emulate the bash ``tput`` helper for ``cr``/``el``/``bel``."""
    if character == "cr":
        sys.stdout.write("\r")
        return 0

    if character == "el":
        sys.stdout.write("\033[K")
        return 0

    if character == "bel":
        if os.environ.get("OO_MAIN_TESTS") == "1":
            return 0
        sys.stdout.write("\a")
        return 0

    if shutil.which("tput") and sys.stdout.isatty():
        subprocess.run(["tput", character], check=False)

    return 0


def green_reset_line(msg: str) -> None:
    """Carriage-return, clear the line, then print ``msg`` in green."""
    _tput("cr")
    _tput("el")
    green(msg)


def red_reset_line(msg: str) -> None:
    """Carriage-return, clear the line, then print ``msg`` in red."""
    _tput("cr")
    _tput("el")
    red_text(msg)


def echo_red(msg: str) -> None:
    """Print ``msg`` in red to stdout with a trailing newline."""
    sys.stdout.write(f"\033[31m{msg}\033[0m\n")


def echo_green(msg: str) -> None:
    """Print ``msg`` in green to stdout with a trailing newline."""
    sys.stdout.write(f"\033[32m{msg}\033[0m\n")


def displaytime(total_seconds: int) -> str:
    """Format ``total_seconds`` like the bash ``displaytime`` function."""
    days = total_seconds // 86400
    hours = (total_seconds // 3600) % 24
    minutes = (total_seconds // 60) % 60
    seconds = total_seconds % 60

    parts: list[str] = []
    if days > 0:
        parts.append(f"{days} days ")
    if hours > 0:
        parts.append(f"{hours} hours ")
    if minutes > 0:
        parts.append(f"{minutes} minutes ")
    if days > 0 or hours > 0 or minutes > 0:
        parts.append("and ")
    parts.append(f"{seconds} seconds\n")
    return "".join(parts)


def _terminal_size() -> tuple[int, int]:
    """Return ``(lines, columns)`` of the attached terminal."""
    try:
        size = os.get_terminal_size()
        return size.lines, size.columns
    except OSError:
        return 24, 80


def error_message(msg: str) -> int:
    """Print ``msg`` in red and show a whiptail error dialog when possible."""
    echo_red(msg)

    if not sys.stdout.isatty() or shutil.which("whiptail") is None:
        return 0

    lines, columns = _terminal_size()
    env = os.environ.copy()
    env["NEWT_COLORS"] = (
        "window=,red\n"
        "border=white,red\n"
        "textbox=white,red\n"
        "button=black,white\n"
    )
    subprocess.run(
        [
            "whiptail",
            "--title",
            "Error Message",
            "--scrolltext",
            "--msgbox",
            msg,
            str(lines),
            str(columns),
            str(max(0, lines - 8)),
        ],
        env=env,
        check=False,
    )
    return 0


def _bar_char() -> str:
    """Return the fill character used for progress bars."""
    if platform.system() == "Linux":
        return "━"
    return "#"


def generate_progress_bar(current: int, maximum: int) -> str:
    """Build the ``[####    ] `` progress bar string (30 cells wide)."""
    bar_length = 30
    filled_length = bar_length * current // maximum
    empty_length = bar_length - filled_length
    bar = _bar_char() * filled_length + " " * empty_length
    return f"[{bar}] \n"


def _frozen_packages() -> str:
    """Return the installed packages in pip ``freeze`` format."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "--disable-pip-version-check",
            "list",
            "--format=freeze",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout


def _module_without_version(module: str) -> str:
    """Strip any version specifier from a requirement string."""
    base = re.split(r"[=<>~]", module)[0]
    return "rich-argparse" if base == "rich_argparse" else base


def get_nr_of_already_installed_modules() -> int:
    """Count how many requirements from the files are already installed."""
    frozen = _frozen_packages().lower()
    count = 0
    for module in install_those:
        if _module_without_version(module) in frozen:
            count += 1
    return count


def generate_progress_bar_setup(total_nr_modules: int) -> str:
    """Build a progress bar showing how many modules are installed."""
    number_installed = get_nr_of_already_installed_modules()
    return generate_progress_bar(number_installed, total_nr_modules)


def _collect_dependencies(module: str) -> list[str]:
    """List the top-level dependencies pip would install for ``module``."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--default-timeout=300",
            "--disable-pip-version-check",
            "--dry-run",
            module,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    dependencies: list[str] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line.startswith("Collecting "):
            continue
        dep = line[len("Collecting "):]
        dep = dep.split()[0]
        if dep and dep != module:
            dependencies.append(dep)
    return dependencies


def _run_pip_install(module: str, quiet: bool, log_file: Path) -> int:
    """Install ``module`` with pip, appending any errors to ``log_file``."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a") as fp:
        args = [
            sys.executable,
            "-m",
            "pip",
            "--default-timeout=300",
            "--disable-pip-version-check",
            "install",
        ]
        if quiet:
            args.append("-q")
        args.append(module)
        result = subprocess.run(
            args,
            stderr=subprocess.STDOUT,
            stdout=fp,
            check=False,
        )
    return result.returncode


def ppip(
    module: str,
    as_requirement_of: str,
    number_of_main_modules: int,
    prefix: str,
) -> int:
    """Install ``module``, recursing into its dependencies first."""
    progressbar = generate_progress_bar_setup(number_of_main_modules)

    if not os.environ.get("CI"):
        green_reset_line(f"{progressbar}➤{prefix}Installing {module} ")

    module_without_version = _module_without_version(module)
    frozen = _frozen_packages().lower()
    already_installed = module_without_version in frozen

    if not already_installed:
        if module != as_requirement_of and as_requirement_of != "-":
            dependencies = _collect_dependencies(module)
            for index, dependency in enumerate(dependencies):
                if index == 0:
                    green_reset_line(
                        f"{progressbar}➤{prefix}Installing requirements "
                        f"for {module}"
                    )
                result = ppip(
                    dependency, module, number_of_main_modules, prefix
                )
                if result != 0:
                    red_reset_line(f"❌Failed to install {dependency}.")
                    return 20

            if dependencies:
                green_reset_line(
                    f"{progressbar}➤{prefix}Installed all requirements "
                    f"for {module}, now installing the package itself..."
                )

        green_reset_line(f"{progressbar}➤{prefix}Installing {module}...")

        install_errors_file = Path("logs") / "install_errors"
        run_uuid = os.environ.get("RUN_UUID")
        if run_uuid:
            install_errors_file = Path("logs") / f"{run_uuid}_install_errors"

        quiet = not os.environ.get("DEBUG")
        exit_code = _run_pip_install(
            module, quiet=quiet, log_file=install_errors_file
        )
        if exit_code != 0:
            red_reset_line(
                f"❌Failed to install {module}. Check {install_errors_file}\n"
            )
            if os.environ.get("CI") or _in_container():
                with open(install_errors_file) as fp:
                    sys.stdout.write(fp.read())
            return 20

        if not os.environ.get("CI"):
            green_reset_line(f"{progressbar}✅{module} installed successfully")

    return 0


def _in_container() -> bool:
    """Best-effort check whether we are running inside docker or lxc."""
    cgroup = Path("/proc/self/cgroup")
    if not cgroup.exists():
        return False
    try:
        content = cgroup.read_text()
    except OSError:
        return False
    return bool(re.search(r"/docker|/lxc", content))


def install_required_modules() -> int:
    """Install every module from the requirements files that is missing."""
    maximum = len(install_those)
    if maximum == 0:
        return 0

    green_reset_line("➤Checking environment...")
    number_of_digits = len(str(maximum))

    for index, module in enumerate(install_those, start=1):
        progressbar = generate_progress_bar(index, maximum)
        prefix = (
            f"Checking {index:>{number_of_digits}}/{maximum}, "
            f"{index * 100 // maximum:>3}%: "
        )

        if not os.environ.get("CI"):
            green_reset_line(f"{progressbar}➤{prefix}{module}...")

        frozen = _frozen_packages().lower()
        if _module_without_version(module) not in frozen:
            result = ppip(module, "-", maximum, prefix)
            if result != 0:
                return result

    _tput("cr")
    _tput("el")
    green_reset_line("✅Environment checking done!")
    _tput("cr")
    _tput("el")
    return 0


def _check_required_programs() -> int:
    """Verify required system programs exist. Returns the number missing."""
    required_programs = [
        "stdbuf",
        "base64",
        "curl",
        "wget",
        "uuidgen",
        "python3",
        "gcc",
        "resize",
        "cat",
        "ls",
        "grep",
        "tput",
        "sed",
    ]
    if shutil.which("apt"):
        required_programs.extend(["findmnt", "whiptail"])

    not_found = 0
    for cmd in required_programs:
        if shutil.which(cmd) is None:
            red_text(
                f"❌{cmd} not found. Try installing it with "
                f"'sudo apt-get install {cmd}' (depending on your distro)\n"
            )
            not_found += 1
    return not_found


def _venv_dir() -> str:
    """Compute the canonical venv directory for this Python version / arch."""
    version = subprocess.run(
        [sys.executable, "--version"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip().replace(" ", "_")
    arch = platform.machine()
    root = os.environ.get("root_venv_dir", str(Path.home()))
    return str(Path(root) / ".omniax_venvs" / version / arch)


def _requirements_hash() -> str:
    """Return the combined md5 of both requirements files."""
    digest = hashlib.md5()
    for requirements_file in (
        SCRIPT_DIR / "requirements.txt",
        SCRIPT_DIR / "test_requirements.txt",
    ):
        digest.update(requirements_file.read_bytes())
    return digest.hexdigest()


def _hash_is_different(hash_file: Path, required_hash: str) -> bool:
    """Return True when the stored hash differs from ``required_hash``."""
    if not hash_file.exists():
        return True
    try:
        return hash_file.read_text().strip() != required_hash
    except OSError:
        return True


def setup_environment() -> int:
    """Prepare the Python environment (venv, requirements) like bash did."""
    requirements = SCRIPT_DIR / "requirements.txt"
    test_requirements = SCRIPT_DIR / "test_requirements.txt"

    if not requirements.exists():
        print(f"The file {requirements} doesn't exist.", file=sys.stderr)
        return 21
    if not test_requirements.exists():
        print(f"The file {test_requirements} doesn't exist.", file=sys.stderr)
        return 21

    modules: list[str] = []
    for line in requirements.read_text().splitlines():
        if line.strip():
            modules.append(line.strip())
    if os.environ.get("install_tests"):
        for line in test_requirements.read_text().splitlines():
            if line.strip():
                modules.append(line.strip())

    install_those.clear()
    install_those.extend(modules)

    os.environ["RUN_VIA_RUNSH"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    if _check_required_programs() != 0:
        return 11

    if shutil.which("python3") is None:
        red_text("python3 not installed. Cannot continue.")
        return 245

    if shutil.which("python3"):
        version_result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; print(1 if sys.version_info >= (3, 10) else 0)",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if version_result.stdout.strip() != "1":
            echo_red(
                f"⚠️  Warning: Python version is less than 3.10. Detected: "
                f"{sys.version}. This may cause issues."
            )

    venv_dir = os.environ.get("VIRTUAL_ENV") or _venv_dir()
    os.environ["VENV_DIR"] = venv_dir
    os.environ["CUSTOM_VIRTUAL_ENV"] = (
        "1" if os.environ.get("VIRTUAL_ENV") else "0"
    )
    os.environ["OMNIAX_VENV_DIR"] = venv_dir

    venv_python = str(Path(venv_dir) / "bin" / "python")

    if not os.environ.get("VIRTUAL_ENV") and not Path(venv_dir).is_dir():
        green_reset_line(
            f"➤Environment {venv_dir} was not found. Creating it..."
        )
        result = subprocess.run(
            [sys.executable, "-m", "venv", venv_dir],
            check=False,
        )
        if result.returncode != 0:
            red_text(f"❌Failed to create Virtual Environment in {venv_dir}\n")
            return 20

        green_reset_line(f"✅Virtual Environment {venv_dir} created.")
        subprocess.run(
            [
                venv_python,
                "-m",
                "pip",
                "--default-timeout=300",
                "--disable-pip-version-check",
                "install",
                "-q",
                "pip==24.0",
            ],
            check=False,
        )
        green_reset_line(
            "✅Virtual Environment created. Now installing software. "
            "This may take some time."
        )

    if (
        not os.environ.get("DONT_INSTALL_MODULES")
        and not os.environ.get("SLURM_JOB_ID")
    ):
        main_hash = _requirements_hash()
        hash_file = Path(venv_dir) / "hash"
        hash_test_file = Path(venv_dir) / "hash_test"

        if _hash_is_different(
            hash_file, main_hash
        ) or _hash_is_different(hash_test_file, main_hash):
            result = install_required_modules()
            if result != 0:
                return result
            hash_file.write_text(main_hash)
            hash_test_file.write_text(main_hash)

    os.environ["PYTHONPATH"] = f"{venv_dir}:{os.environ.get('PYTHONPATH', '')}"
    return 0


if __name__ == "__main__":
    function_name = sys.argv[1] if len(sys.argv) > 1 else ""

    functions: dict[str, Callable[..., object]] = {
        "displaytime": displaytime,
        "generate_progress_bar": generate_progress_bar,
    }

    if function_name not in functions:
        print(
            f"Unknown shellscript function: {function_name}",
            file=sys.stderr,
        )
        sys.exit(1)

    result = functions[function_name](*[int(arg) for arg in sys.argv[2:]])
    if isinstance(result, str):
        sys.stdout.write(result)
