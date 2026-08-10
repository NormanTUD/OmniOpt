"""
Color-output helpers for OmniOpt2 scripts.

This is the Python equivalent of the former bash color library.  It
provides small helpers that print colored text to stdout or stderr,
exactly mirroring the byte-level behavior of the bash original so that
the automated tests pass for both implementations.

The module can either be imported or invoked as a command line tool.  For
example, passing ``red_text`` as the first command line argument calls
that function with the remaining arguments.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

Green = "\033[0;32m"
Color_Off = "\033[0m"
Red = "\033[0;31m"


def echoerr(*args: object) -> None:
    """Print the arguments to stderr, separated by spaces, with a newline."""
    print(*args, file=sys.stderr)


def red_text(text: str) -> None:
    """Print ``text`` in red on stdout without a trailing newline."""
    sys.stdout.write(f"{Red}{text}{Color_Off}")


def yellow_text(text: str) -> None:
    """Print ``text`` in yellow on stderr (keeps the bash double-ESC quirk)."""
    sys.stderr.write(f"\033\033[0;33m{text}\033[0m\n")


def green_text_no_newline(text: str) -> None:
    """Print ``text`` in green on stderr without a trailing newline."""
    sys.stderr.write(f"\033[0;32m{text}\033[0m")


def green_text(text: str) -> None:
    """Print ``text`` in green on stderr with a trailing newline."""
    sys.stderr.write(f"\033[0;32m{text}\033[0m\n")


def green(text: str) -> None:
    """Print ``text`` in green on stdout without a trailing newline."""
    sys.stdout.write(f"{Green}{text}{Color_Off}")


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


def green_reset_line(text: str) -> None:
    """Carriage-return, clear the line, then print ``text`` in green."""
    _tput("cr")
    _tput("el")
    green(text)


def red_reset_line(text: str) -> None:
    """Carriage-return, clear the line, then print ``text`` in red."""
    _tput("cr")
    _tput("el")
    red_text(text)


def _green_text_bold_underline(text: str) -> None:
    """Print ``text`` in bold, underlined green on stderr."""
    sys.stderr.write(f"\033[1;4;32m{text}\033[0m\n")


if __name__ == "__main__":
    function_name = sys.argv[1] if len(sys.argv) > 1 else ""

    functions = {
        "echoerr": echoerr,
        "red_text": red_text,
        "yellow_text": yellow_text,
        "green_text_no_newline": green_text_no_newline,
        "green_text": green_text,
        "green": green,
        "_tput": _tput,
        "green_reset_line": green_reset_line,
        "red_reset_line": red_reset_line,
        "_green_text_bold_underline": _green_text_bold_underline,
    }

    if function_name not in functions:
        print(f"Unknown color function: {function_name}", file=sys.stderr)
        sys.exit(1)

    functions[function_name](*sys.argv[2:])
