"""
General helpers for OmniOpt2 scripts.

This is the Python equivalent of the former general bash library.  It
provides small helpers for colored console output, error dialogs and a
temporary local webserver, mirroring the behavior of the bash original.

The module can either be imported or invoked as a command line tool.  For
example, passing ``echo_red`` as the first command line argument calls
that function with the remaining arguments.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path


def debug_code(msg: str) -> None:
    """Print ``msg`` in yellow to stderr, but only when ``DEBUG=1``."""
    if os.environ.get("DEBUG") == "1":
        sys.stderr.write(f"\033[93m{msg}\033[0m\n")


def echo_red(msg: str) -> None:
    """Print ``msg`` in red to stdout with a trailing newline."""
    sys.stdout.write(f"\033[31m{msg}\033[0m\n")


def echo_green(msg: str) -> None:
    """Print ``msg`` in green to stdout with a trailing newline."""
    sys.stdout.write(f"\033[32m{msg}\033[0m\n")


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


def _find_free_port() -> int:
    """Ask the OS for a currently free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def _hostname_for_forwarding() -> str:
    """Return the hostname with any cluster domain removed."""
    hostname = socket.gethostname()
    return hostname.replace(".taurus.", ".")


def spin_up_temporary_webserver(directory: str, download_path: str) -> int:
    """Serve ``directory`` over a local web server and show a dialog."""
    base_dir = Path(directory).resolve()
    if not base_dir.is_dir():
        error_message(f"Directory not found: {directory}")
        return 1

    free_port = _find_free_port()
    server = subprocess.Popen(
        [sys.executable, "-u", "-m", "http.server", str(free_port)],
        cwd=base_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    hostname = _hostname_for_forwarding()
    user = os.environ.get("USER", "")
    port_forwarding_command = (
        f"ssh -f -N -L {free_port}:{hostname}:{free_port} {user}@{hostname}; "
        f"sensible-browser http://localhost:{free_port}/{download_path}"
    )

    custom_text = (
        f"The webserver has started. Run this command locally to forward "
        f"the port {free_port} to your local system:\n\n"
        f"{port_forwarding_command}\n\n"
        f"When you close this window by clicking OK, the server will be "
        "shut down."
    )

    try:
        if os.environ.get("DISPLAY"):
            xclip = Path("tools") / "xclip"
            if _is_x86_64() and xclip.exists():
                _copy_to_clipboard(port_forwarding_command, xclip)
                copied_text = custom_text + (
                    "\n\nThis command has already been copied to your "
                    "clipboard. Paste it locally on your machine to view "
                    "the file(s)."
                )
                _show_dialog("Webserver", copied_text)
            elif shutil.which("zenity") is not None:
                subprocess.run(
                    ["zenity", "--info", "--width=800", "--height=600",
                     "--text", custom_text],
                    check=False,
                )
            else:
                _show_dialog("Webserver", custom_text)
        else:
            _show_dialog("Webserver", custom_text)
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()

    return 0


def _is_x86_64() -> bool:
    """Check whether the architecture is x86_64."""
    return os.uname().machine == "x86_64"


def _copy_to_clipboard(text: str, xclip: Path) -> None:
    """Copy ``text`` into the clipboard using the bundled xclip binary."""
    subprocess.run(
        [str(xclip), "-sel", "clip"],
        input=text,
        text=True,
        check=False,
    )


def _show_dialog(title: str, text: str) -> None:
    """Show a whiptail message box if whiptail is available."""
    if not sys.stdout.isatty() or shutil.which("whiptail") is None:
        return
    lines, columns = _terminal_size()
    subprocess.run(
        [
            "whiptail",
            "--title",
            title,
            "--msgbox",
            text,
            str(lines),
            str(columns),
            str(max(0, lines - 8)),
        ],
        check=False,
    )


if __name__ == "__main__":
    function_name = sys.argv[1] if len(sys.argv) > 1 else ""

    functions: dict[str, Callable[..., object]] = {
        "debug_code": debug_code,
        "echo_red": echo_red,
        "echo_green": echo_green,
        "error_message": error_message,
        "spin_up_temporary_webserver": spin_up_temporary_webserver,
    }

    if function_name not in functions:
        print(f"Unknown general function: {function_name}", file=sys.stderr)
        sys.exit(1)

    functions[function_name](*sys.argv[2:])
