#!/usr/bin/env python3
"""Run a PHP linter (phpcs)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _framework.helpers import red_text, green_text, yellow_text


REPO_ROOT = THIS_DIR.parent
GUI_DIR = REPO_ROOT / ".gui"

BASECOMMAND = (
    "phpcs --standard=PSR12 "
    "--exclude=Squiz.Commenting.FileComment,Generic.Files.LineLength,"
    "PSR2.ControlStructures.SwitchDeclaration,PSR12.Files.FileHeader -n -s"
)


def _check_or_install_phpcs() -> bool:
    if shutil.which("phpcs"):
        return True

    for needed in ("curl", "sudo", "php"):
        if not shutil.which(needed):
            red_text(f"{needed} is not installed. Cannot continue.")
            return False

    main_dir = Path("/tmp/composer_data")
    main_dir.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            ["curl", "-sS", "https://getcomposer.org/installer",
             "-o", "/tmp/composer-setup.php"],
            check=True,
        )
        subprocess.run(
            ["sudo", "php", "/tmp/composer-setup.php",
             "--install-dir", str(main_dir), "--filename", "composer"],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        red_text(f"Failed to install composer: {exc}")
        return False

    if not shutil.which("composer"):
        red_text("Cannot install php_codesniffer without composer being installed.")
        return False

    env = os.environ.copy()
    env["PATH"] = f"{main_dir}:{env.get('PATH', '')}"
    try:
        subprocess.run(
            ["composer", "global", "require", "squizlabs/php_codesniffer", "--dev"],
            env=env, check=True,
        )
    except subprocess.CalledProcessError:
        red_text("Failed to install squizlabs/php_codesniffer")
        return False

    return shutil.which("phpcs") is not None


def main(argv=None) -> int:
    if not _check_or_install_phpcs():
        return 255

    errors: list[str] = []
    args = sys.argv[1:]
    start = time.time()

    targets: list[Path] = []
    if args:
        for a in args:
            p = Path(a)
            if p.exists() and p.is_file():
                targets.append(p)
            else:
                red_text(f"File {a} not found, skipping.")
    else:
        for php_file in GUI_DIR.rglob("*.php"):
            if php_file.is_file():
                targets.append(php_file)

    for p in targets:
        yellow_text(f"{BASECOMMAND} {p}")
        proc = subprocess.run(
            f"{BASECOMMAND} {p}",
            shell=True,
            cwd=str(REPO_ROOT),
        )
        if proc.returncode != 0:
            errstr = f"Failed linting {p}: Run '{BASECOMMAND} {p}' to see details."
            red_text(errstr)
            errors.append(errstr)

    elapsed = int(time.time() - start)
    hrs = elapsed // 3600
    mins = (elapsed % 3600) // 60
    secs = elapsed % 60
    print(f"phpcs test took: {hrs:02d}:{mins:02d}:{secs:02d}")

    if not errors:
        green_text("No phpcs errors")
        return 0
    red_text("=> PHPCS-ERRORS =>")
    for e in errors:
        red_text(e)
    return len(errors)


if __name__ == "__main__":
    sys.exit(main())
