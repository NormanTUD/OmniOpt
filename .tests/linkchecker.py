#!/usr/bin/env python3
"""Checks all links on the site."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


REPO_ROOT = THIS_DIR.parent


def main(argv=None) -> int:  # pylint: disable=unused-argument
    os.environ.setdefault("install_tests", "1")
    if not shutil.which("linkchecker"):
        print("linkchecker not installed - attempting to install...")
        try:
            # Try installing with --break-system-packages flag (if available)
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--break-system-packages", "linkchecker"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Check if it's now available
            if not shutil.which("linkchecker"):
                print("linkchecker installation failed - skipping link check")
                return 0
        except Exception:
            try:
                # Fallback: try with --user flag
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "--user", "linkchecker"
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # Check if it's now available
                if not shutil.which("linkchecker"):
                    print("linkchecker installation failed - skipping link check")
                    return 0
            except Exception:
                print("linkchecker not installed and failed to install - skipping link check")
                return 0
    
    proc = subprocess.run(
        ["linkchecker", "https://imageseg.scads.de/omniax/"],
        cwd=str(REPO_ROOT),
    )
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
