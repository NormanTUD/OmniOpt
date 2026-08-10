#!/usr/bin/env python3
"""In OmniOpt, there is a parser that allows to parse argparse-like arguments
before python. This tests its output."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _framework.helpers import green_text, red_text


REPO_ROOT = THIS_DIR.parent


def main(argv=None) -> int:
    os.environ["DEBUG_PARAM_EVAL"] = "1"
    os.environ["DONT_SHOW_STARTUP_COMMAND"] = "1"

    cmd = (
        f"{REPO_ROOT}/.tests/start_simple_optimization_run.py "
        "--num_parallel_jobs=2 --gpus=0 --num_random_steps=1 --max_eval=1 "
        "--mem_gb=1 --generate_all_jobs_at_once --follow"
    )
    proc = subprocess.run(cmd, shell=True, cwd=str(REPO_ROOT))

    if proc.returncode == 222:
        green_text("Test test_bash_argparse_clone OK")
        return 0
    red_text("Test test_bash_argparse_clone failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
