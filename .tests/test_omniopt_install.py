#!/usr/bin/env python3
"""Tests for the --install flag in omniopt.

`omniopt --install` is invoked by the Dockerfile to bootstrap the
container's Python dependencies.  It must:

  * exit 0 without any other CLI args
  * skip the argument-validation that requires ``--max_eval``,
    ``--generation_strategy`` or a continuation job
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent


def test_omniopt_install_exits_zero() -> bool:
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "omniopt"), "--install"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    return _check(
        proc.returncode == 0,
        f"--install should exit 0, got {proc.returncode}\nstderr:\n{proc.stderr[-500:]}",
    )


TESTS = [
    test_omniopt_install_exits_zero,
]


def _check(condition: bool, message: str) -> bool:
    if not condition:
        print(f"FAIL: {message}", file=sys.stderr)
        return False
    return True


def main(argv=None) -> int:
    failures = 0
    for t in TESTS:
        print(f"running {t.__name__} ...", end=" ", flush=True)
        try:
            ok = t()
        except Exception as e:
            ok = False
            print(f"\n  EXCEPTION: {e!r}")
        if ok:
            print("ok")
        else:
            failures += 1
            print("FAIL")
    if failures:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
