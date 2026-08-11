#!/usr/bin/env python3
"""Smoke test that omniopt_share works inside the real Docker image.

This is not a full end-to-end (the main image doesn't ship php-zip);
it just verifies that:
  * the Python module imports inside the container
  * the CLI parses arguments correctly
  * the manifest builder produces a valid JSON manifest
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent


def _check(condition: bool, message: str) -> bool:
    if not condition:
        print(f"FAIL: {message}", file=sys.stderr)
        return False
    return True


def _docker_image_exists(image: str) -> bool:
    proc = subprocess.run(
        ["docker", "images", "-q", image],
        capture_output=True, text=True, check=False,
    )
    return bool(proc.stdout.strip())


def test_share_help_inside_docker() -> bool:
    if not _docker_image_exists("omniopt-omniopt2"):
        print("SKIP: omniopt-omniopt2 image not built")
        return True
    proc = subprocess.run(
        [
            "docker", "run", "--rm", "omniopt-omniopt2",
            "bash", "-c",
            "cd /var/opt/omniopt && ./omniopt_share --help",
        ],
        capture_output=True, text=True, timeout=60,
    )
    ok = _check(
        proc.returncode == 0,
        f"omniopt_share --help exited with {proc.returncode}: {proc.stderr}",
    )
    ok &= _check(
        "Usage:" in proc.stdout,
        f"--help output missing Usage: {proc.stdout[:200]!r}",
    )
    return ok


def test_share_manifest_inside_docker() -> bool:
    if not _docker_image_exists("omniopt-omniopt2"):
        print("SKIP: omniopt-omniopt2 image not built")
        return True
    host_run = Path("/tmp/oo_docker_share_test")
    if host_run.exists():
        import shutil
        shutil.rmtree(host_run)
    host_run.mkdir(parents=True)
    (host_run / "results.csv").write_text("a,b\n1,2\n")

    proc = subprocess.run(
        [
            "docker", "run", "--rm",
            "-v", "/tmp/oo_docker_share_test:/data:ro",
            "omniopt-omniopt2",
            "bash", "-c",
            "python3 -c "
            "\"from importlib.machinery import SourceFileLoader; "
            "m = SourceFileLoader('s', '/var/opt/omniopt/omniopt_share').load_module(); "
            "import json; "
            "mani = m.build_manifest(run_dir='/data', user_id='docker', "
            "experiment_name='demo', update=False, update_uuid=None, "
            "password=None, send_single_runs=False); "
            "print('OK', json.dumps({'schema': mani['schema_version'], "
            "'user': mani['user_id'], 'n_files': len(mani['files'])}))\"",
        ],
        capture_output=True, text=True, timeout=120,
    )
    ok = _check(
        proc.returncode == 0,
        f"manifest build inside docker failed: {proc.stderr}",
    )
    if "OK " in proc.stdout:
        line = proc.stdout.split("OK ", 1)[1].strip()
        try:
            data = json.loads(line)
            ok &= _check(
                data.get("schema") == "1.0",
                f"wrong schema: {data}",
            )
            ok &= _check(
                data.get("user") == "docker",
                f"wrong user: {data}",
            )
            ok &= _check(
                data.get("n_files", 0) >= 1,
                f"expected >=1 file, got {data}",
            )
        except json.JSONDecodeError as e:
            ok &= _check(False, f"could not parse manifest output: {e} ({line!r})")
    return ok


TESTS = [
    test_share_help_inside_docker,
    test_share_manifest_inside_docker,
]


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
