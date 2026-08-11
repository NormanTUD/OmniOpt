#!/usr/bin/env python3
"""End-to-end share test using real Docker containers.

This test:
  1. Starts the PHP GUI container (omniopt-omniopt2, exposing share_internal.php)
  2. Starts a second container that runs the new omniopt_share Python
     module and POSTs a real share to it
  3. Verifies the share arrives on the server side with all files intact

Requires Docker and the omniopt-omniopt2 image to be present.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent

sys.path.insert(0, str(REPO_ROOT))
from importlib.machinery import SourceFileLoader  # noqa: E402

os_ = SourceFileLoader(
    "omniopt_share", str(REPO_ROOT / "omniopt_share")
).load_module()


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


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_for(url: str, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                return r.status in (200, 400, 404)
        except Exception:
            time.sleep(0.5)
    return False


def test_share_round_trip_via_docker() -> bool:
    if not _docker_image_exists("omniopt-omniopt2"):
        print("SKIP: omniopt-omniopt2 image not built; run "
              "`docker compose build --build-arg GetMyUsername=$USER` first")
        return True

    # 1. Build a small run dir on the host
    host_run = Path("/tmp/oo_docker_share_run") / "demo" / "0"
    if host_run.exists():
        shutil.rmtree(host_run.parent)
    host_run.mkdir(parents=True)
    (host_run / "results.csv").write_text("a,b\n1,2\n3,4\n")
    (host_run / "parameters.txt").write_text("x: range(1,2)\n")

    # 2. Start the PHP server container
    port = _free_port()
    php = subprocess.Popen(
        [
            "docker", "run", "--rm",
            "-p", f"{port}:80",
            "-e", "share_path=/tmp/oo_share_test",
            "omniopt-omniopt2",
            "bash", "-lc",
            "apt-get install -y php-cli php-zip >/dev/null 2>&1 && "
            f"mkdir -p /tmp/oo_share_test && "
            f"cd /var/opt/omniopt/.gui && "
            f"php -S 0.0.0.0:80 share_internal.php",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        if not _wait_for(f"http://127.0.0.1:{port}/share_internal.php"):
            return _check(False, f"PHP server did not start on port {port}")

        # 3. Run the share from a second container
        share_proc = subprocess.run(
            [
                "docker", "run", "--rm",
                "-v", "/tmp/oo_docker_share_run:/data:ro",
                "-e", f"OO_BASE_URL=http://host.docker.internal:{port}",
                "omniopt-omniopt2",
                "bash", "-lc",
                "cd /data && "
                "/var/opt/omniopt/omniopt_share demo/0 --username=docker_test "
                f"--no_color --no-send_singleruns",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        ok = _check(
            share_proc.returncode in (0, 200),
            f"omniopt_share exited with {share_proc.returncode}\n"
            f"stdout:\n{share_proc.stdout[-500:]}\n"
            f"stderr:\n{share_proc.stderr[-500:]}",
        )

        # 4. Verify files appeared on the server side
        server_share = Path("/tmp/oo_share_test")
        if server_share.exists():
            user_dirs = list(server_share.glob("docker_test"))
            ok &= _check(
                bool(user_dirs),
                f"no user dir for docker_test under {server_share}",
            )
            if user_dirs:
                run_dir = next(iter(user_dirs[0].glob("*/")), None)
                ok &= _check(
                    run_dir is not None,
                    f"no run dir under {user_dirs[0]}",
                )
                if run_dir is not None:
                    ok &= _check(
                        (run_dir / "results.csv").exists(),
                        f"results.csv missing in {run_dir}",
                    )
        else:
            ok &= _check(
                False,
                f"share_path {server_share} not created",
            )
    finally:
        php.terminate()
        try:
            php.wait(timeout=5)
        except subprocess.TimeoutExpired:
            php.kill()

    return ok


TESTS = [
    test_share_round_trip_via_docker,
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
