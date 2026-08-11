#!/usr/bin/env python3
"""Integration tests for the new JSON-Manifest + ZIP share protocol.

These tests spin up the PHP built-in server with ``share_internal.php``
as the router, simulate an upload via Python's urllib, and check that
the server-side share directory contains the expected files with the
expected contents and names.
"""

from __future__ import annotations

import http.client
import json
import mimetypes
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import uuid
import zipfile
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent

sys.path.insert(0, str(REPO_ROOT))
from importlib.machinery import SourceFileLoader  # noqa: E402

os_ = SourceFileLoader(
    "omniopt_share", str(REPO_ROOT / "omniopt_share")
).load_module()

from _framework.helpers import red_text  # noqa: E402


def _check(condition: bool, message: str) -> bool:
    if not condition:
        red_text(f"FAIL: {message}")
        return False
    return True


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_for_server(url: str, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                return r.status in (200, 400, 404)
        except (urllib.error.URLError, ConnectionRefusedError):
            time.sleep(0.1)
    return False


def _make_run_dir(root: Path) -> Path:
    run = root / "share_it_experiment" / "0"
    run.mkdir(parents=True)
    (run / "results.csv").write_text("a,b\n1,2\n")
    (run / "parameters.txt").write_text("x: range\n")
    sf = run / "state_files"
    sf.mkdir()
    (sf / "run_uuid").write_text(f"test-{uuid.uuid4()}")
    return run


def _post_share(port: int, manifest: dict, zip_path: Path) -> tuple[int, str]:
    boundary = "----oo-test-" + uuid.uuid4().hex
    body = b""
    body += f"--{boundary}\r\n".encode()
    body += b'Content-Disposition: form-data; name="manifest"\r\n'
    body += b"Content-Type: application/json\r\n\r\n"
    body += json.dumps(manifest).encode()
    body += b"\r\n"
    body += f"--{boundary}\r\n".encode()
    body += b'Content-Disposition: form-data; name="bundle"; filename="bundle.zip"\r\n'
    body += b"Content-Type: application/zip\r\n\r\n"
    body += zip_path.read_bytes()
    body += b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    conn.request(
        "POST",
        f"/share_internal.php?user_id={manifest['user_id']}"
        f"&experiment_name={manifest['experiment_name']}",
        body=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    resp = conn.getresponse()
    body_text = resp.read().decode("utf-8", errors="replace")
    conn.close()
    return resp.status, body_text


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_php_share_internal_parses() -> bool:
    proc = subprocess.run(
        ["php", "-l", str(REPO_ROOT / ".gui" / "share_internal.php")],
        capture_output=True, text=True,
    )
    return _check(
        proc.returncode == 0 and "No syntax errors" in proc.stdout,
        f"php -l failed: {proc.stdout} {proc.stderr}",
    )


def test_share_end_to_end() -> bool:
    if not shutil.which("php"):
        return _check(False, "php is not installed")
    port = _free_port()
    share_root = Path(tempfile.mkdtemp(prefix="oo_share_int_"))
    share_path = share_root / "shares"
    share_path.mkdir()

    env = os.environ.copy()
    env["share_path"] = str(share_path)

    server = subprocess.Popen(
        [
            "php", "-S", f"127.0.0.1:{port}",
            "-t", str(REPO_ROOT / ".gui"),
            str(REPO_ROOT / ".gui" / "share_internal.php"),
        ],
        cwd=str(REPO_ROOT / ".gui"),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        if not _wait_for_server(f"http://127.0.0.1:{port}/share_internal.php"):
            return _check(False, "PHP server failed to start")

        with tempfile.TemporaryDirectory() as tmp:
            run = _make_run_dir(Path(tmp))
            manifest = os_.build_manifest(
                run_dir=str(run),
                user_id="alice",
                experiment_name="share_it_experiment",
                update=False,
                update_uuid=None,
                password=None,
                send_single_runs=False,
            )
            out_dir = Path(tmp) / "out"
            out_dir.mkdir()
            _, zip_path = os_.write_bundle(manifest, out_dir, source_dir=str(run))

            status, body = _post_share(port, manifest, zip_path)
            ok = _check(
                "Error" not in body,
                f"share_internal.php returned error: status={status} body={body!r}",
            )
            # Server-side share dir should now have the user/experiment/run
            user_dir = share_path / "alice" / "share_it_experiment"
            ok &= _check(user_dir.exists(), f"server did not create {user_dir}")
            if user_dir.exists():
                run_dir_candidates = list(user_dir.iterdir())
                ok &= _check(
                    len(run_dir_candidates) >= 1,
                    f"server did not create a run dir: {run_dir_candidates}",
                )
                if run_dir_candidates:
                    run_dir = run_dir_candidates[0]
                    contents = {p.name for p in run_dir.iterdir()}
                    ok &= _check(
                        "results.csv" in contents,
                        f"results.csv missing on server: {contents}",
                    )
                    ok &= _check(
                        "parameters.txt" in contents,
                        f"parameters.txt missing on server: {contents}",
                    )
                    if "results.csv" in contents:
                        ok &= _check(
                            (run_dir / "results.csv").read_text()
                            == (run / "results.csv").read_text(),
                            "results.csv contents differ",
                        )
    finally:
        server.send_signal(signal.SIGTERM)
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
        shutil.rmtree(share_root, ignore_errors=True)
    return ok


def test_share_rejects_unsafe_archive_path() -> bool:
    if not shutil.which("php"):
        return _check(False, "php is not installed")
    port = _free_port()
    share_root = Path(tempfile.mkdtemp(prefix="oo_share_int_"))
    share_path = share_root / "shares"
    share_path.mkdir()

    env = os.environ.copy()
    env["share_path"] = str(share_path)

    server = subprocess.Popen(
        [
            "php", "-S", f"127.0.0.1:{port}",
            "-t", str(REPO_ROOT / ".gui"),
            str(REPO_ROOT / ".gui" / "share_internal.php"),
        ],
        cwd=str(REPO_ROOT / ".gui"),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        if not _wait_for_server(f"http://127.0.0.1:{port}/share_internal.php"):
            return _check(False, "PHP server failed to start")

        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "evil.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("../../etc/passwd", "evil\n")
            manifest = {
                "schema_version": os_.MANIFEST_SCHEMA_VERSION,
                "user_id": "evil",
                "experiment_name": "evil",
                "update": False,
                "update_uuid": None,
                "password": None,
                "files": [
                    {
                        "name": "x",
                        "archive_path": "../../etc/passwd",
                        "size": 5,
                        "sha256": "0" * 64,
                        "content_type": "text/plain",
                    }
                ],
            }
            status, body = _post_share(port, manifest, zip_path)
            ok = _check(
                "Error" in body and "unsafe" in body.lower(),
                f"expected rejection, got status={status} body={body!r}",
            )
            ok &= _check(
                not (share_path / "evil").exists(),
                "server must NOT create the evil user dir",
            )
    finally:
        server.send_signal(signal.SIGTERM)
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
        shutil.rmtree(share_root, ignore_errors=True)
    return ok


def test_share_rejects_bad_sha() -> bool:
    if not shutil.which("php"):
        return _check(False, "php is not installed")
    port = _free_port()
    share_root = Path(tempfile.mkdtemp(prefix="oo_share_int_"))
    share_path = share_root / "shares"
    share_path.mkdir()

    env = os.environ.copy()
    env["share_path"] = str(share_path)

    server = subprocess.Popen(
        [
            "php", "-S", f"127.0.0.1:{port}",
            "-t", str(REPO_ROOT / ".gui"),
            str(REPO_ROOT / ".gui" / "share_internal.php"),
        ],
        cwd=str(REPO_ROOT / ".gui"),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        if not _wait_for_server(f"http://127.0.0.1:{port}/share_internal.php"):
            return _check(False, "PHP server failed to start")

        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "bad.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("results.csv", "a,b\n1,2\n")
            manifest = {
                "schema_version": os_.MANIFEST_SCHEMA_VERSION,
                "user_id": "alice",
                "experiment_name": "e",
                "update": False,
                "update_uuid": None,
                "password": None,
                "files": [
                    {
                        "name": "results",
                        "archive_path": "results.csv",
                        "size": 8,
                        "sha256": "0" * 64,  # WRONG hash
                        "content_type": "text/csv",
                    }
                ],
            }
            status, body = _post_share(port, manifest, zip_path)
            ok = _check(
                "sha256" in body.lower() or "mismatch" in body.lower(),
                f"expected sha256 rejection, got {body!r}",
            )
    finally:
        server.send_signal(signal.SIGTERM)
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
        shutil.rmtree(share_root, ignore_errors=True)
    return ok


TESTS = [
    test_php_share_internal_parses,
    test_share_end_to_end,
    test_share_rejects_unsafe_archive_path,
    test_share_rejects_bad_sha,
]


def main(argv=None) -> int:
    failures = 0
    for t in TESTS:
        print(f"running {t.__name__} ...", end=" ", flush=True)
        try:
            ok = t()
        except Exception as e:
            ok = False
            red_text(f"\n  EXCEPTION: {e!r}")
        if ok:
            print("ok")
        else:
            failures += 1
            print("FAIL")
    if failures:
        red_text(f"\n{failures} test(s) failed")
        return 1
    print("\nAll share integration tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
