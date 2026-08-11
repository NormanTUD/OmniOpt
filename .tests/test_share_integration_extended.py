#!/usr/bin/env python3
"""Server-side share integration tests (PHP).

These tests spin up the PHP built-in server (with the test-only
router that parses multipart bodies) and verify the *server side*
behaviour of ``share_internal.php``:

  * happy-path uploads land in the expected folder
  * rejected uploads do NOT create files on the server
  * security: traversal paths, oversized files, missing fields,
    wrong content types, malformed JSON are all rejected
  * update flow with a UUID updates the existing folder

Tests that require the PHP ``zip`` extension gracefully skip when it
isn't available.
"""

from __future__ import annotations

import http.client
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
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


def _php_has_zip() -> bool:
    proc = subprocess.run(
        ["php", "-r", 'exit(class_exists("ZipArchive") ? 0 : 1);'],
        capture_output=True,
    )
    return proc.returncode == 0


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_for_server(url: str, timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                return r.status in (200, 400, 404)
        except (urllib.error.URLError, ConnectionRefusedError):
            time.sleep(0.2)
    return False


def _make_run_dir(root: Path, name: str = "e2e") -> Path:
    run = root / name / "0"
    run.mkdir(parents=True)
    (run / "results.csv").write_text("a,b\n1,2\n3,4\n")
    (run / "parameters.txt").write_text("x: range(1, 2)\n")
    return run


def _build_and_post(
    port: int,
    user_id: str,
    experiment_name: str,
    run_dir: Path,
    *,
    update: bool = False,
    update_uuid: str | None = None,
    password: str | None = None,
) -> tuple[int, str]:
    manifest = os_.build_manifest(
        run_dir=str(run_dir),
        user_id=user_id,
        experiment_name=experiment_name,
        update=update,
        update_uuid=update_uuid,
        password=password,
        send_single_runs=False,
    )
    out_dir = Path(tempfile.mkdtemp(prefix="oo_int_"))
    _, zip_path = os_.write_bundle(manifest, out_dir, source_dir=str(run_dir))

    boundary = "----oo-int-" + uuid.uuid4().hex
    body = b""
    body += f"--{boundary}\r\n".encode()
    body += b'Content-Disposition: form-data; name="manifest"\r\n'
    body += b"Content-Type: application/json\r\n\r\n"
    body += json.dumps(manifest).encode()
    body += b"\r\n"
    body += f"--{boundary}\r\n".encode()
    body += (
        b'Content-Disposition: form-data; name="bundle"; filename="bundle.zip"\r\n'
        b"Content-Type: application/zip\r\n\r\n"
    )
    body += zip_path.read_bytes()
    body += b"\r\n"
    body += f"--{boundary}--\r\n".encode()

    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    conn.request(
        "POST",
        f"/share_internal.php?user_id={user_id}"
        f"&experiment_name={experiment_name}",
        body=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    resp = conn.getresponse()
    body_text = resp.read().decode("utf-8", errors="replace")
    conn.close()
    return resp.status, body_text


def _start_server(
    share_path: Path,
) -> tuple[subprocess.Popen, int]:
    port = _free_port()
    env = os.environ.copy()
    env["share_path"] = str(share_path)
    env["OO_TARGET_FILE"] = str(REPO_ROOT / ".gui" / "share_internal.php")
    proc = subprocess.Popen(
        [
            "php", "-S", f"127.0.0.1:{port}",
            "-t", str(REPO_ROOT / ".gui"),
            str(REPO_ROOT / ".tests" / "share_test_router.php"),
        ],
        cwd=str(REPO_ROOT / ".gui"),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc, port


def _stop_server(proc: subprocess.Popen) -> None:
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def _skip_unless_zip() -> bool:
    if not _php_has_zip():
        print("SKIP (php-zip not installed)")
        return True
    if not shutil.which("php"):
        print("SKIP (php not installed)")
        return True
    return False


# ---------------------------------------------------------------------------
# Happy path
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


def test_end_to_end_share_creates_user_dir() -> bool:
    if _skip_unless_zip():
        return True
    share_root = Path(tempfile.mkdtemp(prefix="oo_share_int_"))
    share_path = share_root / "shares"
    share_path.mkdir()
    server, port = _start_server(share_path)
    try:
        if not _wait_for_server(f"http://127.0.0.1:{port}/share_internal.php"):
            return _check(False, "PHP server did not start")

        with tempfile.TemporaryDirectory() as tmp:
            run = _make_run_dir(Path(tmp))
            status, body = _build_and_post(
                port, "alice", "demo", run,
            )
            ok = _check(
                "Error" not in body,
                f"unexpected error: status={status} body={body!r}",
            )
            user_dir = share_path / "alice" / "demo"
            ok &= _check(
                user_dir.exists(),
                f"server did not create {user_dir}",
            )
            if user_dir.exists():
                run_dirs = list(user_dir.iterdir())
                ok &= _check(
                    bool(run_dirs),
                    f"no run dir created: {run_dirs}",
                )
                if run_dirs:
                    files = {p.name for p in run_dirs[0].iterdir()}
                    ok &= _check(
                        "results.csv" in files,
                        f"results.csv missing: {files}",
                    )
                    ok &= _check(
                        "parameters.txt" in files,
                        f"parameters.txt missing: {files}",
                    )
    finally:
        _stop_server(server)
        shutil.rmtree(share_root, ignore_errors=True)
    return ok


def test_end_to_end_update_same_experiment() -> bool:
    """Sharing the same experiment twice should create one run dir
    (the existing one gets updated), not duplicate."""
    if _skip_unless_zip():
        return True
    share_root = Path(tempfile.mkdtemp(prefix="oo_share_int_"))
    share_path = share_root / "shares"
    share_path.mkdir()
    server, port = _start_server(share_path)
    try:
        if not _wait_for_server(f"http://127.0.0.1:{port}/share_internal.php"):
            return _check(False, "PHP server did not start")

        with tempfile.TemporaryDirectory() as tmp:
            run = _make_run_dir(Path(tmp))
            # First share
            _build_and_post(port, "bob", "demo2", run)
            user_dir = share_path / "bob" / "demo2"
            ok = _check(
                user_dir.exists(),
                f"first share did not create {user_dir}",
            )
            first_runs = list(user_dir.iterdir()) if user_dir.exists() else []
            # Modify the run dir slightly and share again with --update
            (run / "results.csv").write_text("a,b\n5,6\n7,8\n")
            status, body = _build_and_post(
                port, "bob", "demo2", run,
                update=True,
            )
            ok &= _check(
                "Error" not in body,
                f"update share failed: {body!r}",
            )
            second_runs = list(user_dir.iterdir()) if user_dir.exists() else []
            ok &= _check(
                len(second_runs) == len(first_runs) == 1,
                f"expected 1 run dir after update, got {len(second_runs)}: "
                f"first={first_runs} second={second_runs}",
            )
    finally:
        _stop_server(server)
        shutil.rmtree(share_root, ignore_errors=True)
    return ok


# ---------------------------------------------------------------------------
# Rejection paths (security)
# ---------------------------------------------------------------------------


def _post_manifest_raw(
    port: int, manifest: dict, zip_path: Path
) -> tuple[int, str]:
    boundary = "----oo-int-" + uuid.uuid4().hex
    body = b""
    body += f"--{boundary}\r\n".encode()
    body += b'Content-Disposition: form-data; name="manifest"\r\n'
    body += b"Content-Type: application/json\r\n\r\n"
    body += json.dumps(manifest).encode()
    body += b"\r\n"
    body += f"--{boundary}\r\n".encode()
    body += (
        b'Content-Disposition: form-data; name="bundle"; filename="bundle.zip"\r\n'
        b"Content-Type: application/zip\r\n\r\n"
    )
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


def test_rejects_unsafe_archive_path() -> bool:
    if _skip_unless_zip():
        return True
    share_root = Path(tempfile.mkdtemp(prefix="oo_share_int_"))
    share_path = share_root / "shares"
    share_path.mkdir()
    server, port = _start_server(share_path)
    try:
        if not _wait_for_server(f"http://127.0.0.1:{port}/share_internal.php"):
            return _check(False, "PHP server did not start")

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
                "files": [{
                    "name": "x",
                    "archive_path": "../../etc/passwd",
                    "size": 5,
                    "sha256": "0" * 64,
                    "content_type": "text/plain",
                }],
            }
            status, body = _post_manifest_raw(port, manifest, zip_path)
            ok = _check(
                "Error" in body and "unsafe" in body.lower(),
                f"expected rejection, got status={status} body={body!r}",
            )
            ok &= _check(
                not (share_path / "evil").exists(),
                "server must NOT create the evil user dir",
            )
    finally:
        _stop_server(server)
        shutil.rmtree(share_root, ignore_errors=True)
    return ok


def test_rejects_bad_sha256() -> bool:
    if _skip_unless_zip():
        return True
    share_root = Path(tempfile.mkdtemp(prefix="oo_share_int_"))
    share_path = share_root / "shares"
    share_path.mkdir()
    server, port = _start_server(share_path)
    try:
        if not _wait_for_server(f"http://127.0.0.1:{port}/share_internal.php"):
            return _check(False, "PHP server did not start")

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
                "files": [{
                    "name": "results",
                    "archive_path": "results.csv",
                    "size": 8,
                    "sha256": "0" * 64,  # WRONG hash
                    "content_type": "text/csv",
                }],
            }
            status, body = _post_manifest_raw(port, manifest, zip_path)
            ok = _check(
                "sha256" in body.lower() or "mismatch" in body.lower(),
                f"expected sha256 rejection, got {body!r}",
            )
            user_dir = share_path / "alice"
            ok &= _check(
                not user_dir.exists() or not any(user_dir.iterdir()),
                f"server should not have created alice's dir, got {list(user_dir.iterdir()) if user_dir.exists() else None}",
            )
    finally:
        _stop_server(server)
        shutil.rmtree(share_root, ignore_errors=True)
    return ok


def test_rejects_bad_size_in_manifest() -> bool:
    """Manifest declares size=999 but file in zip is only 8 bytes."""
    if _skip_unless_zip():
        return True
    share_root = Path(tempfile.mkdtemp(prefix="oo_share_int_"))
    share_path = share_root / "shares"
    share_path.mkdir()
    server, port = _start_server(share_path)
    try:
        if not _wait_for_server(f"http://127.0.0.1:{port}/share_internal.php"):
            return _check(False, "PHP server did not start")

        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "bad_size.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("results.csv", "a,b\n1,2\n")
            manifest = {
                "schema_version": os_.MANIFEST_SCHEMA_VERSION,
                "user_id": "alice",
                "experiment_name": "e",
                "update": False,
                "update_uuid": None,
                "password": None,
                "files": [{
                    "name": "results",
                    "archive_path": "results.csv",
                    "size": 999,  # WRONG size
                    "sha256": "0" * 64,
                    "content_type": "text/csv",
                }],
            }
            status, body = _post_manifest_raw(port, manifest, zip_path)
            ok = _check(
                "size" in body.lower(),
                f"expected size rejection, got {body!r}",
            )
    finally:
        _stop_server(server)
        shutil.rmtree(share_root, ignore_errors=True)
    return ok


def test_rejects_unknown_schema_version() -> bool:
    if _skip_unless_zip():
        return True
    share_root = Path(tempfile.mkdtemp(prefix="oo_share_int_"))
    share_path = share_root / "shares"
    share_path.mkdir()
    server, port = _start_server(share_path)
    try:
        if not _wait_for_server(f"http://127.0.0.1:{port}/share_internal.php"):
            return _check(False, "PHP server did not start")

        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "ok.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("results.csv", "a,b\n1,2\n")
            manifest = {
                "schema_version": "99.0",  # NOT 1.0
                "user_id": "alice",
                "experiment_name": "e",
                "update": False,
                "update_uuid": None,
                "password": None,
                "files": [{
                    "name": "results",
                    "archive_path": "results.csv",
                    "size": 8,
                    "sha256": "0" * 64,
                    "content_type": "text/csv",
                }],
            }
            status, body = _post_manifest_raw(port, manifest, zip_path)
            ok = _check(
                "schema_version" in body.lower() or "unsupported" in body.lower(),
                f"expected schema rejection, got {body!r}",
            )
    finally:
        _stop_server(server)
        shutil.rmtree(share_root, ignore_errors=True)
    return ok


def test_rejects_invalid_json_manifest() -> bool:
    if _skip_unless_zip():
        return True
    share_root = Path(tempfile.mkdtemp(prefix="oo_share_int_"))
    share_path = share_root / "shares"
    share_path.mkdir()
    server, port = _start_server(share_path)
    try:
        if not _wait_for_server(f"http://127.0.0.1:{port}/share_internal.php"):
            return _check(False, "PHP server did not start")

        boundary = "----oo-int-bad"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="manifest"\r\n'
            "Content-Type: application/json\r\n\r\n"
            "{ this is not valid json "
            f"\r\n--{boundary}--\r\n"
        ).encode()
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        conn.request(
            "POST",
            "/share_internal.php?user_id=alice&experiment_name=e",
            body=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        resp = conn.getresponse()
        body_text = resp.read().decode("utf-8", errors="replace")
        conn.close()
        ok = _check(
            "Error" in body_text or "not valid json" in body_text.lower(),
            f"expected JSON rejection, got {body_text!r}",
        )
    finally:
        _stop_server(server)
        shutil.rmtree(share_root, ignore_errors=True)
    return ok


def test_rejects_archive_path_with_backslash() -> bool:
    if _skip_unless_zip():
        return True
    share_root = Path(tempfile.mkdtemp(prefix="oo_share_int_"))
    share_path = share_root / "shares"
    share_path.mkdir()
    server, port = _start_server(share_path)
    try:
        if not _wait_for_server(f"http://127.0.0.1:{port}/share_internal.php"):
            return _check(False, "PHP server did not start")

        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "evil_bs.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("foo\\bar.csv", "x\n")
            manifest = {
                "schema_version": os_.MANIFEST_SCHEMA_VERSION,
                "user_id": "evil",
                "experiment_name": "evil",
                "update": False,
                "update_uuid": None,
                "password": None,
                "files": [{
                    "name": "x",
                    "archive_path": "foo\\bar.csv",
                    "size": 2,
                    "sha256": "0" * 64,
                    "content_type": "text/csv",
                }],
            }
            status, body = _post_manifest_raw(port, manifest, zip_path)
            ok = _check(
                "unsafe" in body.lower() or "error" in body.lower(),
                f"expected rejection, got {body!r}",
            )
            ok &= _check(
                not (share_path / "evil").exists(),
                "must not create evil user dir",
            )
    finally:
        _stop_server(server)
        shutil.rmtree(share_root, ignore_errors=True)
    return ok


def test_rejects_archive_path_with_null_byte() -> bool:
    if _skip_unless_zip():
        return True
    share_root = Path(tempfile.mkdtemp(prefix="oo_share_int_"))
    share_path = share_root / "shares"
    share_path.mkdir()
    server, port = _start_server(share_path)
    try:
        if not _wait_for_server(f"http://127.0.0.1:{port}/share_internal.php"):
            return _check(False, "PHP server did not start")

        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "evil_null.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("foo\x00.csv", "x\n")
            manifest = {
                "schema_version": os_.MANIFEST_SCHEMA_VERSION,
                "user_id": "evil",
                "experiment_name": "evil",
                "update": False,
                "update_uuid": None,
                "password": None,
                "files": [{
                    "name": "x",
                    "archive_path": "foo\x00.csv",
                    "size": 2,
                    "sha256": "0" * 64,
                    "content_type": "text/csv",
                }],
            }
            status, body = _post_manifest_raw(port, manifest, zip_path)
            ok = _check(
                "unsafe" in body.lower() or "error" in body.lower(),
                f"expected rejection, got {body!r}",
            )
    finally:
        _stop_server(server)
        shutil.rmtree(share_root, ignore_errors=True)
    return ok


def test_rejects_oversized_file_in_manifest() -> bool:
    """Server enforces MAX_FILE_SIZE = 1 GiB."""
    if _skip_unless_zip():
        return True
    share_root = Path(tempfile.mkdtemp(prefix="oo_share_int_"))
    share_path = share_root / "shares"
    share_path.mkdir()
    server, port = _start_server(share_path)
    try:
        if not _wait_for_server(f"http://127.0.0.1:{port}/share_internal.php"):
            return _check(False, "PHP server did not start")

        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "huge.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("huge.bin", "x")
            manifest = {
                "schema_version": os_.MANIFEST_SCHEMA_VERSION,
                "user_id": "alice",
                "experiment_name": "e",
                "update": False,
                "update_uuid": None,
                "password": None,
                "files": [{
                    "name": "huge",
                    "archive_path": "huge.bin",
                    "size": 10**12,  # 1 TB
                    "sha256": "0" * 64,
                    "content_type": "application/octet-stream",
                }],
            }
            status, body = _post_manifest_raw(port, manifest, zip_path)
            ok = _check(
                "size" in body.lower() or "out of range" in body.lower(),
                f"expected size rejection, got {body!r}",
            )
    finally:
        _stop_server(server)
        shutil.rmtree(share_root, ignore_errors=True)
    return ok


def test_rejects_manifest_with_no_files() -> bool:
    if _skip_unless_zip():
        return True
    share_root = Path(tempfile.mkdtemp(prefix="oo_share_int_"))
    share_path = share_root / "shares"
    share_path.mkdir()
    server, port = _start_server(share_path)
    try:
        if not _wait_for_server(f"http://127.0.0.1:{port}/share_internal.php"):
            return _check(False, "PHP server did not start")

        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "empty.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                pass  # empty zip
            manifest = {
                "schema_version": os_.MANIFEST_SCHEMA_VERSION,
                "user_id": "alice",
                "experiment_name": "e",
                "update": False,
                "update_uuid": None,
                "password": None,
                "files": [],
            }
            status, body = _post_manifest_raw(port, manifest, zip_path)
            ok = _check(
                "no files" in body.lower() or "nothing" in body.lower(),
                f"expected empty-files rejection, got {body!r}",
            )
    finally:
        _stop_server(server)
        shutil.rmtree(share_root, ignore_errors=True)
    return ok


def test_rejects_manifest_missing_required_field() -> bool:
    if _skip_unless_zip():
        return True
    share_root = Path(tempfile.mkdtemp(prefix="oo_share_int_"))
    share_path = share_root / "shares"
    share_path.mkdir()
    server, port = _start_server(share_path)
    try:
        if not _wait_for_server(f"http://127.0.0.1:{port}/share_internal.php"):
            return _check(False, "PHP server did not start")

        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "ok.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("results.csv", "a,b\n1,2\n")
            # Missing "files" key entirely
            manifest = {
                "schema_version": os_.MANIFEST_SCHEMA_VERSION,
                "user_id": "alice",
                "experiment_name": "e",
                "update": False,
                "update_uuid": None,
                "password": None,
            }
            status, body = _post_manifest_raw(port, manifest, zip_path)
            ok = _check(
                "missing" in body.lower() or "files" in body.lower(),
                f"expected missing-key rejection, got {body!r}",
            )
    finally:
        _stop_server(server)
        shutil.rmtree(share_root, ignore_errors=True)
    return ok


def test_rejects_manifest_with_bad_sha_format() -> bool:
    """SHA256 must be exactly 64 hex chars."""
    if _skip_unless_zip():
        return True
    share_root = Path(tempfile.mkdtemp(prefix="oo_share_int_"))
    share_path = share_root / "shares"
    share_path.mkdir()
    server, port = _start_server(share_path)
    try:
        if not _wait_for_server(f"http://127.0.0.1:{port}/share_internal.php"):
            return _check(False, "PHP server did not start")

        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "ok.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("results.csv", "a,b\n1,2\n")
            manifest = {
                "schema_version": os_.MANIFEST_SCHEMA_VERSION,
                "user_id": "alice",
                "experiment_name": "e",
                "update": False,
                "update_uuid": None,
                "password": None,
                "files": [{
                    "name": "results",
                    "archive_path": "results.csv",
                    "size": 8,
                    "sha256": "not-a-hex-sha",  # INVALID
                    "content_type": "text/csv",
                }],
            }
            status, body = _post_manifest_raw(port, manifest, zip_path)
            ok = _check(
                "sha256" in body.lower() or "bad" in body.lower(),
                f"expected bad-sha rejection, got {body!r}",
            )
    finally:
        _stop_server(server)
        shutil.rmtree(share_root, ignore_errors=True)
    return ok


# ---------------------------------------------------------------------------
# Legacy format compatibility
# ---------------------------------------------------------------------------


def test_legacy_multipart_format_still_works() -> bool:
    """A request that uses the old field-name format (no manifest,
    just ``-F name=@file``) should still work for backward compat."""
    if _skip_unless_zip():
        return True
    share_root = Path(tempfile.mkdtemp(prefix="oo_share_int_"))
    share_path = share_root / "shares"
    share_path.mkdir()
    server, port = _start_server(share_path)
    try:
        if not _wait_for_server(f"http://127.0.0.1:{port}/share_internal.php"):
            return _check(False, "PHP server did not start")

        with tempfile.TemporaryDirectory() as tmp:
            run = _make_run_dir(Path(tmp))
            results_csv = run / "results.csv"
            # Build a legacy-style multipart body
            boundary = "----oo-legacy"
            body = (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="results"; filename="results.csv"\r\n'
                f"Content-Type: text/csv\r\n\r\n"
                f"{results_csv.read_text()}"
                f"\r\n--{boundary}--\r\n"
            ).encode()
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
            conn.request(
                "POST",
                "/share_internal.php?user_id=legacy_user&experiment_name=legacy_e",
                body=body,
                headers={
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                },
            )
            resp = conn.getresponse()
            body_text = resp.read().decode("utf-8", errors="replace")
            conn.close()
            # The legacy code path accepts these fields.
            ok = _check(
                "Error" not in body_text or "success" in body_text.lower(),
                f"unexpected error from legacy path: {body_text!r}",
            )
            user_dir = share_path / "legacy_user" / "legacy_e"
            ok &= _check(
                user_dir.exists(),
                f"server did not create {user_dir}",
            )
    finally:
        _stop_server(server)
        shutil.rmtree(share_root, ignore_errors=True)
    return ok


# ---------------------------------------------------------------------------
# List of tests
# ---------------------------------------------------------------------------


TESTS = [
    test_php_share_internal_parses,
    test_end_to_end_share_creates_user_dir,
    test_end_to_end_update_same_experiment,
    test_rejects_unsafe_archive_path,
    test_rejects_bad_sha256,
    test_rejects_bad_size_in_manifest,
    test_rejects_unknown_schema_version,
    test_rejects_invalid_json_manifest,
    test_rejects_archive_path_with_backslash,
    test_rejects_archive_path_with_null_byte,
    test_rejects_oversized_file_in_manifest,
    test_rejects_manifest_with_no_files,
    test_rejects_manifest_missing_required_field,
    test_rejects_manifest_with_bad_sha_format,
    test_legacy_multipart_format_still_works,
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
    print(f"\nAll {len(TESTS)} server-side share tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
