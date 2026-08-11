#!/usr/bin/env python3
"""Tests for omniopt_share (Python rewrite + new JSON/ZIP format).

Tightly coupled tests for the new omniopt_share module.  The new
module exposes the share as a JSON ``manifest`` + a ``bundle.zip``
upload that the GUI server can verify without trusting any of the
field names sent by the client.

Tested units:

  * :func:`omniopt_share.parse_args`                  -- CLI parsing
  * :func:`omniopt_share.collect_shareable_files`     -- file discovery
  * :func:`omniopt_share.build_manifest`              -- manifest assembly
  * :func:`omniopt_share.write_bundle`                -- zip + manifest on disk
  * :func:`omniopt_share.verify_manifest`             -- security validation
  * :func:`omniopt_share.extract_experiment_name`     -- (re-used)
  * :func:`omniopt_share.is_valid_username`           -- input validation
  * :func:`omniopt_share.sanitize_archive_path`       -- path-traversal block
"""

from __future__ import annotations

import csv
import json
import re
import sys
import tempfile
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


def _make_run_dir(root: Path) -> Path:
    """Build a minimal but realistic run directory tree."""
    run = root / "my_experiment" / "0"
    run.mkdir(parents=True)
    (run / "results.csv").write_text("a,b\n1,2\n")
    (run / "parameters.txt").write_text("x: range\n")
    (run / "best_result.txt").write_text("2.0\n")
    (run / "Constraints.txt").write_text("")
    (run / "ui_url.txt").write_text("http://localhost")
    (run / "0.csv").write_text("start_time,end_time,exit_code\n1,2,0\n")
    (run / "global_vars.json").write_text("{}")
    (run / "git_version").write_text("abc123\n")
    sf = run / "state_files"
    sf.mkdir()
    (sf / "run_uuid").write_text("00000000-0000-0000-0000-000000000000")
    sr = run / "single_runs" / "0"
    sr.mkdir(parents=True)
    (sr / "0.out").write_text("ok\n")
    return run


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------


def test_parse_args_no_args_shows_help_and_exits_nonzero() -> bool:
    rc = os_.main([])
    return _check(rc != 0, f"empty argv must not succeed, got {rc}")


def test_parse_args_minimum_required() -> bool:
    """At minimum, a run dir must be given (no network in tests)."""
    rc = os_.main(["/nonexistent/path/that/is/not/here"])
    return _check(rc != 0, "missing run dir should fail")


def test_parse_args_help() -> bool:
    rc = os_.main(["--help"])
    return _check(rc == 0, f"--help should exit 0, got {rc}")


def test_parse_args_username_password() -> bool:
    args = os_.parse_args(["runs/foo/0", "--username=alice", "--password=secret"])
    return _check(
        args.username == "alice" and args.password == "secret",
        f"got {args}",
    )


def test_parse_args_update_flag() -> bool:
    args = os_.parse_args(["runs/foo/0", "--update"])
    return _check(args.update is True, f"expected update=True, got {args}")


def test_parse_args_force_flag() -> bool:
    args = os_.parse_args(["runs/foo/0", "--force"])
    return _check(args.force is True, f"expected force=True, got {args}")


def test_parse_args_no_color_flag() -> bool:
    args = os_.parse_args(["runs/foo/0", "--no_color"])
    return _check(args.no_color is True, f"got {args}")


def test_parse_args_dont_send_singleruns_flag() -> bool:
    args = os_.parse_args(["runs/foo/0", "--dont_send_singleruns"])
    return _check(
        args.send_single_runs is False, f"got send_single_runs={args.send_single_runs}"
    )


def test_parse_args_outfile() -> bool:
    args = os_.parse_args(["runs/foo/0", "--outfile=foo.out"])
    return _check(args.outfile == "foo.out", f"got outfile={args.outfile!r}")


def test_parse_args_debug_flag() -> bool:
    args = os_.parse_args(["runs/foo/0", "--debug"])
    return _check(args.debug is True, f"got {args}")


# ---------------------------------------------------------------------------
# Username validation
# ---------------------------------------------------------------------------


def test_is_valid_username_ok() -> bool:
    return _check(os_.is_valid_username("alice"), "alice should be valid")


def test_is_valid_username_rejects_empty() -> bool:
    return _check(not os_.is_valid_username(""), "empty must be invalid")


def test_is_valid_username_rejects_whitespace() -> bool:
    return _check(
        not os_.is_valid_username("ali ce"),
        "whitespace must be invalid",
    )


# ---------------------------------------------------------------------------
# Path sanitization
# ---------------------------------------------------------------------------


def test_sanitize_archive_path_ok() -> bool:
    return _check(
        os_.sanitize_archive_path("results.csv") == "results.csv",
        "simple path must round-trip",
    )


def test_sanitize_archive_path_nested_ok() -> bool:
    return _check(
        os_.sanitize_archive_path("state_files/run_uuid") == "state_files/run_uuid",
        "nested path must round-trip",
    )


def test_sanitize_archive_path_blocks_traversal() -> bool:
    try:
        os_.sanitize_archive_path("../etc/passwd")
    except ValueError:
        return True
    return _check(False, "expected ValueError for traversal")


def test_sanitize_archive_path_blocks_absolute() -> bool:
    try:
        os_.sanitize_archive_path("/etc/passwd")
    except ValueError:
        return True
    return _check(False, "expected ValueError for absolute path")


def test_sanitize_archive_path_blocks_null_byte() -> bool:
    try:
        os_.sanitize_archive_path("foo\x00bar")
    except ValueError:
        return True
    return _check(False, "expected ValueError for null byte")


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def test_collect_shareable_files_basic() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(Path(tmp))
        files = os_.collect_shareable_files(str(run), send_single_runs=False)
        names = sorted({f["archive_path"] for f in files})
    ok = _check("results.csv" in names, f"results.csv missing: {names}")
    ok &= _check("parameters.txt" in names, f"parameters.txt missing: {names}")
    ok &= _check("git_version" in names, f"git_version missing: {names}")
    return ok


def test_collect_shareable_files_includes_singleruns_by_default() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(Path(tmp))
        files = os_.collect_shareable_files(str(run), send_single_runs=True)
        names = {f["name"] for f in files}
    ok = _check(
        "single_run_file_0_0.out" in names,
        f"single run file missing: {names}",
    )
    return ok


def test_collect_shareable_files_excludes_singleruns_when_disabled() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(Path(tmp))
        files = os_.collect_shareable_files(str(run), send_single_runs=False)
        names = {f["archive_path"] for f in files}
    return _check(
        not any("single_run_file_" in n for n in names),
        f"single run files must be excluded: {names}",
    )


def test_collect_shareable_files_missing_dir_returns_empty() -> bool:
    files = os_.collect_shareable_files("/nonexistent/that/does/not/exist", True)
    return _check(files == [], f"expected [], got {files!r}")


# ---------------------------------------------------------------------------
# Manifest assembly
# ---------------------------------------------------------------------------


def test_build_manifest_basic() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(Path(tmp))
        manifest = os_.build_manifest(
            run_dir=str(run),
            user_id="alice",
            experiment_name="my_experiment",
            update=False,
            update_uuid=None,
            password=None,
            send_single_runs=False,
        )
    ok = _check(manifest["user_id"] == "alice", f"user_id: {manifest}")
    ok &= _check(
        manifest["experiment_name"] == "my_experiment",
        f"experiment_name: {manifest}",
    )
    ok &= _check(
        manifest["schema_version"] == os_.MANIFEST_SCHEMA_VERSION,
        f"schema_version: {manifest}",
    )
    ok &= _check(isinstance(manifest["files"], list), f"files: {manifest}")
    ok &= _check(len(manifest["files"]) > 0, f"no files: {manifest}")
    for f in manifest["files"]:
        ok &= _check("name" in f and "archive_path" in f and "sha256" in f and "size" in f,
                     f"file missing keys: {f}")
        ok &= _check(len(f["sha256"]) == 64, f"sha256 wrong length: {f['sha256']!r}")
    return ok


def test_build_manifest_includes_run_uuid_when_present() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(Path(tmp))
        manifest = os_.build_manifest(
            run_dir=str(run),
            user_id="alice",
            experiment_name="my_experiment",
            update=True,
            update_uuid=None,
            password=None,
            send_single_runs=False,
        )
    return _check(
        manifest["update_uuid"] is not None,
        f"expected update_uuid to be set, got {manifest}",
    )


def test_build_manifest_uses_explicit_update_uuid() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(Path(tmp))
        manifest = os_.build_manifest(
            run_dir=str(run),
            user_id="alice",
            experiment_name="my_experiment",
            update=False,
            update_uuid="deadbeef-1234",
            password=None,
            send_single_runs=False,
        )
    return _check(
        manifest["update_uuid"] == "deadbeef-1234",
        f"got {manifest}",
    )


def test_build_manifest_rejects_invalid_username() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(Path(tmp))
        try:
            os_.build_manifest(
                run_dir=str(run),
                user_id="bad name",
                experiment_name="my_experiment",
                update=False,
                update_uuid=None,
                password=None,
                send_single_runs=False,
            )
        except ValueError:
            return True
    return _check(False, "expected ValueError for bad username")


# ---------------------------------------------------------------------------
# Bundle writing
# ---------------------------------------------------------------------------


def test_write_bundle_creates_manifest_and_zip() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(Path(tmp))
        manifest = os_.build_manifest(
            run_dir=str(run),
            user_id="alice",
            experiment_name="my_experiment",
            update=False,
            update_uuid=None,
            password=None,
            send_single_runs=False,
        )
        out_dir = Path(tmp) / "out"
        out_dir.mkdir()
        m_path, z_path = os_.write_bundle(manifest, out_dir, source_dir=str(run))

        ok = _check(m_path.exists() and z_path.exists(),
                    f"missing files: {m_path} {z_path}")
        # Manifest is valid JSON with the expected top-level keys
        loaded = json.loads(m_path.read_text())
        ok &= _check(loaded["user_id"] == "alice",
                     f"round-trip user_id: {loaded}")
        # Zip contains every archive_path declared in the manifest
        with zipfile.ZipFile(z_path) as zf:
            names = set(zf.namelist())
        for f in manifest["files"]:
            ok &= _check(
                f["archive_path"] in names,
                f"archive missing from zip: {f['archive_path']}",
            )
        # No zip-slip: every name must be safe
        for n in names:
            ok &= _check(not n.startswith("/") and ".." not in n.split("/"),
                         f"unsafe zip entry: {n!r}")
    return ok


def test_write_bundle_rejects_unsafe_paths() -> bool:
    """A manifest that tries to smuggle an unsafe archive_path must be
    refused."""
    bad_manifest = {
        "schema_version": os_.MANIFEST_SCHEMA_VERSION,
        "user_id": "alice",
        "experiment_name": "e",
        "update": False,
        "update_uuid": None,
        "password": None,
        "files": [
            {
                "name": "evil",
                "archive_path": "../../etc/passwd",
                "size": 0,
                "sha256": "0" * 64,
                "content_type": "text/plain",
            }
        ],
    }
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "out"
        out_dir.mkdir()
        try:
            os_.write_bundle(bad_manifest, out_dir)
        except ValueError:
            return True
    return _check(False, "expected ValueError for unsafe archive_path")


# ---------------------------------------------------------------------------
# Manifest verification
# ---------------------------------------------------------------------------


def test_verify_manifest_ok() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(Path(tmp))
        manifest = os_.build_manifest(
            run_dir=str(run),
            user_id="alice",
            experiment_name="my_experiment",
            update=False,
            update_uuid=None,
            password=None,
            send_single_runs=False,
        )
        err = os_.verify_manifest(manifest)
    return _check(err == "", f"unexpected error: {err!r}")


def test_verify_manifest_rejects_missing_keys() -> bool:
    bad = {"user_id": "alice"}
    err = os_.verify_manifest(bad)
    return _check(err != "", f"expected error, got {err!r}")


def test_verify_manifest_rejects_unknown_schema() -> bool:
    bad = {
        "schema_version": "99.0",
        "user_id": "alice",
        "experiment_name": "e",
        "update": False,
        "update_uuid": None,
        "password": None,
        "files": [],
    }
    err = os_.verify_manifest(bad)
    return _check(err != "", f"expected error, got {err!r}")


def test_verify_manifest_rejects_unsafe_archive_path() -> bool:
    bad = {
        "schema_version": os_.MANIFEST_SCHEMA_VERSION,
        "user_id": "alice",
        "experiment_name": "e",
        "update": False,
        "update_uuid": None,
        "password": None,
        "files": [
            {
                "name": "x",
                "archive_path": "../etc/passwd",
                "size": 0,
                "sha256": "0" * 64,
                "content_type": "text/plain",
            }
        ],
    }
    err = os_.verify_manifest(bad)
    return _check(err != "", f"expected error, got {err!r}")


def test_verify_manifest_rejects_oversized_file() -> bool:
    bad = {
        "schema_version": os_.MANIFEST_SCHEMA_VERSION,
        "user_id": "alice",
        "experiment_name": "e",
        "update": False,
        "update_uuid": None,
        "password": None,
        "files": [
            {
                "name": "huge",
                "archive_path": "huge.bin",
                "size": 10**12,
                "sha256": "0" * 64,
                "content_type": "application/octet-stream",
            }
        ],
    }
    err = os_.verify_manifest(bad)
    return _check(err != "", f"expected error, got {err!r}")


TESTS = [
    # CLI parsing
    test_parse_args_no_args_shows_help_and_exits_nonzero,
    test_parse_args_minimum_required,
    test_parse_args_help,
    test_parse_args_username_password,
    test_parse_args_update_flag,
    test_parse_args_force_flag,
    test_parse_args_no_color_flag,
    test_parse_args_dont_send_singleruns_flag,
    test_parse_args_outfile,
    test_parse_args_debug_flag,
    # Validation
    test_is_valid_username_ok,
    test_is_valid_username_rejects_empty,
    test_is_valid_username_rejects_whitespace,
    test_sanitize_archive_path_ok,
    test_sanitize_archive_path_nested_ok,
    test_sanitize_archive_path_blocks_traversal,
    test_sanitize_archive_path_blocks_absolute,
    test_sanitize_archive_path_blocks_null_byte,
    # File discovery
    test_collect_shareable_files_basic,
    test_collect_shareable_files_includes_singleruns_by_default,
    test_collect_shareable_files_excludes_singleruns_when_disabled,
    test_collect_shareable_files_missing_dir_returns_empty,
    # Manifest
    test_build_manifest_basic,
    test_build_manifest_includes_run_uuid_when_present,
    test_build_manifest_uses_explicit_update_uuid,
    test_build_manifest_rejects_invalid_username,
    # Bundle writing
    test_write_bundle_creates_manifest_and_zip,
    test_write_bundle_rejects_unsafe_paths,
    # Verification
    test_verify_manifest_ok,
    test_verify_manifest_rejects_missing_keys,
    test_verify_manifest_rejects_unknown_schema,
    test_verify_manifest_rejects_unsafe_archive_path,
    test_verify_manifest_rejects_oversized_file,
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
    print("\nAll omniopt_share tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
