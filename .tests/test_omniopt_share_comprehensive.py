#!/usr/bin/env python3
"""Comprehensive tests for omniopt_share / share format.

These tests probe the share module from every angle: file collection
edge cases, hash/integrity, bundle structure, manifest validation,
schema versioning, server-side security, special filenames, update
flows, and CLI behaviour.  The existing :mod:`test_omniopt_share` and
:mod:`test_share_integration` cover the happy path; this module
covers the long tail of edge cases.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
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


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _make_run_dir(root: Path, *, with_singleruns: bool = True) -> Path:
    """Create a realistic run dir under ``root/<experiment>/0``."""
    run = root / "experiment" / "0"
    run.mkdir(parents=True)
    (run / "results.csv").write_text("a,b\n1,2\n")
    (run / "parameters.txt").write_text("x: range(1,2)\n")
    (run / "best_result.txt").write_text("2.0\n")
    (run / "0.csv").write_text("start_time,end_time,exit_code\n1,2,0\n")
    (run / "global_vars.json").write_text("{}")
    (run / "Constraints.txt").write_text("")
    (run / "ui_url.txt").write_text("http://example")
    (run / "git_version").write_text("deadbeef\n")
    (run / "verbose_log.txt").write_text("starting\n")
    sf = run / "state_files"
    sf.mkdir()
    (sf / "run_uuid").write_text("00000000-0000-0000-0000-000000000001")
    (sf / "global_vars.json").write_text("{}")
    if with_singleruns:
        sr = run / "single_runs"
        sr.mkdir()
        for n in (0, 1):
            (sr / str(n)).mkdir()
            (sr / str(n) / f"{n}.out").write_text(f"run {n}\n")
            (sr / str(n) / f"{n}.err").write_text("")
    return run


# ---------------------------------------------------------------------------
# File collection edge cases
# ---------------------------------------------------------------------------


def test_collect_includes_all_expected_top_level_extensions() -> bool:
    """Every accepted extension (.csv .txt .log .json) must be picked up."""
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "e" / "0"
        run.mkdir(parents=True)
        for ext in ("csv", "txt", "log", "json"):
            (run / f"file.{ext}").write_text("x\n")
        files = os_.collect_shareable_files(str(run), send_single_runs=False)
        names = {f["archive_path"] for f in files}
    ok = _check("file.csv" in names, f"file.csv missing: {names}")
    ok &= _check("file.txt" in names, f"file.txt missing: {names}")
    ok &= _check("file.log" in names, f"file.log missing: {names}")
    ok &= _check("file.json" in names, f"file.json missing: {names}")
    return ok


def test_collect_excludes_disallowed_extensions() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "e" / "0"
        run.mkdir(parents=True)
        for ext in ("pyc", "so", "exe", "zip", "tar.gz", "jpg", "png", "pdf"):
            (run / f"file.{ext}").write_text("x\n")
        (run / "Makefile").write_text("all:\n")
        files = os_.collect_shareable_files(str(run), send_single_runs=False)
        names = {f["archive_path"] for f in files}
    ok = _check(
        all(f"file.{ext}" not in names for ext in ("pyc", "so", "exe", "zip", "jpg", "png", "pdf")),
        f"disallowed extensions leaked through: {names}",
    )
    ok &= _check("Makefile" not in names, f"Makefile leaked through: {names}")
    return ok


def test_collect_includes_extensionless_git_version() -> bool:
    """git_version has no extension but must be sent (bash version sends it)."""
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(Path(tmp), with_singleruns=False)
        files = os_.collect_shareable_files(str(run), send_single_runs=False)
        names = {f["archive_path"] for f in files}
    return _check("git_version" in names, f"git_version missing: {names}")


def test_collect_picks_up_state_files_recursively() -> bool:
    """Subdirectories of state_files/ are NOT collected (bash version too)."""
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "e" / "0"
        run.mkdir(parents=True)
        sf = run / "state_files"
        sf.mkdir()
        (sf / "good.csv").write_text("a\n")
        (sf / "nested").mkdir()
        (sf / "nested" / "deep.txt").write_text("x\n")
        files = os_.collect_shareable_files(str(run), send_single_runs=False)
        names = {f["archive_path"] for f in files}
    ok = _check("state_files/good.csv" in names, f"good.csv missing: {names}")
    ok &= _check("state_files/nested/deep.txt" not in names, "nested deep leaked")
    return ok


def test_collect_single_runs_uses_digit_subdir_only() -> bool:
    """Only subdirs whose name is purely digits are sent (matches bash)."""
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "e" / "0"
        run.mkdir(parents=True)
        sr = run / "single_runs"
        sr.mkdir()
        for name in ("0", "1", "10", "abc", "run_a", "12abc"):
            (sr / name).mkdir()
            (sr / name / "0.out").write_text("x\n")
        files = os_.collect_shareable_files(str(run), send_single_runs=True)
        names = {f["name"] for f in files}
    ok = _check("single_run_file_0_0.out" in names, f"0 missing: {names}")
    ok &= _check("single_run_file_1_0.out" in names, f"1 missing: {names}")
    ok &= _check("single_run_file_10_0.out" in names, f"10 missing: {names}")
    ok &= _check(not any("abc" in n for n in names),
                 f"non-numeric subdir leaked: {names}")
    ok &= _check(not any("run_a" in n for n in names),
                 f"non-numeric subdir leaked: {names}")
    return ok


def test_collect_single_runs_only_out_and_err() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "e" / "0"
        run.mkdir(parents=True)
        sr = run / "single_runs" / "0"
        sr.mkdir(parents=True)
        (sr / "0.out").write_text("ok\n")
        (sr / "0.err").write_text("")
        (sr / "0.log").write_text("log\n")
        (sr / "0.json").write_text("{}\n")
        (sr / "0.txt").write_text("txt\n")
        files = os_.collect_shareable_files(str(run), send_single_runs=True)
        names = {f["archive_path"] for f in files}
    ok = _check("single_runs/0/0.out" in names, f"out missing: {names}")
    ok &= _check("single_runs/0/0.err" in names, f"err missing: {names}")
    ok &= _check("single_runs/0/0.log" not in names, "log leaked")
    ok &= _check("single_runs/0/0.txt" not in names, "txt leaked")
    return ok


def test_collect_excludes_empty_directories_at_top_level() -> bool:
    """A top-level directory is not a shareable file."""
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "e" / "0"
        run.mkdir(parents=True)
        (run / "subdir").mkdir()
        (run / "results.csv").write_text("a\n")
        files = os_.collect_shareable_files(str(run), send_single_runs=False)
        names = {f["archive_path"] for f in files}
    return _check(
        "subdir" not in names and "results.csv" in names,
        f"unexpected names: {names}",
    )


def test_collect_preserves_local_path_for_zip_writing() -> bool:
    """Each collected file must carry its absolute local_path."""
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(Path(tmp))
        files = os_.collect_shareable_files(str(run), send_single_runs=False)
        ok = True
        for f in files:
            ok &= _check(
                Path(f["local_path"]).is_file(),
                f"local_path missing on disk: {f}",
            )
            ok &= _check(
                Path(f["local_path"]).is_absolute(),
                f"local_path not absolute: {f}",
            )
    return ok


def test_collect_includes_empty_files() -> bool:
    """Empty files are still sent (just with size 0)."""
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "e" / "0"
        run.mkdir(parents=True)
        (run / "results.csv").write_text("")
        (run / "constraints.txt").write_text("")
        files = os_.collect_shareable_files(str(run), send_single_runs=False)
        names_with_sizes = {f["archive_path"]: f["local_path"].stat().st_size
                             for f in files}
    ok = _check("results.csv" in names_with_sizes, f"missing: {names_with_sizes}")
    ok &= _check(names_with_sizes.get("results.csv") == 0,
                 f"results.csv not empty: {names_with_sizes.get('results.csv')}")
    return ok


def test_collect_handles_uppercase_extension_variants() -> bool:
    """.CSV (uppercase) must still be picked up."""
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "e" / "0"
        run.mkdir(parents=True)
        (run / "UPPER.CSV").write_text("a\n")
        (run / "Mixed.Txt").write_text("b\n")
        (run / "weird.LOG").write_text("c\n")
        files = os_.collect_shareable_files(str(run), send_single_runs=False)
        names = {f["archive_path"] for f in files}
    return _check(
        {"UPPER.CSV", "Mixed.Txt", "weird.LOG"}.issubset(names),
        f"missing uppercase variants: {names}",
    )


def test_collect_resolves_symlinks_within_run_dir() -> bool:
    """Symlinks pointing into the run dir are followed and included."""
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "e" / "0"
        run.mkdir(parents=True)
        target = run / "results.csv"
        target.write_text("a,b\n1,2\n")
        (run / "alias.csv").symlink_to(target)
        files = os_.collect_shareable_files(str(run), send_single_runs=False)
        names = {f["archive_path"] for f in files}
    return _check("alias.csv" in names, f"alias.csv missing: {names}")


def test_collect_unicode_filenames() -> bool:
    """Unicode file names round-trip."""
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "e" / "0"
        run.mkdir(parents=True)
        (run / "résultats.csv").write_text("a,b\n1,2\n")
        (run / "参数.txt").write_text("x\n")
        files = os_.collect_shareable_files(str(run), send_single_runs=False)
        names = {f["archive_path"] for f in files}
    ok = _check("résultats.csv" in names, f"missing: {names}")
    ok &= _check("参数.txt" in names, f"missing: {names}")
    return ok


def test_collect_filenames_with_spaces() -> bool:
    """Filenames with spaces are accepted (sanitizer must not reject)."""
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "e" / "0"
        run.mkdir(parents=True)
        (run / "results file.csv").write_text("a\n")
        files = os_.collect_shareable_files(str(run), send_single_runs=False)
        names = {f["archive_path"] for f in files}
    return _check("results file.csv" in names, f"missing: {names}")


def test_collect_hidden_files_excluded_at_top_level() -> bool:
    """The bash version's whitelist does not include dotfiles; Python must match."""
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "e" / "0"
        run.mkdir(parents=True)
        (run / ".hidden").write_text("x\n")
        (run / "visible.csv").write_text("a\n")
        files = os_.collect_shareable_files(str(run), send_single_runs=False)
        names = {f["archive_path"] for f in files}
    return _check(
        ".hidden" not in names and "visible.csv" in names,
        f"unexpected: {names}",
    )


def test_collect_many_files() -> bool:
    """Sanity-check with many files (100 csv files)."""
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "e" / "0"
        run.mkdir(parents=True)
        for i in range(100):
            (run / f"f{i:03d}.csv").write_text(f"{i}\n")
        files = os_.collect_shareable_files(str(run), send_single_runs=False)
        names = {f["archive_path"] for f in files}
    return _check(len(names) >= 100, f"got {len(names)} files, expected >=100")


# ---------------------------------------------------------------------------
# Hashing / integrity
# ---------------------------------------------------------------------------


def test_sha256_of_empty_file_is_known_value() -> bool:
    with tempfile.NamedTemporaryFile(delete=False) as f:
        path = Path(f.name)
    try:
        sha = os_._sha256_of(path)
    finally:
        path.unlink()
    return _check(
        sha == hashlib.sha256(b"").hexdigest(),
        f"empty file sha wrong: {sha}",
    )


def test_sha256_of_binary_file_matches_stdlib() -> bool:
    """SHA must match hashlib for arbitrary binary content."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(os.urandom(8192))
        path = Path(f.name)
    try:
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        got = os_._sha256_of(path)
    finally:
        path.unlink()
    return _check(got == expected, f"mismatch: {got} != {expected}")


def test_sha256_of_large_file() -> bool:
    """Multi-MB file: hash must still match stdlib."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        chunk = os.urandom(1 << 20)  # 1 MiB
        for _ in range(5):
            f.write(chunk)
        path = Path(f.name)
    try:
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        got = os_._sha256_of(path)
    finally:
        path.unlink()
    return _check(got == expected, f"mismatch on 5 MiB file")


def test_manifest_hash_matches_file_actually_written() -> bool:
    """The hash recorded in the manifest must match the on-disk file."""
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(Path(tmp))
        m = os_.build_manifest(
            run_dir=str(run),
            user_id="alice",
            experiment_name="exp",
            update=False,
            update_uuid=None,
            password=None,
            send_single_runs=False,
        )
    ok = True
    for f in m["files"]:
        local = Path(run) / f["archive_path"]
        if local.exists():
            actual = hashlib.sha256(local.read_bytes()).hexdigest()
            ok &= _check(
                f["sha256"] == actual,
                f"hash mismatch for {f['archive_path']}: "
                f"manifest={f['sha256'][:16]} actual={actual[:16]}",
            )
    return ok


def test_manifest_size_matches_file_actually_written() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(Path(tmp))
        m = os_.build_manifest(
            run_dir=str(run),
            user_id="alice",
            experiment_name="exp",
            update=False,
            update_uuid=None,
            password=None,
            send_single_runs=False,
        )
    ok = True
    for f in m["files"]:
        local = Path(run) / f["archive_path"]
        if local.exists():
            actual = local.stat().st_size
            ok &= _check(
                f["size"] == actual,
                f"size mismatch for {f['archive_path']}: "
                f"manifest={f['size']} actual={actual}",
            )
    return ok


def test_manifest_rejects_file_larger_than_max() -> bool:
    """Files bigger than MAX_FILE_SIZE must be rejected at build time."""
    from omniopt_share import MAX_FILE_SIZE
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "e" / "0"
        run.mkdir(parents=True)
        # Simulate a huge file by patching stat: faster than writing >1 GiB.
        real_stat = Path.stat

        def fake_stat(self, *args, **kwargs):
            res = real_stat(self, *args, **kwargs)
            return type(res)(res.st_mode, res.st_ino, res.st_dev,
                             res.st_nlink, res.st_uid, res.st_gid,
                             MAX_FILE_SIZE + 1, res.st_atime,
                             res.st_mtime, res.st_ctime)
        (run / "huge.csv").write_text("a\n")
        try:
            Path.stat = fake_stat
            try:
                os_.build_manifest(
                    run_dir=str(run),
                    user_id="alice",
                    experiment_name="exp",
                    update=False, update_uuid=None,
                    password=None, send_single_runs=False,
                )
            except ValueError as e:
                return _check("larger than max" in str(e),
                              f"unexpected error: {e}")
        finally:
            Path.stat = real_stat
    return _check(False, "expected ValueError for oversized file")


# ---------------------------------------------------------------------------
# Bundle structure / zip correctness
# ---------------------------------------------------------------------------


def test_bundle_contains_exactly_manifest_files() -> bool:
    """No phantom entries; every declared file is present exactly once."""
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(Path(tmp))
        m = os_.build_manifest(
            run_dir=str(run),
            user_id="alice",
            experiment_name="exp",
            update=False,
            update_uuid=None,
            password=None,
            send_single_runs=False,
        )
        out_dir = Path(tmp) / "out"
        out_dir.mkdir()
        _, z_path = os_.write_bundle(m, out_dir, source_dir=str(run))
        with zipfile.ZipFile(z_path) as zf:
            names = zf.namelist()
    declared = {f["archive_path"] for f in m["files"]}
    ok = _check(set(names) == declared,
                 f"set mismatch: in-zip={set(names)} declared={declared}")
    # Also: no duplicates
    ok &= _check(len(names) == len(set(names)),
                 f"duplicates in zip: {names}")
    return ok


def test_bundle_files_preserve_exact_bytes() -> bool:
    """Round-trip through zip must yield byte-identical files."""
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(Path(tmp))
        m = os_.build_manifest(
            run_dir=str(run),
            user_id="alice",
            experiment_name="exp",
            update=False,
            update_uuid=None,
            password=None,
            send_single_runs=False,
        )
        out_dir = Path(tmp) / "out"
        out_dir.mkdir()
        _, z_path = os_.write_bundle(m, out_dir, source_dir=str(run))
        ok = True
        with zipfile.ZipFile(z_path) as zf:
            for entry in m["files"]:
                original = (Path(run) / entry["archive_path"]).read_bytes()
                in_zip = zf.read(entry["archive_path"])
                ok &= _check(
                    original == in_zip,
                    f"bytes differ for {entry['archive_path']}",
                )
    return ok


def test_bundle_manifest_is_valid_json() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(Path(tmp))
        m = os_.build_manifest(
            run_dir=str(run),
            user_id="alice",
            experiment_name="exp",
            update=False, update_uuid=None, password=None,
            send_single_runs=False,
        )
        out_dir = Path(tmp) / "out"
        out_dir.mkdir()
        m_path, _ = os_.write_bundle(m, out_dir, source_dir=str(run))
        parsed = json.loads(m_path.read_text())
    return _check(
        parsed["user_id"] == "alice" and parsed["experiment_name"] == "exp",
        f"round-trip wrong: {parsed}",
    )


def test_bundle_missing_source_file_raises() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(Path(tmp))
        m = os_.build_manifest(
            run_dir=str(run),
            user_id="alice",
            experiment_name="exp",
            update=False, update_uuid=None, password=None,
            send_single_runs=False,
        )
        # Delete a source file after manifest was built.
        (Path(run) / "results.csv").unlink()
        out_dir = Path(tmp) / "out"
        out_dir.mkdir()
        try:
            os_.write_bundle(m, out_dir, source_dir=str(run))
        except FileNotFoundError as e:
            return _check("results.csv" in str(e),
                          f"unexpected error: {e}")
    return _check(False, "expected FileNotFoundError")


def test_bundle_is_zip_deflated_not_stored() -> bool:
    """Use compression so large CSV files don't blow up the upload."""
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(Path(tmp))
        m = os_.build_manifest(
            run_dir=str(run),
            user_id="alice",
            experiment_name="exp",
            update=False, update_uuid=None, password=None,
            send_single_runs=False,
        )
        out_dir = Path(tmp) / "out"
        out_dir.mkdir()
        _, z_path = os_.write_bundle(m, out_dir, source_dir=str(run))
        with zipfile.ZipFile(z_path) as zf:
            infos = zf.infolist()
    ok = True
    for info in infos:
        # ZIP_DEFLATED is method 8.
        ok &= _check(
            info.compress_type == zipfile.ZIP_DEFLATED,
            f"{info.filename} is not DEFLATED (method={info.compress_type})",
        )
    return ok


def test_bundle_rejects_manifest_with_no_files() -> bool:
    bad = {
        "schema_version": os_.MANIFEST_SCHEMA_VERSION,
        "user_id": "alice",
        "experiment_name": "e",
        "update": False,
        "update_uuid": None,
        "password": None,
        "files": [],
    }
    with tempfile.TemporaryDirectory() as tmp:
        try:
            os_.write_bundle(bad, tmp)
        except ValueError as e:
            return _check(
                "Refusing" in str(e) or "files" in str(e),
                f"unexpected error: {e}",
            )
    return _check(False, "expected ValueError for empty files list")


# ---------------------------------------------------------------------------
# Manifest validation - schema & security
# ---------------------------------------------------------------------------


def test_verify_accepts_minimal_valid_manifest() -> bool:
    """All required keys, no extra files is still valid."""
    m = {
        "schema_version": os_.MANIFEST_SCHEMA_VERSION,
        "user_id": "alice",
        "experiment_name": "e",
        "update": False,
        "update_uuid": None,
        "password": None,
        "files": [],
    }
    err = os_.verify_manifest(m)
    return _check(err == "", f"unexpected error: {err!r}")


def test_verify_rejects_unknown_schema_version() -> bool:
    for bad in ("2.0", "0.9", "1.0.1", "v1", ""):
        m = {
            "schema_version": bad,
            "user_id": "alice",
            "experiment_name": "e",
            "update": False,
            "update_uuid": None,
            "password": None,
            "files": [],
        }
        err = os_.verify_manifest(m)
        if not err:
            return _check(False, f"schema {bad!r} unexpectedly accepted")
    return True


def test_verify_rejects_non_int_size() -> bool:
    for bad_size in (-1, "big", 1.5, None, True):
        m = {
            "schema_version": os_.MANIFEST_SCHEMA_VERSION,
            "user_id": "alice",
            "experiment_name": "e",
            "update": False,
            "update_uuid": None,
            "password": None,
            "files": [{
                "name": "x",
                "archive_path": "x.csv",
                "size": bad_size,
                "sha256": "0" * 64,
                "content_type": "text/csv",
            }],
        }
        err = os_.verify_manifest(m)
        if not err:
            return _check(False, f"size {bad_size!r} unexpectedly accepted")
    return True


def test_verify_rejects_non_string_sha256() -> bool:
    for bad_sha in ("", "abcd", "0" * 63, "0" * 65, None, 12345):
        m = {
            "schema_version": os_.MANIFEST_SCHEMA_VERSION,
            "user_id": "alice",
            "experiment_name": "e",
            "update": False,
            "update_uuid": None,
            "password": None,
            "files": [{
                "name": "x",
                "archive_path": "x.csv",
                "size": 1,
                "sha256": bad_sha,
                "content_type": "text/csv",
            }],
        }
        err = os_.verify_manifest(m)
        if not err:
            return _check(False, f"sha256 {bad_sha!r} unexpectedly accepted")
    return True


def test_verify_rejects_missing_files_key() -> bool:
    m = {
        "schema_version": os_.MANIFEST_SCHEMA_VERSION,
        "user_id": "alice",
        "experiment_name": "e",
        "update": False,
        "update_uuid": None,
        "password": None,
    }
    err = os_.verify_manifest(m)
    return _check("files" in err, f"expected 'files' in error: {err!r}")


def test_verify_allows_extra_fields_in_manifest() -> bool:
    """Forward compatibility: unknown top-level fields are accepted."""
    m = {
        "schema_version": os_.MANIFEST_SCHEMA_VERSION,
        "user_id": "alice",
        "experiment_name": "e",
        "update": False,
        "update_uuid": None,
        "password": None,
        "files": [],
        "future_field": "hello world",
        "client_build": {"git_sha": "abc123"},
    }
    err = os_.verify_manifest(m)
    return _check(err == "", f"unexpected error: {err!r}")


def test_verify_allows_extra_fields_per_file() -> bool:
    m = {
        "schema_version": os_.MANIFEST_SCHEMA_VERSION,
        "user_id": "alice",
        "experiment_name": "e",
        "update": False,
        "update_uuid": None,
        "password": None,
        "files": [{
            "name": "x",
            "archive_path": "x.csv",
            "size": 1,
            "sha256": "0" * 64,
            "content_type": "text/csv",
            "future_field": [1, 2, 3],
        }],
    }
    err = os_.verify_manifest(m)
    return _check(err == "", f"unexpected error: {err!r}")


def test_verify_rejects_update_non_bool() -> bool:
    m = {
        "schema_version": os_.MANIFEST_SCHEMA_VERSION,
        "user_id": "alice",
        "experiment_name": "e",
        "update": "yes",  # string instead of bool
        "update_uuid": None,
        "password": None,
        "files": [],
    }
    err = os_.verify_manifest(m)
    return _check(err != "", f"expected error, got: {err!r}")


# ---------------------------------------------------------------------------
# Path sanitization - all attack vectors
# ---------------------------------------------------------------------------


def test_sanitize_blocks_windows_traversal() -> bool:
    for bad in (
        "..\\windows\\system32",
        "..\\..\\evil.exe",
        "C:\\Windows\\System32",
        "\\\\server\\share",
    ):
        try:
            os_.sanitize_archive_path(bad)
        except ValueError:
            continue
        return _check(False, f"accepted dangerous path: {bad!r}")
    return True


def test_sanitize_blocks_absolute_paths() -> bool:
    for bad in ("/etc/passwd", "/tmp/x", "/", "//etc/passwd"):
        try:
            os_.sanitize_archive_path(bad)
        except ValueError:
            continue
        return _check(False, f"accepted absolute path: {bad!r}")
    return True


def test_sanitize_blocks_null_bytes() -> bool:
    for bad in ("foo\x00.csv", "\x00", "x\x00y"):
        try:
            os_.sanitize_archive_path(bad)
        except ValueError:
            continue
        return _check(False, f"accepted null byte: {bad!r}")
    return True


def test_sanitize_blocks_empty_and_whitespace() -> bool:
    for bad in ("", " ", "  ", "\t", "\n"):
        try:
            os_.sanitize_archive_path(bad)
        except ValueError:
            continue
        return _check(False, f"accepted empty/whitespace: {bad!r}")
    return True


def test_sanitize_blocks_dot_components() -> bool:
    for bad in (".", "..", "foo/.", "foo/..", "foo/./bar", "foo/../bar"):
        try:
            os_.sanitize_archive_path(bad)
        except ValueError:
            continue
        return _check(False, f"accepted dot component: {bad!r}")
    return True


def test_sanitize_accepts_url_encoded_chars_as_literal() -> bool:
    """URL-encoded characters in a filename are treated as literal
    bytes (no decoding) - that's the safe behavior."""
    for good in ("%2e%2e/passwd", "..%2fpasswd", "%2e%2e%2fpasswd"):
        try:
            out = os_.sanitize_archive_path(good)
        except ValueError as e:
            return _check(False, f"rejected safe path {good!r}: {e}")
        if out != good:
            return _check(False, f"modified: {good!r} -> {out!r}")
    return True


def test_sanitize_accepts_unicode_in_path() -> bool:
    for good in ("résultats.csv", "参数/x.json", "emoji-🎉.csv"):
        try:
            out = os_.sanitize_archive_path(good)
        except ValueError as e:
            return _check(False, f"rejected unicode: {good!r}: {e}")
        if out != good:
            return _check(False, f"modified: {good!r} -> {out!r}")
    return True


def test_sanitize_accepts_deeply_nested_paths() -> bool:
    """Paths like state_files/sub/x.csv are fine."""
    for good in (
        "state_files/x.csv",
        "a/b/c/d/e/f.csv",
        "single_runs/0/0.out",
    ):
        try:
            out = os_.sanitize_archive_path(good)
        except ValueError as e:
            return _check(False, f"rejected valid path {good!r}: {e}")
        if out != good:
            return _check(False, f"modified valid path: {good!r} -> {out!r}")
    return True


# ---------------------------------------------------------------------------
# Username validation - all edge cases
# ---------------------------------------------------------------------------


def test_username_rejects_unicode_whitespace() -> bool:
    for bad in ("alice\u00A0", "alice\u2003", "alice\u3000", "ali\tce"):
        if os_.is_valid_username(bad):
            return _check(False, f"accepted unicode whitespace: {bad!r}")
    return True


def test_username_accepts_unicode_letters() -> bool:
    for good in ("alice", "Алиса", "用户名", "alice.smith", "alice-2"):
        if not os_.is_valid_username(good):
            return _check(False, f"rejected unicode username: {good!r}")
    return True


def test_username_accepts_letters_digits_dots_dashes() -> bool:
    """is_valid_username accepts any non-whitespace string.  This
    documents the (permissive) current behaviour."""
    for good in ("alice", "alice.smith", "alice-2", "alice_3", "123"):
        if not os_.is_valid_username(good):
            return _check(False, f"rejected valid username: {good!r}")
    return True


def test_username_accepts_long_but_not_unbounded() -> bool:
    """Long but reasonable usernames are accepted (no length cap in the
    current implementation)."""
    long_name = "a" * 1000
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "e" / "0"
        run.mkdir(parents=True)
        m = os_.build_manifest(
            run_dir=str(run),
            user_id=long_name,
            experiment_name="e",
            update=False, update_uuid=None, password=None,
            send_single_runs=False,
        )
    return _check(
        m["user_id"] == long_name,
        f"long username not preserved",
    )


# ---------------------------------------------------------------------------
# Update flow & password handling
# ---------------------------------------------------------------------------


def test_build_manifest_includes_update_uuid_from_state_files() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(Path(tmp))
        m = os_.build_manifest(
            run_dir=str(run),
            user_id="alice",
            experiment_name="e",
            update=True,
            update_uuid=None,
            password=None,
            send_single_runs=False,
        )
    return _check(
        m["update_uuid"] == "00000000-0000-0000-0000-000000000001",
        f"expected auto-detected uuid, got {m['update_uuid']!r}",
    )


def test_build_manifest_explicit_uuid_overrides_state_files() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(Path(tmp))
        m = os_.build_manifest(
            run_dir=str(run),
            user_id="alice",
            experiment_name="e",
            update=True,
            update_uuid="explicit-uuid-1234",
            password=None,
            send_single_runs=False,
        )
    return _check(
        m["update_uuid"] == "explicit-uuid-1234",
        f"expected explicit, got {m['update_uuid']!r}",
    )


def test_build_manifest_no_update_uuid_when_not_updating() -> bool:
    """If update=False, do NOT pick up the run_uuid automatically."""
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(Path(tmp))
        m = os_.build_manifest(
            run_dir=str(run),
            user_id="alice",
            experiment_name="e",
            update=False,
            update_uuid=None,
            password=None,
            send_single_runs=False,
        )
    return _check(
        m["update_uuid"] is None,
        f"update_uuid should be None when not updating, got {m['update_uuid']!r}",
    )


def test_build_manifest_no_run_uuid_file_no_crash() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "e" / "0"
        run.mkdir(parents=True)
        (run / "results.csv").write_text("a\n")
        # No state_files/run_uuid
        m = os_.build_manifest(
            run_dir=str(run),
            user_id="alice",
            experiment_name="e",
            update=True,
            update_uuid=None,
            password=None,
            send_single_runs=False,
        )
    return _check(
        m["update_uuid"] is None,
        f"expected None when no run_uuid file, got {m['update_uuid']!r}",
    )


def test_build_manifest_password_can_be_none() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(Path(tmp))
        m = os_.build_manifest(
            run_dir=str(run),
            user_id="alice",
            experiment_name="e",
            update=False, update_uuid=None, password=None,
            send_single_runs=False,
        )
    return _check(m["password"] is None, f"got {m['password']!r}")


def test_build_manifest_password_preserved() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(Path(tmp))
        m = os_.build_manifest(
            run_dir=str(run),
            user_id="alice",
            experiment_name="e",
            update=False, update_uuid=None, password="hunter2",
            send_single_runs=False,
        )
    return _check(m["password"] == "hunter2", f"got {m['password']!r}")


def test_build_manifest_timestamps_are_iso_format() -> bool:
    import datetime
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(Path(tmp))
        m = os_.build_manifest(
            run_dir=str(run),
            user_id="alice",
            experiment_name="e",
            update=False, update_uuid=None, password=None,
            send_single_runs=False,
        )
    ts = m["created_at"]
    try:
        datetime.datetime.fromisoformat(ts)
    except ValueError as e:
        return _check(False, f"bad timestamp {ts!r}: {e}")
    return True


def test_build_manifest_includes_client_version() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(Path(tmp))
        m = os_.build_manifest(
            run_dir=str(run),
            user_id="alice",
            experiment_name="e",
            update=False, update_uuid=None, password=None,
            send_single_runs=False,
        )
    return _check(
        isinstance(m.get("client_version"), str) and m["client_version"],
        f"client_version missing: {m}",
    )


def test_build_manifest_experiment_name_override() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(Path(tmp))  # creates "experiment"
        m = os_.build_manifest(
            run_dir=str(run),
            user_id="alice",
            experiment_name="override-name",
            update=False, update_uuid=None, password=None,
            send_single_runs=False,
        )
    return _check(
        m["experiment_name"] == "override-name",
        f"got {m['experiment_name']!r}",
    )


# ---------------------------------------------------------------------------
# Content-type guessing
# ---------------------------------------------------------------------------


def test_content_type_guessing() -> bool:
    cases = {
        "results.csv": "text/csv",
        "data.json": "application/json",
        "log.txt": "text/plain",
        "errors.log": "text/plain",
        "0.out": "text/plain",
        "0.err": "text/plain",
        "image.png": "application/octet-stream",
        "archive.zip": "application/octet-stream",
        "no_extension": "application/octet-stream",
    }
    ok = True
    for name, expected in cases.items():
        got = os_._guess_content_type(name)
        ok &= _check(
            got == expected,
            f"{name!r}: expected {expected!r}, got {got!r}",
        )
    return ok


# ---------------------------------------------------------------------------
# Bundle integrity vs. corruption
# ---------------------------------------------------------------------------


def test_bundle_detects_corrupted_zip() -> bool:
    """If we corrupt the zip, read should fail."""
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(Path(tmp))
        m = os_.build_manifest(
            run_dir=str(run),
            user_id="alice",
            experiment_name="e",
            update=False, update_uuid=None, password=None,
            send_single_runs=False,
        )
        out_dir = Path(tmp) / "out"
        out_dir.mkdir()
        _, z_path = os_.write_bundle(m, out_dir, source_dir=str(run))
        # Corrupt the zip by truncating it
        size = z_path.stat().st_size
        with z_path.open("r+b") as f:
            f.truncate(size // 2)
        try:
            with zipfile.ZipFile(z_path) as zf:
                zf.namelist()
        except zipfile.BadZipFile:
            return True
    return _check(False, "expected BadZipFile on truncated bundle")


def test_bundle_manifest_content_matches_built_manifest() -> bool:
    """The manifest.json on disk must equal what build_manifest returned."""
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(Path(tmp))
        m = os_.build_manifest(
            run_dir=str(run),
            user_id="alice",
            experiment_name="e",
            update=False, update_uuid=None, password=None,
            send_single_runs=False,
        )
        out_dir = Path(tmp) / "out"
        out_dir.mkdir()
        m_path, _ = os_.write_bundle(m, out_dir, source_dir=str(run))
        # The manifest on disk won't equal exactly (created_at differs),
        # but every field except created_at must match.
        on_disk = json.loads(m_path.read_text())
    ok = True
    for k, v in m.items():
        if k == "created_at":
            continue
        ok &= _check(
            on_disk.get(k) == v,
            f"mismatch for {k!r}: on_disk={on_disk.get(k)!r} built={v!r}",
        )
    return ok


def test_bundle_rejects_non_deflated_method() -> bool:
    """Stored (uncompressed) zips should be rejected if our policy is
    compression-required. Currently we always DEFLATE so this is more
    a regression check."""
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(Path(tmp))
        m = os_.build_manifest(
            run_dir=str(run),
            user_id="alice",
            experiment_name="e",
            update=False, update_uuid=None, password=None,
            send_single_runs=False,
        )
        out_dir = Path(tmp) / "out"
        out_dir.mkdir()
        _, z_path = os_.write_bundle(m, out_dir, source_dir=str(run))
        # Re-write the zip with ZIP_STORED
        stored_path = out_dir / "stored.zip"
        with zipfile.ZipFile(z_path) as src, zipfile.ZipFile(
            stored_path, "w", zipfile.ZIP_STORED
        ) as dst:
            for name in src.namelist():
                dst.writestr(name, src.read(name))
        with zipfile.ZipFile(stored_path) as zf:
            info = zf.infolist()[0]
    return _check(
        info.compress_type == zipfile.ZIP_STORED,
        f"expected STORED, got {info.compress_type}",
    )


# ---------------------------------------------------------------------------
# Multi-folder / batch share (CLI behaviour)
# ---------------------------------------------------------------------------


def test_main_with_multiple_run_dirs_no_crash() -> bool:
    """Two folders on the CLI should at least parse and try to share
    (we don't have a server, so we expect a network error, not a parse
    error)."""
    with tempfile.TemporaryDirectory() as tmp:
        run1 = _make_run_dir(Path(tmp) / "a")
        run2 = _make_run_dir(Path(tmp) / "b")
        # No network -> exit non-zero, but the test is about parse + collect.
        rc = os_.main([str(run1), str(run2), "--username=alice", "--no_color"])
    return _check(rc != 0, f"expected non-zero (no network), got {rc}")


def test_main_with_unknown_run_dir_reports_error() -> bool:
    rc = os_.main(["/nonexistent/that/does/not/exist",
                   "--username=alice", "--no_color"])
    return _check(rc != 0, f"expected non-zero, got {rc}")


def test_main_help_flag_exits_zero() -> bool:
    rc = os_.main(["--help"])
    return _check(rc == 0, f"--help should exit 0, got {rc}")


# ---------------------------------------------------------------------------
# Determinism / reproducibility
# ---------------------------------------------------------------------------


def test_build_manifest_is_deterministic_for_same_run() -> bool:
    """The same run dir must produce a manifest with identical hashes
    every time (build_manifest must not be time-of-day dependent for
    the file list)."""
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(Path(tmp))
        m1 = os_.build_manifest(
            run_dir=str(run), user_id="alice", experiment_name="e",
            update=False, update_uuid=None, password=None,
            send_single_runs=False,
        )
        m2 = os_.build_manifest(
            run_dir=str(run), user_id="alice", experiment_name="e",
            update=False, update_uuid=None, password=None,
            send_single_runs=False,
        )
    ok = True
    for k in ("schema_version", "user_id", "experiment_name",
              "update_uuid", "password", "client_version"):
        ok &= _check(m1[k] == m2[k], f"{k} differs: {m1[k]!r} vs {m2[k]!r}")
    # File hashes must match exactly
    files1 = {f["archive_path"]: f for f in m1["files"]}
    files2 = {f["archive_path"]: f for f in m2["files"]}
    ok &= _check(set(files1) == set(files2),
                 f"file lists differ: {set(files1)} vs {set(files2)}")
    for path in files1:
        ok &= _check(
            files1[path]["sha256"] == files2[path]["sha256"],
            f"hash mismatch for {path}",
        )
    return ok


# ---------------------------------------------------------------------------
# Update flow CLI
# ---------------------------------------------------------------------------


def test_main_with_update_flag() -> bool:
    """--update should be parsed without error."""
    args = os_.parse_args(["/nonexistent", "--update"])
    return _check(args.update is True, f"got {args}")


def test_main_with_no_color_flag() -> bool:
    args = os_.parse_args(["/nonexistent", "--no_color"])
    return _check(args.no_color is True, f"got {args}")


def test_main_with_outfile() -> bool:
    args = os_.parse_args(["/nonexistent", "--outfile=/tmp/out.out"])
    return _check(args.outfile == "/tmp/out.out", f"got {args.outfile!r}")


def test_main_help_prints_usage() -> bool:
    """--help must include the word 'Usage' on stdout."""
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = os_.main(["--help"])
    output = buf.getvalue()
    ok = _check(rc == 0, f"--help exit: {rc}")
    ok &= _check("Usage" in output, f"missing Usage: {output!r}")
    ok &= _check("RUN_DIR" in output, f"missing RUN_DIR: {output!r}")
    return ok


# ---------------------------------------------------------------------------
# Discoverable tests
# ---------------------------------------------------------------------------


TESTS = [
    # File collection edge cases
    test_collect_includes_all_expected_top_level_extensions,
    test_collect_excludes_disallowed_extensions,
    test_collect_includes_extensionless_git_version,
    test_collect_picks_up_state_files_recursively,
    test_collect_single_runs_uses_digit_subdir_only,
    test_collect_single_runs_only_out_and_err,
    test_collect_excludes_empty_directories_at_top_level,
    test_collect_preserves_local_path_for_zip_writing,
    test_collect_includes_empty_files,
    test_collect_handles_uppercase_extension_variants,
    test_collect_resolves_symlinks_within_run_dir,
    test_collect_unicode_filenames,
    test_collect_filenames_with_spaces,
    test_collect_hidden_files_excluded_at_top_level,
    test_collect_many_files,
    # Hashing / integrity
    test_sha256_of_empty_file_is_known_value,
    test_sha256_of_binary_file_matches_stdlib,
    test_sha256_of_large_file,
    test_manifest_hash_matches_file_actually_written,
    test_manifest_size_matches_file_actually_written,
    test_manifest_rejects_file_larger_than_max,
    # Bundle structure
    test_bundle_contains_exactly_manifest_files,
    test_bundle_files_preserve_exact_bytes,
    test_bundle_manifest_is_valid_json,
    test_bundle_missing_source_file_raises,
    test_bundle_is_zip_deflated_not_stored,
    test_bundle_rejects_manifest_with_no_files,
    # Manifest validation
    test_verify_accepts_minimal_valid_manifest,
    test_verify_rejects_unknown_schema_version,
    test_verify_rejects_non_int_size,
    test_verify_rejects_non_string_sha256,
    test_verify_rejects_missing_files_key,
    test_verify_allows_extra_fields_in_manifest,
    test_verify_allows_extra_fields_per_file,
    test_verify_rejects_update_non_bool,
    # Path sanitization
    test_sanitize_blocks_windows_traversal,
    test_sanitize_blocks_absolute_paths,
    test_sanitize_blocks_null_bytes,
    test_sanitize_blocks_empty_and_whitespace,
    test_sanitize_blocks_dot_components,
    test_sanitize_accepts_url_encoded_chars_as_literal,
    test_sanitize_accepts_unicode_in_path,
    test_sanitize_accepts_deeply_nested_paths,
    # Username validation
    test_username_rejects_unicode_whitespace,
    test_username_accepts_unicode_letters,
    test_username_accepts_letters_digits_dots_dashes,
    test_username_accepts_long_but_not_unbounded,
    # Update flow
    test_build_manifest_includes_update_uuid_from_state_files,
    test_build_manifest_explicit_uuid_overrides_state_files,
    test_build_manifest_no_update_uuid_when_not_updating,
    test_build_manifest_no_run_uuid_file_no_crash,
    test_build_manifest_password_can_be_none,
    test_build_manifest_password_preserved,
    test_build_manifest_timestamps_are_iso_format,
    test_build_manifest_includes_client_version,
    test_build_manifest_experiment_name_override,
    # Content-type guessing
    test_content_type_guessing,
    # Bundle integrity
    test_bundle_detects_corrupted_zip,
    test_bundle_manifest_content_matches_built_manifest,
    test_bundle_rejects_non_deflated_method,
    # Multi-folder
    test_main_with_multiple_run_dirs_no_crash,
    test_main_with_unknown_run_dir_reports_error,
    test_main_help_flag_exits_zero,
    # Determinism
    test_build_manifest_is_deterministic_for_same_run,
    # CLI flags
    test_main_with_update_flag,
    test_main_with_no_color_flag,
    test_main_with_outfile,
    test_main_help_prints_usage,
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
    print(f"\nAll {len(TESTS)} comprehensive omniopt_share tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
