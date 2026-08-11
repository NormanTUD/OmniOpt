#!/usr/bin/env python3
"""Tests for omniopt_plot (Python rewrite).

Tightly coupled tests for the pure logic of the script:

  * Levenshtein distance / closest-match
  * discovery of plot types by scanning ``.omniopt_plot_*.py``
  * parsing ``# DESCRIPTION`` and ``# EXPECTED FILES`` metadata
  * argument parsing

The actual plot-generation is delegated to existing
``.omniopt_plot_<type>.py`` scripts and is not re-tested here.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent

sys.path.insert(0, str(REPO_ROOT))
from importlib.machinery import SourceFileLoader  # noqa: E402

op = SourceFileLoader(
    "omniopt_plot", str(REPO_ROOT / "omniopt_plot")
).load_module()

from _framework.helpers import red_text  # noqa: E402


def _check(condition: bool, message: str) -> bool:
    if not condition:
        red_text(f"FAIL: {message}")
        return False
    return True


# ---------------------------------------------------------------------------
# Levenshtein / closest-match
# ---------------------------------------------------------------------------


def test_levenshtein_identical() -> bool:
    return _check(op.levenshtein("abc", "abc") == 0, "identical strings")


def test_levenshtein_empty() -> bool:
    return _check(op.levenshtein("", "abc") == 3, "empty vs 'abc'")


def test_levenshtein_one_edit() -> bool:
    return _check(op.levenshtein("abc", "abd") == 1, "abc -> abd is 1 edit")


def test_levenshtein_completely_different() -> bool:
    d = op.levenshtein("abc", "xyz")
    return _check(d == 3, f"abc vs xyz should be 3, got {d}")


def test_find_closest_match_exact_substring() -> bool:
    matches = op.find_closest_match("scat", ["scatter", "scatter_hex", "general"])
    return _check(
        len(matches) == 2 and "scatter" in matches[0],
        f"expected 2 matches starting with scatter, got {matches}",
    )


def test_find_closest_match_levenshtein() -> bool:
    """No exact substring match -> fall back to Levenshtein."""
    matches = op.find_closest_match("scattr", ["scatter", "general", "kde"])
    return _check(
        matches[0] == "scatter",
        f"expected 'scatter' as closest match, got {matches}",
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def test_parse_args_no_args() -> bool:
    args = op.parse_args([])
    return _check(
        args.run_dir is None and args.plot_type == "menu",
        f"defaults wrong: {args}",
    )


def test_parse_args_run_dir() -> bool:
    args = op.parse_args(["--run_dir=runs/test/0"])
    return _check(
        args.run_dir == "runs/test/0",
        f"got run_dir={args.run_dir!r}",
    )


def test_parse_args_plot_type() -> bool:
    args = op.parse_args(["--plot_type=scatter"])
    return _check(args.plot_type == "scatter", f"got {args.plot_type!r}")


def test_parse_args_min_max() -> bool:
    args = op.parse_args(["--min=0.1", "--max=0.9"])
    return _check(
        args.min == "0.1" and args.max == "0.9",
        f"got min={args.min!r} max={args.max!r}",
    )


def test_parse_args_help() -> bool:
    args = op.parse_args(["--help"])
    return _check(args.help is True, f"got help={args.help!r}")


def test_parse_args_save_to_file() -> bool:
    args = op.parse_args(["--save_to_file=plot.svg"])
    return _check(args.save_to_file == "plot.svg", f"got {args.save_to_file!r}")


def test_parse_args_allow_axes() -> bool:
    args = op.parse_args(["--allow_axes=x,y"])
    return _check(args.allow_axes == "x,y", f"got {args.allow_axes!r}")


# ---------------------------------------------------------------------------
# Plot-type discovery / metadata parsing
# ---------------------------------------------------------------------------


def _write_dummy_plot(path: Path, *, description: str = "", expected: str = "") -> None:
    lines = ["#!/usr/bin/env python3"]
    if description:
        lines.append(f"# DESCRIPTION: {description}")
    if expected:
        lines.append(f"# EXPECTED FILES: {expected}")
    lines.append("# add_argument('--save_to_file', action='store_true')")
    lines.append("# args.min = 0")
    path.write_text("\n".join(lines) + "\n")


def test_list_plot_types_finds_all() -> bool:
    """In the real repo, .omniopt_plot_*.py should be discoverable."""
    types = op.list_plot_types(str(REPO_ROOT))
    ok = _check("scatter" in types, f"scatter missing: {types}")
    ok &= _check("general" in types, f"general missing: {types}")
    return ok


def test_list_plot_types_in_tempdir() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / ".omniopt_plot_foo.py").write_text("# nothing\n")
        (Path(tmp) / ".omniopt_plot_bar.py").write_text("# nothing\n")
        types = op.list_plot_types(tmp)
    return _check(
        types == ["foo", "bar"] or set(types) == {"foo", "bar"},
        f"got {types}",
    )


def test_get_plot_description() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        _write_dummy_plot(
            Path(tmp) / ".omniopt_plot_test.py",
            description="This is a test description",
        )
        desc = op.get_plot_description("test", tmp)
    return _check(desc == "This is a test description", f"got {desc!r}")


def test_get_expected_files_single() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        _write_dummy_plot(
            Path(tmp) / ".omniopt_plot_test.py",
            expected="results.csv",
        )
        files = op.get_expected_files("test", tmp)
    return _check(files == ["results.csv"], f"got {files!r}")


def test_get_expected_files_multiple() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        _write_dummy_plot(
            Path(tmp) / ".omniopt_plot_test.py",
            expected="results.csv, cpu_ram_usage.csv",
        )
        files = op.get_expected_files("test", tmp)
    return _check(
        files == ["results.csv", "cpu_ram_usage.csv"],
        f"got {files!r}",
    )


def test_get_expected_files_missing_returns_empty() -> bool:
    files = op.get_expected_files("nonexistent", "/nonexistent")
    return _check(files == [], f"got {files!r}")


def test_plot_type_accepts_min_max_true() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        _write_dummy_plot(Path(tmp) / ".omniopt_plot_test.py")
        ok = op.plot_type_accepts_min_max("test", tmp)
    return _check(ok is True, "should accept min/max")


def test_plot_type_accepts_min_max_false_when_disabled() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / ".omniopt_plot_test.py"
        path.write_text("# no min here\n")
        ok = op.plot_type_accepts_min_max("test", tmp)
    return _check(ok is False, f"got {ok!r}")


def test_plot_type_supports_save_to_file_true() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        _write_dummy_plot(Path(tmp) / ".omniopt_plot_test.py")
        ok = op.plot_type_supports_save_to_file("test", tmp)
    return _check(ok is True, "should support save_to_file")


def test_plot_type_supports_save_to_file_false_when_missing() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / ".omniopt_plot_test.py"
        path.write_text("# no save_to_file\n")
        ok = op.plot_type_supports_save_to_file("test", tmp)
    return _check(ok is False, f"got {ok!r}")


def test_validate_plot_type_valid() -> bool:
    types = op.list_plot_types(str(REPO_ROOT))
    err = op.validate_plot_type(types[0], str(REPO_ROOT))
    return _check(err == "", f"unexpected error: {err!r}")


def test_validate_plot_type_invalid() -> bool:
    err = op.validate_plot_type("does_not_exist", str(REPO_ROOT))
    return _check(err != "", "expected error for invalid plot type")


def test_check_plot_prerequisites_all_files_present() -> bool:
    """All expected files exist in run_dir -> menu entry is available."""
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        run_dir.mkdir()
        (run_dir / "results.csv").write_text("a\n1\n")
        _write_dummy_plot(
            Path(tmp) / ".omniopt_plot_test.py",
            expected="results.csv",
        )
        ok = op.check_plot_prerequisites("test", str(run_dir), tmp)
    return _check(ok is True, "expected True")


def test_check_plot_prerequisites_missing_file() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        run_dir.mkdir()
        _write_dummy_plot(
            Path(tmp) / ".omniopt_plot_test.py",
            expected="results.csv",
        )
        ok = op.check_plot_prerequisites("test", str(run_dir), tmp)
    return _check(ok is False, f"expected False, got {ok!r}")


def test_resolve_run_dir_absolute_unchanged() -> bool:
    abs_path = "/already/absolute"
    return _check(
        op.resolve_run_dir(abs_path, "/some/pwd", False) == abs_path,
        "absolute path must be unchanged",
    )


def test_resolve_run_dir_relative_with_docker_user_dir() -> bool:
    out = op.resolve_run_dir("runs/x/0", "/work", True)
    return _check(out == "/work/docker_user_dir/runs/x/0", f"got {out!r}")


def test_resolve_run_dir_relative_without_docker_user_dir() -> bool:
    out = op.resolve_run_dir("runs/x/0", "/work", False)
    return _check(out == "/work/runs/x/0", f"got {out!r}")


TESTS = [
    # Levenshtein / closest-match
    test_levenshtein_identical,
    test_levenshtein_empty,
    test_levenshtein_one_edit,
    test_levenshtein_completely_different,
    test_find_closest_match_exact_substring,
    test_find_closest_match_levenshtein,
    # Argument parsing
    test_parse_args_no_args,
    test_parse_args_run_dir,
    test_parse_args_plot_type,
    test_parse_args_min_max,
    test_parse_args_help,
    test_parse_args_save_to_file,
    test_parse_args_allow_axes,
    # Discovery / metadata
    test_list_plot_types_finds_all,
    test_list_plot_types_in_tempdir,
    test_get_plot_description,
    test_get_expected_files_single,
    test_get_expected_files_multiple,
    test_get_expected_files_missing_returns_empty,
    test_plot_type_accepts_min_max_true,
    test_plot_type_accepts_min_max_false_when_disabled,
    test_plot_type_supports_save_to_file_true,
    test_plot_type_supports_save_to_file_false_when_missing,
    test_validate_plot_type_valid,
    test_validate_plot_type_invalid,
    test_check_plot_prerequisites_all_files_present,
    test_check_plot_prerequisites_missing_file,
    test_resolve_run_dir_absolute_unchanged,
    test_resolve_run_dir_relative_with_docker_user_dir,
    test_resolve_run_dir_relative_without_docker_user_dir,
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
    print("\nAll omniopt_plot tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
