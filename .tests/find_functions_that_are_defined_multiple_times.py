#!/usr/bin/env python3
"""Find functions that are defined multiple times in *.py and could be moved to .helpers.py."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from collections import defaultdict

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


REPO_ROOT = THIS_DIR.parent
EXCLUDE = {
    "main", "parse_arguments", "plot_graph", "plot_graphs",
    "use_matplotlib", "update_graph", "get_args", "plot_single_graph",
    "print_debug", "filter_data", "plot_multiple_graphs", "set_margins",
    "set_title", "check_args", "save_to_file_or_show_canvas",
}


def main(argv=None) -> int:
    function_files: dict[str, list[str]] = defaultdict(list)

    for py_file in sorted(REPO_ROOT.glob("*.py")):
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for m in re.finditer(r"^def\s+(\w+)", content, re.MULTILINE):
            name = m.group(1)
            if name not in EXCLUDE:
                function_files[name].append(py_file.name)

    duplicates = {name: files for name, files in function_files.items() if len(files) > 1}
    error_count = len(duplicates)

    for name, files in sorted(duplicates.items()):
        print(f"{name}: {' '.join(files)}")

    return error_count


if __name__ == "__main__":
    sys.exit(main())
