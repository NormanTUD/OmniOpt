#!/usr/bin/env python3
"""Remap absolute source paths inside a coverage data file.

CI coverage data may have been produced in a different checkout (for example
``/home/runner/work/OmniOpt/OmniOpt``) or inside the Docker test container
(``/var/opt/omniopt``).  When the report is generated later (or on another
machine) those paths no longer exist.  This helper rewrites them to the
current repository checkout so coverage.py can read the source and generate
HTML reports.
"""

from __future__ import annotations

import sys
from pathlib import Path

from coverage import CoverageData


REPO_ROOT = Path(__file__).resolve().parents[2]
PATH_MARKERS = ("/OmniOpt/", "/omniopt/", "/site-packages/")


def remap_source_path(path: str) -> str:
    candidate_path = Path(path)
    if candidate_path.exists():
        return str(candidate_path)

    normalized = path.replace("\\", "/")
    for marker in PATH_MARKERS:
        index = normalized.rfind(marker)
        if index < 0:
            continue
        tail = normalized[index + len(marker):]
        candidate = REPO_ROOT / tail
        if candidate.exists():
            return str(candidate)
    return path


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print("usage: coverage_remap.py INPUT_DATA OUTPUT_DATA", file=sys.stderr)
        return 2

    source = Path(args[0])
    destination = Path(args[1])
    data = CoverageData(basename=str(source))
    data.read()

    remapped = CoverageData(basename=str(destination))
    if data.has_arcs():
        arcs = {}
        for filename in data.measured_files():
            arcs[remap_source_path(filename)] = data.arcs(filename) or []
        remapped.add_arcs(arcs)
    else:
        lines = {}
        for filename in data.measured_files():
            lines[remap_source_path(filename)] = data.lines(filename) or []
        remapped.add_lines(lines)
    remapped.write()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
