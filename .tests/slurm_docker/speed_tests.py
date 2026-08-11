#!/usr/bin/env python3
"""Run a battery of speed tests through ``run_docker`` (Python
replacement for ``speed_tests.sh``).

Iterates over all non-empty combinations of the ``PARAMS`` list and
invokes ``run_docker.py`` with each combination added to a fixed set of
arguments.  Writes the per-combination output into ``output/`` and
prints an ETA after every run.
"""

from __future__ import annotations

import itertools
import re
import subprocess
import sys
import time
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = THIS_DIR / "output"

ALWAYS_THERE = "--no_normalize_y --fit_out_of_design"

PARAMS = [
    "--fit_abandoned",
]

FIXED_ARGS = (
    "--num_random_steps=5 --max_eval=20 "
    "--num_parallel_jobs=5 --nr_nodes=5 --generate_all_jobs_at_once"
)


def sanitize_filename(input_str: str) -> str:
    cleaned = re.sub(r"--", "", input_str)
    cleaned = cleaned.replace(" ", ":")
    cleaned = cleaned.replace("_", "-")
    return cleaned


def _human_time(seconds: int) -> str:
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours >= 24:
        days, hours = divmod(hours, 24)
        return f"{days} days, {hours} hours and {minutes} minutes"
    return f"{hours} hours, {minutes} minutes and {secs} seconds"


def main() -> int:
    OUTPUT_DIR.mkdir(exist_ok=True)
    combos: list[tuple[str, ...]] = []
    for r in range(1, len(PARAMS) + 1):
        for combo in itertools.combinations(PARAMS, r):
            combos.append(combo)

    total = len(combos)
    durations: list[int] = []

    for index, combo in enumerate(combos):
        additional_args = " ".join(combo)
        filename_part = sanitize_filename(additional_args)
        output_file = OUTPUT_DIR / f"{filename_part}.txt"

        current = index + 1
        progress = current * 100 // total
        bar_width = 40
        filled = progress * bar_width // 100
        empty = bar_width - filled
        bar = "#" * filled + "-" * empty

        print(
            f"\r[{bar}] {progress:3d}% ({current}/{total}) Running with: {additional_args}",
            flush=True,
        )
        print(flush=True)

        cmd = [
            "python3",
            str(THIS_DIR / "run_docker.py"),
            *FIXED_ARGS.split(),
            f"--additional_parameter={ALWAYS_THERE} {additional_args}",
        ]
        start = time.time()
        with output_file.open("w") as fp:
            subprocess.run(cmd, stdout=fp, stderr=subprocess.STDOUT, check=False)
        end = time.time()
        duration = int(end - start)
        durations.append(duration)

        if current > 1:
            sorted_durations = sorted(durations)
            mid = current // 2
            if current % 2 == 0:
                median = (sorted_durations[mid - 1] + sorted_durations[mid]) // 2
            else:
                median = sorted_durations[mid]

            remaining = (total - current) * median
            print(f"\nEstimated remaining time: ~ {_human_time(remaining)}", flush=True)
        else:
            print(flush=True)

    print("\nAll jobs done!", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
