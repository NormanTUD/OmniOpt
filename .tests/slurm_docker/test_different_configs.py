#!/usr/bin/env python3
"""Run the slurm-docker test with different parameter combinations
(Python replacement for ``test_different_configs.sh``).

For each combination of ``max_eval``, ``num_parallel_jobs`` and
``num_random_steps`` between ``--min`` and ``--max`` (with the given
``--stepsize``) the script invokes ``run_docker`` with those flags.
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="test_different_configs.py",
        description="Run run_docker with different parameter combinations.",
    )
    parser.add_argument("--min", type=int, default=1, help="Min value (default: 1)")
    parser.add_argument("--max", type=int, default=10, help="Max value (default: 10)")
    parser.add_argument("--stepsize", type=int, default=2, help="Step size (default: 2)")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle commands before execution")
    args = parser.parse_args(argv)

    commands: list[list[str]] = []
    for max_eval in range(args.min, args.max + 1, args.stepsize):
        for num_parallel_jobs in range(args.min, args.max + 1, args.stepsize):
            for num_random_steps in range(args.min, args.max + 1, args.stepsize):
                commands.append([
                    "python3",
                    str(THIS_DIR / "run_docker.py"),
                    f"--num_random_steps={num_random_steps}",
                    f"--max_eval={max_eval}",
                    f"--num_parallel_jobs={num_parallel_jobs}",
                ])

    if args.shuffle:
        random.shuffle(commands)

    for cmd in commands:
        print("Running:", " ".join(cmd), flush=True)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f">>> Command failed: {' '.join(cmd)} <<<", file=sys.stderr)
            return result.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
