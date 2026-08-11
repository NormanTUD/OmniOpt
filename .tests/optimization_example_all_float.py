#!/usr/bin/env python3
"""Very simple single-objective-optimization-problem based on the shekel-function where all inputs are float."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time


def shekel(args: list[float]) -> float:
    if len(args) != 4:
        raise ValueError(f"need 4 args, got {len(args)}")

    beta = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]
    C = [
        [4, 1, 8, 7, 3, 2, 5, 8, 6, 7],
        [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6],
        [4, 1, 8, 6, 3, 2, 5, 8, 6, 7],
        [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6],
    ]
    outer_sum = 0.0
    for i, beta_i in enumerate(beta):
        inner_sum = 0.0
        for j, x_j in enumerate(args):
            inner_sum += (x_j - C[j][i]) ** 2 + beta_i
        outer_sum += inner_sum
    return -outer_sum


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="All-float optimization example", add_help=False,
    )
    parser.add_argument("--x", type=str, default="")
    parser.add_argument("--y", type=str, default="")
    parser.add_argument("--z", type=str, default="")
    parser.add_argument("--a", type=str, default="")
    parser.add_argument("--random_sem", action="store_true")
    parser.add_argument("--help", "-h", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if args.help:
        parser.print_help()
        return 0

    if not all([args.x, args.y, args.z, args.a]):
        print("All parameters --x, --y, --z, --a must be set", file=sys.stderr)
        return 1

    if shutil.which("sbatch"):
        time.sleep(120)

    x, y, z, a = float(args.x), float(args.y), float(args.z), float(args.a)
    print(f"x: {x}")
    print(f"y: {y}")
    print(f"z: {z}")
    print(f"a: {a}")

    result = shekel([x, y, z, a])
    print(f"RESULT: {result}")
    if args.random_sem or os.environ.get("random_sem"):
        import random
        print(f"SEM-RESULT: {random.random()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
