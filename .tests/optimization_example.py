#!/usr/bin/env python3
"""Simple optimization example based on the shekel-function."""

from __future__ import annotations

import argparse
import os
import sys


def shekel(args: list[float]) -> float:
    if len(args) > 4:
        raise ValueError("too many args, need 4")
    if len(args) < 4:
        raise ValueError("not enough args, need 4")
    for k, v in enumerate(args):
        try:
            float(v)
        except ValueError:
            raise ValueError(f"Invalid parameter {k}: {v} is not a number")

    beta = [1, 2, 2, 4, 4, 6, 3, 7, 5, 5]
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
        description="Simple optimization example", add_help=False,
    )
    parser.add_argument("--int_param", type=str, default="")
    parser.add_argument("--int_param_two", type=str, default="")
    parser.add_argument("--float_param", type=str, default="")
    parser.add_argument("--choice_param", type=str, default="")
    parser.add_argument("--fixed_param", type=str, default="")
    parser.add_argument("--fail_or_not", type=str, default="0")
    parser.add_argument("--random_sem", action="store_true")
    parser.add_argument("--nr_results", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--help", "-h", action="store_true")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if args.help:
        parser.print_help()
        return 0

    if not args.int_param:
        print("Parameter --int_param cannot be empty", file=sys.stderr)
        return 1
    if not args.int_param_two:
        print("Parameter --int_param_two cannot be empty", file=sys.stderr)
        return 1
    if not args.float_param:
        print("Parameter --float_param cannot be empty", file=sys.stderr)
        return 1
    if not args.choice_param:
        print("Parameter --choice_param cannot be empty", file=sys.stderr)
        return 1

    print(f"OO-Info: int_param: {args.int_param}")
    print(f"OO-Info: int_param_two: {args.int_param_two}")
    print(f"OO-Info: float_param: {args.float_param}")
    print(f"OO-Info: choice_param: {args.choice_param}")

    if "1" in args.fail_or_not:
        return 1

    if args.choice_param not in ("1", "2", "4", "8", "16", "hallo"):
        print(
            f"error: Invalid choice_param: {args.choice_param}. "
            "Must be 1, 2, 4, 8, 16 or hallo.",
            file=sys.stderr,
        )
        return 1

    choice = 10 if args.choice_param == "hallo" else float(args.choice_param)
    args_list = [
        float(args.int_param), choice,
        float(args.float_param), float(args.int_param_two),
    ]

    if args.nr_results == 1:
        result = shekel(args_list)
        print(f"RESULT: {result}")
        if args.random_sem or os.environ.get("random_sem"):
            import random
            print(f"SEM-RESULT: {random.random()}")
    else:
        import random
        for r in range(1, args.nr_results + 1):
            print(f"RESULT{r}: {r + random.random()}")
            if args.random_sem or os.environ.get("random_sem"):
                print(f"SEM-RESULT{r}: {random.random()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
