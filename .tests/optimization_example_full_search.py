#!/usr/bin/env python3
"""Optimization example based on the shekel-function (full search variant)."""

from __future__ import annotations

import argparse
import sys

from optimization_example import shekel  # reuse the same shekel function


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Full-search optimization example", add_help=False,
    )
    parser.add_argument("--param", type=str, default="")
    parser.add_argument("--param_two", type=str, default="")
    parser.add_argument("--param_three", type=str, default="")
    parser.add_argument("--param_four", type=str, default="")
    parser.add_argument("--help", "-h", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if args.help:
        parser.print_help()
        return 0

    if not all([args.param, args.param_two, args.param_three, args.param_four]):
        print("All parameters must be set", file=sys.stderr)
        return 1

    args_list = [
        float(args.param), float(args.param_two),
        float(args.param_three), float(args.param_four),
    ]
    result = shekel(args_list)
    print(f"RESULT: {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
