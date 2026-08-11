#!/usr/bin/env python3
"""Test if the seed is properly used and results are consistent and deterministic."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _framework.helpers import green_text, red_text


REPO_ROOT = THIS_DIR.parent


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Seed test", add_help=False)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--hash", dest="wanted_hash", type=str, default=None)
    parser.add_argument("--skip_search", action="store_true")
    parser.add_argument("--help", "-h", action="store_true")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if args.help:
        parser.print_help()
        return 0

    if args.seed is None or args.gpus is None or args.wanted_hash is None:
        print("Error: --seed, --gpus, and --hash are required")
        return 1

    omniopt_call = os.environ.get("OMNIOPT_CALL", "./omniopt")

    test_name = f"seed_test_{args.seed}"
    run_dir = REPO_ROOT / "runs" / test_name
    results_csv = run_dir / "0" / "results.csv"

    if run_dir.exists():
        shutil.rmtree(run_dir)

    cmd = [
        omniopt_call,
        "--partition=alpha",
        f"--experiment_name={test_name}",
        "--mem_gb=1",
        "--time=60",
        "--worker_timeout=5",
        "--max_eval=2",
        "--num_parallel_jobs=1",
        f"--gpus={args.gpus}",
        "--num_random_steps=1",
        "--follow",
        "--result_names", "RESULT=min",
        "--run_program=ZWNobyAiUkVTVUxUOiAlKGVwb2NocyklKGxyKSI=",
        "--cpus_per_task=1",
        "--nodes_per_job=1",
        "--model=BOTORCH_MODULAR",
        "--occ_type=euclid",
        "--run_mode=local",
        "--parameter", "epochs range 1 10 float false",
        "--parameter", "lr range 0 10 float false",
        "--seed", str(args.seed),
    ]
    if args.skip_search:
        cmd.append("--skip_search")

    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    exit_code = 0 if args.skip_search else proc.returncode
    if exit_code != 0:
        red_text(f"OmniOpt2 failed! Exit code should have been 0, but is: {exit_code}")
        return 2

    if not results_csv.exists():
        print(f"{results_csv} does not exist! Something went wrong!")
        return 1

    content = results_csv.read_text(encoding="utf-8", errors="ignore")
    last_col_lines = [line.rsplit(",", 1)[-1] for line in content.splitlines()[1:] if line.strip()]
    real_hash = hashlib.md5("\n".join(last_col_lines).encode("utf-8")).hexdigest()

    if args.seed == 1234:
        if sum(1 for line in content.splitlines() if line.startswith("0,") and "5.15" in line) != 1:
            print("ERROR: The first line does not contain the string 5.15 exactly once.")
            return 1

    print(f"Wanted Hash: '{args.wanted_hash}' (length: {len(args.wanted_hash)})")
    print(f"Real Hash:   '{real_hash}' (length: {len(real_hash)})")

    if real_hash == args.wanted_hash:
        green_text(f"Hash of the '{results_csv}' is {real_hash}")
        return 0

    red_text(f"{results_csv}")
    print(results_csv.read_text(encoding="utf-8", errors="ignore"))
    red_text(
        f"Hash of '{results_csv}' is different from the wanted hash "
        f"(wanted: {args.wanted_hash}, real: {real_hash})"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
