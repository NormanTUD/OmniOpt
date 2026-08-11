#!/usr/bin/env python3
"""Using different raw_samples and num_restarts automatically to test differences."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


REPO_ROOT = THIS_DIR.parent

RAW_SAMPLES_LIST = [64, 512, 1024]
NUM_RESTARTS_LIST = [1, 10, 20]


def main(argv=None) -> int:
    for raw_samples in RAW_SAMPLES_LIST:
        for num_restarts in NUM_RESTARTS_LIST:
            testname = f"raw_samples{raw_samples}__nr_restarts_{num_restarts}"
            additional = (
                f"--username=raw_samples_and_num_restart_tests_6 "
                f"--raw_samples={raw_samples} --num_restarts={num_restarts}"
            )
            cmd = [
                f"{REPO_ROOT}/.tests/start_simple_optimization_run.py",
                "--max_eval=50",
                "--num_parallel_jobs=10",
                "--nr_results=1",
                "--num_random_steps=10",
                "--time=2400",
                f"--additional_parameter={additional} --show_generation_and_submission_sixel",
                "--revert_to_random_when_seemingly_exhausted",
                "--force_choice_for_ranges",
                f"--testname={testname}",
                "--flame_graph",
                "--generate_all_jobs_at_once",
            ]
            print(f"Starting Test: {testname}")
            subprocess.run(cmd, cwd=str(REPO_ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
