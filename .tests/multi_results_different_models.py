#!/usr/bin/env python3
"""Testing if all kinds of different models that support Multi-Objective-Optimization work properly."""

from __future__ import annotations

import base64
import os
import shutil
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _framework.helpers import red_text


REPO_ROOT = THIS_DIR.parent


def _b64(s: str) -> str:
    return base64.b64encode(s.encode("utf-8")).decode("ascii")


def main(argv=None) -> int:
    num_gpus = 1 if shutil.which("nvidia-smi") else 0
    omniopt_call = os.environ.get("OMNIOPT_CALL", "./omniopt")

    omniopt_file = REPO_ROOT / "omniopt"
    if not omniopt_file.exists():
        print("omniopt not found")
        return 1
    content = omniopt_file.read_text(encoding="utf-8", errors="ignore")
    import re
    match = re.search(r"^SUPPORTED_MODELS\s*=\s*\[(.*?)\]", content, re.MULTILINE | re.DOTALL)
    if not match:
        print("Could not find SUPPORTED_MODELS")
        return 1
    raw = match.group(1)
    models = [m.strip().strip('"\'') for m in raw.split(",") if m.strip()]

    errors = 0
    for model in models:
        run_name = f"multi_results_2_results_{model}"
        run_dir = REPO_ROOT / "runs" / run_name
        if run_dir.exists():
            shutil.rmtree(run_dir)

        cmd = [
            omniopt_call,
            "--live_share", "--send_anonymized_usage_stats",
            "--partition", "alpha",
            "--experiment_name", run_name,
            "--mem_gb=4", "--time", "60", "--worker_timeout=5", "--max_eval", "2",
            "--num_parallel_jobs", "1", "--gpus", str(num_gpus),
            "--run_program", _b64(
                "./.tests/optimization_example --int_param='%(int_param)' "
                "--float_param='%(float_param)' --choice_param='%(choice_param)' "
                "--int_param_two='%(int_param_two)' --nr_results=2"
            ),
            "--parameter", "int_param range -100 10 int",
            "--parameter", "float_param range -100 10 float",
            "--parameter", "choice_param choice 1,2,4,8,16,hallo",
            "--parameter", "int_param_two range -100 10 int",
            "--follow", "--num_random_steps", "1",
            "--model", "BOTORCH_MODULAR",
            "--auto_exclude_defective_hosts",
            "--result_names", "RESULT1=min", "RESULT2=min",
            f"--model={model}",
            "--generate_all_jobs_at_once",
        ]
        subprocess.run(cmd, cwd=str(REPO_ROOT))
        proc = subprocess.run(
            [omniopt_call, "--continue", f"{run_dir}/0"],
            cwd=str(REPO_ROOT),
        )
        if proc.returncode != 0:
            red_text(f"multi_results_different_models: {model} failed: {proc.returncode}")
            errors += 1

    return errors


if __name__ == "__main__":
    sys.exit(main())
