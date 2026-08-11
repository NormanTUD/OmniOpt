#!/usr/bin/env python3
"""Find environment variables that change how OmniOpt works that are not documented."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _framework.helpers import red_text


REPO_ROOT = THIS_DIR.parent
MD_FILE = REPO_ROOT / ".gui" / "_tutorials" / "environment_variables.md"

EXCLUDE_VARS = {
    "time", "DISPLAY", "maxvalue", "minvalue", "KSH_VERSION", "SLURM_JOB_ID",
    "ZSH_EVAL_CONTEXT", "config_toml", "config_yaml", "min", "max", "CHAR",
    "CODE", "_reservation", "reservation", "force_local_execution",
    "_force_local_execution", "account", "_account", "exit_code", "git_hash",
    "_follow", "follow", "experiment_name", "current_tag", "BASH_VERSION",
    "gpus", "tag_commit_hash", "ORIGINAL_ARGS", "AVAILABLE_PROJECTS",
    "already_sent_signal", "bash_logname", "old_continue", "previous_job_var",
    "result", "mem_gb", "RUN_DIR", "DONT_ASK_USERNAME", "no_color", "outfile",
    "OUTPUT", "&&", "==", "1", "a", "x", "y", "z", "FROZEN", "LOAD_MODULES",
    "MYCLUSTER_HOST", "output", "specific_linter", "TEMP_BACKUP_FILE",
    "COMMAND", "run_folder", "GITHUB_STEP_SUMMARY", "NAME", "FOUND_FUNCS",
    "msg", "DEBUG", "CLUSTERHOST", "VIRTUAL_ENV", "FILTER_EXPERIMENT",
    "FILTER_USER", "RUN_WITH_COVERAGE", "CI", "ITWORKSONMYMACHINE",
    "LOG_PATH", "OO_MAIN_TESTS", "root_venv_dir", "PRINT_SEPARATOR",
    "RUN_WITH_PYSPY", "RUN_UUID", "NO_RUNTIME", "DONT_ASK_FILE", "!",
    "-f", "RUN_WITH_MEMRAY",
}


def main(argv=None) -> int:
    if not MD_FILE.exists():
        print(f"{MD_FILE} not found")
        return 255

    # Extract described params from the markdown file.
    described_params: set[str] = set()
    md_content = MD_FILE.read_text(encoding="utf-8", errors="ignore")
    for m in re.finditer(
        r'<td><pre class="invert_in_dark_mode"><code class="language-bash">(.*?)</code>',
        md_content,
        re.DOTALL,
    ):
        line = m.group(1).strip()
        if line.startswith("export "):
            line = line[len("export "):]
        line = line.split("=", 1)[0].strip()
        if line:
            described_params.add(line)

    # Find bash files in the repo. Exclude large build/cache dirs so
    # grep doesn't waste time walking .git/ etc.
    proc = subprocess.run(
        "grep -rIl '^#!/usr/bin/env bash' "
        "--exclude-dir=.git --exclude-dir=runs --exclude-dir=logs "
        "--exclude-dir=.mypy_cache --exclude-dir=__pycache__ .",
        shell=True,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    bash_files = [
        f for f in proc.stdout.splitlines()
        if "test_seed" not in f and "docker.sh" not in f
    ]

    errors = 0
    for bash_file in bash_files:
        try:
            content = Path(bash_file).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        found_params: set[str] = set()
        for m in re.finditer(
            r"if\s+\[\[?\s+\$([A-Za-z_][A-Za-z0-9_]*)", content,
        ):
            found_params.add(m.group(1))
        for m in re.finditer(
            r"if\s+\[\[?\s+-n?\s+\$([A-Za-z_][A-Za-z0-9_]*)", content,
        ):
            found_params.add(m.group(1))

        for param in sorted(found_params):
            if re.match(r"^[0-9]+$", param) or param in ("a", "x", "y", "z"):
                continue
            if any(excl in param for excl in EXCLUDE_VARS):
                continue
            if param not in described_params:
                red_text(
                    f"Parameter found {bash_file} but not in {MD_FILE}: {param}"
                )
                errors += 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
