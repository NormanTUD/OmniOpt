#!/usr/bin/env python3
"""Slurm frontend container entrypoint (Python replacement for the bash
``docker-entrypoint.sh``).  Sets ``SLURM_CPUS_ON_NODE`` from
``/proc/cpuinfo``, patches ``/etc/slurm/slurm.conf`` and starts the
services required by an OmniOpt frontend node.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


SLURM_CONF = Path("/etc/slurm/slurm.conf")


def _run(cmd: list[str]) -> None:
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=False)


def _cpu_count() -> int:
    try:
        text = Path("/proc/cpuinfo").read_text()
    except OSError:
        return 1
    return sum(1 for line in text.splitlines() if line.startswith("processor"))


def _patch_slurm_conf(replacement: str) -> None:
    if not SLURM_CONF.exists():
        print(f"WARNING: {SLURM_CONF} does not exist, skipping sed", flush=True)
        return
    text = SLURM_CONF.read_text()
    text = text.replace("REPLACE_IT", replacement)
    SLURM_CONF.write_text(text)


def main() -> int:
    cpus = _cpu_count()
    os.environ["SLURM_CPUS_ON_NODE"] = str(cpus)
    _patch_slurm_conf(str(cpus))

    _run(["sudo", "service", "munge", "start"])
    _run(["sudo", "service", "ssh", "start"])

    _run(["ssh-keygen", "-t", "rsa", "-f", "/home/admin/.ssh/id_rsa", "-N", ""])
    _run(["cp", "/home/admin/.ssh/id_rsa.pub", "/home/admin/.ssh/authorized_keys"])

    _run(["tail", "-f", "/dev/null"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
