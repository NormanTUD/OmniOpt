#!/usr/bin/env python3
"""Spin up a SLURM-in-Docker stack and run an OmniOpt experiment
(Python replacement for the ``run_docker`` bash script).

The script installs ``docker`` and ``docker-compose`` if missing,
generates a ``docker-compose.yml`` and a ``slurm.conf`` for the
requested number of nodes, builds the images, starts the stack and
finally runs an experiment inside the frontend container.

Pass ``--run_tests`` to invoke ``.tests/main.py`` instead of an
optimization run.
"""

from __future__ import annotations

import argparse
import base64
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List


THIS_DIR = Path(__file__).resolve().parent


def _install_if_missing(cmd: str, install_via_apt: bool = True,
                        install_via_curl: bool = False, exit_code: int = 2) -> None:
    if shutil.which(cmd) is not None:
        return
    if install_via_apt:
        subprocess.run(["sudo", "apt-get", "update"], check=False)
    if install_via_curl:
        subprocess.run(["bash", "-c", "curl -fsSL https://get.docker.com | bash"], check=False)
    elif install_via_apt:
        subprocess.run(["sudo", "apt-get", "install", "-y", cmd], check=False)
    if shutil.which(cmd) is None:
        print(f"Failed to install {cmd}", file=sys.stderr)
        sys.exit(exit_code)


def create_docker_compose_yml(nr_nodes: int) -> str:
    nodes = []
    for n in range(1, nr_nodes + 1):
        nodes.append(f"""  slurmnode{n}:
        build:
          context: ./node
          dockerfile: Dockerfile
        container_name: slurmnode{n}
        shm_size: '4g'
        hostname: slurmnode{n}
        user: admin
        volumes:
                - shared-vol:/home/admin:nocopy
                - ../../:/oo_dir
        environment:
                - SLURM_NODENAME=slurmnode{n}
                - SLURM_CPUS_ON_NODE=1
        links:
                - slurmmaster
""")
    nodes_block = "".join(nodes)
    return f"""services:
  slurmfrontend:
        build:
          context: ./frontend
          dockerfile: Dockerfile
        container_name: slurmfrontend
        shm_size: '4g'
        hostname: slurmfrontend
        user: admin
        volumes:
                - shared-vol:/home/admin
                - ../../:/oo_dir
        ports:
                - 8888:8888
  slurmmaster:
        build:
          context: ./master
          dockerfile: Dockerfile
        container_name: slurmmaster
        shm_size: '4g'
        hostname: slurmmaster
        user: admin
        volumes:
                - shared-vol:/home/admin:nocopy
                - ../../:/oo_dir
        environment:
                - SLURM_CPUS_ON_NODE=1
        ports:
                - 6817:6817
                - 6818:6818
                - 6819:6819

{nodes_block}

volumes:
        shared-vol:
"""


def create_slurm_conf(nr_nodes: int) -> str:
    return f"""ClusterName=cluster
SlurmctldHost=slurmmaster
MpiDefault=none
ProctrackType=proctrack/linuxproc
ReturnToService=1
SlurmdParameters=config_overrides
SlurmctldPidFile=/var/run/slurmctld.pid
SlurmctldPort=6817
SlurmdPidFile=/var/run/slurmd.pid
SlurmdPort=6818
SlurmdSpoolDir=/var/spool/slurmd
SlurmUser=slurm
StateSaveLocation=/var/spool/slurmctld
SwitchType=switch/none
TaskPlugin=task/none
InactiveLimit=0
KillWait=30
MinJobAge=300
SlurmctldTimeout=120
SlurmdTimeout=300
Waittime=0
DefMemPerCPU=8192
MaxMemPerCPU=8192
SchedulerType=sched/backfill
SelectType=select/cons_tres
AccountingStorageType=accounting_storage/none
JobAcctGatherType=jobacct_gather/linux
JobAcctGatherFrequency=30
JobCompType=jobcomp/none
SlurmctldDebug=debug2
SlurmctldLogFile=/var/log/slurmctld.log
SlurmdDebug=debug2
SlurmdLogFile=/var/log/slurmd.log
NodeName=DEFAULT State=UNKNOWN Sockets=1 ThreadsPerCore=1 CoresPerSocket=1
NodeName=slurmnode[1-{nr_nodes}] CPUs=1 RealMemory=8192
PartitionName=slurmpar Nodes=ALL Default=YES MaxTime=INFINITE State=UP
SchedulerParameters=MemSpecLimit=YES
"""


def _count_running_slurmnodes() -> int:
    result = subprocess.run(
        ["docker", "ps"],
        capture_output=True,
        text=True,
        check=False,
    )
    return sum(1 for line in result.stdout.splitlines() if "slurmnode" in line)


def _parse_int(value: str, option: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        print(f"Error: {option} must be an integer.", file=sys.stderr)
        sys.exit(1)
    return parsed


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_docker.py",
        description="Spin up a SLURM-in-Docker stack and run an OmniOpt experiment.",
    )
    parser.add_argument("--mem_gb", type=int, default=1)
    parser.add_argument("--time", type=int, default=60)
    parser.add_argument("--worker_timeout", type=int, default=1)
    parser.add_argument("--max_eval", type=int, default=4)
    parser.add_argument("--num_parallel_jobs", type=int, default=2)
    parser.add_argument("--num_random_steps", type=int, default=2)
    parser.add_argument("--max_nr_of_zero", type=int, default=3)
    parser.add_argument("--nr_nodes", type=int, default=4)
    parser.add_argument("--model", default="BOTORCH_MODULAR")
    parser.add_argument("--seed", default="")
    parser.add_argument("--additional_parameter", default="")
    parser.add_argument("--install_slurm", action="store_true")
    parser.add_argument("--force_choice_for_ranges", action="store_true")
    parser.add_argument("--generate_all_jobs_at_once", action="store_true")
    parser.add_argument("--live_share", action="store_true")
    parser.add_argument("--should_deduplicate", action="store_true")
    parser.add_argument("--run_tests", action="store_true")
    parser.add_argument("--send_anonymized_usage_stats", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--stop", action="store_true")
    return parser


def main(argv=None) -> int:
    _install_if_missing("docker", install_via_apt=True, exit_code=2)
    _install_if_missing("docker-compose", install_via_apt=True, exit_code=2)
    _install_if_missing("wget", install_via_apt=True, exit_code=2)
    _install_if_missing("git", install_via_apt=True, exit_code=2)
    _install_if_missing("docker", install_via_apt=True, exit_code=2)

    parser = build_argparser()
    args = parser.parse_args(argv)

    if args.stop:
        compose_path = THIS_DIR / "docker-compose.yml"
        compose_path.write_text(create_docker_compose_yml(_count_running_slurmnodes()))
        subprocess.run(["docker-compose", "-f", str(compose_path), "stop"], check=False)
        compose_path.unlink(missing_ok=True)
        return 0

    if args.num_parallel_jobs > args.nr_nodes:
        print("!!! More parallel jobs than number of nodes-about-to-be-created.")

    slurm_conf = create_slurm_conf(args.nr_nodes)
    (THIS_DIR / "slurm.conf").write_text(slurm_conf)
    for sub in ("frontend", "master", "node"):
        shutil.copy(THIS_DIR / "slurm.conf", THIS_DIR / sub / "slurm.conf")

    compose_path = THIS_DIR / "docker-compose.yml"
    compose_path.write_text(create_docker_compose_yml(args.nr_nodes))

    try:
        if args.install_slurm:
            if shutil.which("docker-compose") is None:
                if shutil.which("apt") is None:
                    print("Cannot install docker-compose. Apt needed, but not found.",
                          file=sys.stderr)
                    return 1
                subprocess.run(["sudo", "apt", "install", "-y", "docker-compose"],
                               check=False)
            subprocess.run(["docker-compose", "build", "slurmmaster"], check=False)
            subprocess.run(["docker-compose", "build", "slurmfrontend"], check=False)
            compose_text = compose_path.read_text()
            import re
            for match in re.findall(r"^(slurmnode\d+):", compose_text, re.MULTILINE):
                subprocess.run(["docker-compose", "build", match], check=False)

        subprocess.run(
            ["docker-compose", "-f", str(compose_path), "up", "-d", "--remove-orphans"],
            check=False,
        )

        for sub in ("frontend", "master", "node"):
            (THIS_DIR / sub / "slurm.conf").unlink(missing_ok=True)

        cmd: List[str]
        if args.run_tests:
            print("Discarding almost all other options because you chose to --run_tests")
            cmd = [
                "docker", "exec", "slurmfrontend",
                "python3", "/oo_dir/.tests/main.py",
                "--max_eval=2",
                "--num_random_steps=1",
                "--exit_on_first_error",
                "--no_linkchecker",
                "--no_linter",
                "--run_with_coverage",
                "--superquick",
                "--run_with_coverage",
                "--skip_test_job_nr",
                "--skip_worker_check",
            ]
        else:
            run_program = base64.b64encode(
                b'echo "RESULT: %(int_param)%(int_param_two)%(float_param)"'
            ).decode().rstrip("=")
            cmd = [
                "docker", "exec", "slurmfrontend", "python3", "/oo_dir/omniopt",
                "--partition", "is_ignored_here",
                "--experiment_name", "slurm_in_docker_test",
                f"--mem_gb={args.mem_gb}",
                f"--time={args.time}",
                f"--worker_timeout={args.worker_timeout}",
                f"--max_eval={args.max_eval}",
                f"--num_parallel_jobs={args.num_parallel_jobs}",
                "--gpus", "0",
                f"--run_program", run_program,
                "--parameter", "int_param", "range", "-100", "10", "int",
                "--parameter", "float_param", "range", "-100", "10", "float",
                "--parameter", "int_param_two", "range", "-100", "10", "int",
                "--follow",
                f"--num_random_steps={args.num_random_steps}",
                "--model", args.model,
                "--auto_exclude_defective_hosts",
                f"--max_nr_of_zero={args.max_nr_of_zero}",
                "--show_generate_time_table",
                "--show_generation_and_submission_sixel",
                "--no_sleep",
            ]

            if args.generate_all_jobs_at_once:
                cmd.append("--generate_all_jobs_at_once")
            if args.send_anonymized_usage_stats:
                cmd.append("--send_anonymized_usage_stats")
            if args.live_share:
                cmd.append("--live_share")
            if args.seed:
                cmd.append(f"--seed={args.seed}")
            if args.verbose:
                cmd.append("--verbose")
            if args.debug:
                cmd.append("--debug")
            if args.should_deduplicate:
                cmd.append("--should_deduplicate")
            if args.force_choice_for_ranges:
                cmd.append("--force_choice_for_ranges")

        if args.additional_parameter:
            cmd.append(args.additional_parameter)

        result = subprocess.run(cmd)
        return result.returncode
    finally:
        for sub in ("frontend", "master", "node"):
            (THIS_DIR / sub / "slurm.conf").unlink(missing_ok=True)
        compose_path.unlink(missing_ok=True)
        (THIS_DIR / "slurm.conf").unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
