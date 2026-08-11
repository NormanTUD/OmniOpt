#!/usr/bin/env python3
"""Tag and push the local slurm-docker images to the GitHub container
registry (Python replacement for ``push_to_docker.sh``).
"""

from __future__ import annotations

import subprocess
import sys

USER = "normantud"
REPO = "omniopt"
IMAGES = [
    "slurm_docker_slurmfrontend",
    "slurm_docker_slurmmaster",
    "slurm_docker_slurmnode1",
    "slurm_docker_slurmnode2",
    "slurm_docker_slurmnode3",
    "slurm_docker_slurmnode4",
]


def main() -> int:
    for image in IMAGES:
        target = f"ghcr.io/{USER}/{REPO}/{image}:latest"
        print(f"→ Tagging and pushing {image}", flush=True)
        tag_result = subprocess.run(["docker", "tag", image, target], check=False)
        if tag_result.returncode != 0:
            print(f"Failed to tag {image}", file=sys.stderr)
            return tag_result.returncode
        push_result = subprocess.run(["docker", "push", target], check=False)
        if push_result.returncode != 0:
            print(f"Failed to push {target}", file=sys.stderr)
            return push_result.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
