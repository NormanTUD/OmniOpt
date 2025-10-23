#!/usr/bin/env bash
set -euo pipefail

USER=normantud
REPO=omniopt
IMAGES=(slurm_docker_slurmfrontend slurm_docker_slurmmaster slurm_docker_slurmnode1 slurm_docker_slurmnode2 slurm_docker_slurmnode3 slurm_docker_slurmnode4)

for name in "${IMAGES[@]}"; do
	echo "→ Tagging and pushing $name"
	docker tag "$name" "ghcr.io/$USER/$REPO/$name:latest"
	docker push "ghcr.io/$USER/$REPO/$name:latest"
done
