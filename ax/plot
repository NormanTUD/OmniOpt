#!/bin/bash

#SBATCH --signal=B:USR1@120

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [ -n "${SLURM_JOB_ID:-}" ] ; then
	SLURM_FILE_SCRIPT_DIR=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
	SLURM_FILE_SCRIPT_DIR=$(dirname $SLURM_FILE_SCRIPT_DIR)

	if [[ -d $SLURM_FILE_SCRIPT_DIR ]]; then
		SCRIPT_DIR="$SLURM_FILE_SCRIPT_DIR"
	else
		echo "SLURM_FILE_SCRIPT_DIR $SLURM_FILE_SCRIPT_DIR not found, even though SLURM_JOB_ID exists ($SLURM_JOB_ID). Using SCRIPT_DIR=$SCRIPT_DIR"
	fi
fi

cd $SCRIPT_DIR

source .shellscript_functions

python3 .plot.py $*
