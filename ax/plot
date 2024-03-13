#!/bin/bash

echo -ne "\033]30;OmniOpt: plot\007"

function echoerr() {
        echo "$@" 1>&2
}

function red_text {
        echoerr -e "\e[31m$1\e[0m"
}

set -e
set -o pipefail

function calltracer () {
        echo 'Last file/last line:'
        caller
}
trap 'calltracer' ERR
help=0

for i in $@; do
        case $i in
                --help|-h)
			help=1
                        ;;
                --debug)
                        set -x
                        ;;
        esac
done


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
source .general.sh

set +e
OUTPUT=$(python3 .plot.py $*)
exit_code=$?
set -e

if [[ "$exit_code" -ne "0" ]]; then
	OUTPUT=$(echo "$OUTPUT" | sed -z 's#.*DONELOADING##')
	if command -v whiptail 2>/dev/null >/dev/null; then
		error_message "$OUTPUT"
	else
		echo -- "$OUTPUT"
	fi
else
	if [[ "$help" -eq "1" ]]; then
		echo "$OUTPUT"
	fi
fi

echo -ne "\033]30;$(basename $(pwd)): $(basename $SHELL)\007"
