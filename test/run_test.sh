#!/bin/bash

function calltracer () {
        echo 'Last file/last line:'
        caller
}
trap 'calltracer' ERR

function help () {
        echo "Possible options:"
        echo "  --nocountdown                                   Don't show countdown"
        echo "  --no_quota_test                                 Disables quota-test"
        echo "  --project=PROJECTNAME                           Using this project"
        echo "  --projectdir=PROJECTDIR                         Using this projectdir"
        echo "  --usegpus(=num)                                 Use GPUs, if num not specified use 1"
        echo "  --help                                          this help"
        echo "  --time=HH:MM:SS                                 Defaults to 01:00:00"
        echo "  --debug                                         Enables debug mode (set -x)"
        exit $1
}
export quota_tests=1
export nocountdown=0
export project=
export projectdir=test/projects/
export usegpus=0
export timeparam_str="01:00:00"

for i in "$@"; do
        case $i in
                --no_quota_test)
                        quota_tests=0
                        shift
                        ;;
                --nocountdown)
                        nocountdown=1
                        shift
                        ;;
                --time=*)
                        timeparam_str="${i#*=}"
                        shift
                        ;;
                --usegpus=*)
                        usegpus="${i#*=}"
                        shift
                        ;;
                --usegpus)
                        usegpus=1
                        shift
                        ;;
                --project=*)
                        project="${i#*=}"
                        shift
                        ;;
                --projectdir=*)
                        projectdir="${i#*=}"
                        shift
                        ;;
                -h|--help)
                        help 0
                        ;;
                --debug)
                        set -x
                        ;;
                *)
                        echo "Unknown parameter $i" >&2
                        help 1
                        ;;
        esac
done


if [[ -z $projectdir ]]; then
        echo "--projectdir is empty"
        exit 2
fi

if [[ -z $project ]]; then
        echo "--project is empty"
        exit 3
fi

if [[ ! -d "$projectdir/$project" ]]; then
        echo "$projectdir/$project is not a directory"
        exit 4
fi

perl projects/cleanprojects.pl $projectdir/$project >/dev/null 2>/dev/null

THISLOGPATH=$HOME/${RANDOM}.out
while [ -e $THISLOGPATH ]; do
    THISLOGPATH=$HOME/${RANDOM}.out
done

echo $THISLOGPATH

countdown () {
        secs=$1 
        shift
        msg=$@ 
        if [[ "$nocountdown" == "0" ]]; then
                while [ $secs -gt 0 ]; do
                        printf "\r\033[KWaiting %.d seconds $msg (waited $SECONDS seconds in total)" $((secs--))
                        sleep 1
                done
                echo
        else
                sleep $secs
        fi
}

slurmlogpath () {
        if command -v scontrol &> /dev/null
        then
                if command -v grep &> /dev/null
                then
                        if command -v sed &> /dev/null
                        then
                                scontrol show job $1 | grep --color=auto --exclude-dir={.bzr,CVS,.git,.hg,.svn,.idea,.tox} StdOut | sed -e 's/^\s*StdOut=//'
                        else
                                red_text "sed not found"
                        fi
                else
                        red_text "grep not found"
                fi
        else
                red_text "scontrol not found"
        fi
}

job_still_running () {
        export SLURMID=$1
        if [[ -z $SLURMID ]]; then
            echo "job_still_running without valid Slurm-ID" >&2
            echo "1"
        else
            if [[ $(squeue -u $USER | grep $SLURMID | wc -l) == 0 ]]; then
                    echo "1"
            else
                    echo "0"
            fi
        fi
}

GPU_STRING=""

if [[ "$usegpus" -ne "0" ]]; then
        GPU_STRING=" --gres=gpu:1 --gpus-per-task=1 "
        sbatch_gpu="--num_gpus_per_worker=$usegpus"
fi

export SBATCH_RESULT=$(sbatch -J $project --cpus-per-task=4 $GPU_STRING --ntasks=1 --time=$timeparam_str --mem-per-cpu=2000 -o $THISLOGPATH sbatch.pl --project=$project --projectdir=$projectdir $sbatch_gpu --no_quota_test)
export SBATCH_ID=$(echo $SBATCH_RESULT | sed -e 's/.* job //')

if [[ -z "$SBATCH_ID" ]]; then
    exit 11
fi

sleep 10

if squeue | grep $SBATCH_ID | grep ReqNodeNotAvail; then
    scancel $SBATCH_ID
    exit 9
fi

if [[ -z $SBATCH_ID ]]; then
    echo "ERROR starting sbatch"
    exit 6
fi

echo ""
while [[ $(job_still_running $SBATCH_ID) -eq "0" ]]; do
    if [[ "$nocountdown" -eq "0" ]]; then
        tput cuu1
        tput el
    fi
    countdown 9 "before checking if the job $SBATCH_ID is still running again"
done

echo ""
echo "Waited $SECONDS seconds before I got the results"

if [[ $(cat $THISLOGPATH | grep "Best result data" | wc -l) == 0 ]]; then
        echo "No jobs were found. The test has failed. Check the log $THISLOGPATH manually."
        exit 7
else
        echo "Jobs were found. The test has succeeded. You can still check the log $THISLOGPATH manually."
        no_quota_test_str=""
        if [[ "$quota_tests" -eq "0" ]]; then
            no_quota_test_str=" --no_quota_test "
        fi
        bash tools/error_analyze.sh --project=$project --projectdir=$projectdir --nowhiptail $no_quota_test_str 2>/dev/null > /dev/null
        exit_code=$?
        if [[ "$exit_code" -eq "0" ]]; then
            exit 0
        else
            cat $THISLOGPATH
            echo "But there seems to be an error with error_analyze.sh. Check the output above. (Exit-Code: $exit_code)"
            exit 8
        fi
fi
