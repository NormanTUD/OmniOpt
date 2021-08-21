#!/bin/bash

cd projects
perl cleanprojects.pl ../test/projects/gpu_test >/dev/null 2>/dev/null
cd -

THISLOGPATH=$HOME/${RANDOM}.out
while [ -e $THISLOGPATH ]; do
    THISLOGPATH=$HOME/${RANDOM}.out
done

echo $THISLOGPATH

function calltracer () {
        echo 'Last file/last line:'
        caller
}
trap 'calltracer' ERR

function help () {
        echo "Possible options:"
        echo "  --nocountdown"
        echo "  --help                                             this help"
        echo "  --debug                                            Enables debug mode (set -x)"
        exit $1
}
export nocountdown=0
for i in $@; do
        case $i in
                --nocountdown)
                        nocountdown=1
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


countdown () {
        secs=$1 
        shift
        msg=$@ 
        if [[ "$nocountdown" == "0" ]]; then
                while [ $secs -gt 0 ]; do
                        printf "\r\033[KWaiting %.d seconds $msg" $((secs--))
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
        if [[ $(squeue -u $USER | grep $SLURMID | wc -l) == 0 ]]; then
                return 1
        else
                return 0
        fi
}

if ! sinfo --noheader -p alpha | grep -v resv 2>/dev/null >/dev/null; then
        echo "Alpha partition has no available nodes, not doing this test"
        exit 1
fi

export SBATCH_RESULT=$(sbatch -J gpu_test \
        --cpus-per-task=4 \
        --gres=gpu:1 \
        --gpus-per-task=1 \
        --ntasks=1 \
        --time=1:00:00 \
        --mem-per-cpu=2000 \
        --partition=alpha \
        -o $THISLOGPATH \
        sbatch.pl --project=gpu_test --projectdir=test/projects/)

export SBATCH_ID=$(echo $SBATCH_RESULT | sed -e 's/.* job //')

export SLURMLOG=$THISLOGPATH

while job_still_running $SBATCH_ID; do
        printf "\r\033[K"
        countdown 9 "before checking if the job $SBATCH_ID is still running again"
done

echo "Waited $SECONDS seconds before I got the results"

if [[ $(cat $SLURMLOG | grep "Best result data" | wc -l) == 0 ]]; then
        echo "No jobs were found. The test has failed. Check the log $SLURMLOG manually."
        exit 1
else
        echo "Jobs were found. The test has succeeded. You can still check the log $SLURMLOG manually."
        exit 0
fi
