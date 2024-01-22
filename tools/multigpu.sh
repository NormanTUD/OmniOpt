#!/bin/bash

set -e

function calltracer () {
        echo 'Last file/last line:'
        caller
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

trap 'calltracer' ERR

export num_gpus=0
export programfile=""
export maxtime="01:00:00"
export cpus_per_task=1
export max_mem_per_cpu=2000
export logfolder=$HOME
export force_redo=0
export reservation=""
export account=""
export projectfolder=""
export jobname=""
export COUNTDOWN=0

function help () {
        echo "Possible options:"
        echo "  --reservation=RESERVATION                       Reservation name"
        echo "  --account=ACCOUNT                               Account name"
        echo "  --num_gpus=NUMBEROFGPUS                         Number of GPUs (default: $num_gpus)"
        echo "  --programfile=PROGRAMFILE                       The program (e.g. bash command) you want to run, in a file"
        echo "  --maxtime=01:00:00                              Max time for this job"
        echo "  --cpus_per_task=CPUS_PER_TASK                   CPUs per task (default: $cpus_per_task)"
        echo "  --max_mem_per_cpu=MAX_MEM_PER_CPU               Max. memory (default: $max_mem_per_cpu)"
        echo "  --logfolder=LOGFOLDER                           Folder where the logs are stored (default: $HOME)"
        echo "  --projectfolder=PROJECTFOLDER                   Folder of the project"
        echo "  --jobname=JOBNAME                               Name of the job"
        echo "  --force_redo                                    Deletes outfile if it already exists and re-does the job"
        echo "  --countdown                                     Show countdown instead of sleeping silently"
        echo "  --help                                          this help"
        echo "  --debug                                         Debug"
        exit $1
}


for i in "$@"; do
        case $i in
                --num_gpus=*)
                        num_gpus="${i#*=}"
                        shift
                        ;;
                --jobname=*)
                        jobname="${i#*=}"
                        shift
                        ;;
                --max_mem_per_cpu=*)
                        max_mem_per_cpu="${i#*=}"
                        shift
                        ;;
                --account=*)
                        account="${i#*=}"
                        shift
                        ;;
                --reservation=*)
                        reservation="${i#*=}"
                        shift
                        ;;
                --cpus_per_task=*)
                        cpus_per_task="${i#*=}"
                        shift
                        ;;
                --maxtime=*)
                        maxtime="${i#*=}"
                        shift
                        ;;
                --projectfolder=*)
                        projectfolder="${i#*=}"
                        shift
                        ;;
                --countdown)
                        COUNTDOWN=1
                        shift
                        ;;
                --logfolder=*)
                        logfolder="${i#*=}"
                        shift
                        ;;
                --programfile=*)
                        programfile="${i#*=}"
                        shift
                        ;;
                --debug)
                        set -x
                        ;;
                --force_redo)
                        force_redo=1
                        ;;
                -h|--help)
                        help 0
                        ;;
                *)
                        echo "Unknown parameter $i" >&2
                        help 1
                        ;;
        esac
done

function sleep_or_countdown () {
    secs=$1
    text=$2
    if [[ "$COUNTDOWN" == 1 ]]; then
        while [ $secs -gt 0 ]; do
            echo -ne "$secs s $text, total elapsed time: $SECONDS s\033[0K\r"
            sleep 1
            : $((secs--))
        done
    else
        sleep $secs
    fi
}


if [[ -z $programfile ]]; then
        echo "--programfile was empty"
        exit 2
fi


if [[ -z $jobname ]]; then
        echo "--jobname was empty"
        exit 3
fi

thislogpath="$logfolder/${jobname}.out"
thisenvpath="$logfolder/${jobname}.env"

env > $thisenvpath

if [[ -e "$thislogpath" ]]; then
        if [[ "$force_redo" -eq "1" ]]; then
            rm "$thislogpath"
        else
            cat "$thislogpath"
            exit 0
        fi
fi

thislogpath=$(echo "$thislogpath" | sed -e 's#//#/#g')

gpu_string=""

echo "hostname (multigpu.sh): $(hostname)"

if [[ "$num_gpus" -lt "2" ]]; then

        bash $programfile > "$thislogpath"
else
        declare -a RESET_VARIABLES=(
                "SLRUM_WORKING_CLUSTER"
                "SLURM_TOPOLOGY_ADDR"
                "SLURM_TASKS_PER_NODE"
                "SLURM_TASK_PID"
                "SLURM_SUBMIT_HOST"
                "SLURM_SUBMIT_DIR"
                "SLURM_SPANK_cloud_type"
                "SLURM_SPANK_cloud_create"
                "SLURM_SPANK_beegfs_mount"
                "SLURM_SPAN_beegfs_meta"
                "SLURM_SPANK__beegfs_jobid"
                "SLURM_SPANK_beegfs_create"
                "SLURM_PROCID"
                "SLURM_PRIO_PROCESS"
                "SLURM_NODELIST"
                "SLURM_NODEID"
                "SLURM_NODE_ALIASES"
                "SLURM_NNODES"
                "SLURM_MEM_PER_CPU"
                "SLURM_LUSTRE"
                "SLURM_LOCALID"
                "SLURM_JOB_USER"
                "SLURM_JOB_UID"
                "SLURM_JOB_QOS"
                "SLURM_JOB_PARTITION"
                "SLURM_JOB_NUM_NODES"
                "SLURM_JOB_NODELIST"
                "SLURM_JOB_NAME"
                "SLURM_JOB_ID"
                "SLURM_JOBID"
                "SLURM_JOB_GID"
                "SLURM_JOB_CPUS_PER_NODE"
                "SLURM_JOB_ACCOUNT"
                "SLURM_GTIDS"
                "SLURMD_NODENAME"
                "SLURM_CPUS_ON_NODE"
                "SLURM_CONF"
                "SLURM_CLUSTER_NAME"
                "ENVIRONMENT"
        )

        for var in "${RESET_VARIABLES[@]}"; do
              eval "declare -g _ORIGINAL_$var=\$$var"
          done

          # Unset values
          for var in "${RESET_VARIABLES[@]}"; do
                unset $var
            done

        gpu_string=" --gres=gpu:$num_gpus"

        reservation_string=""
        if [[ "$reservation" -ne "" ]]; then
                reservation_string=" --reservation=$reservation "
        fi

        account_string=""
        if [[ "$account" -ne "" ]]; then
                account_string=" -A $account "
        fi

        deadline=""
        if [[ ! -z $SLURM_JOB_ID ]]; then
                endofthisjob=$(squeue -o "%e" -j "$SLURM_JOB_ID" | tail -n1)
                deadline=" --deadline=$endofthisjob"
        fi

        STRACE=""
        SBATCH_DEBUG=""
        STRACE_FILE=""
        if [[ -e ~/.oo_multigpu_debug ]]; then
            STRACE_FILE="$HOME/$RANDOM"
            SBATCH_DEBUG=" -vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv "
            STRACE="strace -o $STRACE_FILE "
            echo "strace-log can be found at $STRACE_FILE" >&2
        fi

        SBATCH_COMMAND="${STRACE}sbatch $SBATCH_DEBUG -J $jobname --cpus-per-task=$cpus_per_task $account_string $reservation_string $gpu_string --ntasks=1 $deadline --time=$maxtime --mem-per-cpu=$max_mem_per_cpu $programfile"
        SBATCH_RESULT=$($SBATCH_COMMAND)
        EXIT_CODE=$?

        echo "$SBATCH_COMMAND resulted in EXIT-CODE: $EXIT_CODE"

        if [ "$EXIT_CODE" -eq "0" ]; then
            SBATCH_ID=$(echo $SBATCH_RESULT | sed -e 's/.* job //')

            SLURMLOGPATH=$(slurmlogpath $SBATCH_ID)

            if squeue | grep $SBATCH_ID | grep ReqNodeNotAvail; then
                echo "ReqNodeNotAvail error. Exiting." >&2
                scancel $SBATCH_ID
                exit 9
            fi

            sleep_or_countdown 20 "waiting for job start"

            while [[ $(job_still_running $SBATCH_ID) -eq "0" ]]; do
                    sleep_or_countdown 10 "until checking again whether the job is still running"
            done

            if [[ -e "$SLURMLOGPATH" ]]; then
                cp "$SLURMLOGPATH" "$thislogpath"
            else
                echo "$SLURMLOGPATH could not be found." >&2
            fi
        else
            echo "WARNING: $SBATCH_COMMAND resulted in exit-code $EXIT_CODE" >&2
        fi

        for var in "${RESET_VARIABLES[@]}"; do
            eval "declare -g $var=\$_ORIGINAL_$var"
        done
fi

if [[ -e "$thislogpath" ]]; then
    cp "$thislogpath" "${thislogpath}_original"
    cat "${thislogpath}_original" | perl -le '
        use List::Util qw/sum/;
        my %outputs; 
        my @results = (); 
        while (<>) { 
                chomp; 
                if(/^(.*):\s*([-+]?\d+(?:\.\d+)?)\s*$/) { 
                        my $name = $1;
                        $value = $2;
                        $outputs{$name} = $value;
                        if($name =~ /^RESULT\d+/) {
                                push @results, $value;
                        }
                }
        };
        if(!exists($outputs{RESULT})) {
                print "RESULT: ".sqrt(sum(map { $_ ** 2 } @results));
        }
        for (keys %outputs) {
                print "$_: $outputs{$_}";
        }
    ' > "$thislogpath"

    cat "$thislogpath"
else
    echo "$thislogpath could not be found"
    exit 4
fi
