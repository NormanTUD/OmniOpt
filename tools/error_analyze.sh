#!/bin/bash
function echoerr() {
    echo "$@" 1>&2
}


function red_stdout {
        echo -e "\e[31m$1\e[0m"
}

function red_text {
        echoerr -e "\e[31m$1\e[0m"
}

function join_by { local d=${1-} f=${2-}; if shift 2; then printf %s "$f" "${@/#/$d}"; fi; }


set -e
set -o pipefail

function calltracer () {
        echo 'Last file/last line:'
        caller
}
trap 'calltracer' ERR

function help () {
        echo "Possible options:"
        echo "  --projectdir=(DIREXISTS)                                Project dir"
        echo "  --project                                               Project name"
        echo "  --nowhiptail                                            Disables dying"
        echo "  --help                                                  this help"
        echo "  --debug                                                 Enables debug mode (set -x)"
        exit $1
}
export whiptail=1
export projectdir=projects
export project

for i in $@; do
        case $i in
                --projectdir=*)
                        projectdir="${i#*=}"
                        if [[ ! -d $projectdir ]]; then
                                red_text "error: directory $projectdir does not exist" >&2
                                help 1
                        fi
                        shift
                        ;;
                --project=*)
                        project="${i#*=}"
                        shift
                        ;;
                -h|--help)
                        help 0
                        ;;
                --nowhiptail)
                        export whiptail=0
                        ;;
                --debug)
                        set -x
                        ;;
                *)
                        red_text "Unknown parameter $i" >&2
                        help 1
                        ;;
        esac
done

function error {
    MSG=$1
    EXIT=$2
    red_stdout "$MSG"
    if [[ "$whiptail" -eq "1" ]]; then
        eval `resize`
        SCROLLTEXT=
        if [[ "$(echo "$MSG" | wc -l)" -gt " $(($LINES/4))" ]]; then
        SCROLLTEXT=' --scrolltext '
        fi
        whiptail --title "Error Analyzer" --msgbox $SCROLLTEXT "$MSG" $LINES $COLUMNS
    fi

    if [[ ! -z $EXIT ]]; then
        exit 1
    fi
}

if [[ -z $projectdir ]]; then
        error "no --projectdir given" 1
fi

if [[ -z $project ]]; then
    error "No --project given" 1
fi

export thisprojectdir=$projectdir/$project
if [[ -z $thisprojectdir ]]; then
    error "\$thisprojectdir is empty" 1
fi

if [[ ! -d $thisprojectdir ]]; then
        error "$thisprojectdir could not be found" 1
fi

BUG="This might be a bug in OmniOpt. Create a debug-log-file via 'bash evaluate-run.sh' -> '$project' -> 'Create debug-zip' and send this file to <norman.koch@tu-dresden.de>."

ERRORS=()

if [[ ! -e "$thisprojectdir/config.ini" ]]; then
    ERRORS+=("$thisprojectdir/config.ini does not exist. This is a fatal error on your side.")
else
    OBJECTIVE_PROGRAM=$(cat $thisprojectdir/config.ini | grep "^objective_program" | sed -e 's/objective_program = //')
    if [[ $OBJECTIVE_PROGRAM == /* ]]; then
        ERRORS+=("The objective program starts with a '/', this means that this file must be executable and have a Shebang line. Are you sure this works? If not, add the program to run the script from in the beginning. Objective program:\\n$OBJECTIVE_PROGRAM")
        RUNFILE=$(echo $OBJECTIVE_PROGRAM | sed -e 's# .*##')
        if [[ ! -z $RUNFILE ]]; then
            if [[ ! -x $RUNFILE ]]; then
                ERRORS+=("The file '$RUNFILE' has no -x flag. Use\\nchmod +x $RUNFILE\\nand try again")
            fi
        fi
    fi

    set +e
    perl tools/showconfig.pl $project $thisprojectdir/config.ini 1 2>/dev/null >/dev/null
    exit_code=$?
    set -e
    if [[ "$exit_code" -ne "0" ]]; then
        ERRORS+=("There seem to be problems in the '$thisprojectdir/config.ini' (like missing values)")
    fi
fi

if [[ ! -d "$thisprojectdir/ipfiles" ]]; then
    ERRORS+=("$thisprojectdir/ipfiles could not be found. This may indicate that the job never ran or $BUG")
else
    if [ -z "$(ls -A $thisprojectdir/ipfiles)" ]; then
        ERRORS+=("$thisprojectdir/ipfiles exists, but is empty. $BUG")
    fi
fi

if [[ ! -d "$thisprojectdir/mongodb" ]]; then
    ERRORS+=("$thisprojectdir/mongodb could not be found. This is an indicator that the program has not yet run or starting MongoDB failed. $BUG")
else
    if [[ ! -e "$thisprojectdir/mongodb/WiredTiger" ]]; then
        ERRORS+=("$thisprojectdir/mongodb/WiredTiger could not be found. This is an indicator that the program has not yet run or running MongoDB failed. $BUG")
    fi
fi

singlelogs="$thisprojectdir/singlelogs" 
if [[ ! -d "$singlelogs" ]]; then
    ERRORS+=("$singlelogs could not be found. This is an indicator that the program has not yet run or that worker files could not be found. $BUG")
else
    echo "Choosing a random output file from $singlelogs" >&2
    RANDOM_SINGLE_FILE="$singlelogs/$(ls $singlelogs | grep -vi stdout | grep -vi stderr | shuf -n1)"
    if [[ ! -e $RANDOM_SINGLE_FILE ]]; then
        error "No singe file in $singlelogs could be found"
    else
        echo $RANDOM_SINGLE_FILE >&2
        RETURNCODE=$(cat $RANDOM_SINGLE_FILE | grep RETURNCODE | sed -e 's/.*RETURNCODE:\\n//' | sed -e 's/\\n.*//')
        if [[ "$RETURNCODE" -ne "0" ]]; then
            PROGRAMCODE=$(cat $RANDOM_SINGLE_FILE | head -n1 | sed -e 's/code: //')
            ERRORS+=("The file $RANDOM_SINGLE_FILE contains a non-0-return-status. This may indicate a problem with your program or how your program is called. It was called by using:\\n$PROGRAMCODE\\nCheck if this is correct and fix the config.ini if it is not.")
        fi
    fi
fi


if [[ ! -d "$thisprojectdir/logs/" ]]; then
    ERRORS+=("$thisprojectdir/logs could not be found. This is an indicator that the program has not yet run or that there was an error in OmniOpt")
else
    thislogfolder="$thisprojectdir/logs/$(ls -tr "$thisprojectdir/logs/" | tail -n1)"
    echo "Analyzing only the last created logs folder, called $thislogfolder" >&2
    start_worker_log="$thislogfolder/log-start-worker.log" 
    if [[ -s $start_worker_log ]]; then
        ERRORS+=("$start_worker_log NOT empty. It contains: '$(cat $start_worker_log)'. $BUG")
    else
        echo "$start_worker_log is empty, thats allright" >&2
    fi
fi

mongodblog="$thisprojectdir/mongodb.log"
if [[ ! -e "$mongodblog" ]]; then
    ERRORS+=("$mongodblog NOT found. This may indicate the job has never run or $BUG")
else
    if [[ -s "$mongodblog" ]]; then
        set +e
        EXITCODE=$(cat "$mongodblog" | grep "shutting down with code:" | sed -e 's/.*code://')
        set -e
        if [[ "$EXITCODE" -ne "0" ]]; then
            ERRORS+=("$mongodblog indicates an exit-status of $EXITCODE. It should only be 0. This means something with MongoDB went wrong. $BUG")
        elif [[ -z $EXITCODE ]]; then
            ERRORS+=("$mongodblog contains no exit-code. This means the job is still running something with MongoDB went wrong. $BUG")
        fi
    else
        ERRORS+=("$mongodblog exists but is empty. This means something with MongoDB went wrong. $BUG")
    fi
fi

if pwd | grep "/home/"; then
    QUOTA_DATA=$(quota -u $USER | grep -A1 hrsk_userhome | tail -n1)
    BLOCKS=$(echo $QUOTA_DATA | awk '{print $1}')
    MAX_BLOCKS=$(echo $QUOTA_DATA | awk '{print $2}')
    MAX_USAGE_RATE=0.9
    MAX_SPACE_BEFORE_WARNING=$(echo "scale=1; $MAX_USAGE_RATE*$MAX_BLOCKS" | bc | sed -e 's/\..*//')

    if [[ "$MAX_SPACE_BEFORE_WARNING" -lt "$BLOCKS" ]]; then
        ERRORS+=("$(pwd) lies in /home and you are near your quota-limits (Used $BLOCKS of $MAX_BLOCKS (warning at $MAX_SPACE_BEFORE_WARNING)). You might just not have enough space to run OmniOpt in your home with this quota. Delete some old files in your home.")
    fi
fi

if [ ${#ERRORS[@]} -eq 0 ]; then
    error "No possible errors found. If the problems persist, $BUG"
else
    error "Possible errors: \n\n - $(echo -e $(join_by "\\n\\n - " "${ERRORS[@]}"))"
    exit 1
fi
