#!/bin/bash

function echoerr() {
    echo "$@" 1>&2
}


function red_stdout {
        echo -e "\e[31m$1\e[0m"
}


function yellow_stdout {
        echo -e "\u001b[33m$1\e[0m"
}

function yellow_text {
        echoerr -e "\u001b[33m$1\e[0m"
}

function red_text {
        echoerr -e "\e[31m$1\e[0m"
}

function join_by { local d=${1-} f=${2-}; if shift 2; then printf %s "$f" "${@/#/$d}"; fi; }


function calltracer () {
        echo 'Last file/last line:'
        caller
}
trap 'calltracer' ERR

function help () {
        echo "Possible options:"
        echo "  --projectdir=(DIREXISTS)                                Project dir"
        echo "  --project=PROJECTNAME                                   Project name"
        echo "  --no_quota_test                                         Disables quota-test"
        echo "  --no_ssh_key_errors                                     Disables ssh-key-errors"
        echo "  --nowhiptail                                            Disables dying"
        echo "  --whiptail_only_on_error                                Show whiptail message only on error"
        echo "  --project_has_not_run                                   Disables checks that require that the project has already ran"
        echo "  --help                                                  this help"
        echo "  --debug                                                 Enables debug mode (set -x)"
        exit $1
}

export quota_tests=1
export whiptail=1
export projectdir=projects
export project
export debug=0
export project_has_run=1
export whiptail_only_on_error=0
export no_ssh_key_errors=0

for i in "$@"; do
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
                --whiptail_only_on_error)
                        export whiptail_only_on_error=1
                        ;;
                --project_has_not_run)
                        export project_has_run=0
                        ;;
                --no_quota_test)
                        export quota_tests=0
                        ;;
                --no_ssh_key_errors)
                        export no_ssh_key_errors=1
                        ;;
                --nowhiptail)
                        export whiptail=0
                        ;;
                --debug)
                        set -x
                        export debug=1
                        ;;
                *)
                        red_text "Unknown parameter $i" >&2
                        help 1
                        ;;
        esac
done

function debug {
    MSG=$1
    if [[ "$debug" -eq "1" ]]; then
        echoerr "DEBUG: $MSG"
    fi
}

function warning {
    MSG=$1
    EXIT=$2
    yellow_stdout "$MSG"
    if [[ "$whiptail" -eq "1" ]]; then
        eval `resize`
        SCROLLTEXT=
        if [[ "$(echo "$MSG" | wc -l)" -gt " $(($LINES/4))" ]]; then
        SCROLLTEXT=' --scrolltext '
        fi
        whiptail --title "Error Analyzer" --msgbox $SCROLLTEXT "$MSG" $LINES $COLUMNS
    fi

    if [[ "$EXIT" -ne "0" ]]; then
        exit $EXIT
    fi
}

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
        whiptail --title "Error/warning Analyzer" --msgbox $SCROLLTEXT "$MSG" $LINES $COLUMNS
    fi

    if [[ "$EXIT" -ne "0" ]]; then
        exit $EXIT
    fi
}

if [[ -z $projectdir ]]; then
    error "no --projectdir given" 1
fi

if [[ -z $project ]]; then
    error "No --project given" 2
fi

export thisprojectdir=$projectdir/$project
if [[ -z $thisprojectdir ]]; then
    error "\$thisprojectdir is empty" 3
fi

if [[ ! -d $thisprojectdir ]]; then
    error "$thisprojectdir could not be found" 4
fi

BUG="This might be a bug in OmniOpt. Create a debug-log-file via 'bash evaluate-run.sh' -> '$project' -> 'Create debug-zip' and send this file to <norman.koch@tu-dresden.de>."

ERRORS=()

if [[ ! -e "$thisprojectdir/config.json" ]]; then
    if [[ ! -e "$thisprojectdir/config.ini" ]]; then
        ERRORS+=("$thisprojectdir/config.ini does not exist. This is a fatal error on your side or $BUG.")
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

        perl tools/showconfig.pl $project $thisprojectdir/config.ini 1 2>/dev/null >/dev/null
        exit_code=$?
        if [[ "$exit_code" -ne "0" ]]; then
            ERRORS+=("There seem to be problems in the '$thisprojectdir/config.ini' (like missing values)")
        fi
    fi
fi

if [[ "$project_has_run" -eq "1" ]]; then
    LAST_SLURM_FILE=$(grep "Project: $project" *.out 2>/dev/null | sed -e 's/:.*//' | xargs ls -t1 | tail -n1)
    if [[ ! -z $LAST_SLURM_FILE ]]; then
        if [[ -e $LAST_SLURM_FILE ]]; then
            if [[ $(grep "Permission denied" "$LAST_SLURM_FILE" | wc -l) -ne "0" ]]; then
                LAST_ERROR_LINE=$(grep "Permission denied" $LAST_SLURM_FILE | tail -n1 | sed -e 's/\r//')
                ERRORS+=("There seemed to be problems with ssh: $LAST_ERROR_LINE")
            fi
        fi
    fi
fi

if [[ "$no_ssh_key_errors" -eq "0" ]]; then
    SSH_FOLDER_PERMISSION=$(stat -c '%a' $HOME/.ssh)
    if [[ "$SSH_FOLDER_PERMISSION" == "*700" ]]; then
        ERRORS+=("$HOME/.ssh has permission $SSH_FOLDER_PERMISSION, should be 700")
    fi

    SSH_FILES=(authorized_keys id_rsa id_rsa.pub known_hosts)
    SSH_PERMS=(600             600    644        644)
    cnt=0
    for ssh_file in "${SSH_FILES[@]}"; do
        ssh_file_path="$HOME/.ssh/$ssh_file"
        if [[ -e $ssh_file_path ]]; then
            should_be_permission=${SSH_PERMS[$cnt]}
            is_permissions=$(stat -c '%a' $ssh_file_path)
            if [[ "$is_permissions" -ne "$should_be_permission" ]]; then
                ERRORS+=("$ssh_file_path has permissions $is_permissions, should be: $should_be_permission")
            fi
        else
            ERRORS+=("$ssh_file_path does not exist")
        fi
        cnt=$((cnt+1))
    done
fi

if [[ ! -e "$HOME/.ssh/id_ed25519" ]]; then
    ERRORS+=("The file ~/.ssh/id_ed25519 does not exist. This means you might not be able to internally connect to all nodes. Run 'ssh-keygen -t ed25519' for creation the ed25519 certificate. You can try anyway, this probably won't case any problems.")
fi

if [[ -e $(ls -tr1 *.out 2>/dev/null | tail -n1) ]]; then
    if grep -ri 'oom-kill event' $(ls -tr1 *.out | tail -n1) 2>/dev/null >/dev/null; then
        ERRORS+=("OOM-killer was found. This may mean that the job was cancelled because it requested more RAM than is available.")
    fi
fi

if [[ "$project_has_run" -eq "1" ]]; then
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
        debug "Choosing a random output file from $singlelogs" >&2
        RANDOM_SINGLE_FILE="$singlelogs/$(ls $singlelogs | grep -vi stdout | grep -vi stderr | shuf -n1)"
        if [[ ! -e $RANDOM_SINGLE_FILE ]]; then
            error "No singe file in $singlelogs could be found" 0
        else
            debug $RANDOM_SINGLE_FILE >&2
            RETC=$(cat $RANDOM_SINGLE_FILE | grep "RETURNCODE" | sed -e 's/.*RETURNCODE:\\n//' | sed -e 's/\\n.*//')
            if [[ "$RETC" -ne "0" ]]; then
                PROGRAMCODE=$(cat $RANDOM_SINGLE_FILE | head -n1 | sed -e 's/code: //')
                ERRORS+=("The file $RANDOM_SINGLE_FILE contains a non-0-return-status. This may indicate a problem with your program or how your program is called. It was called by using:\\n$PROGRAMCODE\\nCheck if this is correct and fix the config.ini if it is not.")
            fi
        fi

        if [[ $(grep -m1 --max-count=1 "No such file or directory" $thisprojectdir/singlelogs/*.stderr | wc -l) -ne "0" ]]; then
            PROGRAMCODE="grep 'No such file or directory' $thisprojectdir/singlelogs/*.stderr"
            ERRORS+=("The folder $thisprojectdir/singlelogs/ contains files that contain the words 'No such file or directory'. If this job failed, please check your 'run.sh' file and use absolute paths. Check this command to find those files:\\n'$PROGRAMCODE'")
        fi

        LINES_WITH_IMPORT_ERRORS=$(grep -ri 'Failed to import' $thisprojectdir/singlelogs/*.stderr 2>/dev/null | sed -e "s/.*Error: ('//" | sed -e "s/').*$//" | egrep -v "^\s*$" | head -n1)

        if [[ $(echo "$LINES_WITH_IMPORT_ERRORS" |egrep -v '^\s*$' |  wc -l) -ne "0" ]]; then
            ERRORS+=("Maybe modules are missing. At least one stderr-file contains this line:\\n$LINES_WITH_IMPORT_ERRORS")
        fi

        LINES_WITH_IMPORT_ERRORS_TWO=$(grep -ri 'ModuleNotFoundError' $thisprojectdir/singlelogs/*.stderr 2>/dev/null | egrep -v '^\s*$' | head -n1)

        if [[ $(echo "$LINES_WITH_IMPORT_ERRORS_TWO" |egrep -v '^\s*$' |  wc -l) -ne "0" ]]; then
            ERRORS+=("Maybe modules are missing. At least one stderr-file contains this line:\\n$LINES_WITH_IMPORT_ERRORS_TWO")
        fi
    fi

    if [[ ! -d "$thisprojectdir/logs/" ]]; then
        ERRORS+=("$thisprojectdir/logs could not be found. This is an indicator that the program has not yet run or that there was an error in OmniOpt")
    else
        thislogfolder="$thisprojectdir/logs/$(ls -tr "$thisprojectdir/logs/" | tail -n1)"
        debug "Analyzing only the last created logs folder, called $thislogfolder" >&2
        start_worker_log="$thislogfolder/log-start-worker.log" 
        if [[ -s $start_worker_log ]]; then
            ERRORS+=("$start_worker_log NOT empty. It contains: '$(cat $start_worker_log)'. $BUG")
        else
            debug "$start_worker_log is empty, thats allright" >&2
        fi

        PROCESSLIMIT=$(perl tools/check_process_limit_file.pl $thislogfolder)
        if [[ "$?" -ne "0" ]]; then
            ERRORS+=("$PROCESSLIMIT")
        fi
    fi

    mongodblog="$thisprojectdir/mongodb.log"
    if [[ ! -e "$mongodblog" ]]; then
        ERRORS+=("$mongodblog NOT found. This may indicate the job has never run or $BUG")
    else
        if [[ -s "$mongodblog" ]]; then
            EXITCODE=$(cat "$mongodblog" | grep "shutting down with code:" | sed -e 's/.*code://')
            if [[ "$EXITCODE" -ne "0" ]]; then
                ERRORS+=("$mongodblog indicates an exit-status of $EXITCODE. It should only be 0. This means something with MongoDB went wrong. $BUG")
            elif [[ -z $EXITCODE ]]; then
                ERRORS+=("$mongodblog contains no exit-code. This means the job is still running something with MongoDB went wrong. $BUG")
            fi
        else
            ERRORS+=("$mongodblog exists but is empty. This means something with MongoDB went wrong. $BUG")
        fi
    fi
fi

if [[ "$quota_tests" -eq "1" ]]; then
    if pwd | grep "/home/" && command -v quota 2>/dev/null >/dev/null; then
        QUOTA_DATA=$(quota -u $USER | grep -A1 hrsk_userhome | tail -n1 2>/dev/null)
	exit_code=$?
	if [[ "$exit_code" -eq "0" ]]; then
		BLOCKS=$(echo $QUOTA_DATA | awk '{print $1}' | sed -e 's/\*//')
		MAX_BLOCKS=$(echo $QUOTA_DATA | awk '{print $2}')
		MAX_USAGE_RATE=0.9
		MAX_SPACE_BEFORE_WARNING=$(echo "scale=1; $MAX_USAGE_RATE*$MAX_BLOCKS" | bc | sed -e 's/\..*//')

		if [[ "$MAX_SPACE_BEFORE_WARNING" -lt "$BLOCKS" ]]; then
		    ERRORS+=("$(pwd) lies in /home and you are near your quota-limits (Used $BLOCKS of $MAX_BLOCKS (warning at $MAX_SPACE_BEFORE_WARNING)). You might just not have enough space to run OmniOpt in your home with this quota. Delete some old files in your home.")
		fi
	fi
    fi
fi

if [[ "${#ERRORS[@]}" -eq "0" ]]; then
    STR="No possible errors found."
    if [[ "$project_has_run" -eq "1" ]]; then
        STR="$STR If the problems persist, $BUG" 0
    fi
    warning "$STR"
    exit 0
else
    USE_WHIPTAIL=0

    if [[ $whiptail -eq 1 ]]; then
        USE_WHIPTAIL=1
    fi

    if [[ $whiptail_only_on_error -eq 1 ]]; then
        USE_WHIPTAIL=1
        whiptail=1
    fi

    if [[ "$USE_WHIPTAIL" -eq "1" ]]; then
        error "Possible errors: \n\n - $(echo -e $(join_by "\\n\\n - " "${ERRORS[@]}"))" 6
    else
        red_stdout "Possible errors: \n\n - $(echo -e $(join_by "\\n\\n - " "${ERRORS[@]}"))" 6
    fi
fi
