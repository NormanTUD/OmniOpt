#!/bin/bash

function humanreadabletime {
    date +"%Y-%m-%d %H:%m:%S (%s)"
}

export RED='\033[0;31m'
export GREEN='\033[0;32m'
export YELLOW='\033[0;33m'
export RESET='\033[0m'
export ONWHITE='\033[0;100m'
export SECONDS=0

LMOD_CMD=/usr/share/lmod/lmod/libexec/lmod
module () {
	eval $($LMOD_CMD sh "$@")
}

function print_success {
	echo -e "$GREEN$1$RESET";
}

function print_error {
	echo -e "$RED$1$RESET";
}

function print_info {
	echo -e "$YELLOW$1$RESET";
}

function print_debug {
	echo -e "$ONWHITE$1$RESET";
}

function use_error {
	print_error "Module could not be used (module use $1)"
	return 15
}

function load_error {
	print_error "Module could not be loaded (module load $1)"
	return 10
}

function check_for_program {
    if command -v $1; then
        return 0
    else
        return 1
    fi
}

function module_already_loaded {
	if check_for_module; then
		NUMBEROFMODULESFOUND=$(eval $(/usr/share/lmod/lmod/libexec/lmod $SHELL list) | grep "$1" | wc -l | awk '{print $1}')
		echo $NUMBEROFMODULESFOUND
	else
		echo 0
	fi
}

function check_for_srun {
    if check_for_program "srun"; then
        return 0
    else
        return 1
    fi

}

function check_for_module {
    if check_for_program "module"; then
        return 0
    else
        return 1
    fi
}

function module_load {
    lmodpath=/usr/share/lmod/lmod/libexec/lmod
    if check_for_module; then
        module load $1 && print_success "$1 successfully loaded" || load_error $1
    elif [[ -e $lmodpath ]]; then
        /usr/share/lmod/lmod/libexec/lmod $SHELL load $1 > /dev/null && print_success "$1 successfully loaded" || load_error $1
    else
        print_error "The program module does not exist. Cannot load any modules"
    fi
}

function module_use {
    if check_for_module; then
        module use $1 && print_success "$1 successfully loaded" || use_error $1
    else
        print_error "The program module does not exist. Cannot load any modules"
    fi
}

function load_source {
	if [[ -n $1 ]]; then
		if [ -f $1 ]; then
			if bash -n $1; then
				if source $1; then
					print_success "$1 was successfully sourced"
				else
					print_error "Could not successfully source $1"
				fi
			else
				print_error "$1 exists, but it cannot be loaded due to syntax errors"
			fi
		else
			print_error "$1 does not exist!"
		fi
	else
		print_error "No Parameter for load_source"
	fi
}

function empty {
    local var="$1"

    # Return true if:
    # 1.    var is a null string ("" as empty string)
    # 2.    a non set variable is passed
    # 3.    a declared variable or array but without a value is passed
    # 4.    an empty array is passed
    if test -z "$var";  then
        echo "1"
        return

    # Return true if var is zero (0 as an integer or "0" as a string)
    elif [ "$var" == 0 ]
    then
        echo "1"
        return

    # Return true if var is 0.0 (0 as a float)
    elif [ "$var" == 0 ]
    then
        echo "1"
        return
    fi

    echo ""
}

function print_time {
	date +%Y-%m-%d-%_H:%M:%S
}

function displaytime {
    local T=$1
    local D=$((T/60/60/24))
    local H=$((T/60/60%24))
    local M=$((T/60%60))
    local S=$((T%60))
    (( $D > 0 )) && printf '%d days ' $D
    (( $H > 0 )) && printf '%d hours ' $H
    (( $M > 0 )) && printf '%d minutes ' $M
    (( $D > 0 || $H > 0 || $M > 0 )) && printf 'and '
    printf '%d seconds\n' $S
}

function print_time {
	duration=$SECONDS
    echo "$duration seconds, or:"
    displaytime $duration
    echo "Time now:"
    humanreadabletime
}

function print_error_and_exit {
    echo -e "$RED$1$RESET"
    exit 1
}

function safe_srun {
        if check_for_srun; then
                eval "srun $1"
        else
                eval "$1"
        fi
}

function mymkdir {
    if [ ! -d "$1" ]; then
        print_debug "Trying to create folder $1"
        mkdir -p $1 && print_success "$1 was successfully created" || print_error_and_exit "There was an error creating $1"
    else
        print_success "The folder $1 already exists"
    fi
}

function mycopy {
    print_debug "Trying to copy from $1 to $2"
    cp $1 $2 && print_success "$1 was successfully copied to $2" || print_error "There was an error copying $1 to $2"
}

function run_python3 {
        print_debug "python3 $*"
        python3 "$*"
}

function mysrunmultiplegpu {
        n=$3
        if check_for_module; then
            if [ ${#SLURM_JOB_ID} -ge 1 ]; then
                for i in $(seq 1 $n); do 
                    SRUNCODE="srun -n 1 --gres=gpu:1 --mpi=none --mem-per-cpu=$2 --no-kill $1";
                    print_debug "srun found. Executing $SRUNCODE nr $i";
                    eval $SRUNCODE;
                    echo "RETURN-CODE: $?";
                done
                sleep 30
            else
                print_debug "srun found, but not in slurm job. Executing '$1'"
                eval "$1"
                echo $?
            fi
        else
            print_debug "srun not found. Executing '$1' $3 times"
            for i in {1..$3}; do eval "$1" & done
            echo $?
        fi
}


function mysrunmultiple {
        n=$3
        if check_for_module; then
            if [ ${#SLURM_JOB_ID} -ge 1 ]; then
                #SRUNCODE="srun -n $n --cpus-per-task=5 --overcommit --mpi=none --mem-per-cpu=$2 --no-kill $1"
                SRUNCODE="srun -n $n --mpi=none --mem-per-cpu=$2 --no-kill $1"
                print_debug "srun found. Executing $SRUNCODE"
                eval $SRUNCODE 
                echo "RETURN-CODE: $?"
                sleep 30
            else
                print_debug "srun found, but not in slurm job. Executing '$1'"
                eval "$1"
                echo $?
            fi
        else
            print_debug "srun not found. Executing '$1' $3 times"
            for i in {1..$3}; do eval "$1" & done
            echo $?
        fi
}

function mysrun {
        if check_for_module; then
            if [ ${#SLURM_JOB_ID} -ge 1 ]; then
                #SRUNCODE="srun --exclusive --mpi=none --mem-per-cpu=$2 --no-kill $1"
                #SRUNCODE="srun --overcommit --mpi=none --mem-per-cpu=$2 --no-kill $1"
                SRUNCODE="for i in {1..$3}; do eval "$1" & done"
                print_debug "srun found. Executing $SRUNCODE"
                eval $SRUNCODE 
                echo $?
                sleep 10
            else
                print_debug "srun found, but not in slurm job. Executing '$1'"
                eval "$1"
                echo $?
            fi
        else
            print_debug "srun not found. Executing '$1'"
            eval "$1"
            echo $?
        fi
}

humanreadabletime
print_debug "Real path on Cluster --> sh $0 $*"
export STATUSTEXT="Invalid status code"
export PROJECTID=$1

SYSTEMNAME=`uname -a`
print_debug "Systemname: $SYSTEMNAME"

if [ -n "$SLURM_JOB_ID" ]; then
    print_debug "\$SLURM_JOB_ID = $SLURM_JOB_ID"
	print_debug "Partition: $SLURM_JOB_PARTITION"
else
	print_debug "Not running as a SLURM-Job (or at least \$SLURM_JOB_ID is not defined)"
fi

print_debug "User: $USER"
