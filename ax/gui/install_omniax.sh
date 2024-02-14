#!/bin/bash

START_COMMAND_BASE64=$1

if [[ -z $START_COMMAND_BASE64 ]]; then
	echo "Missing argument for start-command (must be in base64)"
	exit 1
fi

set -o pipefail
set -u

function calltracer () {
        echo 'Last file/last line:'
        caller
}
trap 'calltracer' ERR

INTERACTIVE=0
if [[ $- == *i* ]]; then
	INTERACTIVE=1
fi

export LC_ALL=en_US.UTF-8

function echo_green {
        echo -e "\e[42m\e[97m$1\e[0m"
}

function echo_yellow {
        echo -e "\e[43m\e[97m$1\e[0m"
}

function echo_red {
        echo -e "\e[41m\e[97m$1\e[0m"
}

function echo_headline {
        echo -e "\e[4m\e[96m$1\e[0m"
}

COPY_FROM="https://github.com/NormanTUD/OmniOpt.git"

TO_DIR_BASE=omniopt
TO_DIR=$TO_DIR_BASE
TO_DIR_NR=0

while [[ -d $TO_DIR ]]; do
	TO_DIR_NR=$((TO_DIR_NR + 1))
	TO_DIR=${TO_DIR_BASE}_${TO_DIR_NR}
done

total=0
CLONECOMMAND="git clone --depth=1 $COPY_FROM $TO_DIR"

if [[ "$INTERACTIVE" == "1" ]] && command -v whiptail >/dev/null 2>/dev/null; then
	$CLONECOMMAND 2>&1 | tr \\r \\n | {
		while read -r line ; do
			cur=`grep -oP '\d+(?=%)' <<< ${line}`
			total=$((total+cur))
			percent=$(bc <<< "scale=2;100*($total/100)")
			echo "$percent/1" | bc
		done
	} | whiptail --title "Cloning" --gauge "Cloning OmniOpt for optimizing project '$PROJECTNAME'" 8 78 0 && echo_green 'Cloning successful' || echo_red 'Cloning failed'
else
	$CLONECOMMAND
fi

cd $TO_DIR/ax/

START_COMMAND=$(echo $START_COMMAND_BASE64 | base64 --decode)

if [[ $? -eq 0 ]]; then
	$START_COMMAND
else
	echo_red "Error: $START_COMMAND_BASE64 was not valid base64 code"
fi
