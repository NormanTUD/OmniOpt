#!/bin/bash

{
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

function set_debug {
	trap 'echo -e "${CYAN}$(date +"%Y-%m-%d %H:%M:%S")${NC} ${MAGENTA}| Line: $LINENO | Exit: $? ${NC}${YELLOW}-> ${NC}${BLUE}[DEBUG]${NC} ${GREEN}$BASH_COMMAND${NC}"' DEBUG
}

function unset_debug {
	trap - DEBUG
}

function echoerr() {
	echo "$@" 1>&2
}

function red_text {
	echoerr -e "\e[31m$1\e[0m"
}

START_COMMAND_BASE64=$1

export reservation=""
for i in $@; do
	case $i in
		--reservation=*)
			reservation="${i#*=}"
			;;
		--debug)
			set_debug
			;;
	esac
done

if [[ -z $START_COMMAND_BASE64 ]]; then
	red_text "Missing argument for start-command (must be in base64)"
	exit 1
fi

set -o pipefail
set -u

function calltracer () {
        yellow_text 'Last file/last line:'
        caller
}
trap 'calltracer' ERR

INTERACTIVE=1

if ! tty 2>/dev/null >/dev/null; then
	INTERACTIVE=0
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

if [[ ! -z $reservation ]]; then
	export SBATCH_RESERVATION=$reservation
fi

COPY_FROM="https://github.com/NormanTUD/OmniOpt.git"

TO_DIR_BASE=omniopt
TO_DIR=$TO_DIR_BASE
TO_DIR_NR=0

while [[ -d $TO_DIR ]]; do
	TO_DIR_NR=$((TO_DIR_NR + 1))
	TO_DIR=${TO_DIR_BASE}_${TO_DIR_NR}
done

if ! command -v base64 >/dev/null 2>/dev/null; then
	red_text "❌base64 not found. Try installing it with 'sudo apt-get install base64' (depending on your distro)"
fi

if ! command -v curl >/dev/null 2>/dev/null; then
	red_text "❌curl not found. Try installing it with 'sudo apt-get install curl' (depending on your distro)"
fi

if ! command -v wget >/dev/null 2>/dev/null; then
	red_text "❌wget not found. Try installing it with 'sudo apt-get install wget' (depending on your distro)"
fi

if ! command -v uuidgen >/dev/null 2>/dev/null; then
	red_text "❌uuidgen not found. Try installing it with 'sudo apt-get install uuid-runtime' (depending on your distro)"
fi

if ! command -v git >/dev/null 2>/dev/null; then
	echo_red "❌git not found. Try installing it with 'sudo apt-get install python3' (depending on your distro)"
fi

if ! command -v python3 >/dev/null 2>/dev/null; then
	echo_red "❌python3 not found. Try installing it with 'sudo apt-get install python3' (depending on your distro)"
fi

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
	} | whiptail --title "Cloning" --gauge "Cloning OmniOpt for optimizing project..." 8 78 0 && echo_green 'Cloning successful' || echo_red 'Cloning failed'
else
	$CLONECOMMAND || {
		echo_red "Git cloning failed."
		exit 2
	}
fi

cd $TO_DIR/ax/

START_COMMAND=$(echo $START_COMMAND_BASE64 | base64 --decode)

if [[ $? -eq 0 ]]; then
	$START_COMMAND
else
	echo_red "Error: $START_COMMAND_BASE64 was not valid base64 code"
fi
}
