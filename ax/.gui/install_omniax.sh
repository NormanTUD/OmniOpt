#!/bin/bash -i

{
start_command_base64=$1

if [[ -z $start_command_base64 ]]; then
	red_text "Missing argument for start-command (must be in base64)"
	exit 1
fi

green='\033[0;32m'
yellow='\033[0;33m'
blue='\033[0;34m'
cyan='\033[0;36m'
magenta='\033[0;35m'
nc='\033[0m'
is_interactive=1
reservation=""

function set_debug {
	trap 'echo -e "${cyan}$(date +"%Y-%m-%d %H:%M:%S")${nc} ${magenta}| Line: $LINENO ${nc}${yellow}-> ${nc}${blue}[DEBUG]${nc} ${green}$BASH_COMMAND${nc}"' DEBUG
}

function echoerr() {
	echo -e "$@" 1>&2
}

function red_text {
	echoerr "\e[31m$1\e[0m"
}

function yellow_text {
	echoerr "\e\033[0;33m$1\e[0m"
}

function calltracer {
	yellow_text 'Last file/last line:'
	caller
}

function check_command {
	local cmd=$1
	local install_hint=$2
	if ! command -v "$cmd" >/dev/null 2>/dev/null; then
		red_text "❌$cmd not found. Try installing it with 'sudo apt-get install $install_hint' (depending on your distro)"
		exit 1
	fi
}

function check_if_everything_is_installed {
	check_command base64 "base64"
	check_command curl "curl"
	check_command wget "wget"
	check_command uuidgen "uuid-runtime"
	check_command git "git"
	check_command python3 "python3"
}

function set_interactive {
	if ! tty 2>/dev/null >/dev/null; then
		is_interactive=0
	fi
}

check_if_everything_is_installed

set_interactive

trap 'calltracer' ERR

export reservation=""
for i in "$@"; do
	case $i in
		--reservation=*)
			reservation="${i#*=}"
			;;
		--debug)
			set_debug
			;;
	esac
done

if [[ $reservation != "" ]]; then
	export SBATCH_RESERVATION=$reservation
fi

set -o pipefail
set -u

export LC_ALL=en_US.UTF-8

github_repo="https://github.com/NormanTUD/OmniOpt.git"

to_dir_base=omniopt
to_dir=$to_dir_base
to_dir_nr=0

while [[ -d $to_dir ]]; do
	to_dir_nr=$((to_dir_nr + 1))
	to_dir=${to_dir_base}_${to_dir_nr}
done

total=0
clone_command="git clone --depth=1 $github_repo $to_dir"

if [[ "$is_interactive" == "1" ]] && command -v whiptail >/dev/null 2>/dev/null; then
	$clone_command 2>&1 | tr \\r \\n | {
		while read -r line ; do
			cur=`grep -oP '\d+(?=%)' <<< ${line}`
			total=$((total+cur))
			percent=$(bc <<< "scale=2;100*($total/100)")
			echo "$percent/1" | bc
		done
	} | whiptail --title "Cloning" --gauge "Cloning OmniOpt2 for optimizing project..." 8 78 0 && green_text 'Cloning successful' || red_text 'Cloning failed'
else
	$clone_command || {
		red_text "Git cloning failed."
		exit 2
	}
fi

ax_dir="$to_dir/ax/"

if [[ -d $ax_dir ]]; then
	cd $ax_dir

	start_command=$(echo $start_command_base64 | base64 --decode)

	if [[ $? -eq 0 ]]; then
		$start_command
	else
		red_text "Error: $start_command_base64 was not valid base64 code"
	fi
else
	red_text "Error: $ax_dir could not be found!"
fi
}
