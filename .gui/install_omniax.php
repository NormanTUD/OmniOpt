<?php
	include_once("_functions.php");

	function read_requirements_file($file_path) {
		if (!file_exists($file_path)) {
			trigger_error("File not found: " . $file_path, E_USER_ERROR);
		}

		$lines = file($file_path, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES);
		if ($lines === false) {
			trigger_error("Error reading the file: " . $file_path, E_USER_ERROR);
		}

		$cleaned_lines = array();
		foreach ($lines as $line) {
			$trimmed = trim($line);
			if ($trimmed !== '' && $trimmed[0] !== '#') {
				$cleaned_lines[] = $trimmed;
			}
		}

		return $cleaned_lines;
	}

	$pip_requirements = read_requirements_file("../requirements.txt");
	$pip_requirements[] = "omniopt2";

	$pip_commands = array();

	foreach ($pip_requirements as $requirement) {
		$requirement = trim($requirement);
		if ($requirement !== '') {
			$pip_commands[] = 'pip install -q ' . escapeshellarg($requirement);
		}
	}

	$total = count($pip_requirements);
	$bash_lines = array();

	$bash_lines[] = 'spin=("⠇" "⠏" "⠋" "⠙" "⠹" "⠸" "⠴" "⠦" "⠧")';
	$bash_lines[] = 'GREEN="\033[0;32m"';
	$bash_lines[] = 'RED="\033[0;31m"';
	$bash_lines[] = 'YELLOW="\033[1;33m"';
	$bash_lines[] = 'NC="\033[0m"';
	$bash_lines[] = 'total=' . $total;
	$bash_lines[] = 'current=0';
	$bash_lines[] = 'spin_index=0';
	$bash_lines[] = 'start_time=$(date +%s)';
	$bash_lines[] = 'declare -a failed=()';
	$bash_lines[] = 'installed=$(pip freeze --all | cut -d "=" -f 1 | tr "[:upper:]" "[:lower:]")';
	$bash_lines[] = '';
	$bash_lines[] = 'clear_line() { echo -ne "\r\033[K"; }';
	$bash_lines[] = 'cursor_up() { echo -ne "\033[A"; }';
	$bash_lines[] = 'progress_bar() {';
	$bash_lines[] = '  local filled=$(( ($current * 40) / $total ))';
	$bash_lines[] = '  local empty=$(( 40 - filled ))';
	$bash_lines[] = '  printf "[%s%s]" "$(printf "#%.0s" $(seq 1 $filled))" "$(printf " %.0s" $(seq 1 $empty))"';
	$bash_lines[] = '}';
	$bash_lines[] = 'estimate_time() {';
	$bash_lines[] = '  local now=$(date +%s)';
	$bash_lines[] = '  local elapsed=$((now - start_time))';
	$bash_lines[] = '  if [ $current -eq 0 ]; then echo "--:--"; return; fi';
	$bash_lines[] = '  local avg=$((elapsed / current))';
	$bash_lines[] = '  local remaining=$((avg * (total - current)))';
	$bash_lines[] = '  local min=$((remaining / 60))';
	$bash_lines[] = '  local sec=$((remaining % 60))';
	$bash_lines[] = '  printf "%02d:%02d" $min $sec';
	$bash_lines[] = '}';
	$bash_lines[] = '';
	$bash_lines[] = 'echo ""';
	$bash_lines[] = '';

	foreach ($pip_requirements as $index => $req) {
	    $bash_lines[] = 'current=' . ($index + 1);
	    $bash_lines[] = 'pkgname="' . strtolower(preg_replace('/[^a-zA-Z0-9_\[\]-]/', '', $req)) . '"';
	    $bash_lines[] = 'if echo "$installed" | grep -qx "$pkgname"; then';
	    $bash_lines[] = '  clear_line';
	    $bash_lines[] = '  echo -ne "$(progress_bar) ${GREEN}✔ Already installed: ' . $req . ' ($current/$total)${NC}"';
	    $bash_lines[] = 'else';
	    $bash_lines[] = '  i=0';
	    $bash_lines[] = '  while true; do';
	    $bash_lines[] = '    clear_line';
	    $bash_lines[] = '    echo -ne "$(progress_bar) ${GREEN}→ Installing ' . $req . ' ($current/$total) (ETA: $(estimate_time))${NC}"';
	    $bash_lines[] = '    sleep 0.25';
	    $bash_lines[] = '    i=$((i+1))';
	    $bash_lines[] = '    if [ $i -ge 4 ]; then break; fi';
	    $bash_lines[] = '  done';
	    $bash_lines[] = '  clear_line';
	    $bash_lines[] = '  if pip install -q ' . escapeshellarg($req) . '; then';
	    $bash_lines[] = '    echo -ne "$(progress_bar) ${GREEN}✔ Installed: ' . $req . ' ($current/$total)${NC}"';
	    $bash_lines[] = '  else';
	    $bash_lines[] = '    echo -e "${RED}✘ Failed: ' . $req . ' ($current/$total)${NC}"';
	    $bash_lines[] = '    failed+=(' . escapeshellarg($req) . ')';
	    $bash_lines[] = '    exit 1';
	    $bash_lines[] = '  fi';
	    $bash_lines[] = 'fi';
	    $bash_lines[] = 'echo ""';
	}

	$bash_lines[] = 'echo ""';
	$bash_lines[] = 'if [ ${#failed[@]} -eq 0 ]; then';
	$bash_lines[] = '  echo -e "${GREEN}✔ All packages installed successfully.${NC}"';
	$bash_lines[] = 'else';
	$bash_lines[] = '  echo -e "${RED}✘ The following packages failed to install:${NC}"';
	$bash_lines[] = '  for pkg in "${failed[@]}"; do echo -e "  ${RED}- $pkg${NC}"; done';
	$bash_lines[] = 'fi';

	$bash_script = implode("\n", $bash_lines);

	#dier($bash_script);

	header('Content-Type: application/bash');
?>
#!/usr/bin/env bash -i

{
set -euo pipefail

github_repo_url="https://github.com/NormanTUD/OmniOpt.git"
green='\033[0;32m'
yellow='\033[0;33m'
blue='\033[0;34m'
cyan='\033[0;36m'
magenta='\033[0;35m'
red='\033[0;31m'
nc='\033[0m'
is_interactive=1
depth=1
debug=0
reservation=""
omniopt_venv=omniopt_venv
no_whiptail=0
installation_method="clone"
dryrun=0

start_command_base64=""

function help {
	echo "OmniOpt2-Installer"
	echo ""
	echo "<start-command>                                             Command that should be started after cloning (decoded in base64)"
	echo "--depth=N                                                   Depth of git clone (default: 1, only used for --installation_method=clone)"
	echo "--reservation=str                                           Name of your reservation, if any"
	echo "--installation_method=str                                   How to install OmniOpt2 (default: clone, other option: pip)"
	echo "--no_whiptail                                               Disable whiptail for cloning"
	echo "--omniopt_venv=str                                          Path to virtual env dir (only used for --installation_method=pip, default: omniopt_venv)"
	echo "--dryrun                                                    Clone, download, install and then run in dryrun mode"
	echo "--debug                                                     Enable debug mode"
	echo "--help                                                      This help"

	exit 0
}

function set_debug {
	trap 'echo -e "${cyan}$(date +"%Y-%m-%d %H:%M:%S")${nc} ${magenta}| Line: $LINENO ${nc}${yellow}-> ${nc}${blue}[DEBUG]${nc} ${green}$BASH_COMMAND${nc}"' DEBUG
}

function echoerr() {
	echo -e "$@" 1>&2
}

function green_text {
	echoerr "${green}$1${nc}"
}

function red_text {
	echoerr "${red}$1${nc}"
}

function yellow_text {
	echoerr "${yellow}$1${nc}"
}

function calltracer {
	yellow_text 'Last file/last line:'
	caller
}

function dbg {
	msg="$1"
	if [[ $debug -eq 1 ]]; then
		yellow_text "DEBUG: $msg"
	fi
}

trap 'calltracer' ERR

function check_command {
	local cmd="$1"
	local install_hint="$2"

	dbg "check_command $cmd $install_hint"

	if ! command -v "$cmd" >/dev/null 2>/dev/null; then
		red_text "❌$cmd not found. Try installing it with 'sudo apt-get install $install_hint' (depending on your distro)"
		exit 14
	fi
}

function check_if_everything_is_installed {
	dbg "check_if_everything_is_installed"

	check_command bc "bc"
	check_command base64 "base64"
	check_command curl "curl"
	check_command wget "wget"
	check_command uuidgen "uuid-runtime"
	check_command git "git"
	check_command python3 "python3"
	check_command whiptail "whiptail"
}

function check_interactive {
	dbg "check_interactive"

	if [[ $- == *i* ]]; then
		is_interactive=0
		dbg "check_interactive: \$- ($-) does not contain 'i', setting is_interactive to 0"
	fi
}

function get_to_dir {
	dbg "get_to_dir"

	to_dir_base=omniopt
	to_dir=$to_dir_base
	to_dir_nr=0

	while [[ -d $to_dir ]]; do
		to_dir_nr=$((to_dir_nr + 1))
		to_dir=${to_dir_base}_${to_dir_nr}
	done

	echo "$to_dir"
}

function parse_parameters {
	args=$@

	for i in $args; do
		case $i in
			--omniopt_venv=*)
				omniopt_venv="${i#*=}"
				;;
			--installation_method=*)
				installation_method="${i#*=}"
				;;
			--depth=*)
				depth="${i#*=}"
				;;
			--reservation=*)
				reservation="${i#*=}"
				;;
			--no_whiptail)
				no_whiptail=1
				;;
			--dryrun)
				dryrun=1
				;;
			--debug)
				set_debug
				debug=1
				;;
			-h)
				help
				;;
			--help)
				help
				;;
			*)
				set +e
				echo "$i" | base64 --decode 2>/dev/null >/dev/null
				start_command_base64="$i"
				start_command_exit_code=$?
				set -e

				if [[ $start_command_exit_code -ne 0 ]]; then
					red_text "Invalid parameter $i."
					help
				fi
				;;
		esac
	done

	if [[ $reservation != "" ]]; then
		export SBATCH_RESERVATION=$reservation
	fi
}

function git_clone_interactive {
	_command="$1"

	dbg "git_clone_interactive $_command"

	total=0

	if [[ $no_whiptail -eq 0 ]]; then
		$_command 2>&1 | tr \\r \\n | {
			while read -r line ; do
				cur=`grep -oP '\d+(?=%)' <<< ${line}`
				total=$((total+cur))
				percent=$(bc <<< "scale=2;100*($total/100)")
				echo "$percent/1" | bc
			done
		} | whiptail --title "Cloning" --gauge "Cloning OmniOpt2 for optimizing project..." 8 78 0 && green_text 'Cloning successful' || red_text 'Cloning failed'
	else
		$_command
	fi
}

function git_clone_non_interactive {
	_command="$1"

	dbg "git_clone_non_interactive $_command"

	$_command || {
		red_text "Git cloning failed."
		exit 14
	}
}

function cd_and_run_command {
	_to_dir="$1"
	_start_command="$2"

	dbg "cd_and_run_command $_to_dir $_start_command"

	ax_dir="$_to_dir/"

	if [[ -d $ax_dir ]]; then
		cd $ax_dir

		$start_command
	else
		red_text "Error: $ax_dir could not be found!"
	fi
}

function install_and_run {
	_start_command_base64="$1"

	dbg "install_and_run $_start_command_base64"

	if [[ -z $_start_command_base64 ]] || [[ $_start_command_base64 == "" ]]; then
		red_text "Missing argument for start-command (must be in base64)"
		exit 14
	fi

	set +e
	start_command=$(echo "$_start_command_base64" | base64 --decode)
	start_command_exit_code=$?
	set -e

	if [[ $dryrun -eq 1 ]]; then
		start_command="$start_command --dryrun"
	fi

	if [[ $installation_method == "clone" ]]; then
		if [[ $start_command_exit_code -eq 0 ]]; then
			to_dir=$(get_to_dir)

			clone_command="git clone --depth=$depth $github_repo_url $to_dir"

			if [[ "$is_interactive" == "1" ]] && command -v whiptail >/dev/null 2>/dev/null; then
				git_clone_interactive "$clone_command"
			else
				git_clone_non_interactive "$clone_command"
			fi

			cd_and_run_command "$to_dir" "$start_command"
		else
			red_text "Error: '$_start_command_base64' was not valid base64 code (base64 --decode exited with $start_command_exit_code)"
		fi
	elif [[ $installation_method == "pip" ]]; then
		if [[ ! -d $omniopt_venv ]]; then
			mkdir -p $omniopt_venv
		fi

		venv_activate_file="$omniopt_venv/bin/activate"

		if [[ ! -d $omniopt_venv ]] || [[ ! -e $venv_activate_file ]]; then
			dbg "Creating venv $omniopt_venv"
			python3 -mvenv $omniopt_venv
		fi

		if [[ -e "$venv_activate_file" ]]; then
			dbg "Activating venv $venv_activate_file"
			source "$venv_activate_file"

<?php
			echo $bash_script;
?>
		else
			red_text "Could not find $venv_activate_file. Cannot activate environment. OmniOpt2 installation cancelled."
			exit 14
		fi

		if [[ -e "$venv_activate_file" ]]; then
			dbg "Activating venv $venv_activate_file"
			source "$venv_activate_file"

<?php
			echo $bash_script;
?>

			run_command=$(echo "$start_command" | sed -e 's#^\./##')

			dbg "Run-command: $run_command"

			$run_command
		fi
	else
		red_text "Unknown installation method '$installation_method'. Valid ones are: clone, pip"
	fi
}

parse_parameters $@

check_if_everything_is_installed

check_interactive

install_and_run "$start_command_base64"
}
