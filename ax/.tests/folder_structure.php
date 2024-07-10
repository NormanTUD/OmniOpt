#!/bin/bash

errors=()

function join_by {
	local d=${1-} f=${2-}
	if shift 2; then
		printf %s "$f" "${@/#/$d}"
	fi
}

function echoerr() {
        echo "$@" 1>&2
}

function yellow_text {
        echoerr -e "\e\033[0;33m$1\e[0m"
}

function red_text {
        echoerr -e "\e[31m$1\e[0m"
}

function green_text {
        echoerr -e "\e\033[0;32m$1\e[0m"
}

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd $SCRIPT_DIR

cd ..

expected_plot_types=()

for state_file in $(ls runs/*/*/state_files | grep -v "^runs" | sort | uniq | grep -v '^\s*$'); do
	expected_plot_types+=("$state_file")
done

for state_file in "${expected_plot_types[@]}"; do
	if ! grep "$state_file" .gui/tutorials/folder_structure.php 2>&1 >/dev/null; then
		errmsg="State file $state_file does not appear in .gui/tutorials/folder_structure.php"
		red_text "$errmsg"
		errors+=("$errmsg")
	else
		green_text "$state_file does appear in .gui/tutorials/folder_structure.php"
	fi
done

secs=$SECONDS
hrs=$(( secs/3600 )); mins=$(( (secs-hrs*3600)/60 )); secs=$(( secs-hrs*3600-mins*60 ))
printf 'folder_structure has all the items took: %02d:%02d:%02d\n' $hrs $mins $secs

if [ ${#errors[@]} -eq 0 ]; then
	green_text "No plot errors"
	exit 0
else
	red_text "=> PLOT-ERRORS => PLOT-ERRORS => PLOT-ERRORS =>"
	for i in "${errors[@]}"; do
		red_text "$i"
	done

	exit ${#errors[@]}
fi
