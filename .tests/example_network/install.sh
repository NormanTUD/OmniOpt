#!/usr/bin/env bash

export install_tests=1

FROZEN=""

function ppip {
	set +e
	echo "$FROZEN" | grep -i "$1" 2>/dev/null >/dev/null
	_exit_code=$?

	if [[ "$_exit_code" != "0" ]]; then
		if [[ "$UPGRADED_PIP" -eq "0" ]]; then
			pip install --upgrade pip
			UPGRADED_PIP=1
		fi

		pip3 install -q $1 || {
			red_text "Failed to install $1. Deleting $VENV_DIR..."
			rm -rf $VENV_DIR || {
				red_text "Failed to delete $VENV_DIR"
				exit 4
			}

			exit 3
		}

		green_text "$1 installed successfully"

		FROZEN=$(pip3 freeze)
	fi
	set -e
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

LMOD_DIR=/software/foundation/$(uname -m)/lmod/lmod/libexec

if [[ -d $LMOD_DIR ]]; then
	ml () {
		eval "$($LMOD_DIR/ml_cmd "$@")"
	}

	ml release/24.04 GCC/11.3.0 OpenMPI/4.1.4 TensorFlow/2.9.1 SciPy-bundle/2022.05
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/.shellscript_functions
