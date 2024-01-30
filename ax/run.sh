#!/bin/bash

IFS=$'\n'

Green='\033[0;32m'
Color_Off='\033[0m'
Red='\033[0;31m'

VENV_DIR=$HOME/.omniax

function red {
	echo -e "${Red}$1${Color_Off}"
}

function green {
	echo -e "${Green}$1${Color_Off}"
}

function ppip {
	pip3 install $* || {
		red "Failed to install $*. Deleting $VENV_DIR..."
		rm -rf $VENV_DIR || {
			red "Failed to delete $VENV_DIR"
			exit 4
		}

		exit 3
	}

	green "$* installed successfully"
}

set -e

LMOD_DIR=/software/foundation/x86_64/lmod/lmod/libexec

ml () {
        eval "$($LMOD_DIR/ml_cmd "$@")"
}

green "Loading modules..."

ml release/23.04 GCCcore/12.2.0 Python/3.10.8

if [[ ! -d "$VENV_DIR" ]]; then
	green "Environment $VENV_DIR was not found. Creating it..."
	python3 -mvenv $VENV_DIR/ || {
		red "Failed to create Virtual Environment in $VENV_DIR"
		exit 1
	}

	green "Virtual Environment $VENV_DIR created. Activating it..."

	source $VENV_DIR/bin/activate || {
		red "Failed to activate $VENV_DIR"
		exit 2
	}

	green "Virtual Environment activated. Now installing software. This may take some time."

	ppip submitit
	ppip ax
	ppip ax-platform
fi

source $VENV_DIR/bin/activate

set -x

python3 main.py $*
