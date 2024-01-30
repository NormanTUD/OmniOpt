#!/bin/bash

Green='\033[0;32m'
Color_Off='\033[0m'
Red='\033[0;31m'

function red {
	echo -e "${Red}$1${Color_Off}"
}

function green {
	echo -e "${Green}$1${Color_Off}"
}

function ppip {
	pip3 install $* || {
		red "Failed to install $*"
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

VENV_DIR=$HOME/.omniax

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

	green "Virtual Environment activated. Now installing software"

	ppip submitit
	ppip pprint
	ppip logging
	ppip argparse
	ppip ax
fi

source $VENV_DIR/bin/activate

python3 main.py $*
