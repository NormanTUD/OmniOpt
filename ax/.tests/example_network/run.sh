#!/bin/bash

export PYENV_ROOT="$HOME/.pyenv"
if [[ -d $PYENV_ROOT ]]; then
	[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
	eval "$(pyenv init - bash)"

	pyenv local 3.10.0
fi

export install_tests=1

GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

function set_debug {
	trap 'echo -e "${CYAN}$(date +"%Y-%m-%d %H:%M:%S")${NC} ${MAGENTA}| Line: $LINENO ${NC}${YELLOW}-> ${NC}${BLUE}[DEBUG]${NC} ${GREEN}$BASH_COMMAND${NC}"' DEBUG
}

function unset_debug {
	trap - DEBUG
}

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd $SCRIPT_DIR

function echoerr() {
        echo "$@" 1>&2
}

function green_text {
        echo -e "\033[0;32m$1\e[0m"
}

function red_text {
        echoerr -e "\e[31m$1\e[0m"
}

set -e
set -o pipefail

function calltracer () {
        red_text 'Last file/last line:'
        caller
}
trap 'calltracer' ERR

LMOD_DIR=/software/foundation/$(uname -m)/lmod/lmod/libexec

myml () {
	if [[ -e $LMOD_DIR/ml_cmd ]]; then
		eval "$($LMOD_DIR/ml_cmd "$@")" 2>/dev/null >/dev/null
	fi
}

if [ -z "$LOAD_MODULES" ] || [ "$LOAD_MODULES" -eq 1 ]; then
	myml release/23.04 GCCcore/12.2.0 Python/3.10.8 GCCcore/11.3.0 Tkinter/3.10.4 PostgreSQL/14.4
fi

source $SCRIPT_DIR/.shellscript_functions

function help () {
	python3 train.py --help
	exit $?
}

train=1
predict=0

for i in "$@"; do
        case $i in
                --train)
                        train=1
                        predict=0
                        shift
                        ;;
                --predict)
                        train=0
                        predict=1
                        shift
                        ;;
                -h|--help)
                        help
                        ;;
                --debug)
			set_debug
                        ;;
        esac
done

if [[ "$train" == 1 ]]; then
        python3 train.py $*
elif [[ "$predict" == 1 ]]; then
        python3 predict.py $*
else
        red_text "Neither predict nor train was set."
fi

echo "RUNTIME: $SECONDS"
