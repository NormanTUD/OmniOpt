#!/bin/bash

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

function help () {
        echo "Possible options:"
        echo "  --train                                            Start training"
        echo "  --predict                                          Start predicting"
        echo "  --learning_rate=FLOAT                              The learning rate"
        echo "  --epochs=INT                                       The number of epochs"
	echo "  --validation_split=FLOAT                           The validation split (between 0 and 1)"
        echo "  --width=INT                                        Image width"
        echo "  --height=INT                                       Image height"
        echo "  --data=DIRNAME                                     Data dir"
	echo "  --conv                                             Number of convolution layers"
	echo "  --conv_filters                                     Number of convolution filters"
	echo "  --dense                                            Number of dense layers"
	echo "  --dense_units                                      Number of dense neurons"
        echo "  --help                                             This help"
        echo "  --debug                                            Enables debug mode"
        exit $1
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
                --data=*)
                        shift
			;;
                --data)
                        shift
                        shift
                        ;;
                --predict)
                        train=0
                        predict=1
                        shift
                        ;;
                -h|--help)
                        help 0
                        ;;
                --debug)
			set_debug
                        ;;
        esac
done

source $SCRIPT_DIR/.shellscript_functions

if [[ "$train" == 1 ]]; then
        python3 train.py $*
elif [[ "$predict" == 1 ]]; then
        python3 predict.py $*
else
        red_text "Neither predict nor train was set."
fi
