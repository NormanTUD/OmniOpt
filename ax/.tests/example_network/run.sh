#!/bin/bash

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
        echo "  --train"
        echo "  --predict"
        echo "  --learning_rate=FLOAT"
        echo "  --epochs=INT"
        echo "  --validation_split=FLOAT"
        echo "  --width=INT"
        echo "  --height=INT"
        echo "  --data=DIRNAME"
	echo "  --conv"
	echo "  --conv_filters"
	echo "  --dense"
	echo "  --dense_units"
        echo "  --help                                             this help"
        echo "  --debug                                            Enables debug mode (set -x)"
        exit $1
}

train=1
predict=0

for i in $@; do
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
                        set -x
                        ;;
        esac
done

source $SCRIPT_DIR/install.sh

if [[ "$train" == 1 ]]; then
        python3 train.py $*
elif [[ "$predict" == 1 ]]; then
        python3 predict.py $*
else
        red_text "Neither predict nor train was set."
fi
