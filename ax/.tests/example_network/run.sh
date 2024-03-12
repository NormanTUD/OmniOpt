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
        echo "  --width=(INT)=INT"
        echo "  --height=(INT)=INT"
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

LMOD_DIR=/software/foundation/$(uname -m)/lmod/lmod/libexec

if [[ -d $LMOD_DIR ]]; then
	ml () {
		eval "$($LMOD_DIR/ml_cmd "$@")"
	}

	ml release/23.10 GCC/11.3.0 OpenMPI/4.1.4 TensorFlow/2.11.0-CUDA-11.7.0
fi

ENV_DIR=$HOME/.test_env
if [[ ! -d "$ENV_DIR" ]]; then
        green_text "$ENV_DIR not found. Creating virtual environment."
        python3 -m venv $ENV_DIR
        source $ENV_DIR/bin/activate

        pip install tensorflow tensorflowjs protobuf scikit-image opencv-python keras termcolor pyyaml h5py
fi

source $ENV_DIR/bin/activate

if [[ "$train" == 1 ]]; then
        python3 train.py $*
elif [[ "$predict" == 1 ]]; then
        python3 predict.py $*
else
        red_text "Neither predict nor train was set."
fi
