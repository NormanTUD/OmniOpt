#!/bin/bash

LMOD_DIR=/software/foundation/$(uname -m)/lmod/lmod/libexec

if [[ -d $LMOD_DIR ]]; then
	ml () {
		eval "$($LMOD_DIR/ml_cmd "$@")"
	}

	ml release/23.10 GCC/11.3.0 OpenMPI/4.1.4 TensorFlow/2.11.0-CUDA-11.7.0
fi

ENV_DIR=$HOME/.omniopt_test_install_$(uname -m)_$(python3 --version | sed -e 's# #_#g')
if [[ ! -d "$ENV_DIR" ]]; then
        green_text "$ENV_DIR not found. Creating virtual environment."
        python3 -m venv $ENV_DIR
        source $ENV_DIR/bin/activate

        pip install tensorflow tensorflowjs protobuf scikit-image opencv-python keras termcolor pyyaml h5py

	deactivate
else
	green_text "$ENV_DIR already exists"
fi
