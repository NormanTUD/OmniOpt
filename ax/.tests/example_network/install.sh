#!/bin/bash

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
