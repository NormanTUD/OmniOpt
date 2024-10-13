#!/bin/bash

LMOD_DIR=/usr/share/lmod/lmod/libexec/
LMOD_CMD=/usr/share/lmod/lmod/libexec/lmod

module () {
    eval `$LMOD_CMD sh "$@"`
}


ml () {
    eval $($LMOD_DIR/ml_cmd "$@")
}

mml () {
        if ! ml is-loaded $1; then
                ml $1
        fi
}

ml release/23.04 GCC/11.3.0 OpenMPI/4.1.4 TensorFlow/2.11.0-CUDA-11.7.0

python3 test/list_num_of_gpus.py
