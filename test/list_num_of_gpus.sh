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

mml TensorFlow/2.3.1-fosscuda-2019b-Python-3.7.4

python3 test/list_num_of_gpus.py
