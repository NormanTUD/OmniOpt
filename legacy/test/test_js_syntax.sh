#!/bin/bash

LMOD_DIR=/usr/share/lmod/lmod/libexec/
LMOD_CMD=/usr/share/lmod/lmod/libexec/lmod
ml () {
        eval $($LMOD_DIR/ml_cmd "$@")
}
module () {
        eval `$LMOD_CMD sh "$@"`
}

ml release/23.04 GCCcore/11.2.0 nodejs/14.17.6

node --check $1

exit $?
