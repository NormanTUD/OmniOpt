#!/bin/bash

EXITCODE=0

export PLOTPATH=$RANDOM.svg
while [[ -e $PLOTPATH ]]; do
    export PLOTPATH=$RANDOM.svg
done

LMOD_DIR=/usr/share/lmod/lmod/libexec/
LMOD_CMD=/usr/share/lmod/lmod/libexec/lmod

ml () {
        eval $($LMOD_DIR/ml_cmd "$@")
}
module () {
        eval `$LMOD_CMD sh "$@"`
}

ml release/23.04 GCCcore/11.3.0 ImageMagick/7.1.0-37

perl tools/plot.pl --project=allparamtypes --projectdir=test/projects/ 2>&1 | grep -v "DEBUG:matplotlib"

convert $PLOTPATH ${PLOTPATH}.png

if grep "loguniform" $PLOTPATH; then
    if grep "loss" $PLOTPATH; then
        echo "Everything seemed to worked fine"
    else
        echo "ERROR! Plot does not contain 'loss'"
        EXITCODE=3
    fi
else
    echo "ERROR! Plot does not contain 'loguniform'"
    EXITCODE=2
fi

if [[ $EXITCODE -eq "0" ]]; then
    rm "$PLOTPATH" "${PLOTPATH}.png"
fi

exit $EXITCODE
