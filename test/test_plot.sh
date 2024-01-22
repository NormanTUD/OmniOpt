#!/bin/bash

EXITCODE=0

#set -x

LMOD_DIR=/usr/share/lmod/lmod/libexec/
LMOD_CMD=/usr/share/lmod/lmod/libexec/lmod

ml () {
        eval $($LMOD_DIR/ml_cmd "$@")
}
module () {
        eval `$LMOD_CMD sh "$@"`
}

ml release/23.04 GCCcore/11.3.0 ImageMagick/7.1.0-37

export PLOTPATH=$RANDOM.svg
while [[ -e $PLOTPATH ]]; do
    export PLOTPATH=$RANDOM.svg
done

perl tools/plot.pl --project=cpu_test --projectdir=test/projects/ 2>&1 | grep -v "DEBUG:matplotlib"

convert $PLOTPATH ${PLOTPATH}.png

if [[ $(file ${PLOTPATH}.png) =~ "PNG image data, 1920 x 1440, 8-bit grayscale, non-interlaced" ]]; then
    if grep "cpuparam" $PLOTPATH; then
        if grep "loss" $PLOTPATH; then
            echo "Everything seemed to worked fine"
        else
            echo "ERROR! Plot does not contain 'loss'"
            EXITCODE=3
        fi
    else
        echo "ERROR! Plot does not contain 'cpuparam'"
        EXITCODE=2
    fi
else
    echo "ERROR! Check the logs for more details"
    EXITCODE=1
fi

if [[ $EXITCODE -eq "0" ]]; then
    rm "$PLOTPATH" "${PLOTPATH}.png"
fi

exit $EXITCODE
