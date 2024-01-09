#!/bin/bash

EXITCODE=0

#set -x

export PLOTPATH=$RANDOM.svg
while [[ -e $PLOTPATH ]]; do
    export PLOTPATH=$RANDOM.svg
done

perl tools/plot.pl --project=cpu_test --projectdir=test/projects/

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
