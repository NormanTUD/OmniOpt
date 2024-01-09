#!/bin/bash

LMOD_DIR=/usr/share/lmod/lmod/libexec/
LMOD_CMD=/usr/share/lmod/lmod/libexec/lmod

ml () {
        eval $($LMOD_DIR/ml_cmd "$@")
}
module () {
        eval `$LMOD_CMD sh "$@"`
}



EXITCODE=0

#set -x

export PLOTPATH=$RANDOM.svg
while [[ -e $PLOTPATH ]]; do
    export PLOTPATH=$RANDOM.svg
done

perl tools/plot.pl --project=cpu_test2 --projectdir=test/projects/

convert $PLOTPATH ${PLOTPATH}.png

if [[ $(file ${PLOTPATH}.png) =~ "1920 x 1440" ]]; then
    if grep "cpuparam" $PLOTPATH; then
        if grep "cpuparam2" $PLOTPATH; then
            echo "Everything seemed to worked fine"
        else
            echo "ERROR! Plot does not contain 'cpuparam2'"
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
