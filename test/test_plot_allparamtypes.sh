#!/bin/bash

EXITCODE=0

export PLOTPATH=$RANDOM.svg
while [[ -e $PLOTPATH ]]; do
    export PLOTPATH=$RANDOM.svg
done

perl tools/plot.pl --project=allparamtypes --projectdir=test/projects/

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
