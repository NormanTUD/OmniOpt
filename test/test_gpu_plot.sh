#!/bin/bash

EXITCODE=0

LMOD_DIR=/usr/share/lmod/lmod/libexec/
LMOD_CMD=/usr/share/lmod/lmod/libexec/lmod

ml () {
        eval $($LMOD_DIR/ml_cmd "$@")
}
module () {
        eval `$LMOD_CMD sh "$@"`
}



set -ex

export PLOTPATH=$RANDOM.svg
while [[ -e $PLOTPATH ]]; do
    export PLOTPATH=$RANDOM.svg
done

ml purge
ml modenv/scs5
ml MongoDB/4.0.3
ml Hyperopt/0.2.2-fosscuda-2019b-Python-3.7.4
ml Python/3.7.4-GCCcore-8.3.0
ml matplotlib/3.1.1-foss-2019b-Python-3.7.4

python3 script/plot_gpu.py gpu_test test/gpu_plot_data/

if grep "Utilization" $PLOTPATH; then
    if grep "gpu_test" $PLOTPATH; then
        echo "Everything seemed to worked fine"
    else
        echo "ERROR! Plot does not contain 'gpu_test'"
        EXITCODE=3
    fi
else
    echo "ERROR! Plot does not contain 'Utilization'"
    EXITCODE=2
fi

if [[ $EXITCODE -eq "0" ]]; then
    rm "$PLOTPATH"
fi

exit $EXITCODE
