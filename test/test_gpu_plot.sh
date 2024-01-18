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



set -e

export PLOTPATH=$RANDOM.svg
while [[ -e $PLOTPATH ]]; do
    export PLOTPATH=$RANDOM.svg
done

ml purge

ml release/23.04 2>&1 | grep -v loaded
ml MongoDB/4.0.3 2>&1 | grep -v loaded
ml GCC/11.3.0 2>&1 | grep -v loaded
ml OpenMPI/4.1.4 2>&1 | grep -v loaded
ml Hyperopt/0.2.7 2>&1 | grep -v loaded
ml matplotlib/3.5.2 2>&1 | grep -v loaded

python3 script/plot_gpu.py gpu_test_alpha test/gpu_plot_data/

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
