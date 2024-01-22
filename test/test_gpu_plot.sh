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

export PLOTPATH=$RANDOM.svg
while [[ -e $PLOTPATH ]]; do
    export PLOTPATH=$RANDOM.svg
done

module --force purge

ml release/23.04
ml MongoDB/4.0.3
ml GCC/11.3.0
ml OpenMPI/4.1.4
ml Hyperopt/0.2.7
ml matplotlib/3.5.2

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
