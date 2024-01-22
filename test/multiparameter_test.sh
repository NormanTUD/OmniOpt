#!/bin/bash

RESULT=$(bash tools/multigpu.sh --num_gpus=0 --force_redo --maxtime=00:05:00 --programfile=test/multiparameter_test_script.sh --jobname="multiparamtestjob" | grep "RESULT:" | sed -e 's/RESULT: //')

if [[ "$RESULT" == "1.77482393492988" ]]; then
    exit 0
else
    exit 1
fi
