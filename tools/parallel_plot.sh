#!/bin/bash

CSVFILE=$1
OUTPUTPATH=$2

set -x

source tools/parallel_venv/bin/activate

python3 tools/create_parallel_plot.py $CSVFILE $OUTPUTPATH && firefox $OUTPUTPATH || echo "Creating parallel plot failed"
