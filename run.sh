#!/bin/bash

LMOD_DIR=/software/foundation/x86_64/lmod/lmod/libexec

ml () { 
    eval "$($LMOD_DIR/ml_cmd "$@")"
}

ml release/23.10 GCCcore/11.3.0 Perl/5.34.1  GCC/11.3.0  OpenMPI/4.1.4 numpy/1.21.6-Python-3.10.4

pip3 install hyperopt

perl sbatch.pl $*
