#!/bin/bash

LMOD_CMD=/usr/share/lmod/lmod/libexec/lmod
LMOD_DIR=/software/foundation/x86_64/lmod/lmod/libexec

ml () { 
	eval "$($LMOD_DIR/ml_cmd "$@")"
}

module () {
	eval `$LMOD_CMD sh "$@"`
}

ml release/23.10 GCCcore/11.3.0 Perl/5.34.1 GCC/11.3.0 OpenMPI/4.1.4 numpy/1.21.6-Python-3.10.4 MongoDB/6.0.4 2>&1 | grep -v loaded

pip3 install hyperopt

perl sbatch.pl $*
