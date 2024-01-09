#!/bin/bash -l

LMOD_DIR=/usr/share/lmod/lmod/libexec/
LMOD_CMD=$LMOD_DIR/lmod

ml () {
	eval $($LMOD_DIR/ml_cmd "$@")
}
module () {
	eval `$LMOD_CMD sh "$@"`
}


if [[ $(uname -r) =~ "86_64" ]]; then
	ml release/23.04
	ml GCC/11.3.0
	ml OpenMPI/4.1.4
	ml Hyperopt/0.2.7
	ml MongoDB/4.0.3
else
	ml modenv/ml
	ml MongoDB/4.0.3
	ml Python/3.7.4-GCCcore-8.3.0
	ml Hyperopt/0.2.2-fosscuda-2019b-Python-3.7.4
fi
ml matplotlib/3.5.2

python3 script/test_packages.py 2>&1 | grep -v 'DEBUG:matplotlib'

exit $?
