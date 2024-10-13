#!/bin/bash -l

PARAM=$1

if [[ -z $PARAM ]]; then
	echo "Missing parameter"
	exit 2
fi

LMOD_DIR=/usr/share/lmod/lmod/libexec/
LMOD_CMD=/usr/share/lmod/lmod/libexec/lmod
module () {
        eval `$LMOD_CMD sh "$@"`
}
ml () {
        eval $($LMOD_DIR/ml_cmd "$@")
}


ml release/23.04
ml GCC/11.3.0
ml OpenMPI/4.1.4
ml TensorFlow/2.11.0-CUDA-11.7.0

PYTHONOUTPUT=$(python3 -c "import tensorflow as tf; print(tf.test.gpu_device_name())" 2>&1)
echo $PYTHONOUTPUT
PCI_ID=$(echo $PYTHONOUTPUT | grep 'pci bus id' | sed -e "s/^//")
if echo $PCI_ID | egrep "pci bus id: ([0-9]*:?)+.[0-9]*" >/dev/null 2>/dev/null; then
        python3 -c "print('RESULT: $PARAM')"
else
        echo "NO GPU FOUND"

	echo "hostname: $(hostname)"

	echo "RESULT: 0"
	exit 0
fi

