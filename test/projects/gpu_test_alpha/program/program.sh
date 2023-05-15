#!/bin/bash -l

PARAM=$1

LMOD_DIR=/usr/share/lmod/lmod/libexec/
LMOD_CMD=/usr/share/lmod/lmod/libexec/lmod
module () {
        eval `$LMOD_CMD sh "$@"`
}
ml () {
        eval $($LMOD_DIR/ml_cmd "$@")
}


ml modenv/hiera
ml fosscuda/2020b
ml scikit-learn/0.23.2
ml TensorFlow/2.4.1

PYTHONOUTPUT=$(python3 -c "import tensorflow as tf; print(tf.test.gpu_device_name())" 2>&1)
echo $PYTHONOUTPUT
PCI_ID=$(echo $PYTHONOUTPUT | grep pciBusID | sed -e "s/^//")
if echo $PCI_ID | egrep "pciBusID: ([0-9]*:?)+.[0-9]*" >/dev/null 2>/dev/null; then
        python3 -c "print('RESULT: $PARAM')"
else
        echo "NO GPU FOUND"
        exit 1
fi

