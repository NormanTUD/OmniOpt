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


ml TensorFlow/1.15.0-fosscuda-2019b-Python-3.7.4 2>/dev/null
PYTHONOUTPUT=$(python3 -c "import tensorflow as tf; print(tf.test.gpu_device_name())" 2>&1)
echo $PYTHONOUTPUT
PCI_ID=$(echo $PYTHONOUTPUT | grep pciBusID | sed -e "s/^//")
ml unload TensorFlow/1.15.0-fosscuda-2019b-Python-3.7.4 2>/dev/null
if echo $PCI_ID | egrep "pciBusID: ([0-9]*:?)+.[0-9]*" >/dev/null 2>/dev/null; then
        python3 -c "print('RESULT: $PARAM')"
else
        echo "NO GPU FOUND"
        exit 1
fi

