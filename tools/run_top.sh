#!/bin/bash

GENERALLOGDIR=$1
LOGDIR=$GENERALLOGDIR/

mkdir -p $LOGDIR

DATE=$(date +'%Y-%m-%d_%H_%M_%S')
LOGFILE=$LOGDIR/${DATE}.log

top -b -n1 -bcn1 -w512 >> $LOGFILE
