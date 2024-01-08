#!/bin/bash

LOGFILENAME=$1

CURRENTNUMPROCESSES=$(ps h -Lu $USER | wc -l)
HUMANDATE=$(date '+%Y.%m.%d %H:%M:%S')
UNIXTIMESTAMP=$(date +%s)

if [[ ! -s $LOGFILENAME ]]; then
    echo "time,timestamp,currentnumprocesses"
fi

echo "$HUMANDATE,$UNIXTIMESTAMP,$CURRENTNUMPROCESSES"
