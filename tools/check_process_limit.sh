#!/bin/bash

LOGFILENAME=$1

CURRENTNUMPROCESSES=$(ps h -Lu $USER | wc -l)
MAXNUMPROCESSES=$(cat /etc/security/limits.d/20-nproc.conf | grep "^\*" | sed -e 's/.*nproc\s*//')
HUMANDATE=$(date '+%Y.%m.%d %H:%M:%S')
UNIXTIMESTAMP=$(date +%s)

if [[ ! -s $LOGFILENAME ]]; then
    echo "time,timestamp,currentnumprocesses,maxnumprocesses"
fi

echo "$HUMANDATE,$UNIXTIMESTAMP,$CURRENTNUMPROCESSES,$MAXNUMPROCESSES"
