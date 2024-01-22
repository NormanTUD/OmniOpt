#!/bin/bash

logpathdate=$1

hn=$(hostname | sed -e 's/\.taurus.*//')
logpath="$logpathdate/$hn"

mkdir -p $logpath

i=0
logfile="$logpath/lsof_$i.txt"

while [[ -e $logfile ]]; do
    i=$((i+1))
    logfile="$logpath/lsof_$i.txt"
done

lsof -w -u $USER 2>&1 > $logfile
