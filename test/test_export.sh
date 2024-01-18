#!/bin/bash

#set -x
#pwd

echoerr() {
	echo "$@" 1>&2
}

function red_text {
	echoerr -e "\e[31m$1\e[0m"
}

function green_text {
	echoerr -e "\e[92m$1\e[0m"
}

TESTEXPORTFILE=${RANDOM}.csv
while [[ -e $TESTEXPORTFILE ]]; do
    TESTEXPORTFILE=${RANDOM}.csv
done

green_text "Using test file $TESTEXPORTFILE"
perl script/runningdbtocsv.pl --project=$1 --projectdir=test/projects > $TESTEXPORTFILE

echo ""

if [[ $? -ne "0" ]]; then
    red_text "runningdbtocsv.pl failed with $?"
    exit $?
fi

NUMBEROFLINES=$(cat $TESTEXPORTFILE | wc -l)
MINNUMBEROFLINES=2
MINHEADERITEMS=5

if [[ $NUMBEROFLINES -gt $MINNUMBEROFLINES ]]; then
        green_text "More than $MINNUMBEROFLINES lines"
else
        red_text "Not enough lines in $TESTEXPORTFILE"
        exit 1
fi

NUMBEROFHEADERITEMS=$(cat $TESTEXPORTFILE | head -n1 | perl -e 'while (<>) { for my $x (split //, $_) { print "$x\n" } }' | grep ";" | wc -l)

if [[ $NUMBEROFHEADERITEMS -gt $MINHEADERITEMS ]]; then
        green_text "More than $MINHEADERITEMS header items"
else
        red_text "Not enough ($MINHEADERITEMS) header items in line $(cat $TESTEXPORTFILE | head -n1)"
        exit 2
fi


rm $TESTEXPORTFILE
