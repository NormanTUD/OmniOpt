#!/bin/bash

echoerr() {
	echo "$@" 1>&2
}

function red_text {
	echoerr -e "\e[31m$1\e[0m"
}

function green_text {
	echoerr -e "\e[92m$1\e[0m"
}

WCT_RESULT=$(bash tools/get_wallclock_time.sh --projectdir=test/projects --project=gpu_test_alpha)

echo -e "\n"
echo "WCT_RESULT: $WCT_RESULT"
echo -e "\n"

if echo "$WCT_RESULT" | egrep "WallclockTime: [0-9][0-9]:[0-9][0-9]:[0-9][0-9]"; then
    green_text "\nTest OK\n"
    exit 0
else
    red_text "\nTest NOT ok\n"
    exit 1
fi
