#!/bin/bash

VERBOSE=false
PROG="$(basename $0)"
DIR="${HOME}/.cache/${PROG}"
mkdir -p "${DIR}"
EXPIRY=2678400
# check if first argument is a number, if so use it as expiration (seconds)
[ "$1" -eq "$1" ] 2>/dev/null && EXPIRY=$1 && shift
[ "$VERBOSE" = true ] && echo "Using expiration $EXPIRY seconds"
CMD="$@"
echo $CMD
HASH=$(echo "$CMD" | md5sum | awk '{print $1}')
CACHE="$DIR/$HASH"
test -f "${CACHE}" && [ $(expr $(date +%s) - $(date -r "$CACHE" +%s)) -le $EXPIRY ] || eval "$CMD 2>&1" > "${CACHE}"
cat "${CACHE}"
