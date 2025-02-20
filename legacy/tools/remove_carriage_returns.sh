#!/bin/sh
tmpfile=$(mktemp)
tr '\r' "\n" <"$1" >"$tmpfile"
mv "$tmpfile" "$1"
