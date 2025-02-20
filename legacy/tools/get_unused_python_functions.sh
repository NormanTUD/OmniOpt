#!/usr/bin/bash

for defname in $(egrep -h --exclude-dir=tests "^\\s*def .*:" **/*.py | sed -e 's/^\s*def //' | sed -e 's/\s*(.*://' | grep -v "^test_" | grep -v "^_"); do
    if ! egrep -h "$defname\(" **/*.py | grep -v "^\\s*def " | egrep "(^|.*=).*$defname\s*\(" 2>/dev/null >/dev/null; then
        echo "'$defname' could not be found in any way that looks like it's used somewhere"
    fi
done
