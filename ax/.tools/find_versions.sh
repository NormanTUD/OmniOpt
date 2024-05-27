#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR
cd ..

for i in $(cat .omniop*.py | grep import | grep -v with | sed -e 's#^\s*##' -e 's#^import ##' -e 's#.* import ##' | grep -v ImportError | grep -v -i barc | sort | grep -v "=" | uniq | grep -v Gene | grep -v Ax | sed -e 's# as .*##' | grep -v "," | sort | uniq); do 
    python3 -c "
import importlib
import importlib.metadata
module_name = '$i'
exists = True
_str = ''
try:
    importlib.import_module(module_name)
    _str += f'{module_name}='
    _str += importlib.metadata.version(module_name)
except (Exception, ModuleNotFoundError):
    exists = False

if exists:
    print(_str)
"
done
