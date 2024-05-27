#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR
cd ..

for i in $(cat .omniop*.py | grep import | grep -v with | sed -e 's#^\s*##' -e 's#^import ##' -e 's#.* import ##' | grep -v ImportError | grep -v -i barc | sort | grep -v "=" | uniq | grep -v Gene | grep -v Ax | sed -e 's# as .*##' | grep -v "," | sort | uniq); do 
    python3 -c "
import importlib
import importlib.metadata
module_name = '$i'
try:
    importlib.import_module(module_name)
    print(f'{module_name}=', end='')
    print(importlib.metadata.version(module_name))
except ModuleNotFoundError:
    print(f'{module_name} not found')
except Exception as e:
    print(e)
"
done
