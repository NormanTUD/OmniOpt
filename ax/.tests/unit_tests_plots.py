#!/usr/bin/python

import glob
import sys
import os
import importlib.util
import json

def _is_equal(name, _input, output):
    _equal_types = [
        int, str, float, bool
    ]
    for equal_type in _equal_types:
        if type(_input) is equal_type and type(output) and _input != output:
            print(f"Failed test (1): {name}")
            return True

    if type(_input) is not type(output):
        print(f"Failed test (4): {name}")
        return True

    if isinstance(_input, bool) and _input != output:
        print(f"Failed test (6): {name}")
        return True

    if (output is None and _input is not None) or (output is not None and _input is None):
        print(f"Failed test (7): {name}")
        return True

    print(f"Test OK: {name}")
    return False

def print_diff(i, o):
    if isinstance(i, str):
        print("Should be:", i.strip())
    else:
        print("Should be:", i)

    if isinstance(o, str):
        print("Is:", o.strip())
    else:
        print("Is:", o)
    if isinstance(i, str) or isinstance(o, str):
        print("Diff:", _unidiff_output(json.dumps(i), json.dumps(o)))

def is_equal(n, o, i):
    r = _is_equal(n, i, o)

    if r:
        print_diff(i, o)

    return r

script_dir = os.path.dirname(os.path.realpath(__file__))

helpers_file = f"{script_dir}/../.helpers.py"
spec = importlib.util.spec_from_file_location(
    name="helpers",
    location=helpers_file,
)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)

dier = helpers.dier

def clean_filename(filename):
    # Entfernt das "_" am Anfang, falls vorhanden
    if filename.startswith('_'):
        filename = filename[1:]

    # Entfernt die Dateiendung (alles nach dem letzten Punkt)
    filename = os.path.splitext(filename)[0]

    return filename

path = f'{script_dir}/../'
_glob = f'{path}/.omniopt_plot_*.py'

files = glob.glob(_glob)

mods = {}
loaded_files = []

to_test = {
    ".omniopt_plot_kde.py": {
        "get_num_rows_cols(1, 1, 1)": (1, 1)
    }
}

for file in files:
    print(f"Processing file: {file}")
    filename = os.path.basename(file)
    loaded_files.append(f"{filename}")
    spec = importlib.util.spec_from_file_location(
        name=clean_filename(file),
        location=loaded_files[len(loaded_files) - 1],
    )
    mods[filename] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mods[filename])

errors = 0

for modname in mods:
    if modname in to_test:
        mod = mods[modname]
        for test in to_test[modname]:
            expected_ret_val = to_test[modname][test]
            ret_val = eval(f"mod.{test}")
            errors += is_equal(test, ret_val, expected_ret_val)
    else:
        print(f"Not testing {modname}")

print(f"{errors} error(s)")
sys.exit(errors)
