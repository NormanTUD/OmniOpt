#!/usr/bin/python

import glob
import os
import importlib.util

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

for file in files:
    print(f"Processing file: {file}")
    loaded_files.append(f"{file}")
    spec = importlib.util.spec_from_file_location(
        name=clean_filename(file),
        location=loaded_files[len(loaded_files) - 1],
    )
    mods[file] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mods[file])

    dier(help(mods[file]))

print(mods)
