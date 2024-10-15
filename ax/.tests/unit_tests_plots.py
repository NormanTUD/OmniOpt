#!/usr/bin/python

import glob
import sys
import os
import importlib.util
import json
from rich.progress import Progress
from rich.console import Console
from rich.table import Table
from rich.text import Text

console = Console()

def _is_equal(name, _input, output):
    _equal_types = [int, str, float, bool]
    for equal_type in _equal_types:
        if type(_input) is equal_type and _input != output:
            console.print(f"Failed test (1): {name}", style="bold red")
            return True

    if type(_input) is not type(output):
        console.print(f"Failed test (4): {name}", style="bold red")
        return True

    if isinstance(_input, bool) and _input != output:
        console.print(f"Failed test (6): {name}", style="bold red")
        return True

    if (output is None and _input is not None) or (output is not None and _input is None):
        console.print(f"Failed test (7): {name}", style="bold red")
        return True

    console.print(f"Test OK: {name}", style="bold green")
    return False

def print_diff(i, o):
    console.print("Should be:", i if not isinstance(i, str) else i.strip(), style="bold yellow")
    console.print("Is:", o if not isinstance(o, str) else o.strip(), style="bold yellow")
    if isinstance(i, str) or isinstance(o, str):
        console.print("Diff:", _unidiff_output(json.dumps(i), json.dumps(o)), style="bold cyan")

def is_equal(n, o, i):
    r = _is_equal(n, i, o)
    if r:
        print_diff(i, o)
    return r

script_dir = os.path.dirname(os.path.realpath(__file__))

helpers_file = f"{script_dir}/../.helpers.py"
spec = importlib.util.spec_from_file_location(name="helpers", location=helpers_file)
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

errors = 0
error_list = []

# Ladeprozess mit Progress Bar
with Progress(transient=True) as progress:
    load_task = progress.add_task("[cyan]Loading files...", total=len(files))
    
    for file in files:
        filename = os.path.basename(file)
        loaded_files.append(f"{filename}")
        spec = importlib.util.spec_from_file_location(
            name=clean_filename(file),
            location=loaded_files[len(loaded_files) - 1],
        )
        mods[filename] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mods[filename])
        progress.update(load_task, advance=1)

# Testen der geladenen Module mit Progress Bar
with Progress(transient=True) as progress:
    test_task = progress.add_task("[cyan]Running tests...", total=len(mods))

    for modname in mods:
        if modname in to_test:
            mod = mods[modname]
            for test in to_test[modname]:
                expected_ret_val = to_test[modname][test]
                ret_val = eval(f"mod.{test}")
                if is_equal(test, ret_val, expected_ret_val):
                    error_list.append((modname, test, ret_val, expected_ret_val))
                    errors += 1
        progress.update(test_task, advance=1)

# Fehleranzeige
if errors > 0:
    console.print(f"{errors} error(s) found", style="bold red")

    # Tabelle mit Fehlern
    table = Table(title="Test Errors")

    table.add_column("Module", justify="left", style="cyan", no_wrap=True)
    table.add_column("Test", justify="left", style="magenta")
    table.add_column("Result", justify="left", style="green")
    table.add_column("Expected", justify="left", style="red")

    for modname, test, ret_val, expected_ret_val in error_list:
        table.add_row(modname, test, str(ret_val), str(expected_ret_val))

    console.print(table)
else:
    console.print("All tests passed successfully!", style="bold green")

sys.exit(errors)
