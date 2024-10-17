#!/usr/bin/python

import glob
import sys
import os
import importlib.util
from rich.progress import Progress
from rich.console import Console
from rich.table import Table

console = Console()

script_dir = os.path.dirname(os.path.realpath(__file__))

helpers_file = f"{script_dir}/../.helpers.py"
spec = importlib.util.spec_from_file_location(name="helpers", location=helpers_file)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)

dier = helpers.dier

def clean_filename(_filename):
    _filename = os.path.splitext(_filename)[0]
    return _filename

path = f'{script_dir}/../'
_glob = f'{path}/.*.py'

files = glob.glob(_glob)

mods = {}
loaded_files = []

to_test = {
    ".omniopt_plot_kde.py": {
        "get_num_rows_cols(1, 1, 1)": (1, 1)
    },
    ".omniopt_plot_get_next_trials.py": {
        "is_valid_time_format('hallo')": False,
        "is_valid_time_format('2024-01-01 20:20:02')": True
    },
    ".helpers.py": {
        "check_environment_variable('I_DO_NOT_EXIST')": False,
        "looks_like_int(1)": True,
        "to_int_when_possible('hallo')": "hallo",
        "print_diff('hallo', 'hallo')": None
    }
}

errors = 0
error_list = []

# Ladeprozess mit Progress Bar
with Progress(transient=True) as progress:
    load_task = progress.add_task("[cyan]Loading files...", total=len(files))

    for file in files:
        filename = os.path.basename(file)
        if filename in to_test:
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
                if helpers.is_equal(test, ret_val, expected_ret_val):
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
