import math
import os
import platform
import re
import sys
import traceback
from importlib.metadata import version
from pprint import pprint

import numpy as np


def check_environment_variable(variable_name):
    try:
        value = os.environ[variable_name]
        if value == "1":
            return True
    except KeyError:
        pass

    return False

if not check_environment_variable("RUN_VIA_RUNSH"):
    print("Must be run via the bash script, cannot be run as standalone.")

    sys.exit(16)

def in_venv():
    return sys.prefix != sys.base_prefix


if not in_venv():
    print("No venv loaded. Cannot continue.")
    sys.exit(19)

def warn_versions():
    wrns = []

    supported_versions = {
        "ax": ["0.36.0", "0.3.7", "0.3.8.dev133", "0.52.0"],
        "botorch": ["0.10.0", "0.10.1.dev46+g7a844b9e", "0.11.0", "0.8.5", "0.9.5", "0.11.3"],
        "torch": ["2.3.0", "2.3.1", "2.4.0", "2.4.1"],
        "seaborn": ["0.12.2", "0.13.2"],
        "pandas": ["1.5.3", "2.0.3", "2.2.2"],
        "psutil": ["5.9.4", "5.9.8", "6.0.0"],
        "numpy": ["1.24.4", "1.26.4"],
        "matplotlib": ["3.6.3", "3.7.5", "3.9.0", "3.9.1", "3.9.1.post1", "3.9.2"],
        "submitit": ["1.5.1"],
        "tqdm": ["4.64.1", "4.66.4", "4.66.5"]
    }

    for key in supported_versions.keys():
        _supported_versions = supported_versions[key]
        try:
            _real_version = version(key)
            if _real_version not in _supported_versions:
                wrns.append(f"Possibly unsupported {key}-version: {_real_version} not in supported version(s): {', '.join(_supported_versions)}")
        except Exception:
            pass

    if len(wrns):
        print("- " + ("\n- ".join(wrns)))

def looks_like_float(x):
    if isinstance(x, (int, float)):
        return True  # int and float types are directly considered as floats

    if isinstance(x, str):
        try:
            float(x)  # Try converting string to float
            return True
        except ValueError:
            return False  # If conversion fails, it's not a float-like string

    return False  # If x is neither str, int, nor float, it's not float-like

def looks_like_int(x):
    if isinstance(x, bool):
        return False

    if isinstance(x, int):
        return True

    if isinstance(x, float):
        return x.is_integer()

    if isinstance(x, str):
        return bool(re.match(r'^\d+$', x))

    return False

def looks_like_number (x):
    return looks_like_float(x) or looks_like_int(x) or type(x) is int or type(x) is float or type(x) is np.int64

def to_int_when_possible(val):
    # Überprüfung, ob der Wert ein Integer ist oder ein Float, der eine ganze Zahl sein könnte
    if type(val) is int or (type(val) is float and val.is_integer()) or (type(val) is str and val.isdigit()):
        return int(val)

    # Überprüfung auf nicht-numerische Zeichenketten
    if type(val) is str and re.match(r'^-?\d+(?:\.\d+)?$', val) is None:
        return val

    try:
        # Versuche, den Wert als Float zu interpretieren
        val = float(val)
        # Bestimmen der Anzahl der Dezimalstellen, um die Genauigkeit der Ausgabe zu steuern
        if '.' in str(val):
            decimal_places = len(str(val).split('.')[1])
            # Formatieren des Floats mit der exakten Anzahl der Dezimalstellen, ohne wissenschaftliche Notation
            formatted_value = format(val, f'.{decimal_places}f').rstrip('0').rstrip('.')
            return formatted_value if formatted_value else '0'
        return int(val)
    except Exception:
        # Falls ein Fehler auftritt, gebe den ursprünglichen Wert zurück
        return val

def dier (msg):
    pprint(msg)
    sys.exit(1)

def flatten_extend(matrix):
    flat_list = []
    for row in matrix:
        flat_list.extend(row)
    return flat_list

def convert_string_to_number(input_string):
    try:
        assert isinstance(input_string, str), "Input must be a string"

        # Replace commas with dots
        input_string = input_string.replace(",", ".")

        # Regular expression patterns for int and float
        float_pattern = re.compile(r"[+-]?\d*\.\d+")
        int_pattern = re.compile(r"[+-]?\d+")

        # Search for float pattern
        float_match = float_pattern.search(input_string)
        if float_match:
            number_str = float_match.group(0)
            try:
                number = float(number_str)
                return number
            except ValueError as e:
                print(f"Failed to convert {number_str} to float: {e}")

        # If no float found, search for int pattern
        int_match = int_pattern.search(input_string)
        if int_match:
            number_str = int_match.group(0)
            try:
                number = int(number_str)
                return number
            except ValueError as e:
                print(f"Failed to convert {number_str} to int: {e}")

        return None

    except AssertionError as e:
        print(f"Assertion error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")

        tb = traceback.format_exc()
        print(tb)

        return None

def log_error(error_text):
    print(f"Error: {error_text}", file=sys.stderr)

def check_if_results_are_empty(result_column_values, csv_file_path):
    filtered_data = list(filter(lambda x: not math.isnan(x), result_column_values.tolist()))

    number_of_non_nan_results = len(filtered_data)

    if number_of_non_nan_results == 0:
        print(f"No values were found. Every evaluation found in {csv_file_path} evaluated to NaN.")
        sys.exit(11)

def get_result_column_values(df, csv_file_path):
    result_column_values = df["result"]

    check_if_results_are_empty(result_column_values, csv_file_path)

    return result_column_values

def check_path(_path):
    global args
    if not os.path.exists(_path):
        print(f'The folder {_path} does not exist.')
        sys.exit(1)

class bcolors:
    header = '\033[95m'
    blue = '\033[94m'
    cyan = '\033[96m'
    green = '\033[92m'
    warning = '\033[93m'
    red = '\033[91m'
    endc = '\033[0m'
    bold = '\033[1m'
    underline = '\033[4m'
    yellow = '\033[33m'

def print_color(color, text):
    color_codes = {
        "header": bcolors.header,
        "blue": bcolors.blue,
        "cyan": bcolors.cyan,
        "green": bcolors.green,
        "warning": bcolors.warning,
        "red": bcolors.red,
        "bold": bcolors.bold,
        "underline": bcolors.underline,
        "yellow": bcolors.yellow,
    }
    end_color = bcolors.endc

    try:
        assert color in color_codes, f"Color '{color}' is not supported."
        print(f"{color_codes[color]}{text}{end_color}")
    except AssertionError as e:
        print(f"Error: {e}")
        print(text)

def check_python_version():
    # python3 version
    python_version = platform.python_version()
    supported_versions = ["3.8.10", "3.10.4", "3.10.12", "3.11.2", "3.11.9", "3.9.2", "3.12.3", "3.12.4", "3.12.5", "3.12.6"]
    if python_version not in supported_versions:
        print_color("yellow", f"Warning: Supported python versions are {', '.join(supported_versions)}, but you are running {python_version}. This may or may not cause problems. Just is just a warning.")

check_python_version()

warn_versions()
