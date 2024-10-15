import json
import math
import difflib
import logging
import os
import platform
import re
import sys
import traceback
from importlib.metadata import version
from pprint import pprint
from matplotlib.widgets import Button, TextBox
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
        "botorch": ["0.10.0", "0.10.1.dev46+g7a844b9e", "0.11.0", "0.8.5", "0.9.5", "0.11.3", "0.12.0"],
        "torch": ["2.3.0", "2.3.1", "2.4.0", "2.4.1"],
        "seaborn": ["0.12.2", "0.13.2"],
        "pandas": ["1.5.3", "2.0.3", "2.2.2", "2.2.3"],
        "psutil": ["5.9.4", "5.9.8", "6.0.0"],
        "numpy": ["1.24.4", "1.26.4", "2.1.1", "2.1.2", "2.1.3"],
        "matplotlib": ["3.6.3", "3.7.5", "3.9.0", "3.9.1", "3.9.1.post1", "3.9.2"],
        "submitit": ["1.5.1", "1.5.2"],
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
    if type(val) is int or (type(val) is float and val.is_integer()) or (type(val) is str and val.isdigit()):
        return int(val)

    if type(val) is str and re.match(r'^-?\d+(?:\.\d+)?$', val) is None:
        return val

    try:
        val = float(val)
        if '.' in str(val):
            decimal_places = len(str(val).split('.')[1])
            formatted_value = format(val, f'.{decimal_places}f').rstrip('0').rstrip('.')
            return formatted_value if formatted_value else '0'
        return int(val)
    except Exception:
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

        input_string = input_string.replace(",", ".")

        float_pattern = re.compile(r"[+-]?\d*\.\d+")
        int_pattern = re.compile(r"[+-]?\d+")

        float_match = float_pattern.search(input_string)
        if float_match:
            number_str = float_match.group(0)
            try:
                number = float(number_str)
                return number
            except ValueError as e:
                print(f"Failed to convert {number_str} to float: {e}")

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
        "yellow": bcolors.yellow
    }
    end_color = bcolors.endc

    try:
        assert color in color_codes, f"Color '{color}' is not supported."
        print(f"{color_codes[color]}{text}{end_color}")
    except AssertionError as e:
        print(f"Error: {e}")
        print(text)

def check_python_version():
    python_version = platform.python_version()
    supported_versions = ["3.8.10", "3.10.4", "3.10.12", "3.11.2", "3.11.9", "3.9.2", "3.12.3", "3.12.4", "3.12.5", "3.12.6", "3.12.7"]
    if python_version not in supported_versions:
        print_color("yellow", f"Warning: Supported python versions are {', '.join(supported_versions)}, but you are running {python_version}. This may or may not cause problems. Just is just a warning.")

def create_widgets(_data):
    _plt, button, MAXIMUM_TEXTBOX, MINIMUM_TEXTBOX, args, TEXTBOX_MINIMUM, TEXTBOX_MAXIMUM, update_graph = _data

    button_ax = _plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(button_ax, 'Update Graph')

    button.on_clicked(update_graph)

    max_string, min_string = "", ""

    if looks_like_float(args.max):
        max_string = str(args.max)

    if looks_like_float(args.min):
        min_string = str(args.min)

    TEXTBOX_MINIMUM = _plt.axes([0.2, 0.025, 0.1, 0.04])
    MINIMUM_TEXTBOX = TextBox(TEXTBOX_MINIMUM, 'Minimum result:', initial=min_string)

    TEXTBOX_MAXIMUM = _plt.axes([0.5, 0.025, 0.1, 0.04])
    MAXIMUM_TEXTBOX = TextBox(TEXTBOX_MAXIMUM, 'Maximum result:', initial=max_string)

    return button, MAXIMUM_TEXTBOX, MINIMUM_TEXTBOX, TEXTBOX_MINIMUM, TEXTBOX_MAXIMUM

def die_if_no_nonempty_graph (non_empty_graphs, _exit):
    if not non_empty_graphs:
        print('No non-empty graphs to display.')
        if _exit:
            sys.exit(2)

def get_r(df_filtered):
    r = 2

    if len(list(df_filtered.columns)) == 1:
        r = 1

    return r

def save_to_file (_fig, _args, _plt):
    _fig.set_size_inches(15.5, 9.5)

    _path = os.path.dirname(_args.save_to_file)
    if _path:
        os.makedirs(_path, exist_ok=True)
    try:
        _plt.savefig(_args.save_to_file)
    except OSError as e:
        print(f"Error: {e}. This may happen on unstable file systems or in docker containers.")
        sys.exit(199)

def check_dir_and_csv(_args, csv_file_path):
    if not os.path.isdir(_args.run_dir):
        print(f"The path {_args.run_dir} does not point to a folder. Must be a folder.")
        sys.exit(11)

    if not os.path.exists(csv_file_path):
        print(f'The file {csv_file_path} does not exist.')
        sys.exit(39)

def get_csv_file_path(_args):
    pd_csv = "results.csv"
    csv_file_path = os.path.join(_args.run_dir, pd_csv)
    check_dir_and_csv(_args, csv_file_path)

    return csv_file_path

def drop_empty_results (NO_RESULT, df):
    negative_rows_to_remove = df[df["result"].astype(str) == '-' + NO_RESULT].index
    positive_rows_to_remove = df[df["result"].astype(str) == NO_RESULT].index

    df.drop(negative_rows_to_remove, inplace=True)
    df.drop(positive_rows_to_remove, inplace=True)

    return df

def hide_empty_plots(parameter_combinations, num_rows, num_cols, axs):
    for i in range(len(parameter_combinations), num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        axs[row, col].set_visible(False)

    return axs

def get_title(_args, result_column_values, df_filtered, num_entries, _min, _max):
    _mean = result_column_values.mean()

    extreme_index = None
    if os.path.exists(_args.run_dir + "/state_files/maximize"):
        extreme_index = result_column_values.idxmax()
    else:
        extreme_index = result_column_values.idxmin()

    extreme_values = df_filtered.loc[extreme_index].to_dict()

    title = "Minimum"
    if os.path.exists(_args.run_dir + "/state_files/maximize"):
        title = "Maximum"

    extreme_values_items = extreme_values.items()

    title_values = []

    for _l in extreme_values_items:
        if "result" not in _l:
            key = _l[0]
            value = to_int_when_possible(_l[1])
            title_values.append(f"{key} = {value}")

    title += " of f("
    title += ', '.join(title_values)
    title += f") = {to_int_when_possible(result_column_values[extreme_index])}"

    title += f"\nNumber of evaluations shown: {num_entries}"

    if _min is not None:
        title += f", show min = {to_int_when_possible(_min)}"

    if _max is not None:
        title += f", show max = {to_int_when_possible(_max)}"

    if _mean is not None:
        title += f", mean result = {to_int_when_possible(_mean)}"

    return title

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

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


def _unidiff_output(expected, actual):
    """
    Helper function. Returns a string containing the unified diff of two multiline strings.
    """

    expected = expected.splitlines(1)
    actual = actual.splitlines(1)

    diff = difflib.unified_diff(expected, actual)

    return ''.join(diff)

def _is_equal(name, _input, output):
    _equal_types = [
        int, str, float, bool
    ]
    for equal_type in _equal_types:
        if type(_input) is equal_type and type(output) and _input != output:
            print_color("red", f"Failed test (1): {name}")
            return True

    if type(_input) is not type(output):
        print_color("red", f"Failed test (4): {name}")
        return True

    if isinstance(_input, bool) and _input != output:
        print_color("red", f"Failed test (6): {name}")
        return True

    if (output is None and _input is not None) or (output is not None and _input is None):
        print_color("red", f"Failed test (7): {name}")
        return True

    print_color("green", f"Test OK: {name}")
    return False

def is_equal(n, o, i):
    r = _is_equal(n, i, o)

    if r:
        print_diff(i, o)

    return r

def _is_not_equal(name, _input, output):
    _equal_types = [
        int, str, float, bool
    ]
    for equal_type in _equal_types:
        if isinstance(_input, equal_type) and isinstance(output, equal_type) and _input == output:
            print_color("red", f"Failed test (1): {name}")
            return True

    if isinstance(_input, bool) and _input == output:
        print_color("red", f"Failed test (2): {name}")
        return True

    if not (output is not None and _input is not None):
        print_color("red", f"Failed test (3): {name}")
        return True

    print_color("green", f"Test OK: {name}")
    return False

def is_not_equal(n, i, o):
    r = _is_not_equal(n, i, o)

    if r:
        print_diff(i, o)

    return r

def set_min_max(MINIMUM_TEXTBOX, MAXIMUM_TEXTBOX, _min, _max):
    if MINIMUM_TEXTBOX and looks_like_float(MINIMUM_TEXTBOX.text):
        _min = convert_string_to_number(MINIMUM_TEXTBOX.text)

    if MAXIMUM_TEXTBOX and looks_like_float(MAXIMUM_TEXTBOX.text):
        _max = convert_string_to_number(MAXIMUM_TEXTBOX.text)

    return _min, _max

def get_num_subplots_rows_and_cols(non_empty_graphs):
    num_subplots = len(non_empty_graphs)
    num_cols = math.ceil(math.sqrt(num_subplots))
    num_rows = math.ceil(num_subplots / num_cols)

    return num_subplots, num_cols, num_rows

def remove_widgets(fig, button, MAXIMUM_TEXTBOX, MINIMUM_TEXTBOX):
    for widget in fig.axes:
        if widget not in [button.ax, MAXIMUM_TEXTBOX.ax, MINIMUM_TEXTBOX.ax]:
            widget.remove()

def get_non_empty_graphs(parameter_combinations, df_filtered, _exit):
    non_empty_graphs = []

    if len(parameter_combinations[0]) == 1:
        param = parameter_combinations[0][0]
        if param in df_filtered and df_filtered[param].notna().any():
            non_empty_graphs = [(param,)]
    else:
        if len(parameter_combinations) > 1 or type(parameter_combinations[0]) is tuple:
            non_empty_graphs = [param_comb for param_comb in parameter_combinations if df_filtered[param_comb[0]].notna().any() and df_filtered[param_comb[1]].notna().any()]
        elif len(parameter_combinations) == 1:
            non_empty_graphs = [param_comb for param_comb in parameter_combinations if df_filtered[param_comb].notna().any()]
        else:
            print("Error: No non-empty parameter combinations")
            sys.exit(75)

    if not non_empty_graphs:
        print('No non-empty graphs to display.')
        if _exit:
            sys.exit(2)

    return non_empty_graphs

def get_df_filtered(_args, df):
    all_columns_to_remove = ['trial_index', 'arm_name', 'trial_status', 'generation_method']
    columns_to_remove = []
    existing_columns = df.columns.values.tolist()

    for col in existing_columns:
        if col in all_columns_to_remove:
            columns_to_remove.append(col)

    if len(_args.allow_axes):
        for col in existing_columns:
            if col != "result" and col not in helpers.flatten_extend(_args.allow_axes):
                columns_to_remove.append(col)

    df_filtered = df.drop(columns=columns_to_remove)

    return df_filtered

def check_min_and_max(num_entries, nr_of_items_before_filtering, csv_file_path, _min, _max, _exit=True):
    if num_entries is None or num_entries == 0:
        if nr_of_items_before_filtering:
            if _min and not _max:
                print("Using --min filtered out all results")
            elif not _min and _max:
                print("Using --max filtered out all results")
            elif _min and _max:
                print("Using --min and --max filtered out all results")
            else:
                print("For some reason, there were values in the beginning but not after filtering")
        else:
            if not os.environ.get("NO_NO_RESULT_ERROR"):
                print(f"No applicable values could be found in {csv_file_path}.")
        if _exit:
            sys.exit(4)

check_python_version()

warn_versions()
