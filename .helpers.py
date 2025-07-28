import sys

try:
    import json
    from typing import Union, Tuple, Any, Optional
    from datetime import datetime
    from itertools import combinations
    from pprint import pprint
    import math
    import difflib
    import logging
    import os
    import platform
    import re
    import traceback
    import numpy as np
    import pandas as pd
    import matplotlib
    from matplotlib.widgets import Button, TextBox
    from matplotlib.colors import LinearSegmentedColormap
except OSError as e:
    print(f"Error loading module: {e}")
    sys.exit(109)

all_columns_to_remove = ['trial_index', 'arm_name', 'trial_status', 'generation_method', 'generation_node']
val_if_nothing_found = 99999999999999999999999999999999999999999999999999999999999
NO_RESULT = "{:.0e}".format(val_if_nothing_found)

def starts_with_OO_Info(s: str) -> bool:
    if not isinstance(s, str):
        raise TypeError(f"Expected input of type str, got {type(s).__name__}")

    return s.startswith("OO_Info_")

def dier(*args: Any, exit: Union[bool, int] = True) -> None:
    for msg in args:
        pprint(msg)
    if exit is False or exit == 0:
        return
    sys.exit(exit if isinstance(exit, int) else 1)

def check_environment_variable(variable_name: str) -> bool:
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

def in_venv() -> bool:
    return sys.prefix != sys.base_prefix

if not in_venv():
    print("No venv loaded. Cannot continue.")
    sys.exit(19)

def looks_like_float(x: Union[float, int, str, None]) -> bool:
    if isinstance(x, (int, float)):
        return True  # int and float types are directly considered as floats

    if isinstance(x, str):
        try:
            float(x)  # Try converting string to float
            return True
        except ValueError:
            return False  # If conversion fails, it's not a float-like string

    return False  # If x is neither str, int, nor float, it's not float-like

def looks_like_int(x: Union[float, int, str, None]) -> bool:
    if isinstance(x, bool):
        return False

    if isinstance(x, int):
        return True

    if isinstance(x, float):
        return x.is_integer()

    if isinstance(x, str):
        return bool(re.match(r'^\d+$', x))

    return False

def looks_like_number (x: Union[float, int, str, None]) -> bool:
    return looks_like_float(x) or looks_like_int(x) or type(x) is int or type(x) is float or type(x) is np.int64

def to_int_when_possible(val: Any) -> Union[None, int, float, str]:
    if isinstance(val, int):
        return val

    if isinstance(val, float):
        if val.is_integer():
            return int(val)
        return val

    if isinstance(val, str):
        val = val.strip()

        if re.fullmatch(r'-?\d+', val):
            return int(val)

        if re.fullmatch(r'-?\d+\.\d+', val):
            try:
                fval = float(val)
                return fval if not fval.is_integer() else int(fval)
            except Exception:
                return val

        if re.fullmatch(r'-?\d+(?:\.\d+)?[eE][-+]?\d+', val):
            try:
                fval = float(val)
                return fval if not fval.is_integer() else int(fval)
            except Exception:
                return val

        return val

    try:
        fval = float(val)
        return fval if not fval.is_integer() else int(fval)
    except Exception:
        return val

def flatten_extend(matrix: list) -> list:
    flat_list = []
    for row in matrix:
        flat_list.extend(row)
    return flat_list

def convert_string_to_number(input_string: str) -> Union[int, float, None]:
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
            return int(int_match.group(0))
    except AssertionError as e:
        print(f"Assertion error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")

        tb = traceback.format_exc()
        print(tb)

    return None

def log_error(error_text: str) -> None:
    print(f"Error: {error_text}", file=sys.stderr)

def check_if_results_are_empty(result_column_values: Any, csv_file_path: str) -> None:
    filtered_data = list(filter(lambda x: not math.isnan(x), result_column_values.tolist()))

    number_of_non_nan_results = len(filtered_data)

    if number_of_non_nan_results == 0:
        print(f"No values were found. Every evaluation found in {csv_file_path} evaluated to NaN.")
        sys.exit(11)

def get_result_column_values(df: pd.DataFrame, csv_file_path: str) -> Any:
    res_col_name = get_result_name_or_default_from_csv_file_path(csv_file_path)

    result_column_values = df[res_col_name]

    check_if_results_are_empty(result_column_values, csv_file_path)

    return result_column_values

def check_path(_path: str) -> None:
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

def print_color(color: str, text: str) -> None:
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
        print(f"{color_codes[color]}{text}{end_color}", file=sys.stderr)
    except AssertionError as e:
        print(f"Error: {e}")
        print(text)

def check_python_version() -> None:
    python_version = platform.python_version()
    supported_versions = ["3.8.10", "3.8.20", "3.9.21", "3.10.4", "3.10.12", "3.11.2", "3.11.9", "3.9.2", "3.11.3", "3.12.3", "3.12.4", "3.12.5", "3.12.6", "3.12.7", "3.12.7+", "3.12.8", "3.13.1", "3.13.2", "3.13.3", "3.13.5"]
    if python_version not in supported_versions:
        print_color("yellow", f"Warning: Supported python versions are {', '.join(supported_versions)}, but you are running {python_version}. This may or may not cause problems. Just is just a warning.")

def create_widgets(_data: Any) -> Any:
    _plt, button, MAXIMUM_TEXTBOX, MINIMUM_TEXTBOX, _args, TEXTBOX_MINIMUM, TEXTBOX_MAXIMUM, update_graph = _data

    button_ax = _plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(button_ax, 'Update Graph')

    button.on_clicked(update_graph)

    max_string, min_string = "", ""

    if looks_like_float(_args.max):
        max_string = str(_args.max)

    if looks_like_float(_args.min):
        min_string = str(_args.min)

    TEXTBOX_MINIMUM = _plt.axes([0.2, 0.025, 0.1, 0.04])
    MINIMUM_TEXTBOX = TextBox(TEXTBOX_MINIMUM, 'Minimum result:', initial=min_string)

    TEXTBOX_MAXIMUM = _plt.axes([0.5, 0.025, 0.1, 0.04])
    MAXIMUM_TEXTBOX = TextBox(TEXTBOX_MAXIMUM, 'Maximum result:', initial=max_string)

    return button, MAXIMUM_TEXTBOX, MINIMUM_TEXTBOX, TEXTBOX_MINIMUM, TEXTBOX_MAXIMUM

def die_if_no_nonempty_graph(non_empty_graphs: Any, _exit: Any) -> None:
    if not non_empty_graphs:
        print('No non-empty graphs to display.')
        if _exit:
            sys.exit(2)

def get_r(df_filtered: pd.DataFrame) -> int:
    r = 2

    if len(list(df_filtered.columns)) == 1:
        r = 1

    return r

def save_to_file (_fig: Any, _args: Any, _plt: Any) -> None:
    _fig.set_size_inches(15.5, 9.5)

    _path = os.path.dirname(_args.save_to_file)
    if _path:
        os.makedirs(_path, exist_ok=True)
    try:
        _plt.savefig(_args.save_to_file)
    except OSError as e:
        print(f"Error: {e}. This may happen on unstable file systems or in docker containers.")
        sys.exit(199)

def check_dir_and_csv(_args: Any, csv_file_path: str) -> None:
    if not os.path.isdir(_args.run_dir):
        print(f"The path {_args.run_dir} does not point to a folder. Must be a folder.")
        sys.exit(11)

    if not os.path.exists(csv_file_path):
        print(f'The file {csv_file_path} does not exist.')
        sys.exit(39)

def get_csv_file_path(_args: Any) -> str:
    pd_csv = "results.csv"
    csv_file_path = os.path.join(_args.run_dir, pd_csv)
    check_dir_and_csv(_args, csv_file_path)

    return csv_file_path

def drop_empty_results (df: pd.DataFrame, res_col_name: str) -> pd.DataFrame:
    negative_rows_to_remove = df[df[res_col_name].astype(str) == '-' + NO_RESULT].index
    positive_rows_to_remove = df[df[res_col_name].astype(str) == NO_RESULT].index

    df.drop(negative_rows_to_remove, inplace=True)
    df.drop(positive_rows_to_remove, inplace=True)

    return df

def hide_empty_plots(parameter_combinations: list, num_rows: int, num_cols: int, axs: Any) -> Any:
    for i in range(len(parameter_combinations), num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        axs[row, col].set_visible(False)

    return axs

def check_first_line_max(run_dir: str) -> bool:
    file_path = f"{run_dir}/result_min_max.txt"

    try:
        # Open the file and read the first line
        with open(file_path, mode='r', encoding="utf-8") as file:
            first_line = file.readline().strip()  # Removes leading and trailing whitespace

        # Check if the first line is "max"
        return first_line.lower() == "max"

    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

def get_title(_args: Any, result_column_values: pd.DataFrame, df_filtered: pd.DataFrame, num_entries: int, _min: Union[float, int, None], _max: Union[float, int, None]) -> str:
    res_col_name = get_result_name_or_default_from_csv_file_path(_args.run_dir + "/results.csv")

    maximize = check_first_line_max(_args.run_dir)

    extreme_index = None
    if maximize:
        extreme_index = result_column_values.idxmax()
    else:
        extreme_index = result_column_values.idxmin()

    extreme_values = df_filtered.loc[extreme_index].to_dict()

    title = "Minimum"
    if maximize:
        title = "Maximum"

    extreme_values_items = extreme_values.items()

    title_values = []

    for _l in extreme_values_items:
        if res_col_name not in _l:
            key = _l[0]
            value = to_int_when_possible(_l[1])
            if key not in ["generation_node", res_col_name]:
                title_values.append(f"{key} = {value}")

    title += " of f("
    title += ', '.join(title_values)
    title += f") = {to_int_when_possible(result_column_values[extreme_index])}"

    title += f"\nNumber of evaluations shown: {num_entries}"

    if _min is not None:
        title += f", show min = {to_int_when_possible(_min)}"

    if _max is not None:
        title += f", show max = {to_int_when_possible(_max)}"

    return title

def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def _unidiff_output(expected: str, actual: str) -> str:
    """
    Helper function. Returns a string containing the unified diff of two multiline strings.
    """

    diff = difflib.unified_diff(expected, actual)

    return ''.join(diff)

def print_diff(n: str, i: str, o: str) -> None:
    if isinstance(i, str):
        print(f"{n} Should be:", i.strip())
    else:
        print(f"{n} Should be:", i)

    if isinstance(o, str):
        print("Is:", o.strip())
    else:
        print("Is:", o)

    if isinstance(i, str) or isinstance(o, str):
        output = _unidiff_output(str(json.dumps(i)), str(json.dumps(o)))
        if output:
            print("Diff:", output)

def _is_equal(name: str, _input: Any, output: Any) -> bool:
    _equal_types = [
        int, str, float, bool
    ]
    for equal_type in _equal_types:
        if type(_input) is equal_type and type(output) and _input != output:
            print_color("red", f"\n\nFailed test (1): {name}")
            return True

    if type(_input) is not type(output):
        print_color("red", f"\n\nFailed test (4): {name}")
        return True

    if isinstance(_input, bool) and _input != output:
        print_color("red", f"\n\nFailed test (6): {name}")
        return True

    if (output is None and _input is not None) or (output is not None and _input is None):
        print_color("red", f"\n\nFailed test (7): {name}")
        return True

    #print_color("green", f"Test OK: {name}")
    return False

def is_equal(n: str, o: Any, i: Any) -> bool:
    r = _is_equal(n, i, o)

    if r:
        print_diff(n, i, o)

    if os.path.exists("None"):
        print("Folder 'None' exists! Exiting.")
        sys.exit(255)

    return r

def _is_not_equal(name: str, _input: Any, output: Any) -> bool:
    _equal_types = [
        int, str, float, bool
    ]
    for equal_type in _equal_types:
        if isinstance(_input, equal_type) and isinstance(output, equal_type) and _input == output:
            print_color("red", f"\n\nFailed test (1): {name}")
            return True

    if isinstance(_input, bool) and _input == output:
        print_color("red", f"\n\nFailed test (2): {name}")
        return True

    if not (output is not None and _input is not None):
        print_color("red", f"\n\nFailed test (3): {name}")
        return True

    #print_color("green", f"Test OK: {name}")
    return False

def is_not_equal(n: str, i: Any, o: Any) -> bool:
    r = _is_not_equal(n, i, o)

    if r:
        print_diff(n, i, o)

    return r

def set_min_max(MINIMUM_TEXTBOX: Any, MAXIMUM_TEXTBOX: Any, _min: Union[None, int, float], _max: Union[None, int, float]) -> Tuple[Union[int, float, None], Union[int, float, None]]:
    if MINIMUM_TEXTBOX and looks_like_float(MINIMUM_TEXTBOX.text):
        _min = convert_string_to_number(MINIMUM_TEXTBOX.text)

    if MAXIMUM_TEXTBOX and looks_like_float(MAXIMUM_TEXTBOX.text):
        _max = convert_string_to_number(MAXIMUM_TEXTBOX.text)

    return _min, _max

def get_num_subplots_rows_and_cols(non_empty_graphs: list) -> Tuple[int, int, int]:
    num_subplots = len(non_empty_graphs)
    num_cols = math.ceil(math.sqrt(num_subplots))
    num_rows = math.ceil(num_subplots / num_cols)

    return num_subplots, num_cols, num_rows

def remove_widgets(fig: Any, button: Any, MAXIMUM_TEXTBOX: Any, MINIMUM_TEXTBOX: Any) -> None:
    for widget in fig.axes:
        if widget not in [button.ax, MAXIMUM_TEXTBOX.ax, MINIMUM_TEXTBOX.ax]:
            widget.remove()

def get_non_empty_graphs(parameter_combinations: list, df_filtered: pd.DataFrame, _exit: Union[bool, None]) -> list:
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

def get_df_filtered(_args: Any, df: pd.DataFrame) -> pd.DataFrame:
    res_col_name = get_result_name_or_default_from_csv_file_path(_args.run_dir + "/results.csv")

    columns_to_remove = []
    existing_columns = df.columns.values.tolist()

    for col in existing_columns:
        if col in all_columns_to_remove:
            columns_to_remove.append(col)

    if len(_args.allow_axes):
        for col in existing_columns:
            if col != res_col_name and col not in flatten_extend(_args.allow_axes):
                columns_to_remove.append(col)

    df_filtered = df.drop(columns=columns_to_remove)

    return df_filtered

def print_filtering_message(_min: Union[int, float, None], _max: Union[int, float, None]) -> None:
    if _min and not _max:
        print("Using --min filtered out all results")
    elif not _min and _max:
        print("Using --max filtered out all results")
    elif _min and _max:
        print("Using --min and --max filtered out all results")
    else:
        print("For some reason, there were values in the beginning but not after filtering")

def print_no_results_message(csv_file_path: str, _min: Union[int, float, None], _max: Union[int, float, None]) -> None:
    if _min is not None and _max is not None:
        print(f"No applicable values could be found in {csv_file_path} (min: {_min}, max: {_max}).")
    elif _min is not None:
        print(f"No applicable values could be found in {csv_file_path} (min: {_min}).")
    elif _max is not None:
        print(f"No applicable values could be found in {csv_file_path} (max: {_max}).")
    else:
        print(f"No applicable values could be found in {csv_file_path}.")

def check_min_and_max(num_entries: int, nr_of_items_before_filtering: int, csv_file_path: str, _min: Union[int, float, None] = None, _max: Union[int, float, None] = None, _exit: bool = True) -> None:
    if num_entries is None or num_entries == 0:
        if nr_of_items_before_filtering:
            print_filtering_message(_min, _max)
        else:
            if not os.environ.get("NO_NO_RESULT_ERROR"):
                print_no_results_message(csv_file_path, _min, _max)

        if _exit:
            sys.exit(4)

def contains_strings(series: Any) -> bool:
    return series.apply(lambda x: isinstance(x, str)).any()

def file_exists(csv_file_path: Optional[str]) -> bool:
    return bool(csv_file_path) and isinstance(csv_file_path, str) and os.path.exists(csv_file_path)

def load_csv(csv_file_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_file_path, index_col=0)

def headers_match(df: pd.DataFrame, old_headers_string: Optional[str]) -> bool:
    if old_headers_string is None:
        return True
    df_header_string = ','.join(sorted(df.columns))
    return df_header_string == old_headers_string

def filter_by_result_range(df: pd.DataFrame, res_col_name: str, _min: Optional[Union[int, float]], _max: Optional[Union[int, float]]) -> pd.DataFrame:
    if res_col_name not in df:
        handle_missing_result_column(res_col_name)

    if _min is not None:
        df = df[df[res_col_name] >= _min]
    if _max is not None:
        df = df[df[res_col_name] <= _max]
    return df.dropna(subset=[res_col_name])

def handle_missing_result_column(res_col_name: str) -> None:
    if not os.environ.get("NO_NO_RESULT_ERROR"):
        print(f"There was no '{res_col_name}' column. This may mean all tests failed. Cannot continue.")
    sys.exit(10)

def drop_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns_with_strings = [col for col in df.columns if contains_strings(df[col])]
    df = df.drop(columns=columns_with_strings)
    if len(df.columns) <= 1 and columns_with_strings:
        print("All available columns contained strings instead of numbers. Cannot plot.")
        sys.exit(19)
    return df

def handle_csv_exceptions(csv_file_path: str, error: Exception) -> None:
    error_messages = {
        pd.errors.EmptyDataError: f"{csv_file_path} has no lines to parse.",
        pd.errors.ParserError: f"{csv_file_path} is invalid CSV. Parsing error: {str(error).rstrip()}",
        UnicodeDecodeError: f"{csv_file_path} does not seem to be a text-file or has invalid UTF-8 encoding."
    }
    if not os.environ.get("PLOT_TESTS"):
        print(error_messages.get(type(error), "Unknown error."))
    sys.exit({pd.errors.EmptyDataError: 19, pd.errors.ParserError: 12, UnicodeDecodeError: 7}.get(type(error), 1))

def get_result_name_or_default_from_csv_file_path(csv_file_path: str) -> str:
    res_col_name = "result"

    dir_path = '/'.join(csv_file_path.split('/')[:-1]) + '/'

    result_names_txt = f"{dir_path}/result_names.txt"

    if os.path.exists(result_names_txt):
        with open(result_names_txt, mode='r', encoding="utf-8") as file:
            lines = file.readlines()
            if len(lines) > 1:
                raise ValueError(f"The file >{result_names_txt} contains more than one line<")
            res_col_name = lines[0].strip()

    return res_col_name

def get_df_without_special_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_remove = []
    existing_columns = df.columns.values.tolist()

    for col in existing_columns:
        if col in all_columns_to_remove or starts_with_OO_Info(col) or col in ["signal", "hostname", "queue_time", "submit_time", "exit_code", "end_time", "run_time", "program_string", "start_time"]:
            columns_to_remove.append(col)

    df_filtered = df.drop(columns=columns_to_remove)

    return df_filtered

def get_data(
    csv_file_path: str,
    _min: Optional[Union[int, float]],
    _max: Optional[Union[int, float]],
    old_headers_string: Optional[str] = None,
    drop_columns_with_strings: Union[str, bool] = False
) -> Optional[pd.DataFrame]:
    res_col_name = get_result_name_or_default_from_csv_file_path(csv_file_path)

    if not isinstance(csv_file_path, str):
        return None

    if not file_exists(csv_file_path):
        return None

    try:
        df = load_csv(csv_file_path)
        df = get_df_without_special_columns(df)
        if not headers_match(df, old_headers_string):
            print(f"Cannot merge {csv_file_path}. Old headers: {old_headers_string}, new headers: {','.join(sorted(df.columns))}")
            return None
        df = filter_by_result_range(df, res_col_name, _min, _max)
        if drop_columns_with_strings:
            df = drop_string_columns(df)
        return drop_empty_results(df, res_col_name)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as e:
        handle_csv_exceptions(csv_file_path, e)
    except KeyError as e:
        print(f"Column '{res_col_name}' could not be found in {csv_file_path}: {e}.")
        sys.exit(6)

    return None

def show_legend(_args: Any, _fig: Any, _scatter: Any, axs: Any) -> None:
    res_col_name = get_result_name_or_default_from_csv_file_path(_args.run_dir + "/results.csv")

    try:
        if not _args.no_legend:
            cbar = _fig.colorbar(_scatter, ax=axs, orientation='vertical', fraction=0.02, pad=0.05)
            cbar.set_label(res_col_name, rotation=270, labelpad=15)

            cbar.formatter.set_scientific(False)
            cbar.formatter.set_useMathText(False)
    except Exception as e:
        print_color("red", f"ERROR: show_legend failed with error: {e}")

def get_parameter_combinations(df_filtered: pd.DataFrame, csv_file_path: str) -> list:
    res_col_name = get_result_name_or_default_from_csv_file_path(csv_file_path)

    r = get_r(df_filtered)

    df_filtered_cols = df_filtered.columns.tolist()

    del df_filtered_cols[df_filtered_cols.index(res_col_name)]

    parameter_combinations: list = list(combinations(df_filtered_cols, r))

    if len(parameter_combinations) == 0:
        parameter_combinations = list([*df_filtered_cols])

    return parameter_combinations

def get_colors(df: pd.DataFrame, csv_file_path: str) -> Any:
    res_col_name = get_result_name_or_default_from_csv_file_path(csv_file_path)

    colors = None

    try:
        colors = df[res_col_name]
    except KeyError as e:
        if str(e) == f"'{res_col_name}'":
            print(f"get_colors: Could not find any results for column {res_col_name}")
            sys.exit(3)
        else:
            print(f"Key-Error: {e}")
            sys.exit(8)

    return colors

def get_color_list(df: pd.DataFrame, _args: Any, _plt: Any, csv_file_path: str) -> Any:
    colors = get_colors(df, csv_file_path)

    if colors is None:
        print_color("yellow", "colors is None. Cannot plot.")
        sys.exit(3)

    maximize = check_first_line_max(_args.run_dir)
    if maximize:
        colors = -1 * colors  # Negate colors for maximum result

    norm = None
    try:
        norm = _plt.Normalize(colors.min(), colors.max())
    except Exception as e:
        print_color("red", f"Wrong values in CSV or error parsing CSV file: {e}")
        sys.exit(16)

    c = ["darkred", "red", "lightcoral", "palegreen", "green", "darkgreen"]
    c = c[::-1]
    v = [0, 0.3, 0.5, 0.7, 0.9, 1]
    _l = list(zip(v, c))

    cmap = LinearSegmentedColormap.from_list('rg', _l, N=256)

    return cmap, norm, colors

def merge_df_with_old_data(_args: Any, df: pd.DataFrame, _min: Union[int, float, None], _max: Union[int, float, None], old_headers_string: str) -> pd.DataFrame:
    if len(_args.merge_with_previous_runs):
        for prev_run in _args.merge_with_previous_runs:
            prev_run_csv_path = prev_run[0] + "/results.csv"
            prev_run_df = get_data(prev_run_csv_path, _min, _max, old_headers_string)
            if prev_run_df:
                df = df.merge(prev_run_df, how='outer')
    return df

def print_if_not_plot_tests_and_exit(msg: str, exit_code: int) -> str:
    if not os.environ.get("PLOT_TESTS"):
        print(msg)
    if exit_code is not None:
        sys.exit(exit_code)

    return msg

def load_and_merge_data(_args: Any, _min: Union[int, float, None], _max: Union[int, float, None], filter_out_strings: str, csv_file_path: str) -> Union[pd.DataFrame, None]:
    df = get_data(csv_file_path, _min, _max, None, filter_out_strings)

    if df is not None and not df.empty:
        old_headers_string = ','.join(sorted(df.columns))
        return merge_df_with_old_data(_args, df, _min, _max, old_headers_string)

    return None

def _update_graph(_params: list) -> None:
    csv_file_path, plt, fig, MINIMUM_TEXTBOX, MAXIMUM_TEXTBOX, _min, _max, _args, filter_out_strings, set_title, plot_graphs, button = _params

    try:
        csv_file_path = get_csv_file_path(_args)
        _min, _max = set_min_max(MINIMUM_TEXTBOX, MAXIMUM_TEXTBOX, _min, _max)
        df = load_and_merge_data(_args, _min, _max, filter_out_strings, csv_file_path)
        if df is not None and not df.empty:
            df_filtered = get_df_filtered(_args, df)

            check_filtering(df, df_filtered, csv_file_path, _min, _max)
            plot_parameters([csv_file_path, df, df_filtered, _args, fig, button, MINIMUM_TEXTBOX, MAXIMUM_TEXTBOX, plot_graphs, set_title, filter_out_strings, _min, _max])

            plt.draw()
        else:
            print("Failed to get df")

    except Exception as e:
        _handle_update_graph_exception(e)

def check_filtering(df: pd.DataFrame, df_filtered: pd.DataFrame, csv_file_path: str, _min: Union[int, float, None], _max: Union[int, float, None]) -> None:
    nr_of_items_before_filtering = len(df)
    check_min_and_max(len(df_filtered), nr_of_items_before_filtering, csv_file_path, _min, _max)

def plot_parameters(_params: list) -> None:
    csv_file_path, df, df_filtered, _args, fig, button, MINIMUM_TEXTBOX, MAXIMUM_TEXTBOX, plot_graphs, set_title, filter_out_strings, _min, _max = _params
    parameter_combinations = get_parameter_combinations(df_filtered, csv_file_path)
    non_empty_graphs = get_non_empty_graphs(parameter_combinations, df_filtered, filter_out_strings)

    num_subplots, num_cols, num_rows = get_num_subplots_rows_and_cols(non_empty_graphs)
    remove_widgets(fig, button, MAXIMUM_TEXTBOX, MINIMUM_TEXTBOX)

    axs = fig.subplots(num_rows, num_cols)
    result_column_values = get_result_column_values(df, get_csv_file_path(_args))

    plot_graphs([df, fig, axs, df_filtered, non_empty_graphs, num_subplots, parameter_combinations, num_rows, num_cols, result_column_values, csv_file_path])
    set_title(df_filtered, result_column_values, len(df_filtered), _min, _max)

def _handle_update_graph_exception(e: Union[str, Exception]) -> None:
    if "invalid command name" not in str(e):
        print(f"Failed to update graph: {e}")

def set_margins(fig: Any) -> None:
    left = 0.04
    right = 0.864
    bottom = 0.171
    top = 0.9
    wspace = 0.27
    hspace = 0.31

    fig.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

def use_matplotlib(_args: Any) -> None:
    try:
        if not _args.save_to_file:
            matplotlib.use('TkAgg')
    except Exception as e:
        print(f"An error occurred while loading TkAgg. This may happen when you forgot to add -X to your ssh-connection: {e}.")
        sys.exit(33)

def filter_data(_args: Any, dataframe: pd.DataFrame, min_value: Union[int, float, None] = None, max_value: Union[int, float, None] = None, csv_file_path: str = "") -> pd.DataFrame:
    res_col_name = get_result_name_or_default_from_csv_file_path(csv_file_path)

    try:
        if min_value is not None:
            dataframe = dataframe[dataframe[res_col_name] >= min_value]
        if max_value is not None:
            dataframe = dataframe[dataframe[res_col_name] <= max_value]
    except KeyError:
        print_if_not_plot_tests_and_exit(f"{_args.run_dir}/results.csv seems to have no results column.", 19)

    return dataframe

def print_traceback() -> None:
    tb = traceback.format_exc()
    print(tb)

def is_valid_time_format(time_string: str) -> bool:
    try:
        datetime.strptime(time_string, '%Y-%m-%d %H:%M:%S')
        return True
    except ValueError:
        return False

def check_args(_args: Any) -> None:
    if _args.min and _args.max:
        if _args.min > _args.max:
            _args.max, _args.min = _args.min, _args.max
        elif _args.min == _args.max:
            print("Max and min value are the same. May result in empty data")

    check_path(_args.run_dir)

def can_be_plotted(path: str) -> bool:
    result_file = os.path.join(path, "result_names.txt")

    if not os.path.exists(result_file):
        return True

    with open(result_file, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]

    return len(lines) == 1

def die_if_cannot_be_plotted(run_dir: Optional[str]) -> None:
    if run_dir is None:
        log_error("run_dir was empty")
        sys.exit(2)

    if not can_be_plotted(run_dir):
        log_error(f"{run_dir} contains multiple RESULTS and thus can only be plotted by parallel plot")
        sys.exit(2)

check_python_version()
