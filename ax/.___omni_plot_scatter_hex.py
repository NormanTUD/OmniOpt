import re
import os
import importlib.util
import numpy as np
import sys
import argparse
import math
import time
import threading
import signal
import seaborn as sns
from rich.pretty import pprint
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from itertools import combinations

from rich.traceback import install
install(show_locals=True)

# Constants
VAL_IF_NOTHING_FOUND = 99999999999999999999999999999999999999999999999999999999999
NO_RESULT = "{:.0e}".format(VAL_IF_NOTHING_FOUND)
BUBBLESIZEINPX = int(os.environ.get('BUBBLESIZEINPX', 15))
ORIGINAL_PWD = os.environ.get("ORIGINAL_PWD", "")
script_dir = os.path.dirname(os.path.realpath(__file__))
helpers_file = f"{script_dir}/.helpers.py"

# Import helpers
spec = importlib.util.spec_from_file_location("helpers", helpers_file)
my_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(my_module)

if ORIGINAL_PWD:
    os.chdir(ORIGINAL_PWD)

# Global variables
args = None
fig = None
maximum_textbox = None
minimum_textbox = None

# Signal handling
signal.signal(signal.SIGINT, signal.SIG_DFL)

# Utility functions
def print_debug(msg):
    global args

    if args and args.debug:
        print("DEBUG: ", end="")
        pprint(msg)

def dier(msg):
    pprint(msg)
    sys.exit(9)

def check_path():
    global args
    print_debug("check_path")
    if not os.path.exists(args.run_dir):
        print(f'The folder {args.run_dir} does not exist.')
        sys.exit(1)

def get_current_time():
    print_debug("get_current_time()")
    return time.time()

def check_csv_modified(last_modified_time, csv_file_path):
    print_debug("check_csv_modified()")
    current_modified_time = os.path.getmtime(csv_file_path)
    return current_modified_time > last_modified_time

def to_int_when_possible(val):
    print_debug("to_int_when_possible")
    if isinstance(val, (int, float)) and (isinstance(val, int) or val.is_integer()):
        return int(val)
    if isinstance(val, str) and val.isdigit():
        return int(val)
    if isinstance(val, str) and re.match(r'^-?\d+(?:\.\d+)?$', val) is None:
        return val
    try:
        val = float(val)
        return '{:.{}f}'.format(val, len(str(val).split('.')[1])).rstrip('0').rstrip('.')
    except:
        return val

def set_margins(fig):
    print_debug("set_margins()")
    fig.subplots_adjust(left=0.04, bottom=0.171, right=0.864, top=0.9, wspace=0.27, hspace=0.31)

def check_if_results_are_empty(result_column_values):
    print_debug("check_if_results_are_empty()")
    filtered_data = list(filter(lambda x: not math.isnan(x), result_column_values.tolist()))
    number_of_non_nan_results = len(filtered_data)
    if number_of_non_nan_results == 0:
        print(f"No values were found. Every evaluation found in {csv_file_path} evaluated to NaN.")
        sys.exit(11)

def set_title(fig, df_filtered, result_column_values, num_entries, _min, _max):
    print_debug("set_title()")
    _mean = result_column_values.mean()
    extreme_index = result_column_values.idxmax() if os.path.exists(args.run_dir + "/maximize") else result_column_values.idxmin()
    extreme_values = df_filtered.loc[extreme_index].to_dict()
    title = "Maximum" if os.path.exists(args.run_dir + "/maximize") else "Minimum"
    title_values = [f"{key} = {to_int_when_possible(value)}" for key, value in extreme_values.items() if "result" not in key]
    title += " of f(" + ', '.join(title_values) + f") = {to_int_when_possible(result_column_values[extreme_index])}"
    title += f"\nNumber of evaluations shown: {num_entries}"
    if _min is not None:
        title += f", show min = {to_int_when_possible(_min)}"
    if _max is not None:
        title += f", show max = {to_int_when_possible(_max)}"
    if _mean is not None:
        title += f", mean result = {to_int_when_possible(_mean)}"
    fig.suptitle(title)

def check_args():
    print_debug("check_args()")
    global args
    if args.min or args.max:
        if args.min and args.max and args.min > args.max:
            args.min, args.max = args.max, args.min
        elif args.min == args.max:
            print("Max and min value are the same. May result in empty data")

    check_path()

def check_dir_and_csv(csv_file_path):
    print_debug("check_dir_and_csv()")
    if not os.path.isdir(args.run_dir):
        print(f"The path {args.run_dir} does not point to a folder. Must be a folder.")
        sys.exit(11)
    if not os.path.exists(csv_file_path):
        print(f'The file {csv_file_path} does not exist.')
        sys.exit(39)

def check_min_and_max(num_entries, nr_of_items_before_filtering, csv_file_path, _min, _max, _exit=True):
    print_debug("check_min_and_max()")
    if num_entries is None or num_entries == 0:
        if nr_of_items_before_filtering:
            if _min and not _max:
                print(f"Using --min filtered out all results")
            elif not _min and _max:
                print(f"Using --max filtered out all results")
            elif _min and _max:
                print(f"Using --min and --max filtered out all results")
            else:
                print(f"For some reason, there were values in the beginning but not after filtering")
        else:
            print(f"No applicable values could be found in {csv_file_path}.")
        if _exit:
            sys.exit(4)

def get_data(csv_file_path, result_column, _min, _max, old_headers_string=None):
    print_debug("get_data()")
    try:
        df = pd.read_csv(csv_file_path, index_col=0)
        if old_headers_string:
            df_header_string = ','.join(sorted(df.columns))
            if df_header_string != old_headers_string:
                print(f"Cannot merge {csv_file_path}. Old headers: {old_headers_string}, new headers {df_header_string}")
                return None
        if _min is not None:
            df = df[df[result_column] >= _min]
        if _max is not None:
            df = df[df[result_column] <= _max]
        if result_column not in df:
            print(f"There was no {result_column} in {csv_file_path}. This may mean all tests failed. Cannot continue.")
            sys.exit(10)
        df.dropna(subset=[result_column], inplace=True)
    except pd.errors.EmptyDataError:
        print(f"{csv_file_path} has no lines to parse.")
        sys.exit(5)
    except pd.errors.ParserError as e:
        print(f"{csv_file_path} is invalid CSV. Parsing error: {str(e).rstrip()}")
        sys.exit(12)
    except UnicodeDecodeError:
        print(f"{csv_file_path} does not seem to be a text-file or it has invalid UTF8 encoding.")
        sys.exit(7)
    try:
        negative_rows_to_remove = df[df[result_column].astype(str) == '-' + NO_RESULT].index
        positive_rows_to_remove = df[df[result_column].astype(str) == NO_RESULT].index
        df.drop(negative_rows_to_remove, inplace=True)
        df.drop(positive_rows_to_remove, inplace=True)
    except KeyError:
        print(f"column named `{result_column}` could not be found in {csv_file_path}.")
        sys.exit(6)
    return df

def plot_with_seaborn(df_filtered, param1, param2, result_column):
    print_debug("plot_with_seaborn()")
    #print("DataFrame:")
    #print(df_filtered.to_string(index=False))

    ignored_params = ['trial_status', 'arm_name', 'generation_method']

    # Überprüfen, ob die Parameter zu den ignorierten Parametern gehören
    if param1 in ignored_params or param2 in ignored_params:
        print("One or both parameters are ignored.")
        print(f"Ignored parameters: {param1}, {param2}")
        return
    
    if param1 in df_filtered.columns and param2 in df_filtered.columns:
        print("Both parameters are found in dataframe columns.")
        
        # Überprüfen, ob die Parameter numerisch sind
        if pd.api.types.is_numeric_dtype(df_filtered[param1]) and pd.api.types.is_numeric_dtype(df_filtered[param2]):
            print("Both parameters are numeric.")
            print(df_filtered.to_string(index=False))
            param1_data = pd.to_numeric(df_filtered[param1], errors='coerce')
            param2_data = pd.to_numeric(df_filtered[param2], errors='coerce')
            param1_data.dropna(inplace=True)
            param2_data.dropna(inplace=True)
            
            if len(param1_data) > 0 and len(param2_data) > 0:
                sns.jointplot(x=param1_data, y=param2_data, kind="hex", color="k")
                plt.xlabel(param1)
                plt.ylabel(param2)
            else:
                print(f"No data available for plotting {param1} vs {param2}.")
        else:
            print("One or both parameters are not numeric. Unable to plot.")
    else:
        print(f"One or both of the parameters ({param1}, {param2}) not found in dataframe columns.")


def parse_args():
    print_debug("parse_args()")
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", help="The directory to search for results in")
    parser.add_argument("--csv", default="pd.csv", help="The name of the csv-file to plot")
    parser.add_argument("--min", type=float, default=None, help="Minimum value to include")
    parser.add_argument("--max", type=float, default=None, help="Maximum value to include")
    parser.add_argument("--rows", type=int, default=3, help="Number of rows of subplots")
    parser.add_argument("--cols", type=int, default=3, help="Number of columns of subplots")
    parser.add_argument("--result_column", type=str, default="result", help="Column name for result values")
    parser.add_argument("--exclude_params", type=str, nargs=1, help="Comma-separated list of parameter names to exclude")
    parser.add_argument("--include_params", type=str, nargs=1, help="Comma-separated list of parameter names to include")
    parser.add_argument("--resizable", action="store_true", help="Allow the plot to be resizable")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    return args

def main():
    global args, fig, maximum_textbox, minimum_textbox

    args = parse_args()
    check_args()
    csv_file_path = os.path.join(args.run_dir, args.csv)
    check_dir_and_csv(csv_file_path)

    last_modified_time = 0
    df_filtered = None

    last_modified_time = os.path.getmtime(csv_file_path)
    df = get_data(csv_file_path, args.result_column, args.min, args.max)
    nr_of_items_before_filtering = df.shape[0]
    num_entries = df.shape[0]
    check_min_and_max(num_entries, nr_of_items_before_filtering, csv_file_path, args.min, args.max)

    if args.include_params:
        include_params_list = args.include_params[0].split(',')
        df_filtered = df[include_params_list + [args.result_column]]
    elif args.exclude_params:
        exclude_params_list = args.exclude_params[0].split(',')
        df_filtered = df.drop(columns=exclude_params_list)
    else:
        df_filtered = df

    # Check if df_filtered is a DataFrame before plotting
    if isinstance(df_filtered, pd.DataFrame):
        param_combinations = list(combinations([col for col in df_filtered.columns if col != args.result_column], 2))

        fig, axes = plt.subplots(args.rows, args.cols, figsize=(15, 15))
        fig.subplots_adjust(hspace=0.4, wspace=0.4)

        for idx, (param1, param2) in enumerate(param_combinations):
            if idx >= args.rows * args.cols:
                break
            ax = axes[idx // args.cols, idx % args.cols]
            plot_with_seaborn(df_filtered, param1, param2, args.result_column)  # Modified line
            ax.set_xlabel(param1)
            ax.set_ylabel(param2)

        set_margins(fig)
        set_title(fig, df_filtered, df_filtered[args.result_column], num_entries, args.min, args.max)
        plt.show()
    else:
        print("Error: df_filtered is not a valid DataFrame.")

if __name__ == "__main__":
    main()
