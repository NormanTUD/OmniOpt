# DESCRIPTION: Scatter plot
# EXPECTED FILES: results.csv
# TEST_OUTPUT_MUST_CONTAIN: Number of evaluations shown
# TEST_OUTPUT_MUST_CONTAIN: mean result
# TEST_OUTPUT_MUST_CONTAIN: result

from rich.traceback import install
#install(show_locals=True)
install()

import os
script_dir = os.path.dirname(os.path.realpath(__file__))
helpers_file = f"{script_dir}/.helpers.py"
import importlib.util
spec = importlib.util.spec_from_file_location(
    name="helpers",
    location=helpers_file,
)
my_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(my_module)

val_if_nothing_found = 99999999999999999999999999999999999999999999999999999999999
NO_RESULT = "{:.0e}".format(val_if_nothing_found)

# idee: single plots zeichnen und mit plotext anzeigen, so dass man einen überblick kriegt

args = None

def print_debug(msg):
    if args.debug:
        print("DEBUG: ", end="")
        pprint(msg)

import numpy as np
import sys
import argparse
import math
import time
import threading

fig = None
maximum_textbox = None
minimum_textbox = None

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

try:
    from rich.pretty import pprint
except ModuleNotFoundError:
    from pprint import pprint

def dier(msg):
    pprint(msg)
    sys.exit(9)

import importlib.util
spec = importlib.util.spec_from_file_location(
    name="helpers",
    location=".helpers.py",
)
my_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(my_module)

try:
    import re
    import pandas as pd
    from matplotlib.colors import LinearSegmentedColormap

    import matplotlib
    import matplotlib.pyplot as plt

    from matplotlib.widgets import Button, TextBox

    from itertools import combinations
except ModuleNotFoundError as e:
    print(f"Error: {e}")
    sys.exit(244)

# Get shell variables or use default values
BUBBLESIZEINPX = int(os.environ.get('BUBBLESIZEINPX', 15))
ORIGINAL_PWD = os.environ.get("ORIGINAL_PWD", "")

if ORIGINAL_PWD:
    os.chdir(ORIGINAL_PWD)

def get_current_time():
    print_debug("get_current_time()")
    return time.time()

def check_csv_modified(last_modified_time, csv_file_path):
    print_debug("check_csv_modified()")
    current_modified_time = os.path.getmtime(csv_file_path)
    return current_modified_time > last_modified_time

def to_int_when_possible(val):
    print_debug("to_int_when_possible")
    if type(val) == int or (type(val) == float and val.is_integer()) or (type(val) == str and val.isdigit()):
        return int(val)
    if type(val) == str and re.match(r'^-?\d+(?:\.\d+)?$', val) is None:
        return val

    try:
        val = float(val)

        return '{:.{}f}'.format(val, len(str(val).split('.')[1])).rstrip('0').rstrip('.')
    except:
        return val


def set_margins (fig):
    print_debug("set_margins()")
    left  = 0.04
    if args.save_to_file and len(args.allow_axes):
        left  = 0.1
    right = 0.864
    bottom = 0.171
    top = 0.9
    wspace = 0.27
    hspace = 0.31

    fig.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

def check_if_results_are_empty(result_column_values):
    print_debug("check_if_results_are_empty()")
    filtered_data = list(filter(lambda x: not math.isnan(x), result_column_values.tolist()))

    number_of_non_nan_results = len(filtered_data)

    if number_of_non_nan_results == 0:
        print(f"No values were found. Every evaluation found in {csv_file_path} evaluated to NaN.")
        sys.exit(11)

def set_title(fig, df_filtered, result_column_values, num_entries, _min, _max):
    _mean = result_column_values.mean()
    print_debug("set_title")
    #extreme_index = result_column_values.idxmax() if args.run_dir + "/state_files/maximize" in os.listdir(args.run_dir) else result_column_values.idxmin()
    extreme_index = result_column_values.idxmin()
    if os.path.exists(args.run_dir + "/state_files/maximize"):
        extreme_index = result_column_values.idxmax()

    extreme_values = df_filtered.loc[extreme_index].to_dict()

    title = "Minimum"
    if os.path.exists(args.run_dir + "/state_files/maximize"):
        title = "Maximum"

    extreme_values_items = extreme_values.items()

    filtered_extreme_values_items = {}

    title_values = []

    for l in extreme_values_items:
        if not "result" in l:
            key = l[0]
            value = to_int_when_possible(l[1])
            title_values.append(f"{key} = {value}")

    #title_values = [f"{key} = {value}" for key, value in filtered_extreme_values_items]

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

    fig.suptitle(title)

def check_args ():
    print_debug("check_args()")
    global args

    if args.min and args.max:
        if args.min > args.max:
            tmp = args.max
            args.max = args.min
            args.min = tmp
        elif args.min == args.max:
            print("Max and min value are the same. May result in empty data")

    check_path()

def check_dir_and_csv (csv_file_path):
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
            if not os.environ.get("NO_NO_RESULT_ERROR"):
                print(f"No applicable values could be found in {csv_file_path}.")
        if _exit:
            sys.exit(4)

def get_data (csv_file_path, _min, _max, old_headers_string=None):
    print_debug("get_data")
    result_column = "result"
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
        if not "result" in df:
            if not os.environ.get("NO_NO_RESULT_ERROR"):
                print(f"There was no 'result' in {csv_file_path}. This may means all tests failed. Cannot continue.")
            sys.exit(10)
        df.dropna(subset=["result"], inplace=True)
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
        negative_rows_to_remove = df[df["result"].astype(str) == '-' + NO_RESULT].index
        positive_rows_to_remove = df[df["result"].astype(str) == NO_RESULT].index

        df.drop(negative_rows_to_remove, inplace=True)
        df.drop(positive_rows_to_remove, inplace=True)
    except KeyError:
        print(f"column named `result` could not be found in {csv_file_path}.")
        sys.exit(6)

    return df

def hide_empty_plots(parameter_combinations, num_rows, num_cols, axs):
    print_debug("hide_empty_plots()")
    for i in range(len(parameter_combinations), num_rows*num_cols):
        row = i // num_cols
        col = i % num_cols
        axs[row, col].set_visible(False)

def plot_multiple_graphs(fig, non_empty_graphs, num_cols, axs, df_filtered, colors, cmap, norm, parameter_combinations, num_rows):
    print_debug("plot_multiple_graphs")
    for i, (param1, param2) in enumerate(non_empty_graphs):
        row = i // num_cols
        col = i % num_cols
        if (len(args.exclude_params) and not param1 in args.exclude_params[0] and not param2 in args.exclude_params[0]) or len(args.exclude_params) == 0:
            try:
                scatter = axs[row, col].scatter(df_filtered[param1], df_filtered[param2], c=colors, cmap=cmap, norm=norm, s=BUBBLESIZEINPX)
                axs[row, col].set_xlabel(param1)
                axs[row, col].set_ylabel(param2)
            except Exception as e:
                if "" in str(e):
                    scatter = axs.scatter(df_filtered[param1], df_filtered[param2], c=colors, cmap=cmap, norm=norm, s=BUBBLESIZEINPX)
                    axs.set_xlabel(param1)
                    axs.set_ylabel(param2)
                else:
                    print("ERROR: " + str(e))

                    import traceback
                    tb = traceback.format_exc()
                    print(tb)

                    sys.exit(17)

    for i in range(len(parameter_combinations), num_rows*num_cols):
        row = i // num_cols
        col = i % num_cols
        axs[row, col].set_visible(False)

    show_legend(scatter, axs)
    
def show_legend(_scatter, axs):
    print_debug("show_legend")
    global args, fig

    if not args.no_legend:
        try:
            cbar = fig.colorbar(_scatter, ax=axs, orientation='vertical', fraction=0.02, pad=0.05)
            cbar.set_label("result", rotation=270, labelpad=15)

            cbar.formatter.set_scientific(False)
            cbar.formatter.set_useMathText(False)
        except Exception as e:
            print_debug(f"ERROR: show_legend failed with error: {e}")

def plot_two_graphs(axs, df_filtered, non_empty_graphs, colors, cmap, norm):
    print_debug("plot_two_graphs()")
    _x = df_filtered[non_empty_graphs[0][0]]
    try:
        _y = df_filtered[non_empty_graphs[0][1]]
    except Exception as e:
        print(f"Error in plot_two_graphs: {e}")

        import traceback
        tb = traceback.format_exc()
        print("Traceback ==>", tb, "<==")

        print("df_filtered:")
        print(df_filtered.to_string(index=False))

        print("non_empty_graphs:")
        print(non_empty_graphs)
        sys.exit(45)

    scatter = axs.scatter(_x, _y, c=colors, cmap=cmap, norm=norm, s=BUBBLESIZEINPX)
    axs.set_xlabel(non_empty_graphs[0][0])
    axs.set_ylabel(non_empty_graphs[0][1])

    # Farbgebung und Legende für das einzelne Scatterplot
    show_legend(_y, axs)

def plot_single_graph (fig, axs, df_filtered, colors, cmap, norm, non_empty_graphs):
    print_debug("plot_single_graph()")
    _range = range(len(df_filtered))
    _data = df_filtered

    _data = _data[:].values

    _x = []
    _y = []

    for l in _data:
        _x.append(l[0])
        _y.append(l[1])

    scatter = axs.scatter(_x, _y, c=colors, cmap=cmap, norm=norm, s=BUBBLESIZEINPX)
    axs.set_xlabel(non_empty_graphs[0][0])
    axs.set_ylabel("result")

def plot_graphs(df, fig, axs, df_filtered, non_empty_graphs, num_subplots, parameter_combinations, num_rows, num_cols):
    print_debug("plot_graphs")
    colors = get_colors(df)

    if os.path.exists(args.run_dir + "/maximize"):
        colors = -colors  # Negate colors for maximum result

    norm = None
    try:
        norm = plt.Normalize(colors.min(), colors.max())
    except:
        print(f"Wrong values in csv file")
        sys.exit(16)

    c = ["darkred", "red","lightcoral", "palegreen", "green", "darkgreen"]
    c = c[::-1]
    v = [0, 0.3, 0.5, 0.7, 0.9, 1]
    l = list(zip(v,c))

    cmap = LinearSegmentedColormap.from_list('rg', l, N=256)

    if num_subplots == 1 and (type(non_empty_graphs[0]) == str or len(non_empty_graphs[0]) == 1):
        plot_single_graph(fig, axs, df_filtered, colors, cmap, norm, non_empty_graphs)
    else:
        plot_multiple_graphs(fig, non_empty_graphs, num_cols, axs, df_filtered, colors, cmap, norm, parameter_combinations, num_rows)

    hide_empty_plots(parameter_combinations, num_rows, num_cols, axs)

def get_args ():
    global args
    parser = argparse.ArgumentParser(description='Plot optimization runs.', prog="plot")

    parser.add_argument('--run_dir', type=str, help='Path to a CSV file', required=True)
    parser.add_argument('--save_to_file', type=str, help='Save the plot to the specified file', default=None)
    parser.add_argument('--max', type=float, help='Maximum value', default=None)
    parser.add_argument('--min', type=float, help='Minimum value', default=None)
    parser.add_argument('--delete_temp', help='Delete temp files (useless here)', action='store_true', default=False)
    parser.add_argument('--darkmode', help='Enable darktheme', action='store_true', default=False)
    parser.add_argument('--single', help='Print plot to command line', action='store_true', default=False)
    parser.add_argument('--bubblesize', type=int, help='Size of the bubbles', default=15)
    parser.add_argument('--merge_with_previous_runs', action='append', nargs='+', help="Run-Dirs to be merged with", default=[])
    parser.add_argument('--exclude_params', action='append', nargs='+', help="Params to be ignored", default=[])

    parser.add_argument('--allow_axes', action='append', nargs='+', help="Allow specific axes only (parameter names)", default=[])
    parser.add_argument('--plot_type', action='append', nargs='+', help="Params to be ignored", default=[])
    parser.add_argument('--debug', help='Enable debug', action='store_true', default=False)

    parser.add_argument('--alpha', type=float, help='Transparency of plot bars (useless here)', default=0.5)
    parser.add_argument('--no_legend', help='Disables legend', action='store_true', default=False)
    parser.add_argument('--bins', type=int, help='Number of bins for distribution of results (useless here)', default=10)
    parser.add_argument('--dpi', type=int, help='Set DPI', default=300)
    parser.add_argument('--no_plt_show', help='Disable showing the plot', action='store_true', default=False)

    args = parser.parse_args()

    if args.bubblesize:
        global BUBBLESIZEINPX
        BUBBLESIZEINPX = args.bubblesize

    check_args()

    return args

def get_csv_file_path():
    global args
    print_debug("get_csv_file_path")
    pd_csv = "results.csv"
    csv_file_path = os.path.join(args.run_dir, pd_csv)
    check_dir_and_csv(csv_file_path)

    return csv_file_path

def flatten_extend(matrix):
    flat_list = []
    for row in matrix:
        flat_list.extend(row)
    return flat_list

def get_df_filtered(args, df):
    print_debug("get_df_filtered")
    all_columns_to_remove = ['trial_index', 'arm_name', 'trial_status', 'generation_method']
    columns_to_remove = []
    existing_columns = df.columns.values.tolist()

    for col in existing_columns:
        if col in all_columns_to_remove:
            columns_to_remove.append(col)

    if len(args.allow_axes):
        for col in existing_columns:
            if col != "result" and col not in flatten_extend(args.allow_axes):
                columns_to_remove.append(col)

    df_filtered = df.drop(columns=columns_to_remove)

    return df_filtered

def get_colors(df):
    print_debug("get_colors")
    colors = None
    try:
        colors = df["result"]
    except KeyError as e:
        if str(e) == "'result'":
            print(f"Could not find any results in {csv_file_path}")
            sys.exit(3)
        else:
            print(f"Key-Error: {e}")
            sys.exit(8)

    return colors

def check_path():
    global args
    print_debug("check_path")
    if not os.path.exists(args.run_dir):
        print(f'The folder {args.run_dir} does not exist.')
        sys.exit(1)

def get_non_empty_graphs(parameter_combinations, df_filtered, _exit):
    print_debug("get_non_empty_graphs")
    non_empty_graphs = []

    if len(parameter_combinations[0]) == 1:
        param = parameter_combinations[0][0]
        if param in df_filtered and df_filtered[param].notna().any():
            non_empty_graphs = [(param,)]
    else:
        if len(parameter_combinations) > 1 or type(parameter_combinations[0]) == tuple:
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

def get_r (df_filtered):
    print_debug("get_r")
    r = 2

    if len(list(df_filtered.columns)) == 1:
        r = 1

    return r

def get_parameter_combinations (df_filtered):
    print_debug("get_parameter_combinations")
    r = get_r(df_filtered)

    df_filtered_cols = df_filtered.columns.tolist()

    del df_filtered_cols[df_filtered_cols.index("result")]

    parameter_combinations = list(combinations(df_filtered_cols, r))

    if len(parameter_combinations) == 0:
        parameter_combinations = [*df_filtered_cols]

    return parameter_combinations

def use_matplotlib():
    global args
    print_debug("use_matplotlib")
    try:
        if not args.save_to_file:
            matplotlib.use('TkAgg')
    except Exception as e:
        print("An error occured while loading TkAgg. This may happen when you forgot to add -X to your ssh-connection")
        sys.exit(33)

def get_result_column_values(df):
    print_debug("get_result_column_values")
    result_column_values = df["result"]

    check_if_results_are_empty(result_column_values)

    return result_column_values

def create_new_figure_from_axes(ax):
    # Extrahiere Daten aus dem vorhandenen Axes-Element
    lines = ax.get_lines()
    x_data = []
    y_data = []

    for line in lines:
        x_data.append(line.get_xdata())
        y_data.append(line.get_ydata())

    # Erstelle eine neue Figur und Achsen
    new_fig, new_ax = plt.subplots()

    # Plotte die extrahierten Daten in der neuen Figur
    for x, y in zip(x_data, y_data):
        new_ax.plot(x, y)

    # Kopiere Achsentitel und Labels
    new_ax.set_xlabel(ax.get_xlabel())
    new_ax.set_ylabel(ax.get_ylabel())
    new_ax.set_title(ax.get_title())

    return new_fig, new_ax

def main():
    global args
    use_matplotlib()

    csv_file_path = get_csv_file_path()

    df = get_data(csv_file_path, args.min, args.max)

    old_headers_string = ','.join(sorted(df.columns))

    if len(args.merge_with_previous_runs):
        for prev_run in args.merge_with_previous_runs:
            prev_run_csv_path = prev_run[0] + "/results.csv"
            prev_run_df = get_data(prev_run_csv_path, result_column, args.min, args.max, old_headers_string)
            if prev_run_df is not None:
                print(f"Loading {prev_run_csv_path} into the dataset")
                df = df.merge(prev_run_df, how='outer')

    nr_of_items_before_filtering = len(df)
    df_filtered = get_df_filtered(args, df)

    check_min_and_max(len(df_filtered), nr_of_items_before_filtering, csv_file_path, args.min, args.max)

    parameter_combinations = get_parameter_combinations(df_filtered)

    non_empty_graphs = get_non_empty_graphs(parameter_combinations, df_filtered, True)

    num_subplots = len(non_empty_graphs)

    num_cols = math.ceil(math.sqrt(num_subplots))
    num_rows = math.ceil(num_subplots / num_cols)

    global fig

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15*num_cols, 7*num_rows))

    plot_graphs(df, fig, axs, df_filtered, non_empty_graphs, num_subplots, parameter_combinations, num_rows, num_cols)

    result_column_values = get_result_column_values(df)

    set_title(fig, df_filtered, result_column_values, len(df_filtered), args.min, args.max)

    set_margins(fig)

    fig.canvas.manager.set_window_title("Scatter: " + str(args.run_dir))
    if args.save_to_file:
        fig.set_size_inches(15.5, 9.5)

        _path = os.path.dirname(args.save_to_file)
        if _path:
            os.makedirs(_path, exist_ok=True)
        plt.savefig(args.save_to_file, dpi=args.dpi)
    else:
        create_widgets()

        if not args.no_plt_show:
            plt.show()

        update_graph(args.min, args.max)

def convert_string_to_number(input_string):
    print_debug("convert_string_to_number")
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

        import traceback
        tb = traceback.format_exc()
        print(tb)

        return None


# Define update function for the button
def update_graph(event=None, _min=None, _max=None):
    print_debug("update_graph")
    global fig, ax, button, maximum_textbox, minimum_textbox, args

    try:
        if minimum_textbox and looks_like_float(minimum_textbox.text):
            _min = convert_string_to_number(minimum_textbox.text)

        if maximum_textbox and looks_like_float(maximum_textbox.text):
            _max = convert_string_to_number(maximum_textbox.text)

        print_debug(f"update_graph: _min = {_min}, _max = {_max}")

        result_column = "result"
        csv_file_path = get_csv_file_path()
        df = get_data(csv_file_path, _min, _max)

        old_headers_string = ','.join(sorted(df.columns))

        # Redo previous run merges if needed
        if len(args.merge_with_previous_runs):
            for prev_run in args.merge_with_previous_runs:
                prev_run_csv_path = prev_run[0] + "/results.csv"
                prev_run_df = get_data(prev_run_csv_path, _min, _max, old_headers_string)
                if prev_run_df:
                    df = df.merge(prev_run_df, how='outer')

        nr_of_items_before_filtering = len(df)
        df_filtered = get_df_filtered(args, df)

        check_min_and_max(len(df_filtered), nr_of_items_before_filtering, csv_file_path, _min, _max, False)

        parameter_combinations = get_parameter_combinations(df_filtered)
        non_empty_graphs = get_non_empty_graphs(parameter_combinations, df_filtered, False)

        num_subplots = len(non_empty_graphs)
        num_cols = math.ceil(math.sqrt(num_subplots))
        num_rows = math.ceil(num_subplots / num_cols)

        # Clear the figure, but keep the widgets
        for widget in fig.axes:
            if widget not in [button.ax, maximum_textbox.ax, minimum_textbox.ax]:
                widget.remove()

        axs = fig.subplots(num_rows, num_cols)  # Create new subplots

        plot_graphs(df, fig, axs, df_filtered, non_empty_graphs, num_subplots, parameter_combinations, num_rows, num_cols)
        
        result_column_values = get_result_column_values(df)
        set_title(fig, df_filtered, result_column_values, len(df_filtered), _min, _max)

        plt.draw()
    except Exception as e:
        if not "invalid command name" in str(e):
            print(f"Failed to update graph: {e}")

def looks_like_float(x):
    print_debug(f"looks_like_float(x = {x})")
    if isinstance(x, (int, float)):
        return True  # int and float types are directly considered as floats
    elif isinstance(x, str):
        try:
            float(x)  # Try converting string to float
            return True
        except ValueError:
            return False  # If conversion fails, it's not a float-like string
    return False  # If x is neither str, int, nor float, it's not float-like


def change_min_max(expression):
    print_debug("change_min_max")
    global args

    try:
        has_params = False
        # Assuming the expression is a filter value update for min/max
        if textbox_maximum.text and  looks_like_float(textbox_maximum.text):
            args.min = float(textbox_maximum.text)
            print(f"set arg min to {args.min}")
            has_params = True
        if textbox_minimum.text and looks_like_float(textbox_minimum.text):
            args.min = float(textbox_minimum.text)
            print(f"set arg max to {args.max}")
            has_params = True
        if has_params:
            args.min = None
            args.max = None
        
        update_graph(None)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(tb)

        print(f"Failed to update graph with expression '{expression}': {e}")

def create_widgets():
    print_debug("create_widgets()")
    global button, maximum_textbox, minimum_textbox, args

    # Create a Button and set its position
    button_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(button_ax, 'Update Graph')
    button.on_clicked(update_graph)

    # Create TextBoxes and set their positions
    max_string = ""
    min_string = ""

    if looks_like_float(args.max):
        max_string = str(args.max)

    if looks_like_float(args.min):
        min_string = str(args.min)

    textbox_minimum = plt.axes([0.2, 0.025, 0.1, 0.04])
    minimum_textbox = TextBox(textbox_minimum, 'Minimum result:', initial=min_string)

    textbox_maximum = plt.axes([0.5, 0.025, 0.1, 0.04])
    maximum_textbox = TextBox(textbox_maximum, 'Maximum result:', initial=max_string)
     
if __name__ == "__main__":
    try:
        get_args()

        theme = "fast"

        if args.darkmode:
            theme = "dark_background"

        with plt.style.context(theme):
            main()
    except KeyboardInterrupt as e:
        sys.exit(0)
