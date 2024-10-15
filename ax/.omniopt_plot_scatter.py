# DESCRIPTION: Scatter plot
# EXPECTED FILES: results.csv
# TEST_OUTPUT_MUST_CONTAIN: Number of evaluations shown
# TEST_OUTPUT_MUST_CONTAIN: mean result
# TEST_OUTPUT_MUST_CONTAIN: result

# TODO: Check if this script is able to react properly to --maximize'd runs

import argparse
import importlib.util
import math
import os
import signal
import sys
import traceback

#from rich.traceback import install
from rich.pretty import pprint

#install(show_locals=True)

button = None

TEXTBOX_MINIMUM = None
TEXTBOX_MAXIMUM = None

MAXIMUM_TEXTBOX = None
MINIMUM_TEXTBOX = None

script_dir = os.path.dirname(os.path.realpath(__file__))
helpers_file = f"{script_dir}/.helpers.py"
spec = importlib.util.spec_from_file_location(
    name="helpers",
    location=helpers_file,
)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)

val_if_nothing_found = 99999999999999999999999999999999999999999999999999999999999
NO_RESULT = "{:.0e}".format(val_if_nothing_found)

args = None

def print_debug(msg):
    if args.debug:
        print("DEBUG: ", end="")
        pprint(msg)

fig = None
MAXIMUM_TEXTBOX = None
MINIMUM_TEXTBOX = None

signal.signal(signal.SIGINT, signal.SIG_DFL)

try:
    from itertools import combinations

    import matplotlib
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.widgets import Button, TextBox
except ModuleNotFoundError as ee:
    print(f"Error: {ee}")
    sys.exit(244)

# Get shell variables or use default values
BUBBLESIZEINPX = int(os.environ.get('BUBBLESIZEINPX', 15))
ORIGINAL_PWD = os.environ.get("ORIGINAL_PWD", "")

if ORIGINAL_PWD:
    os.chdir(ORIGINAL_PWD)

def set_margins():
    print_debug("set_margins()")
    left = 0.04
    if args.save_to_file and len(args.allow_axes):
        left = 0.1
    right = 0.864
    bottom = 0.171
    top = 0.9
    wspace = 0.27
    hspace = 0.31

    fig.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

def set_title(df_filtered, result_column_values, num_entries, _min, _max):
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

    title_values = []

    for _l in extreme_values_items:
        if "result" not in _l:
            key = _l[0]
            value = helpers.to_int_when_possible(_l[1])
            title_values.append(f"{key} = {value}")

    title += " of f("
    title += ', '.join(title_values)
    title += f") = {helpers.to_int_when_possible(result_column_values[extreme_index])}"

    title += f"\nNumber of evaluations shown: {num_entries}"

    if _min is not None:
        title += f", show min = {helpers.to_int_when_possible(_min)}"

    if _max is not None:
        title += f", show max = {helpers.to_int_when_possible(_max)}"

    if _mean is not None:
        title += f", mean result = {helpers.to_int_when_possible(_mean)}"

    fig.suptitle(title)

def check_args():
    print_debug("check_args()")
    global args

    if args.min and args.max:
        if args.min > args.max:
            tmp = args.max
            args.max = args.min
            args.min = tmp
        elif args.min == args.max:
            print("Max and min value are the same. May result in empty data")

    helpers.check_path(args.run_dir)

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

def get_data(csv_file_path, _min, _max, old_headers_string=None):
    print_debug("get_data")
    try:
        df = pd.read_csv(csv_file_path, index_col=0)

        if old_headers_string:
            df_header_string = ','.join(sorted(df.columns))
            if df_header_string != old_headers_string:
                print(f"Cannot merge {csv_file_path}. Old headers: {old_headers_string}, new headers {df_header_string}")
                return None

        try:
            if _min is not None:
                df = df[df["result"] >= _min]
            if _max is not None:
                df = df[df["result"] <= _max]
        except KeyError:
            if not os.environ.get("NO_NO_RESULT_ERROR"):
                print(f"There was no 'result' in {csv_file_path}. This may means all tests failed. Cannot continue.")
            sys.exit(10)

        if "result" not in df:
            if not os.environ.get("NO_NO_RESULT_ERROR"):
                print(f"There was no 'result' in {csv_file_path}. This may means all tests failed. Cannot continue.")
            sys.exit(10)
        df.dropna(subset=["result"], inplace=True)
    except pd.errors.EmptyDataError:
        if not os.environ.get("PLOT_TESTS"):
            print(f"{csv_file_path} has no lines to parse.")
        sys.exit(19)
    except pd.errors.ParserError as e:
        if not os.environ.get("PLOT_TESTS"):
            print(f"{csv_file_path} is invalid CSV. Parsing error: {str(e).rstrip()}")
        sys.exit(12)
    except UnicodeDecodeError:
        if not os.environ.get("PLOT_TESTS"):
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
    for i in range(len(parameter_combinations), num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        axs[row, col].set_visible(False)

def plot_multiple_graphs(_params):
    non_empty_graphs, num_cols, axs, df_filtered, colors, cmap, norm, parameter_combinations, num_rows = _params

    scatter = None

    print_debug("plot_multiple_graphs")
    for i, (param1, param2) in enumerate(non_empty_graphs):
        row = i // num_cols
        col = i % num_cols
        if (len(args.exclude_params) and param1 not in args.exclude_params[0] and param2 not in args.exclude_params[0]) or len(args.exclude_params) == 0:
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

                    tb = traceback.format_exc()
                    print(tb)

                    sys.exit(17)

    for i in range(len(parameter_combinations), num_rows * num_cols):
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

def plot_single_graph(_params):
    axs, df_filtered, colors, cmap, norm, non_empty_graphs = _params
    print_debug("plot_single_graph()")
    _data = df_filtered

    _data = _data[:].values

    _x = []
    _y = []

    for _l in _data:
        _x.append(_l[0])
        _y.append(_l[1])

    scatter = axs.scatter(_x, _y, c=colors, cmap=cmap, norm=norm, s=BUBBLESIZEINPX)
    axs.set_xlabel(non_empty_graphs[0][0])
    axs.set_ylabel("result")

    return scatter

def get_colors(df):
    print_debug("get_colors")
    colors = None
    try:
        colors = df["result"]
    except KeyError as e:
        if str(e) == "'result'":
            print("Could not find any results")
            sys.exit(3)
        else:
            print(f"Key-Error: {e}")
            sys.exit(8)

    return colors

def plot_graphs(params):
    print_debug("plot_graphs")

    df, axs, df_filtered, non_empty_graphs, num_subplots, parameter_combinations, num_rows, num_cols = params

    colors = get_colors(df)

    if colors is None:
        print_debug("colors is None. Cannot plot.")
        sys.exit(3)

    if os.path.exists(args.run_dir + "/maximize"):
        colors = -1 * colors  # Negate colors for maximum result

    norm = None
    try:
        norm = plt.Normalize(colors.min(), colors.max())
    except Exception:
        print("Wrong values in csv file")
        sys.exit(16)

    c = ["darkred", "red", "lightcoral", "palegreen", "green", "darkgreen"]
    c = c[::-1]
    v = [0, 0.3, 0.5, 0.7, 0.9, 1]
    _l = list(zip(v, c))

    cmap = LinearSegmentedColormap.from_list('rg', _l, N=256)

    if num_subplots == 1 and (type(non_empty_graphs[0]) is str or len(non_empty_graphs[0]) == 1):
        plot_single_graph([axs, df_filtered, colors, cmap, norm, non_empty_graphs])
    else:
        plot_multiple_graphs([non_empty_graphs, num_cols, axs, df_filtered, colors, cmap, norm, parameter_combinations, num_rows])

    hide_empty_plots(parameter_combinations, num_rows, num_cols, axs)

def get_args():
    global args
    parser = argparse.ArgumentParser(description='Plot optimization runs.', prog="plot")

    parser.add_argument('--run_dir', type=str, help='Path to a CSV file', required=True)
    parser.add_argument('--save_to_file', type=str, help='Save the plot to the specified file', default=None)
    parser.add_argument('--max', type=float, help='Maximum value', default=None)
    parser.add_argument('--min', type=float, help='Minimum value', default=None)
    parser.add_argument('--darkmode', help='Enable darktheme', action='store_true', default=False)
    parser.add_argument('--bubblesize', type=int, help='Size of the bubbles', default=15)
    parser.add_argument('--merge_with_previous_runs', action='append', nargs='+', help="Run-Dirs to be merged with", default=[])
    parser.add_argument('--exclude_params', action='append', nargs='+', help="Params to be ignored", default=[])

    parser.add_argument('--allow_axes', action='append', nargs='+', help="Allow specific axes only (parameter names)", default=[])
    parser.add_argument('--debug', help='Enable debug', action='store_true', default=False)

    parser.add_argument('--no_legend', help='Disables legend', action='store_true', default=False)
    parser.add_argument('--no_plt_show', help='Disable showing the plot', action='store_true', default=False)

    args = parser.parse_args()

    if args.bubblesize:
        global BUBBLESIZEINPX
        BUBBLESIZEINPX = args.bubblesize

    check_args()

    return args

def get_df_filtered(df):
    print_debug("get_df_filtered")
    all_columns_to_remove = ['trial_index', 'arm_name', 'trial_status', 'generation_method']
    columns_to_remove = []
    existing_columns = df.columns.values.tolist()

    for col in existing_columns:
        if col in all_columns_to_remove:
            columns_to_remove.append(col)

    if len(args.allow_axes):
        for col in existing_columns:
            if col != "result" and col not in helpers.flatten_extend(args.allow_axes):
                columns_to_remove.append(col)

    df_filtered = df.drop(columns=columns_to_remove)

    return df_filtered

def get_non_empty_graphs(parameter_combinations, df_filtered, _exit):
    print_debug("get_non_empty_graphs")
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

    helpers.die_if_no_nonempty_graph(non_empty_graphs, _exit)

    return non_empty_graphs

def get_parameter_combinations(df_filtered):
    print_debug("get_parameter_combinations")
    r = helpers.get_r(df_filtered)

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
    except Exception:
        print("An error occurred while loading TkAgg. This may happen when you forgot to add -X to your ssh-connection")
        sys.exit(33)

def main():
    global button, MAXIMUM_TEXTBOX, MINIMUM_TEXTBOX, TEXTBOX_MINIMUM, TEXTBOX_MAXIMUM, args, fig
    use_matplotlib()

    csv_file_path = helpers.get_csv_file_path(args)

    df = get_data(csv_file_path, args.min, args.max)

    old_headers_string = ','.join(sorted(df.columns))

    if len(args.merge_with_previous_runs):
        for prev_run in args.merge_with_previous_runs:
            prev_run_csv_path = prev_run[0] + "/results.csv"
            prev_run_df = get_data(prev_run_csv_path, args.min, args.max, old_headers_string)
            if prev_run_df is not None:
                print(f"Loading {prev_run_csv_path} into the dataset")
                df = df.merge(prev_run_df, how='outer')

    nr_of_items_before_filtering = len(df)
    df_filtered = get_df_filtered(df)

    check_min_and_max(len(df_filtered), nr_of_items_before_filtering, csv_file_path, args.min, args.max)

    parameter_combinations = get_parameter_combinations(df_filtered)

    non_empty_graphs = get_non_empty_graphs(parameter_combinations, df_filtered, True)

    num_subplots = len(non_empty_graphs)

    num_cols = math.ceil(math.sqrt(num_subplots))
    num_rows = math.ceil(num_subplots / num_cols)

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15 * num_cols, 7 * num_rows))

    plot_graphs([df, axs, df_filtered, non_empty_graphs, num_subplots, parameter_combinations, num_rows, num_cols])

    result_column_values = helpers.get_result_column_values(df, csv_file_path)

    set_title(df_filtered, result_column_values, len(df_filtered), args.min, args.max)

    set_margins()

    fig.canvas.manager.set_window_title("Scatter: " + str(args.run_dir))

    if args.save_to_file:
        helpers.save_to_file(fig, args, plt)
    else:
        button, MAXIMUM_TEXTBOX, MINIMUM_TEXTBOX, TEXTBOX_MINIMUM, TEXTBOX_MAXIMUM = helpers.create_widgets([plt, button, MAXIMUM_TEXTBOX, MINIMUM_TEXTBOX, args, TEXTBOX_MINIMUM, TEXTBOX_MAXIMUM, Button, update_graph])

        if not args.no_plt_show:
            plt.show()

        update_graph(args.min, args.max)

# Define update function for the button
def update_graph(event=None, _min=None, _max=None):
    if event: # only for fooling pylint...
        pass

    global fig, ax, button, MAXIMUM_TEXTBOX, MINIMUM_TEXTBOX, args

    try:
        axs, df_filtered, num_rows, num_cols = prepare_graph_update(MINIMUM_TEXTBOX, MAXIMUM_TEXTBOX, _args, get_df_filtered, check_min_and_max, get_parameter_combinations, get_non_empty_graphs, button, check_dir_and_csv)

        plot_graphs([df, fig, axs, df_filtered, non_empty_graphs, num_subplots, parameter_combinations, num_rows, num_cols])

        result_column_values = helpers.get_result_column_values(df, csv_file_path)
        set_title(df_filtered, result_column_values, len(df_filtered), _min, _max)

        plt.draw()
    except Exception as e:
        if "invalid command name" not in str(e):
            print(f"Failed to update graph: {e}")

if __name__ == "__main__":
    try:
        get_args()

        theme = "fast"

        if args.darkmode:
            theme = "dark_background"

        with plt.style.context(theme):
            main()
    except KeyboardInterrupt:
        sys.exit(0)
