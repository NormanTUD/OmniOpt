# DESCRIPTION: Scatter plot
# EXPECTED FILES: results.csv
# TEST_OUTPUT_MUST_CONTAIN: Number of evaluations shown
# TEST_OUTPUT_MUST_CONTAIN: mean result
# TEST_OUTPUT_MUST_CONTAIN: result

# TODO: Check if this script is able to react properly to --maximize'd runs

import argparse
import importlib.util
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
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
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
    print_debug("set_title")

    title = helpers.get_title(args, result_column_values, df_filtered, num_entries, _min, _max)

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

    helpers.show_legend(args, fig, scatter, axs)

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

    axs = helpers.hide_empty_plots(parameter_combinations, num_rows, num_cols, axs)

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
    global args, fig
    use_matplotlib()

    csv_file_path = helpers.get_csv_file_path(args)

    df = helpers.get_data(NO_RESULT, csv_file_path, args.min, args.max)

    old_headers_string = ','.join(sorted(df.columns))

    if len(args.merge_with_previous_runs):
        for prev_run in args.merge_with_previous_runs:
            prev_run_csv_path = prev_run[0] + "/results.csv"
            prev_run_df = helpers.get_data(NO_RESULT, prev_run_csv_path, args.min, args.max, old_headers_string)
            if prev_run_df is not None:
                print(f"Loading {prev_run_csv_path} into the dataset")
                df = df.merge(prev_run_df, how='outer')

    nr_of_items_before_filtering = len(df)
    df_filtered = helpers.get_df_filtered(args, df)

    helpers.check_min_and_max(len(df_filtered), nr_of_items_before_filtering, csv_file_path, args.min, args.max)

    parameter_combinations = helpers.get_parameter_combinations(df_filtered)

    non_empty_graphs = helpers.get_non_empty_graphs(parameter_combinations, df_filtered, True)

    num_subplots, num_cols, num_rows = helpers.get_num_subplots_rows_and_cols(non_empty_graphs)

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15 * num_cols, 7 * num_rows))

    plot_graphs([df, axs, df_filtered, non_empty_graphs, num_subplots, parameter_combinations, num_rows, num_cols])

    result_column_values = helpers.get_result_column_values(df, csv_file_path)

    set_title(df_filtered, result_column_values, len(df_filtered), args.min, args.max)

    set_margins()

    fig.canvas.manager.set_window_title("Scatter: " + str(args.run_dir))
    if args.save_to_file:
        fig.set_size_inches(15.5, 9.5)

        _path = os.path.dirname(args.save_to_file)
        if _path:
            os.makedirs(_path, exist_ok=True)
        try:
            plt.savefig(args.save_to_file)
        except OSError as e:
            print(f"Error: {e}. This may happen on unstable file systems or in docker containers.")
            sys.exit(199)

    else:
        global button, MAXIMUM_TEXTBOX, MINIMUM_TEXTBOX, TEXTBOX_MINIMUM, TEXTBOX_MAXIMUM

        button, MAXIMUM_TEXTBOX, MINIMUM_TEXTBOX, TEXTBOX_MINIMUM, TEXTBOX_MAXIMUM = helpers.create_widgets([plt, button, MAXIMUM_TEXTBOX, MINIMUM_TEXTBOX, args, TEXTBOX_MINIMUM, TEXTBOX_MAXIMUM, update_graph])

        if not args.no_plt_show:
            plt.show()

        update_graph(args.min, args.max)

# Define update function for the button
def update_graph(event=None, _min=None, _max=None):
    if event: # only for fooling pylint...
        pass

    global fig, ax, button, MAXIMUM_TEXTBOX, MINIMUM_TEXTBOX, args

    try:
        _min, _max = helpers.set_min_max(MINIMUM_TEXTBOX, MAXIMUM_TEXTBOX, _min, _max)

        print_debug(f"update_graph: _min = {_min}, _max = {_max}")

        csv_file_path = helpers.get_csv_file_path(args)
        df = helpers.get_data(NO_RESULT, csv_file_path, _min, _max)

        old_headers_string = ','.join(sorted(df.columns))

        # Redo previous run merges if needed
        if len(args.merge_with_previous_runs):
            for prev_run in args.merge_with_previous_runs:
                prev_run_csv_path = prev_run[0] + "/results.csv"
                prev_run_df = helpers.get_data(NO_RESULT, prev_run_csv_path, _min, _max, old_headers_string)
                if prev_run_df:
                    df = df.merge(prev_run_df, how='outer')

        nr_of_items_before_filtering = len(df)
        df_filtered = helpers.get_df_filtered(args, df)

        helpers.check_min_and_max(len(df_filtered), nr_of_items_before_filtering, csv_file_path, _min, _max, False)

        parameter_combinations = helpers.get_parameter_combinations(df_filtered)
        non_empty_graphs = helpers.get_non_empty_graphs(parameter_combinations, df_filtered, False)

        num_subplots, num_cols, num_rows = helpers.get_num_subplots_rows_and_cols(non_empty_graphs)

        helpers.remove_widgets(fig, button, MAXIMUM_TEXTBOX, MINIMUM_TEXTBOX)

        axs = fig.subplots(num_rows, num_cols)  # Create new subplots

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
