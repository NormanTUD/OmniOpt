# DESCRIPTION: Hex-Scatter plot
# TITLE: Hex-Scatter plot
# FULL_DESCRIPTION: <p>Similar to scatter plot, but here many runs are grouped into hexagonal subspaces of the parameter combinations, and the groups are coloured by their average result, and as such you can see an approximation of the function space. This allows you to quickly grasp 'good' areas of your hyperparameter space.</p>
# EXPECTED FILES: results.csv
# TEST_OUTPUT_MUST_CONTAIN: Number of evaluations shown
# TEST_OUTPUT_MUST_CONTAIN: mean result
# TEST_OUTPUT_MUST_CONTAIN: result

# TODO: [SHK, 3/10] Check if this script is able to react properly to --maximize'd runs

import argparse
import importlib.util
import os
import signal
import sys
import traceback
from typing import Any, Union
import pandas as pd
from beartype import beartype

button = None

TEXTBOX_MINIMUM = None
TEXTBOX_MAXIMUM = None

MAXIMUM_TEXTBOX = None
MINIMUM_TEXTBOX = None

bins = None

res_col_name = None

script_dir = os.path.dirname(os.path.realpath(__file__))
helpers_file = f"{script_dir}/.helpers.py"
spec = importlib.util.spec_from_file_location(
    name="helpers",
    location=helpers_file,
)
if spec is not None and spec.loader is not None:
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
else:
    raise ImportError(f"Could not load module from {helpers_file}")

dier = helpers.dier

args = None
fig = None

signal.signal(signal.SIGINT, signal.SIG_DFL)

parser = argparse.ArgumentParser(description='Plot optimization runs.', prog="plot")

parser.add_argument('--run_dir', type=str, help='Path to a CSV file', required=True)
parser.add_argument('--save_to_file', type=str, help='Save the plot to the specified file', default=None)
parser.add_argument('--max', type=float, help='Maximum value', default=None)
parser.add_argument('--min', type=float, help='Minimum value', default=None)
parser.add_argument('--merge_with_previous_runs', action='append', nargs='+', help="Run-Dirs to be merged with", default=[])
parser.add_argument('--exclude_params', action='append', nargs='+', help="Params to be ignored", default=[])

parser.add_argument('--allow_axes', action='append', nargs='+', help="Allow specific axes only (parameter names)", default=[])

parser.add_argument('--no_legend', help='Disables legend', action='store_true', default=False)
parser.add_argument('--bins', type=str, help='Number of bins for distribution of results', default=None)

parser.add_argument('--gridsize', type=int, help='Gridsize for hex plots', default=5)
parser.add_argument('--no_plt_show', help='Disable showing the plot', action='store_true', default=False)

args = parser.parse_args()

if args.bins:
    if not (args.bins == "log" or helpers.looks_like_int(args.bins)):
        print(f"Error: --bin must be 'log' or a number, or left out entirely. Is: {args.bins}")
        sys.exit(193)

    if helpers.looks_like_int(args.bins):
        bins = int(args.bins)
    else:
        bins = args.bins

helpers.check_args(args)

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as ee:
    print(f"Error: {ee}")
    sys.exit(244)

# Get shell variables or use default values
ORIGINAL_PWD = os.environ.get("ORIGINAL_PWD", "")

if ORIGINAL_PWD:
    os.chdir(ORIGINAL_PWD)

@beartype
def set_title(df_filtered: pd.DataFrame, result_column_values: pd.core.series.Series, num_entries: int, _min: Union[float, int, None] = None, _max: Union[float, int, None] = None) -> None:
    title = helpers.get_title(args, result_column_values, df_filtered, num_entries, _min, _max)

    if fig:
        fig.suptitle(title)

@beartype
def plot_multiple_graphs(_params: list) -> None:
    if args is not None:
        non_empty_graphs, num_cols, axs, df_filtered, cmap, norm, parameter_combinations, num_rows, result_column_values = _params

        scatter = None

        for i, (param1, param2) in enumerate(non_empty_graphs):
            row = i // num_cols
            col = i % num_cols
            if (args.exclude_params is not None and len(args.exclude_params) and param1 not in args.exclude_params[0] and param2 not in args.exclude_params[0]) or len(args.exclude_params) == 0:
                try:
                    _x = df_filtered[param1]
                    _y = df_filtered[param2]

                    gridsize: int = args.gridsize

                    if bins:
                        scatter = axs[row][col].hexbin(_x, _y, result_column_values, gridsize=gridsize, cmap=cmap, bins=bins)
                    else:
                        scatter = axs[row][col].hexbin(_x, _y, result_column_values, norm=norm, gridsize=gridsize, cmap=cmap)
                    axs[row][col].set_xlabel(param1)
                    axs[row][col].set_ylabel(param2)
                except Exception as e:
                    if "'Axes' object is not subscriptable" in str(e):
                        if bins:
                            scatter = axs.hexbin(_x, _y, result_column_values, gridsize=args.gridsize, cmap=cmap, bins=bins)
                        else:
                            scatter = axs.hexbin(_x, _y, result_column_values, norm=norm, gridsize=args.gridsize, cmap=cmap)
                        axs.set_xlabel(param1)
                        axs.set_ylabel(param2)
                    elif "could not convert string to float" in str(e):
                        print("ERROR: " + str(e))

                        tb = traceback.format_exc()
                        print(tb)

                        sys.exit(177)
                    else:
                        print("ERROR: " + str(e))

                        tb = traceback.format_exc()
                        print(tb)

                        sys.exit(17)

        axs = helpers.hide_empty_plots(parameter_combinations, num_rows, num_cols, axs)

        helpers.show_legend(args, fig, scatter, axs)

@beartype
def plot_single_graph(_params: list) -> None:
    if args is not None:
        axs, df_filtered, cmap, norm, non_empty_graphs, result_column_values = _params

        df_filtered = df_filtered.drop(res_col_name, axis=1)

        col_names = list(df_filtered.columns)

        if len(col_names) <= 1:
            print(f"Error: Cannot plot run folder with 1 or less column names. Has {len(col_names)}.")
            sys.exit(17)

        _data = df_filtered

        _data = _data[:].values

        _x = []
        _y = []

        for _l in _data:
            _x.append(_l[0])
            _y.append(_l[1])

        if bins:
            axs.hexbin(_x, _y, result_column_values, cmap=cmap, gridsize=args.gridsize, bins=bins)
        else:
            axs.hexbin(_x, _y, result_column_values, cmap=cmap, gridsize=args.gridsize, norm=norm)

        axs.set_xlabel(non_empty_graphs[0][0])
        axs.set_ylabel(non_empty_graphs[0][1])

@beartype
def plot_graphs(_params: list) -> None:
    global fig

    df, fig, axs, df_filtered, non_empty_graphs, num_subplots, parameter_combinations, num_rows, num_cols, result_column_values, csv_file_path = _params

    cmap, norm, colors = helpers.get_color_list(df, args, plt, csv_file_path)

    if colors is not None:
        pass # for fooling linter

    if num_subplots == 1:
        plot_single_graph([axs, df_filtered, cmap, norm, non_empty_graphs, result_column_values])
    else:
        plot_multiple_graphs([non_empty_graphs, num_cols, axs, df_filtered, cmap, norm, parameter_combinations, num_rows, result_column_values])

    axs = helpers.hide_empty_plots(parameter_combinations, num_rows, num_cols, axs)

@beartype
def main() -> None:
    global fig, res_col_name

    if args is not None:
        helpers.die_if_cannot_be_plotted(args.run_dir)

        helpers.use_matplotlib(args)

        csv_file_path = helpers.get_csv_file_path(args)

        df = helpers.get_data(csv_file_path, args.min, args.max, None, True)

        res_col_name = helpers.get_result_name_or_default_from_csv_file_path(args.run_dir + "/results.csv")

        old_headers_string = ','.join(sorted(df.columns))

        df = helpers.merge_df_with_old_data(args, df, args.min, args.max, old_headers_string)

        nr_of_items_before_filtering = len(df)
        df_filtered = helpers.get_df_filtered(args, df)

        helpers.check_min_and_max(len(df_filtered), nr_of_items_before_filtering, csv_file_path, args.min, args.max)

        parameter_combinations = helpers.get_parameter_combinations(df_filtered, csv_file_path)

        non_empty_graphs = helpers.get_non_empty_graphs(parameter_combinations, df_filtered, True)

        num_subplots, num_cols, num_rows = helpers.get_num_subplots_rows_and_cols(non_empty_graphs)

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(15 * num_cols, 7 * num_rows))

        result_column_values = helpers.get_result_column_values(df, csv_file_path)

        plot_graphs([df, fig, axs, df_filtered, non_empty_graphs, num_subplots, parameter_combinations, num_rows, num_cols, result_column_values, csv_file_path])

        if not args.no_legend:
            set_title(df_filtered, result_column_values, len(df_filtered), args.min, args.max)

            helpers.set_margins(fig)

            fig.canvas.manager.set_window_title("Hex-Scatter: " + str(args.run_dir))

        if args.save_to_file:
            helpers.save_to_file(fig, args, plt)
        else:
            global button, MAXIMUM_TEXTBOX, MINIMUM_TEXTBOX, TEXTBOX_MINIMUM, TEXTBOX_MAXIMUM

            button, MAXIMUM_TEXTBOX, MINIMUM_TEXTBOX, TEXTBOX_MINIMUM, TEXTBOX_MAXIMUM = helpers.create_widgets([plt, button, MAXIMUM_TEXTBOX, MINIMUM_TEXTBOX, args, TEXTBOX_MINIMUM, TEXTBOX_MAXIMUM, update_graph])

            if not args.no_plt_show:
                plt.show()

            update_graph(csv_file_path, None, args.min, args.max)

# Define update function for the button
def update_graph(csv_file_path: str, event: Any = None, _min: Union[int, float, None] = None, _max: Union[int, float, None] = None) -> None:
    if event:
        # only for fooling pylint...
        pass

    filter_out_strings = True
    helpers._update_graph([csv_file_path, plt, fig, MINIMUM_TEXTBOX, MAXIMUM_TEXTBOX, _min, _max, args, filter_out_strings, set_title, plot_graphs, button])

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
