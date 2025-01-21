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
from typing import Any, Union
import pandas as pd
from typeguard import typechecked

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
if spec is not None and spec.loader is not None:
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
else: # pragma: no cover
    raise ImportError(f"Could not load module from {helpers_file}")

val_if_nothing_found = 99999999999999999999999999999999999999999999999999999999999
NO_RESULT = "{:.0e}".format(val_if_nothing_found)

args = None
fig = None
MAXIMUM_TEXTBOX = None
MINIMUM_TEXTBOX = None

signal.signal(signal.SIGINT, signal.SIG_DFL)

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

parser.add_argument('--no_legend', help='Disables legend', action='store_true', default=False)
parser.add_argument('--no_plt_show', help='Disable showing the plot', action='store_true', default=False)

args = parser.parse_args()

if args.bubblesize:
    BUBBLESIZEINPX = args.bubblesize

helpers.check_args(args)

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as ee: # pragma: no cover
    print(f"Error: {ee}")
    sys.exit(244)

# Get shell variables or use default values
BUBBLESIZEINPX = int(os.environ.get('BUBBLESIZEINPX', 15))
ORIGINAL_PWD = os.environ.get("ORIGINAL_PWD", "")

if ORIGINAL_PWD:
    os.chdir(ORIGINAL_PWD)

@typechecked
def set_title(df_filtered: pd.DataFrame, result_column_values: pd.core.series.Series, num_entries: int, _min: Union[int, float, None], _max: Union[int, float, None]) -> None:
    title = helpers.get_title(args, result_column_values, df_filtered, num_entries, _min, _max)

    if fig:
        fig.suptitle(title)

@typechecked
def plot_multiple_graphs(_params: list) -> None:
    if args is not None:
        non_empty_graphs, num_cols, axs, df_filtered, colors, cmap, norm, parameter_combinations, num_rows = _params

        scatter = None

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
                    else: # pragma: no cover
                        print("ERROR: " + str(e))

                        tb = traceback.format_exc()
                        print(tb)

                        sys.exit(17)

        axs = helpers.hide_empty_plots(parameter_combinations, num_rows, num_cols, axs)

        helpers.show_legend(args, fig, scatter, axs)

@typechecked
def plot_single_graph(_params: list) -> Any:
    axs, df_filtered, colors, cmap, norm, non_empty_graphs = _params
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

@typechecked
def plot_graphs(_params: list) -> None:
    global fig
    df, fig, axs, df_filtered, non_empty_graphs, num_subplots, parameter_combinations, num_rows, num_cols, result_column_values = _params

    cmap, norm, colors = helpers.get_color_list(df, args, plt)

    if num_subplots == 1 and (type(non_empty_graphs[0]) is str or len(non_empty_graphs[0]) == 1):
        plot_single_graph([axs, df_filtered, colors, cmap, norm, non_empty_graphs])
    else:
        plot_multiple_graphs([non_empty_graphs, num_cols, axs, df_filtered, colors, cmap, norm, parameter_combinations, num_rows])

    axs = helpers.hide_empty_plots(parameter_combinations, num_rows, num_cols, axs)

@typechecked
def main() -> None:
    global args, fig

    if args is not None:
        if not helpers.can_be_plotted(args.run_dir):
            print(f"{args.run_dir} contains multiple RESULTS and thus can only be plotted by parallel plot")
            sys.exit(2)

        helpers.use_matplotlib(args)

        csv_file_path = helpers.get_csv_file_path(args)

        df = helpers.get_data(NO_RESULT, csv_file_path, args.min, args.max)

        old_headers_string = ','.join(sorted(df.columns))

        df = helpers.merge_df_with_old_data(args, df, NO_RESULT, args.min, args.max, old_headers_string)

        nr_of_items_before_filtering = len(df)
        df_filtered = helpers.get_df_filtered(args, df)

        helpers.check_min_and_max(len(df_filtered), nr_of_items_before_filtering, csv_file_path, args.min, args.max)

        parameter_combinations = helpers.get_parameter_combinations(df_filtered)

        non_empty_graphs = helpers.get_non_empty_graphs(parameter_combinations, df_filtered, True)

        num_subplots, num_cols, num_rows = helpers.get_num_subplots_rows_and_cols(non_empty_graphs)

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(15 * num_cols, 7 * num_rows))

        result_column_values = helpers.get_result_column_values(df, csv_file_path)

        plot_graphs([df, fig, axs, df_filtered, non_empty_graphs, num_subplots, parameter_combinations, num_rows, num_cols, result_column_values])

        set_title(df_filtered, result_column_values, len(df_filtered), args.min, args.max)

        helpers.set_margins(fig)

        fig.canvas.manager.set_window_title("Scatter: " + str(args.run_dir))

        if args.save_to_file:
            helpers.save_to_file(fig, args, plt)
        else: # pragma: no cover
            global button, MAXIMUM_TEXTBOX, MINIMUM_TEXTBOX, TEXTBOX_MINIMUM, TEXTBOX_MAXIMUM

            button, MAXIMUM_TEXTBOX, MINIMUM_TEXTBOX, TEXTBOX_MINIMUM, TEXTBOX_MAXIMUM = helpers.create_widgets([plt, button, MAXIMUM_TEXTBOX, MINIMUM_TEXTBOX, args, TEXTBOX_MINIMUM, TEXTBOX_MAXIMUM, update_graph])

            if not args.no_plt_show:
                plt.show()

            update_graph(args.min, args.max)

# Define update function for the button
def update_graph(event: Any = None, _min: Union[int, float, None] = None, _max: Union[int, float, None] = None) -> None: # pragma: no cover
    global fig, ax, button, MAXIMUM_TEXTBOX, MINIMUM_TEXTBOX, args

    if event: # only for fooling pylint...
        pass

    filter_out_strings = False
    helpers._update_graph([plt, fig, MINIMUM_TEXTBOX, MAXIMUM_TEXTBOX, _min, _max, args, NO_RESULT, filter_out_strings, set_title, plot_graphs, button])

if __name__ == "__main__":
    try:
        theme = "fast"

        if args is not None and args.darkmode: # pragma: no cover
            theme = "dark_background"

        with plt.style.context(theme):
            main()
    except KeyboardInterrupt: # pragma: no cover
        sys.exit(0)
