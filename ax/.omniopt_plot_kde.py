# DESCRIPTION: Kernel-Density estimation plot
# EXPECTED FILES: results.csv
# TEST_OUTPUT_MUST_CONTAIN: Histogram for
# TEST_OUTPUT_MUST_CONTAIN: Count

import argparse
import importlib.util
import logging
import math
import os
import sys
import traceback
import warnings
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fig = None
args = None

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

parser = argparse.ArgumentParser(description='Plotting tool for analyzing trial data.')
parser.add_argument('--run_dir', type=str, help='Path to a run dir', required=True)
parser.add_argument('--bins', type=int, help='Number of bins for distribution of results', default=10)
parser.add_argument('--alpha', type=float, help='Transparency of plot bars (between 0 and 1)', default=0.5)
parser.add_argument('--no_legend', help='Disables legend', action='store_true', default=False)
parser.add_argument('--save_to_file', type=str, help='Save the plot to the specified file', default=None)
parser.add_argument('--no_plt_show', help='Disable showing the plot', action='store_true', default=False)
args = parser.parse_args()

def get_num_rows_cols(num_plots: int, num_rows: int, num_cols: int) -> Tuple[int, int]:
    if num_plots > 1:
        num_rows = int(num_plots ** 0.5)
        num_cols = int(math.ceil(num_plots / num_rows))

    return num_rows, num_cols

def plot_histograms(dataframe: pd.DataFrame) -> None:
    exclude_columns = ['trial_index', 'arm_name', 'trial_status', 'generation_method', 'result']
    numeric_columns = [col for col in dataframe.select_dtypes(include=['float64', 'int64']).columns if col not in exclude_columns]

    num_plots = len(numeric_columns)
    num_rows = 1
    num_cols = num_plots

    num_rows, num_cols = get_num_rows_cols(num_plots, num_rows, num_cols)

    if num_rows == 0 or num_cols == 0:
        if not os.environ.get("NO_NO_RESULT_ERROR"): # pragma: no cover
            print(f"Num rows ({num_rows}) or num cols ({num_cols}) is 0. Cannot plot an empty graph.")
        sys.exit(42)

    global fig

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))
    try:
        axes = axes.flatten()
    except Exception as e:
        if "'Axes' object has no attribute 'flatten'" not in str(e): # pragma: no cover
            print(e)
            tb = traceback.format_exc()
            print(tb)

            sys.exit(145)

    for i, col in enumerate(numeric_columns):
        try:
            ax = axes[i]
        except TypeError:
            ax = axes

        values = dataframe[col]
        if "result" not in dataframe:
            if not os.environ.get("NO_NO_RESULT_ERROR"): # pragma: no cover
                print("KDE: Result column not found in dataframe. That may mean that the job had no valid runs")
            sys.exit(169)
        result_values = dataframe['result']
        if args is not None:
            bin_edges = np.linspace(result_values.min(), result_values.max(), args.bins + 1)  # Divide the range into 10 equal bins
            colormap = plt.get_cmap('RdYlGn_r')

            for j in range(args.bins):
                color = colormap(j / 9)  # Calculate color based on colormap
                bin_mask = (result_values >= bin_edges[j]) & (result_values <= bin_edges[j + 1])
                bin_range = f'{bin_edges[j]:.2f}-{bin_edges[j + 1]:.2f}'
                ax.hist(values[bin_mask], bins=args.bins, alpha=args.alpha, color=color, label=f'{bin_range}')

            ax.set_title(f'Histogram for {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Count')
            if not args.no_legend:
                ax.legend(loc='upper right')

    # Hide any unused subplots

    nr_axes = 1

    try:
        nr_axes = len(axes)
    except Exception:
        pass

    for j in range(num_plots, nr_axes):
        axes[j].axis('off')

    warnings.filterwarnings("ignore")

    plt.tight_layout()
    save_to_file_or_show_canvas()

def save_to_file_or_show_canvas() -> None:
    if args is not None:
        if args.save_to_file:
            helpers.save_to_file(fig, args, plt)
        else: # pragma: no cover
            if fig is not None and fig.canvas is not None and fig.canvas.manager is not None:
                fig.canvas.manager.set_window_title("KDE: " + str(args.run_dir))
                if not args.no_plt_show:
                    plt.show()

def update_graph() -> None:
    if args is not None:
        pd_csv = args.run_dir + "/results.csv"

        try:
            dataframe = None

            try:
                dataframe = pd.read_csv(pd_csv)
            except pd.errors.EmptyDataError:
                if not os.environ.get("PLOT_TESTS"): # pragma: no cover
                    print(f"{pd_csv} seems to be empty.")
                sys.exit(19)
            except UnicodeDecodeError:
                if not os.environ.get("PLOT_TESTS"): # pragma: no cover
                    print(f"{args.run_dir}/results.csv seems to be invalid utf8.")
                sys.exit(7)

            plot_histograms(dataframe)
        except FileNotFoundError: # pragma: no cover
            logging.error("File not found: %s", pd_csv)
        except Exception as exception: # pragma: no cover
            logging.error("An unexpected error occurred: %s", str(exception))

            tb = traceback.format_exc()
            print(tb)

if __name__ == "__main__":
    helpers.setup_logging()

    if not args.alpha: # pragma: no cover
        logging.error("--alpha cannot be left unset.")
        sys.exit(2)

    if args.alpha > 1 or args.alpha < 0: # pragma: no cover
        logging.error("--alpha must between 0 and 1")
        sys.exit(3)

    if not os.path.exists(args.run_dir): # pragma: no cover
        logging.error("Specified --run_dir does not exist")
        sys.exit(1)

    update_graph()
