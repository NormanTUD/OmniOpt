# DESCRIPTION: Plot general job info
# EXPECTED FILES: results.csv
# TEST_OUTPUT_MUST_CONTAIN: generation_method

import argparse
import importlib.util
import logging
import os
import sys
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from beartype import beartype

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
parser.add_argument('--min', type=float, help='Minimum value for result filtering')
parser.add_argument('--max', type=float, help='Maximum value for result filtering')
parser.add_argument('--save_to_file', nargs='?', const='plot', type=str, help='Path to save the plot(s)')
parser.add_argument('--run_dir', type=str, help='Path to a CSV file', required=True)
parser.add_argument('--no_plt_show', help='Disable showing the plot', action='store_true', default=False)
args = parser.parse_args()

@beartype
def plot_graph(dataframe: pd.DataFrame, save_to_file: Union[None, str] = None) -> None:
    exclude_columns: list = ['trial_index', 'arm_name', 'trial_status', 'generation_method']

    numeric_columns: list[str] = [
        col for col in dataframe.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if col not in exclude_columns
    ]

    pair_plot = sns.pairplot(dataframe, hue='generation_method', vars=numeric_columns)
    pair_plot.fig.suptitle('Pair Plot of Numeric Variables by Generation Method', y=1.02)

    if save_to_file:
        helpers.save_to_file(pair_plot.fig, args, plt)
    else:
        if args is not None:
            if not args.no_plt_show:
                plt.show()

@beartype
def handle_empty_data(filepath: str) -> None:
    if not os.environ.get("NO_NO_RESULT_ERROR"):
        print(f"Could not find values in file {filepath}")
    sys.exit(19)

@beartype
def handle_unicode_error(filepath: str) -> None:
    if not os.environ.get("PLOT_TESTS"):
        print(f"{filepath} seems to be invalid utf8.")
    sys.exit(7)

@beartype
def update_graph() -> None:
    if args is None:
        return

    csv_path = os.path.join(args.run_dir, "results.csv")

    try:
        dataframe = pd.read_csv(csv_path)

        if args.min is not None or args.max is not None:
            dataframe = helpers.filter_data(args, dataframe, args.min, args.max)

        if dataframe.empty:
            if not os.environ.get("NO_NO_RESULT_ERROR"):
                print("DataFrame is empty after filtering.")
            return

        if args.save_to_file:
            ensure_directory_exists(args.save_to_file)

        plot_graph(dataframe, args.save_to_file)

    except FileNotFoundError:
        print(f"File not found: {csv_path}")
    except pd.errors.EmptyDataError:
        handle_empty_data(csv_path)
    except UnicodeDecodeError:
        handle_unicode_error(csv_path)
    except KeyError:
        if not os.environ.get("PLOT_TESTS"):
            print(f"{csv_path} seems to have no 'result' column.")
    except Exception as exception:
        print(f"An unexpected error occurred: {exception}")

@beartype
def ensure_directory_exists(file_path: str) -> None:
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

if __name__ == "__main__":
    helpers.setup_logging()

    if not os.path.exists(args.run_dir):
        logging.error("Specified --run_dir does not exist")
        sys.exit(1)

    helpers.die_if_cannot_be_plotted(args.run_dir)

    update_graph()
