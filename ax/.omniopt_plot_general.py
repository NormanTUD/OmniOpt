# DESCRIPTION: Plot general job info
# EXPECTED FILES: results.csv
# TEST_OUTPUT_MUST_CONTAIN: Sobol

import argparse
import importlib.util
import logging
import os
import signal
import sys
from typing import Any, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

args = None

signal.signal(signal.SIGINT, signal.SIG_DFL)

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

def parse_arguments() -> Any:
    parser = argparse.ArgumentParser(description='Plotting tool for analyzing trial data.')
    parser.add_argument('--min', type=float, help='Minimum value for result filtering')
    parser.add_argument('--max', type=float, help='Maximum value for result filtering')
    parser.add_argument('--save_to_file', nargs='?', const='plot', type=str, help='Path to save the plot(s)')
    parser.add_argument('--run_dir', type=str, help='Path to a CSV file', required=True)
    parser.add_argument('--bins', type=int, help='Number of bins for distribution of results', default=10)
    parser.add_argument('--alpha', type=float, help='Transparency of plot bars', default=0.5)
    parser.add_argument('--no_plt_show', help='Disable showing the plot', action='store_true', default=False)
    return parser.parse_args()

def plot_graph(dataframe: pd.DataFrame, save_to_file: Union[str, None] = None) -> None:
    if args is not None:
        if "result" not in dataframe:
            if not os.environ.get("NO_NO_RESULT_ERROR"): # pragma: no cover
                print("General: Result column not found in dataframe. That may mean that the job had no valid runs")
            sys.exit(169)

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        sns.boxplot(x='generation_method', y='result', data=dataframe)
        plt.title('Results by Generation Method')
        plt.xlabel('Generation Method')
        plt.ylabel('Result')

        plt.subplot(2, 2, 2)
        sns.countplot(x='trial_status', data=dataframe)
        plt.title('Distribution of job status')
        plt.xlabel('Trial Status')
        plt.ylabel('Nr. of jobs')

        plt.subplot(2, 2, 3)
        exclude_columns = ['trial_index', 'arm_name', 'trial_status', 'generation_method']
        numeric_columns = dataframe.select_dtypes(include=['float64', 'int64']).columns
        numeric_columns = [col for col in numeric_columns if col not in exclude_columns]
        correlation_matrix = dataframe[numeric_columns].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", cbar=False)
        plt.title('Correlation Matrix')

        plt.subplot(2, 2, 4)
        histogram = sns.histplot(data=dataframe, x='result', hue='generation_method', multiple="stack", kde=False, bins=args.bins)
        for patch in histogram.patches:
            patch.set_alpha(args.alpha)
        plt.title('Distribution of Results by Generation Method')
        plt.xlabel('Result')
        plt.ylabel('Nr. of jobs')

        plt.tight_layout()

        if save_to_file:
            fig = plt.figure(1)
            helpers.save_to_file(fig, args, plt)
        else:
            if not args.no_plt_show: # pragma: no cover
                plt.show()

def update_graph() -> None:
    if args is not None:
        try:
            dataframe = None

            try:
                dataframe = pd.read_csv(args.run_dir + "/results.csv")
            except pd.errors.EmptyDataError:
                helpers.print_if_not_plot_tests_and_exit(f"{args.run_dir}/results.csv seems to be empty.", 19)
            except UnicodeDecodeError:
                helpers.print_if_not_plot_tests_and_exit(f"{args.run_dir}/results.csv seems to be invalid utf8.", 7)

            if args.min is not None or args.max is not None:
                dataframe = helpers.filter_data(args, dataframe, args.min, args.max)

            if dataframe.empty:
                helpers.print_if_not_plot_tests_and_exit("No applicable values could be found.", None)
                return

            if args.save_to_file:
                _path = os.path.dirname(args.save_to_file)
                if _path: # pragma: no cover
                    os.makedirs(_path, exist_ok=True)

            plot_graph(dataframe, args.save_to_file)

        except FileNotFoundError: # pragma: no cover
            logging.error("File not found: %s", args.run_dir + "/results.csv")
        except Exception as exception: # pragma: no cover
            logging.error("An unexpected error occurred: %s", str(exception))

            helpers.print_traceback()

if __name__ == "__main__":
    args = parse_arguments()

    helpers.setup_logging()

    if not os.path.exists(args.run_dir): # pragma: no cover
        logging.error("Specified --run_dir does not exist")
        sys.exit(1)

    update_graph()
