# DESCRIPTION: Plot general job info
# EXPECTED FILES: results.csv
# TEST_OUTPUT_MUST_CONTAIN: Sobol

import os
import sys
import importlib.util
import logging
import signal
import traceback
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

signal.signal(signal.SIGINT, signal.SIG_DFL)

script_dir = os.path.dirname(os.path.realpath(__file__))
helpers_file = f"{script_dir}/.helpers.py"
spec = importlib.util.spec_from_file_location(
    name="helpers",
    location=helpers_file,
)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Plotting tool for analyzing trial data.')
    parser.add_argument('--min', type=float, help='Minimum value for result filtering')
    parser.add_argument('--max', type=float, help='Maximum value for result filtering')
    parser.add_argument('--save_to_file', nargs='?', const='plot', type=str, help='Path to save the plot(s)')
    parser.add_argument('--run_dir', type=str, help='Path to a CSV file', required=True)
    parser.add_argument('--bins', type=int, help='Number of bins for distribution of results', default=10)
    parser.add_argument('--alpha', type=float, help='Transparency of plot bars', default=0.5)
    parser.add_argument('--no_plt_show', help='Disable showing the plot', action='store_true', default=False)
    return parser.parse_args()

args = parse_arguments()

def filter_data(dataframe, min_value=None, max_value=None):
    try:
        if min_value is not None:
            dataframe = dataframe[dataframe['result'] >= min_value]
        if max_value is not None:
            dataframe = dataframe[dataframe['result'] <= max_value]
    except KeyError:
        print_if_not_plot_tests_and_exit(f"{args.run_dir}/results.csv seems to have no results column.", 19)

    return dataframe

def plot_graph(dataframe, save_to_file=None):
    if "result" not in dataframe:
        if not os.environ.get("NO_NO_RESULT_ERROR"):
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
        _path = os.path.dirname(args.save_to_file)
        if _path:
            os.makedirs(_path, exist_ok=True)
        try:
            plt.savefig(args.save_to_file)
        except OSError as e:
            print(f"Error: {e}. This may happen on unstable file systems or in docker containers.")
            sys.exit(199)
    else:
        if not args.no_plt_show:
            plt.show()

def print_if_not_plot_tests_and_exit(msg, exit_code):
    if not os.environ.get("PLOT_TESTS"):
        print(msg)
    if exit_code is not None:
        sys.exit(exit_code)

def print_traceback():
    tb = traceback.format_exc()
    print(tb)

def update_graph():
    try:
        dataframe = None

        try:
            dataframe = pd.read_csv(args.run_dir + "/results.csv")
        except pd.errors.EmptyDataError:
            print_if_not_plot_tests_and_exit(f"{args.run_dir}/results.csv seems to be empty.", 19)
        except UnicodeDecodeError:
            print_if_not_plot_tests_and_exit(f"{args.run_dir}/results.csv seems to be invalid utf8.", 7)

        if args.min is not None or args.max is not None:
            dataframe = filter_data(dataframe, args.min, args.max)

        if dataframe.empty:
            print_if_not_plot_tests_and_exit("No applicable values could be found.", None)
            return

        if args.save_to_file:
            _path = os.path.dirname(args.save_to_file)
            if _path:
                os.makedirs(_path, exist_ok=True)

        plot_graph(dataframe, args.save_to_file)

    except FileNotFoundError:
        logging.error("File not found: %s", args.run_dir + "/results.csv")
    except Exception as exception:
        logging.error("An unexpected error occurred: %s", str(exception))

        print_traceback()

if __name__ == "__main__":
    setup_logging()

    if not os.path.exists(args.run_dir):
        logging.error("Specified --run_dir does not exist")
        sys.exit(1)

    update_graph()
