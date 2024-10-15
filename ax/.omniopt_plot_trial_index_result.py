# DESCRIPTION: Plot trial index/result
# EXPECTED FILES: results.csv
# TEST_OUTPUT_MUST_CONTAIN: Results over Trial Index
# TEST_OUTPUT_MUST_CONTAIN: Trial Index
# TEST_OUTPUT_MUST_CONTAIN: Result

import argparse
import importlib.util
import logging
import os
import signal
import sys
import traceback

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
    parser.add_argument('--no_plt_show', help='Disable showing the plot', action='store_true', default=False)
    return parser.parse_args()

def filter_data(dataframe, min_value=None, max_value=None):
    if min_value is not None:
        dataframe = dataframe[dataframe['result'] >= min_value]
    if max_value is not None:
        dataframe = dataframe[dataframe['result'] <= max_value]
    return dataframe

def plot_graph(dataframe, save_to_file=None):
    if "result" not in dataframe:
        if not os.environ.get("NO_NO_RESULT_ERROR"):
            print("General: Result column not found in dataframe. That may mean that the job had no valid runs")
        sys.exit(169)

    plt.figure(figsize=(12, 8))

    # Lineplot der Ergebnisse über trial_index
    sns.lineplot(x='trial_index', y='result', data=dataframe)
    plt.title('Results over Trial Index')
    plt.xlabel('Trial Index')
    plt.ylabel('Result')

    if save_to_file:
        _path = os.path.dirname(args.save_to_file)
        if _path:
            os.makedirs(_path, exist_ok=True)
        try:
            plt.savefig(args.save_to_file)
        except OSError as e:
            print(f"Error: {e}. This may happen on unstable file systems.")
            sys.exit(199)

    else:
        if not args.no_plt_show:
            plt.show()

def update_graph():
    try:
        dataframe = None

        try:
            dataframe = pd.read_csv(args.run_dir + "/results.csv")
        except pd.errors.EmptyDataError:
            if not os.environ.get("PLOT_TESTS"):
                print(f"{args.run_dir}/results.csv seems to be empty.")
            sys.exit(19)
        except UnicodeDecodeError:
            if not os.environ.get("PLOT_TESTS"):
                print(f"{args.run_dir}/results.csv seems to be invalid utf8.")
            sys.exit(7)

        if args.min is not None or args.max is not None:
            try:
                dataframe = filter_data(dataframe, args.min, args.max)
            except KeyError:
                if not os.environ.get("PLOT_TESTS"):
                    print(f"{args.run_dir}/results.csv seems have no result column.")
                sys.exit(10)

        if dataframe.empty:
            if not os.environ.get("NO_NO_RESULT_ERROR"):
                logging.warning("DataFrame is empty after filtering.")
            return

        plot_graph(dataframe, args.save_to_file)

    except FileNotFoundError:
        logging.error("File not found: %s", args.run_dir + "/results.csv")
    except Exception as exception:
        logging.error("An unexpected error occurred: %s", str(exception))

        tb = traceback.format_exc()
        print(tb)

if __name__ == "__main__":
    args = parse_arguments()

    setup_logging()

    if not os.path.exists(args.run_dir):
        logging.error("Specified --run_dir does not exist")
        sys.exit(1)

    update_graph()
