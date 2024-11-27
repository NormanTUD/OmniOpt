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
if spec is not None:
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
else:
    raise ImportError(f"Could not load module from {helpers_file}")


def parse_arguments():
    parser = argparse.ArgumentParser(description='Plotting tool for analyzing trial data.')
    parser.add_argument('--min', type=float, help='Minimum value for result filtering')
    parser.add_argument('--max', type=float, help='Maximum value for result filtering')
    parser.add_argument('--save_to_file', nargs='?', const='plot', type=str, help='Path to save the plot(s)')
    parser.add_argument('--run_dir', type=str, help='Path to a CSV file', required=True)
    parser.add_argument('--no_plt_show', help='Disable showing the plot', action='store_true', default=False)
    return parser.parse_args()

def plot_graph(dataframe, save_to_file=None):
    if "result" not in dataframe:
        if not os.environ.get("NO_NO_RESULT_ERROR"): # pragma: no cover
            print("General: Result column not found in dataframe. That may mean that the job had no valid runs")
        sys.exit(169)

    plt.figure(figsize=(12, 8))

    # Lineplot der Ergebnisse über trial_index
    sns.lineplot(x='trial_index', y='result', data=dataframe)
    plt.title('Results over Trial Index')
    plt.xlabel('Trial Index')
    plt.ylabel('Result')

    if save_to_file:
        fig = plt.figure(1)
        helpers.save_to_file(fig, args, plt)
    else: # pragma: no cover
        if not args.no_plt_show:
            plt.show()

def update_graph():
    try:
        dataframe = None

        try:
            dataframe = pd.read_csv(args.run_dir + "/results.csv")
        except pd.errors.EmptyDataError:
            if not os.environ.get("PLOT_TESTS"): # pragma: no cover
                print(f"{args.run_dir}/results.csv seems to be empty.")
            sys.exit(19)
        except UnicodeDecodeError:
            if not os.environ.get("PLOT_TESTS"): # pragma: no cover
                print(f"{args.run_dir}/results.csv seems to be invalid utf8.")
            sys.exit(7)

        if args.min is not None or args.max is not None:
            try:
                dataframe = helpers.filter_data(args, dataframe, args.min, args.max)
            except KeyError: # pragma: no cover
                if not os.environ.get("PLOT_TESTS"):
                    print(f"{args.run_dir}/results.csv seems have no result column.")
                sys.exit(10)

        if dataframe.empty:
            if not os.environ.get("NO_NO_RESULT_ERROR"):
                logging.warning("DataFrame is empty after filtering.") # pragma: no cover
            return

        plot_graph(dataframe, args.save_to_file)

    except FileNotFoundError: # pragma: no cover
        logging.error("File not found: %s", args.run_dir + "/results.csv")
    except Exception as exception: # pragma: no cover
        logging.error("An unexpected error occurred: %s", str(exception))

        tb = traceback.format_exc()
        print(tb)

if __name__ == "__main__":
    args = parse_arguments()

    helpers.setup_logging()

    if not os.path.exists(args.run_dir): # pragma: no cover
        logging.error("Specified --run_dir does not exist")
        sys.exit(1)

    update_graph()
