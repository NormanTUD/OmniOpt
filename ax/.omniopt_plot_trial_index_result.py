# DESCRIPTION: Plot trial index/result
# EXPECTED FILES: results.csv
# TEST_OUTPUT_MUST_CONTAIN: Results over Trial Index
# TEST_OUTPUT_MUST_CONTAIN: Trial Index
# TEST_OUTPUT_MUST_CONTAIN: Result

import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import logging
import pandas as pd

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Plotting tool for analyzing trial data.')
    parser.add_argument('--min', type=float, help='Minimum value for result filtering')
    parser.add_argument('--max', type=float, help='Maximum value for result filtering')
    parser.add_argument('--save_to_file', nargs='?', const='plot', type=str, help='Path to save the plot(s)')
    parser.add_argument('--run_dir', type=str, help='Path to a CSV file', required=True)
    parser.add_argument('--darkmode', help='Enable darktheme', action='store_true', default=False)
    parser.add_argument('--no_plt_show', help='Disable showing the plot', action='store_true', default=False)
    return parser.parse_args()

def filter_data(dataframe, min_value=None, max_value=None):
    if min_value is not None:
        dataframe = dataframe[dataframe['result'] >= min_value]
    if max_value is not None:
        dataframe = dataframe[dataframe['result'] <= max_value]
    return dataframe

def plot_graph(dataframe, save_to_file=None):
    window_title = f"Results over trial index: {args.run_dir}"

    if not "result" in dataframe:
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
        plt.savefig(save_to_file)
    else:
        if not args.no_plt_show:
            plt.show()

def update_graph():
    try:
        dataframe = None

        try:
            dataframe = pd.read_csv(args.run_dir + "/results.csv")
        except pd.errors.EmptyDataError:
            print(f"{args.run_dir}/results.csv seems to be empty.")
            sys.exit(19)

        if args.min is not None or args.max is not None:
            dataframe = filter_data(dataframe, args.min, args.max)

        if dataframe.empty:
            if not os.environ.get("NO_NO_RESULT_ERROR"):
                logging.warning("DataFrame is empty after filtering.")
            return

        plot_graph(dataframe, args.save_to_file)

    except FileNotFoundError:
        logging.error("File not found: %s", args.run_dir + "/results.csv")
    except Exception as exception:
        logging.error("An unexpected error occurred: %s", str(exception))

        import traceback
        tb = traceback.format_exc()
        print(tb)

if __name__ == "__main__":
    setup_logging()
    args = parse_arguments()

    if not os.path.exists(args.run_dir):
        logging.error("Specified --run_dir does not exist")
        sys.exit(1)

    update_graph()
