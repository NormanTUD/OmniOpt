# DESCRIPTION: Plot general job info
# EXPECTED FILES: results.csv
# TEST_OUTPUT_MUST_CONTAIN: generation_method

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import logging

script_dir = os.path.dirname(os.path.realpath(__file__))
helpers_file = f"{script_dir}/.helpers.py"
import importlib.util
spec = importlib.util.spec_from_file_location(
    name="helpers",
    location=helpers_file,
)
my_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(my_module)


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Plotting tool for analyzing trial data.')
    parser.add_argument('--min', type=float, help='Minimum value for result filtering')
    parser.add_argument('--max', type=float, help='Maximum value for result filtering')
    parser.add_argument('--save_to_file', nargs='?', const='plot', type=str, help='Path to save the plot(s)')
    parser.add_argument('--exclude_params', action='append', nargs='+', help="Params to be ignored", default=[])
    parser.add_argument('--run_dir', type=str, help='Path to a CSV file', required=True)
    parser.add_argument('--delete_temp', help='Delete temp files', action='store_true', default=False)
    parser.add_argument('--darkmode', help='Enable darktheme', action='store_true', default=False)
    parser.add_argument('--single', help='Print plot to command line', action='store_true', default=False)
    parser.add_argument('--bubblesize', type=int, help='Size of the bubbles', default=7)
    parser.add_argument('--merge_with_previous_runs', action='append', nargs='+', help="Run-Dirs to be merged with", default=[])
    parser.add_argument('--plot_type', action='append', nargs='+', help="Params to be ignored", default=[])
    parser.add_argument('--bins', type=int, help='Number of bins for distribution of results', default=10)
    parser.add_argument('--alpha', type=float, help='Transparency of plot bars', default=0.5)
    parser.add_argument('--no_legend', help='Disables legend (useless here)', action='store_true', default=False)
    parser.add_argument('--debug', help='Enable debug', action='store_true', default=False)
    parser.add_argument('--no_plt_show', help='Disable showing the plot', action='store_true', default=False)
    return parser.parse_args()

def filter_data(dataframe, min_value=None, max_value=None):
    if min_value is not None:
        dataframe = dataframe[dataframe['result'] >= min_value]
    if max_value is not None:
        dataframe = dataframe[dataframe['result'] <= max_value]
    return dataframe

def plot_graph(dataframe, save_to_file=None):
    exclude_columns = ['trial_index', 'arm_name', 'trial_status', 'generation_method']
    numeric_columns = dataframe.select_dtypes(include=['float64', 'int64']).columns
    numeric_columns = [col for col in numeric_columns if col not in exclude_columns]

    pair_plot = sns.pairplot(dataframe, hue='generation_method', vars=numeric_columns)
    pair_plot.fig.suptitle('Pair Plot of Numeric Variables by Generation Method', y=1.02)

    if save_to_file:
        _path = os.path.dirname(args.save_to_file)
        if _path:
            os.makedirs(_path, exist_ok=True)
        try:
            pair_plot.savefig(args.save_to_file)
        except OSError as e:
            print(f"Error: {e}. This may happen on unstable file systems or in docker containers.")
            sys.exit(199)

    else:
        if not args.no_plt_show:
            plt.show()

def update_graph():
    try:
        dataframe = pd.read_csv(args.run_dir + "/results.csv")

        if args.min is not None or args.max is not None:
            dataframe = filter_data(dataframe, args.min, args.max)

        if dataframe.empty:
            if not os.environ.get("NO_NO_RESULT_ERROR"):
                print("DataFrame is empty after filtering.")
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

if __name__ == "__main__":
    setup_logging()
    args = parse_arguments()

    if not os.path.exists(args.run_dir):
        logging.error("Specified --run_dir does not exist")
        sys.exit(1)

    update_graph()
