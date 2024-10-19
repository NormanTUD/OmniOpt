# DESCRIPTION: Plot general job info
# EXPECTED FILES: results.csv
# TEST_OUTPUT_MUST_CONTAIN: generation_method

import argparse
import importlib.util
import logging
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

args = None

script_dir = os.path.dirname(os.path.realpath(__file__))
helpers_file = f"{script_dir}/.helpers.py"
spec = importlib.util.spec_from_file_location(
    name="helpers",
    location=helpers_file,
)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Plotting tool for analyzing trial data.')
    parser.add_argument('--min', type=float, help='Minimum value for result filtering')
    parser.add_argument('--max', type=float, help='Maximum value for result filtering')
    parser.add_argument('--save_to_file', nargs='?', const='plot', type=str, help='Path to save the plot(s)')
    parser.add_argument('--run_dir', type=str, help='Path to a CSV file', required=True)
    parser.add_argument('--no_plt_show', help='Disable showing the plot', action='store_true', default=False)
    return parser.parse_args()

def plot_graph(dataframe, save_to_file=None):
    exclude_columns = ['trial_index', 'arm_name', 'trial_status', 'generation_method']
    numeric_columns = dataframe.select_dtypes(include=['float64', 'int64']).columns
    numeric_columns = [col for col in numeric_columns if col not in exclude_columns]

    pair_plot = sns.pairplot(dataframe, hue='generation_method', vars=numeric_columns)
    pair_plot.fig.suptitle('Pair Plot of Numeric Variables by Generation Method', y=1.02)

    if save_to_file:
        helpers.save_to_file(pair_plot.fig, args, plt)
    else:
        if not args.no_plt_show:
            plt.show()

def update_graph():
    try:
        dataframe = pd.read_csv(args.run_dir + "/results.csv")

        if args.min is not None or args.max is not None:
            dataframe = helpers.filter_data(dataframe, args.min, args.max)

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
        print("File not found: %s", args.run_dir + "/results.csv")
    except pd.errors.EmptyDataError:
        if not os.environ.get("NO_NO_RESULT_ERROR"):
            print("The file to be parsed was empty")
        sys.exit(19)
    except UnicodeDecodeError:
        if not os.environ.get("PLOT_TESTS"):
            print(f"{args.run_dir}/results.csv seems to be invalid utf8.")
        sys.exit(7)
    except KeyError:
        if not os.environ.get("PLOT_TESTS"):
            print(f"{args.run_dir}/results.csv seems have no result column.")
    except Exception as exception:
        print("An unexpected error occurred: %s" % str(exception))

if __name__ == "__main__":
    args = parse_arguments()

    helpers.setup_logging()

    if not os.path.exists(args.run_dir):
        logging.error("Specified --run_dir does not exist")
        sys.exit(1)

    update_graph()
