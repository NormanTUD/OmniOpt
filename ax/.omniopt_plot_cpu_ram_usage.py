# DESCRIPTION: Plot RAM and CPU usage
# EXPECTED FILES: cpu_ram_usage.csv
# TEST_OUTPUT_MUST_CONTAIN: CPU and RAM Usage over Time

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

def parse_arguments():
    parser = argparse.ArgumentParser(description='Plotting tool for analyzing CPU and RAM usage data.')
    parser.add_argument('--save_to_file', nargs='?', const='plot', type=str, help='Path to save the plot(s)')
    parser.add_argument('--run_dir', type=str, help='Path to a CSV file', required=True)
    parser.add_argument('--no_plt_show', help='Disable showing the plot', action='store_true', default=False)
    return parser.parse_args()

def plot_graph(dataframe, save_to_file=None):
    plt.figure(figsize=(12, 8))

    # Plotting RAM usage over time
    sns.lineplot(x='timestamp', y='ram_usage_mb', data=dataframe, label='RAM Usage (MB)')
    # Plotting CPU usage over time
    sns.lineplot(x='timestamp', y='cpu_usage_percent', data=dataframe, label='CPU Usage (%)')

    plt.title('CPU and RAM Usage over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Usage')

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
            dataframe = pd.read_csv(args.run_dir + "/cpu_ram_usage.csv")
        except pd.errors.EmptyDataError:
            if not os.environ.get("NO_NO_RESULT_ERROR"):
                print(f"{args.run_dir}/cpu_ram_usage.csv seems to be empty.")
            sys.exit(19)
        except UnicodeDecodeError:
            if not os.environ.get("NO_NO_RESULT_ERROR"):
                print(f"{args.run_dir}/cpu_ram_usage.csv seems to be invalid utf8.")
            sys.exit(7)

        if dataframe.empty:
            if not os.environ.get("NO_NO_RESULT_ERROR"):
                logging.warning("DataFrame is empty after reading.")
            return

        plot_graph(dataframe, args.save_to_file)

    except FileNotFoundError:
        logging.error("File not found: %s", args.run_dir + "/cpu_ram_usage.csv")
    except Exception as exception:
        logging.error("An unexpected error occurred: %s", str(exception))

        tb = traceback.format_exc()
        print(tb)

if __name__ == "__main__":
    args = parse_arguments()

    helpers.setup_logging()

    if not os.path.exists(args.run_dir):
        logging.error("Specified --run_dir does not exist")
        sys.exit(1)

    update_graph()
