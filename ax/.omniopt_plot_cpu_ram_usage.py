import importlib.util
import argparse
import logging
import os
import signal
import sys
import traceback

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Setup signal handling for interrupt
signal.signal(signal.SIGINT, signal.SIG_DFL)

# Global variable for parsed arguments
args = None
helpers = None

def load_helpers(script_dir):
    """Loads the helper module."""
    global helpers
    helpers_file = os.path.join(script_dir, ".helpers.py")
    spec = importlib.util.spec_from_file_location("helpers", helpers_file)
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)

def parse_arguments():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description='Plotting tool for analyzing CPU and RAM usage data.')
    parser.add_argument('--save_to_file', nargs='?', const='plot', type=str, help='Path to save the plot(s)')
    parser.add_argument('--run_dir', type=str, help='Path to a CSV file', required=True)
    parser.add_argument('--no_plt_show', help='Disable showing the plot', action='store_true', default=False)
    return parser.parse_args()

def load_data(csv_path):
    """Loads data from the given CSV file."""
    try:
        dataframe = pd.read_csv(csv_path)
        if dataframe.empty:
            logging.warning("DataFrame is empty after reading.")
            return None
        return dataframe
    except pd.errors.EmptyDataError:
        if not os.environ.get("NO_NO_RESULT_ERROR"):
            logging.error(f"CSV file {csv_path} is empty.")
        sys.exit(19)
    except UnicodeDecodeError:
        if not os.environ.get("NO_NO_RESULT_ERROR"):
            logging.error(f"CSV file {csv_path} contains invalid UTF-8.")
        sys.exit(7)
    except FileNotFoundError:
        if not os.environ.get("NO_NO_RESULT_ERROR"):
            logging.error(f"CSV file not found: {csv_path}")
        sys.exit(1)

def plot_graph(dataframe, save_to_file=None):
    """Generates and optionally saves/plots the graph."""
    plt.figure(figsize=(12, 8))

    # Plot RAM usage over time
    sns.lineplot(x='timestamp', y='ram_usage_mb', data=dataframe, label='RAM Usage (MB)')
    # Plot CPU usage over time
    sns.lineplot(x='timestamp', y='cpu_usage_percent', data=dataframe, label='CPU Usage (%)')

    plt.title('CPU and RAM Usage over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Usage')

    if save_to_file:
        fig = plt.figure(1)
        helpers.save_to_file(fig, args, plt)
    elif not args.no_plt_show:
        plt.show()

def update_graph(csv_path):
    """Updates the graph by loading data and plotting."""
    dataframe = load_data(csv_path)
    if dataframe is not None:
        plot_graph(dataframe, args.save_to_file)

def main():
    """Main function for handling the overall logic."""
    global args
    args = parse_arguments()

    load_helpers(os.path.dirname(os.path.realpath(__file__)))
    helpers.setup_logging()

    if not os.path.exists(args.run_dir):
        logging.error("Specified --run_dir does not exist")
        sys.exit(1)

    csv_path = os.path.join(args.run_dir, "cpu_ram_usage.csv")
    update_graph(csv_path)

if __name__ == "__main__":
    main()
