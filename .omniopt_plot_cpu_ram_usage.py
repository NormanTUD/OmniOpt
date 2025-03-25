# DESCRIPTION: Plot CPU and RAM Usage
# TITLE: CPU/RAM Usage
# FULL_DESCRIPTION: Plot CPU and RAM Usage over time for the main worker
# EXPECTED FILES: cpu_ram_usage.csv

import importlib.util
from typing import Union
import argparse
import logging
import os
import signal
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from beartype import beartype

# Setup signal handling for interrupt
signal.signal(signal.SIGINT, signal.SIG_DFL)

# Global variable for parsed arguments
args = None
helpers = None

@beartype
def load_helpers(script_dir: str) -> None:
    """Loads the helper module."""
    global helpers
    helpers_file = os.path.join(script_dir, ".helpers.py")
    spec = importlib.util.spec_from_file_location("helpers", helpers_file)
    if spec is not None and spec.loader is not None:
        helpers = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helpers)
    else:
        raise ImportError(f"Could not load module from {helpers_file}")

parser = argparse.ArgumentParser(description='Plotting tool for analyzing CPU and RAM usage data.')
parser.add_argument('--save_to_file', nargs='?', const='plot', type=str, help='Path to save the plot(s)')
parser.add_argument('--run_dir', type=str, help='Path to a CSV file', required=True)
parser.add_argument('--no_plt_show', help='Disable showing the plot', action='store_true', default=False)
args = parser.parse_args()

@beartype
def load_data(csv_path: str) -> Union[pd.DataFrame, None]:
    """Loads data from the given CSV file."""
    try:
        dataframe = pd.read_csv(csv_path)
        if dataframe.empty:
            logging.warning("DataFrame is empty after reading.")
            return None
        return dataframe
    except pd.errors.EmptyDataError:
        if not os.environ.get("NO_NO_RESULT_ERROR"):
            logging.error("CSV file %s is empty.", csv_path)
        sys.exit(19)
    except UnicodeDecodeError:
        if not os.environ.get("NO_NO_RESULT_ERROR"):
            logging.error("CSV file %s contains invalid UTF-8.", csv_path)
        sys.exit(7)
    except FileNotFoundError:
        if not os.environ.get("NO_NO_RESULT_ERROR"):
            logging.error("CSV file not found: %s", csv_path)
        sys.exit(1)

@beartype
def plot_graph(dataframe: pd.DataFrame, save_to_file: Union[str, None] = None) -> None:
    """Generates and optionally saves/plots the graph."""
    plt.figure("Plot CPU and RAM Usage", figsize=(12, 8))

    dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp'], unit='s')

    # Plot RAM usage over time
    sns.lineplot(x='timestamp', y='ram_usage_mb', data=dataframe, label='RAM Usage (MB)')
    # Plot CPU usage over time
    sns.lineplot(x='timestamp', y='cpu_usage_percent', data=dataframe, label='CPU Usage (%)')

    plt.title('CPU and RAM Usage over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Usage')

    if save_to_file:
        fig = plt.figure(1)
        if fig is not None and args is not None and plt is not None and helpers is not None:
            helpers.save_to_file(fig, args, plt)
    elif args is not None and not args.no_plt_show:
        if plt is not None:
            plt.show()

@beartype
def update_graph(csv_path: str) -> None:
    """Updates the graph by loading data and plotting."""
    dataframe = load_data(csv_path)
    if dataframe is not None:
        if args is not None:
            plot_graph(dataframe, args.save_to_file)

@beartype
def main() -> None:
    """Main function for handling the overall logic."""

    load_helpers(os.path.dirname(os.path.realpath(__file__)))
    if helpers:
        helpers.setup_logging()

    if args:
        if not os.path.exists(args.run_dir):
            print("Specified --run_dir does not exist")
            sys.exit(1)

        csv_path = os.path.join(args.run_dir, "cpu_ram_usage.csv")
        update_graph(csv_path)

if __name__ == "__main__":
    main()
