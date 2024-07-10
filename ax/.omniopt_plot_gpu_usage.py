# DESCRIPTION: Plot GPU usage over time on different hosts
# EXPECTED FILES: gpu_usage_
# TEST_OUTPUT_MUST_CONTAIN: GPU Usage Over Time
# TEST_OUTPUT_MUST_CONTAIN: pci.bus_id
# TEST_OUTPUT_MUST_CONTAIN: GPU Usage
# TEST_OUTPUT_MUST_CONTAIN: Time

import os
script_dir = os.path.dirname(os.path.realpath(__file__))
helpers_file = f"{script_dir}/.helpers.py"
import importlib.util
spec = importlib.util.spec_from_file_location(
    name="helpers",
    location=helpers_file,
)
my_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(my_module)

import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

def dier(msg):
    pprint(msg)
    sys.exit(1)

def plot_gpu_usage(run_dir):
    gpu_data = []
    num_plots = 0
    plot_rows = 2
    plot_cols = 2  # standard number of columns for subplot grid
    _paths = []
    gpu_data_len = 0

    for root, dirs, files in os.walk(run_dir):
        for file in files:
            if file.startswith("gpu_usage_") and file.endswith(".csv"):
                file_path = os.path.join(root, file)
                _paths.append(file_path)
                df = pd.read_csv(file_path,
                                 names=["timestamp", "name", "pci.bus_id", "driver_version", "pstate",
                                        "pcie.link.gen.max", "pcie.link.gen.current", "temperature.gpu",
                                        "utilization.gpu [%]", "utilization.memory [%]", "memory.total [MiB]",
                                        "memory.free [MiB]", "memory.used [MiB]"])
                df = df.dropna()
                gpu_data.append(df)
                num_plots += 1
                gpu_data_len += len(df)

    if len(_paths) == 0:
        print(f"No gpu_usage_*.csv files could be found in {run_dir}")
        sys.exit(10)

    if not gpu_data:
        print("No GPU usage data found.")
        sys.exit(44)

    if gpu_data_len < 1:
        print(f"No valid GPU usage data foundf (len = {gpu_data_len}).")
        sys.exit(55)

    plot_cols = min(num_plots, plot_cols)  # Adjusting number of columns based on available plots
    plot_rows = (num_plots + plot_cols - 1) // plot_cols  # Calculating number of rows based on columns

    fig, axs = plt.subplots(plot_rows, plot_cols, figsize=(10, 5*plot_rows))
    if num_plots > 1:
        axs = axs.flatten()  # Flatten the axs array to handle both 1D and 2D subplots

    for i, df in enumerate(gpu_data):
        _ax = axs
        try:
            _ax = axs[i]
        except Exception as e:
            pass
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y/%m/%d %H:%M:%S.%f', errors='coerce')
        df = df.sort_values(by='timestamp')

        grouped_data = df.groupby('pci.bus_id')
        for bus_id, group in grouped_data:
            group['utilization.gpu [%]'] = pd.to_numeric(group['utilization.gpu [%]'].str.replace('%', ''), errors='coerce')
            group = group.dropna(subset=['timestamp', 'utilization.gpu [%]'])
            _ax.scatter(group['timestamp'], group['utilization.gpu [%]'], label=f'pci.bus_id: {bus_id}')

        _ax.set_xlabel('Time')
        _ax.set_ylabel('GPU Usage (%)')
        _ax.set_title(f'GPU Usage Over Time - {os.path.basename(_paths[i])}')
        if not args.no_legend:
            _ax.legend(loc='upper right')


    # Hide empty subplots
    for j in range(num_plots, plot_rows * plot_cols):
        axs[j].axis('off')

    plt.subplots_adjust(bottom=0.086, hspace=0.35)
    if args.save_to_file:
        _path = os.path.dirname(args.save_to_file)
        if _path:
            os.makedirs(_path, exist_ok=True)
        try:
            plt.savefig(args.save_to_file)
        except OSError as e:
            print(f"Error: {e}. This may happen on unstable file systems or in docker containers.")
            sys.exit(199)

    else:
        fig.canvas.manager.set_window_title("GPU-Usage: " + str(args.run_dir))
        if not args.no_plt_show:
            plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--run_dir', type=str, help='Directory where to search for CSV files')
    parser.add_argument('--plot_type', type=str, default="irgendwas", help='Type of plot (ignored)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (ignored)')
    parser.add_argument('--alpha', type=float, help='Transparency of plot bars', default=0.5)

    parser.add_argument('--no_legend', help='Disables legend (useless here)', action='store_true', default=False)
    parser.add_argument('--min', type=float, help='Minimum value for result filtering (useless here)')
    parser.add_argument('--max', type=float, help='Maximum value for result filtering (useless here)')
    parser.add_argument('--save_to_file', nargs='?', const='plot', type=str, help='Path to save the plot(s)')
    parser.add_argument('--single', help='Print plot to command line (useless here)', action='store_true', default=False)
    parser.add_argument('--bins', type=int, help='Number of bins for distribution of results (useless here)', default=10)
    parser.add_argument('--merge_with_previous_runs', action='append', nargs='+', help="Run-Dirs to be merged with (useless here)", default=[])
    parser.add_argument('--delete_temp', help='Delete temp files (useless here)', action='store_true', default=False)
    parser.add_argument('--darkmode', help='Enable darktheme (useless here)', action='store_true', default=False)
    parser.add_argument('--exclude_params', action='append', nargs='+', help="Params to be ignored (useless here)", default=[])
    parser.add_argument('--bubblesize', type=int, help='Size of the bubbles (useless here)', default=7)
    parser.add_argument('--no_plt_show', help='Disable showing the plot', action='store_true', default=False)

    args = parser.parse_args()

    plot_gpu_usage(args.run_dir)
