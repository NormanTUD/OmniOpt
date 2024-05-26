import sys
import os
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
                gpu_data.append(df)
                num_plots += 1

    if not gpu_data:
        print("No GPU usage data found.")
        return

    plot_cols = min(num_plots, plot_cols)  # Adjusting number of columns based on available plots
    plot_rows = (num_plots + plot_cols - 1) // plot_cols  # Calculating number of rows based on columns

    fig, axs = plt.subplots(plot_rows, plot_cols, figsize=(10, 5*plot_rows))
    axs = axs.flatten()  # Flatten the axs array to handle both 1D and 2D subplots

    for i, df in enumerate(gpu_data):
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y/%m/%d %H:%M:%S.%f', errors='coerce')
        df = df.sort_values(by='timestamp')

        grouped_data = df.groupby('pci.bus_id')
        for bus_id, group in grouped_data:
            group['utilization.gpu [%]'] = group['utilization.gpu [%]'].str.replace('%', '').astype(float)
            group = group.dropna(subset=['timestamp'])
            axs[i].plot(group['timestamp'], group['utilization.gpu [%]'], label=f'pci.bus_id: {bus_id}')

        axs[i].set_xlabel('Time')
        axs[i].set_ylabel('GPU Usage (%)')
        axs[i].set_title(f'GPU Usage Over Time - {os.path.basename(_paths[i])}')
        if not args.no_legend:
            axs[i].legend(loc='upper right')

    # Hide empty subplots
    for j in range(num_plots, plot_rows * plot_cols):
        axs[j].axis('off')

    plt.subplots_adjust(bottom=0.086, hspace=0.35)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--run_dir', type=str, help='Directory where to search for CSV files')
    parser.add_argument('--plot_type', type=str, default="irgendwas", help='Type of plot (ignored)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (ignored)')
    parser.add_argument('--no_legend', help='Disables legend', action='store_true', default=False)
    args = parser.parse_args()

    plot_gpu_usage(args.run_dir)

