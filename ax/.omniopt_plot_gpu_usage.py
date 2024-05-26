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

    _paths = []

    for root, dirs, files in os.walk(run_dir):
        for file in files:
            if file.startswith("gpu_usage_") and file.endswith(".csv"):
                file_path = os.path.join(root, file)
                _paths.append(file_path)
                # Einlesen der Daten mit expliziter Angabe der Spaltennamen
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

    fig, axs = plt.subplots(num_plots, 1, figsize=(10, 5*num_plots))
    if num_plots == 1:
        axs = [axs]  # Wenn nur ein Plot vorhanden ist, wandeln wir axs in eine Liste um

    for i, df in enumerate(gpu_data):
        # Sortieren der Daten nach Datum
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y/%m/%d %H:%M:%S.%f', errors='coerce')
        df = df.sort_values(by='timestamp')

        # Gruppieren der Daten nach pci.bus_id
        grouped_data = df.groupby('pci.bus_id')
        for bus_id, group in grouped_data:
            group['utilization.gpu [%]'] = group['utilization.gpu [%]'].str.replace('%', '').astype(float)
            group = group.dropna(subset=['timestamp'])  # Entfernen von ungültigen Zeitstempeln
            axs[i].plot(group['timestamp'], group['utilization.gpu [%]'], label=f'pci.bus_id: {bus_id}')

        axs[i].set_xlabel('Time')
        axs[i].set_ylabel('GPU Usage (%)')
        axs[i].set_title(f'GPU Usage Over Time - {os.path.basename(_paths[i])}')
        if not args.no_legend:
            axs[i].legend(loc='upper right')

    plt.subplots_adjust(bottom=0.2, hspace=0.35)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--run_dir', type=str, help='Directory where to search for CSV files')
    parser.add_argument('--plot_type', type=str, default="irgendwas", help='Type of plot (ignored)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (ignored)')
    parser.add_argument('--no_legend', help='Disables legend', action='store_true', default=False)
    args = parser.parse_args()

    plot_gpu_usage(args.run_dir)

