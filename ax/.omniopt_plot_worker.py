import sys
from datetime import datetime
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

def dier(msg):
    pprint(msg)
    sys.exit(1)

def plot_worker_usage(pd_csv):
    #try:
    data = pd.read_csv(pd_csv)

    duplicate_mask = (data[data.columns.difference(['time'])].shift() == data[data.columns.difference(['time'])]).all(axis=1)
    data = data[~duplicate_mask].reset_index(drop=True)
    
    data['time'] = data['time'].apply(lambda x: datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))

    plt.plot(data['time'], data['num_parallel_jobs'], label='Requested Number of Workers')
    plt.plot(data['time'], data['nr_current_workers'], label='Number of Current Workers')
    plt.xlabel('Time')
    plt.ylabel('Count')
    plt.title('Worker Usage Plot')
    plt.legend()

    # Reduziere die Anzahl der x-Achsenbeschriftungen
    num_ticks = min(10, len(data['time']))
    x_ticks_indices = range(0, len(data['time']), len(data['time']) // num_ticks)
    x_tick_labels = [data['time'][i] for i in x_ticks_indices]
    plt.xticks(x_ticks_indices, x_tick_labels, rotation=45)

    plt.ylim(bottom=0.238)  # Setze bottom auf 0.238

    plt.tight_layout()  # Optimiere Layout, um Überlappungen zu minimieren
    plt.show()
    #except FileNotFoundError:
    #    print(f"File '{pd_csv}' not found.")
    #except Exception as e:
    #    print(f"An error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description='Plot worker usage from CSV file')
    parser.add_argument('--run_dir', type=str, help='Directory containing worker usage CSV file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    parser.add_argument('--plot_type', action='append', nargs='+', help="Params to be ignored", default=[])
    args = parser.parse_args()

    if args.debug:
        print(f"Debug mode enabled. Run directory: {args.run_dir}")

    if args.run_dir:
        pd_csv = os.path.join(args.run_dir, "worker_usage.csv")
        if os.path.exists(pd_csv):
            plot_worker_usage(pd_csv)
        else:
            print(f"File '{pd_csv}' does not exist.")

if __name__ == "__main__":
    main()
