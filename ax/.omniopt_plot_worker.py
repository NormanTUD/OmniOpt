import re
import traceback
import sys
from datetime import datetime
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

def assert_condition(condition, error_text):
    if not condition:
        raise AssertionError(error_text)

def log_error(error_text):
    print(f"Error: {error_text}", file=sys.stderr)

def looks_like_number(x):
    return looks_like_float(x) or looks_like_int(x)

def looks_like_float(x):
    if isinstance(x, (int, float)):
        return True
    elif isinstance(x, str):
        try:
            float(x)
            return True
        except ValueError:
            return False
    return False

def looks_like_int(x):
    if isinstance(x, int):
        return True
    elif isinstance(x, float):
        return x.is_integer()
    elif isinstance(x, str):
        return bool(re.match(r'^\d+$', x))
    return False

def plot_worker_usage(pd_csv):
    try:
        data = pd.read_csv(pd_csv)

        assert_condition(len(data.columns) > 0, "CSV file has no columns.")
        assert_condition("time" in data.columns, "The 'time' column is missing.")
        assert_condition(data is not None, "No data could be found in the CSV file.")

        duplicate_mask = (data[data.columns.difference(['time'])].shift() == data[data.columns.difference(['time'])]).all(axis=1)
        data = data[~duplicate_mask].reset_index(drop=True)

        # Filter out invalid 'time' entries
        valid_times = data['time'].apply(looks_like_number)
        data = data[valid_times]

        data['time'] = data['time'].apply(lambda x: datetime.utcfromtimestamp(int(float(x))).strftime('%Y-%m-%d %H:%M:%S') if looks_like_number(x) else x)
        data['time'] = pd.to_datetime(data['time'])

        plt.figure(figsize=(12, 6))
        plt.plot(data['time'], data['num_parallel_jobs'], label='Requested Number of Workers', color='blue')
        plt.plot(data['time'], data['nr_current_workers'], label='Number of Current Workers', color='orange')
        plt.xlabel('Time')
        plt.ylabel('Count')
        plt.title('Worker Usage Plot')
        plt.legend()

        num_ticks = min(10, len(data['time']))
        x_ticks_indices = range(0, len(data['time']), max(1, len(data['time']) // num_ticks))
        x_tick_labels = [data['time'].dt.strftime('%Y-%m-%d %H:%M:%S').iloc[i] for i in x_ticks_indices]
        plt.xticks(x_ticks_indices, x_tick_labels, rotation=45)

        plt.ylim(bottom=0.238)

        plt.tight_layout()
        plt.show()
    except FileNotFoundError:
        log_error(f"File '{pd_csv}' not found.")
    except AssertionError as e:
        log_error(str(e))
    except Exception as e:
        log_error(f"An unexpected error occurred: {e}")
        print(traceback.format_exc(), file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description='Plot worker usage from CSV file')
    parser.add_argument('--run_dir', type=str, help='Directory containing worker usage CSV file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    parser.add_argument('--save_to_file', type=str, help='Save the plot to the specified file', default=None)
    parser.add_argument('--plot_type', action='append', nargs='+', help="Params to be ignored", default=[])
    args = parser.parse_args()

    if args.debug:
        print(f"Debug mode enabled. Run directory: {args.run_dir}")

    if args.run_dir:
        worker_usage_csv = os.path.join(args.run_dir, "worker_usage.csv")
        if os.path.exists(worker_usage_csv):
            try:
                plot_worker_usage(worker_usage_csv)
            except Exception as e:
                log_error(f"Error: {e}")
                sys.exit(3)
        else:
            log_error(f"File '{worker_usage_csv}' does not exist.")

if __name__ == "__main__":
    main()

