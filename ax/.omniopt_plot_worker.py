# DESCRIPTION: Plot number of workers over time
# EXPECTED FILES: worker_usage.csv
# TEST_OUTPUT_MUST_CONTAIN: Requested Number of Workers
# TEST_OUTPUT_MUST_CONTAIN: Number of Current Workers
# TEST_OUTPUT_MUST_CONTAIN: Worker Usage Plot

import argparse
import importlib.util
import os
import sys
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import pandas as pd

# Load helper functions
def load_helpers(script_dir):
    helpers_file = f"{script_dir}/.helpers.py"
    spec = importlib.util.spec_from_file_location(name="helpers", location=helpers_file)
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    return helpers

def read_csv_data(file_path, helpers):
    try:
        data = pd.read_csv(file_path, names=['time', 'num_parallel_jobs', 'nr_current_workers', 'percentage'])
        assert len(data.columns) > 0, "CSV file has no columns."
        assert "time" in data.columns, "The 'time' column is missing."
        assert data is not None, "No data could be found in the CSV file."
        return data
    except FileNotFoundError:
        helpers.log_error(f"File '{file_path}' not found.")
        sys.exit(19)
    except AssertionError as e:
        helpers.log_error(str(e))
        sys.exit(19)

def filter_valid_data(data, helpers):
    # Remove duplicate entries
    duplicate_mask = (data[data.columns.difference(['time'])].shift() == data[data.columns.difference(['time'])]).all(axis=1)
    data = data[~duplicate_mask].reset_index(drop=True)

    # Filter out invalid 'time' entries
    valid_times = data['time'].apply(helpers.looks_like_number)
    data = data[valid_times]

    if "time" not in data:
        print("time could not be found in data")
        sys.exit(19)

    # Convert 'time' to datetime format
    data['time'] = data['time'].apply(lambda x: datetime.fromtimestamp(int(float(x)), timezone.utc).strftime('%Y-%m-%d %H:%M:%S') if helpers.looks_like_number(x) else x)
    data['time'] = pd.to_datetime(data['time'])

    # Sort data by time
    return data.sort_values(by='time')

def plot_data(data, args):
    plt.figure(figsize=(12, 6))

    # Plot Requested Number of Workers
    plt.plot(data['time'], data['num_parallel_jobs'], label='Requested Number of Workers', color='blue')

    # Plot Number of Current Workers
    plt.plot(data['time'], data['nr_current_workers'], label='Number of Current Workers', color='orange')

    plt.xlabel('Time')
    plt.ylabel('Count')
    plt.title('Worker Usage Plot')
    plt.legend()

    plt.gcf().autofmt_xdate()  # Rotate and align the x labels
    plt.tight_layout()

    if args.save_to_file:
        save_plot(args.save_to_file)
    elif not args.no_plt_show:
        plt.show()

def save_plot(save_to_file):
    _path = os.path.dirname(save_to_file)
    if _path:
        os.makedirs(_path, exist_ok=True)

    try:
        plt.savefig(save_to_file)
    except OSError as e:
        print(f"Error: {e}. This may happen on unstable file systems or in docker containers.")
        sys.exit(199)

def main():
    parser = argparse.ArgumentParser(description='Plot worker usage from CSV file')
    parser.add_argument('--run_dir', type=str, help='Directory containing worker usage CSV file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--save_to_file', type=str, help='Save the plot to the specified file', default=None)
    parser.add_argument('--no_plt_show', help='Disable showing the plot', action='store_true', default=False)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    helpers = load_helpers(script_dir)

    if args.debug:
        print(f"Debug mode enabled. Run directory: {args.run_dir}")

    if args.run_dir:
        worker_usage_csv = os.path.join(args.run_dir, "worker_usage.csv")
        if os.path.exists(worker_usage_csv):
            try:
                data = read_csv_data(worker_usage_csv, helpers)
                valid_data = filter_valid_data(data, helpers)
                plot_data(valid_data, args)
            except Exception as e:
                helpers.log_error(f"Error: {e}")
                sys.exit(3)
        else:
            helpers.log_error(f"File '{worker_usage_csv}' does not exist.")
            sys.exit(19)

if __name__ == "__main__":
    main()
