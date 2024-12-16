# DESCRIPTION: Plot number of workers over time
# EXPECTED FILES: worker_usage.csv
# TEST_OUTPUT_MUST_CONTAIN: Requested Number of Workers
# TEST_OUTPUT_MUST_CONTAIN: Number of Current Workers
# TEST_OUTPUT_MUST_CONTAIN: Worker Usage Plot

import argparse
import importlib.util
import os
import sys
import traceback
from datetime import datetime, timezone
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

script_dir = os.path.dirname(os.path.realpath(__file__))
helpers_file = f"{script_dir}/.helpers.py"
spec = importlib.util.spec_from_file_location(
    name="helpers",
    location=helpers_file,
)
if spec is not None and spec.loader is not None:
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
else: # pragma: no cover
    raise ImportError(f"Could not load module from {helpers_file}")

def plot_worker_usage(args: Any, pd_csv: str) -> None:
    try:
        data = pd.read_csv(pd_csv, names=['time', 'num_parallel_jobs', 'nr_current_workers', 'percentage'])

        assert len(data.columns) > 0, "CSV file has no columns."
        assert "time" in data.columns, "The 'time' column is missing."
        assert data is not None, "No data could be found in the CSV file."

        duplicate_mask = (data[data.columns.difference(['time'])].shift() == data[data.columns.difference(['time'])]).all(axis=1)
        data = data[~duplicate_mask].reset_index(drop=True)

        # Filter out invalid 'time' entries
        valid_times = data['time'].apply(helpers.looks_like_number)
        data = data[valid_times]

        if "time" not in data:
            if not os.environ.get("NO_NO_RESULT_ERROR"): # pragma: no cover
                print("time could not be found in data")
            sys.exit(19)

        data['time'] = data['time'].apply(lambda x: datetime.fromtimestamp(int(float(x)), timezone.utc).strftime('%Y-%m-%d %H:%M:%S') if helpers.looks_like_number(x) else x)
        data['time'] = pd.to_datetime(data['time'])

        # Sort data by time
        data = data.sort_values(by='time')

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
            fig = plt.figure(1)
            helpers.save_to_file(fig, args, plt)
        else:
            if not args.no_plt_show: # pragma: no cover
                plt.show()
    except FileNotFoundError: # pragma: no cover
        helpers.log_error(f"File '{pd_csv}' not found.")
    except AssertionError as e: # pragma: no cover
        helpers.log_error(str(e))
    except UnicodeDecodeError: # pragma: no cover
        if not os.environ.get("PLOT_TESTS"): # pragma: no cover
            print(f"{args.run_dir}/results.csv seems to be invalid utf8.")
        sys.exit(7)
    except Exception as e: # pragma: no cover
        helpers.log_error(f"An unexpected error occurred: {e}")
        print(traceback.format_exc(), file=sys.stderr)

def main() -> None:
    parser = argparse.ArgumentParser(description='Plot worker usage from CSV file')
    parser.add_argument('--run_dir', type=str, help='Directory containing worker usage CSV file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    parser.add_argument('--save_to_file', type=str, help='Save the plot to the specified file', default=None)

    parser.add_argument('--no_plt_show', help='Disable showing the plot', action='store_true', default=False)
    args = parser.parse_args()

    if args.debug: # pragma: no cover
        print(f"Debug mode enabled. Run directory: {args.run_dir}")

    if args.run_dir:
        worker_usage_csv = os.path.join(args.run_dir, "worker_usage.csv")
        if os.path.exists(worker_usage_csv):
            try:
                plot_worker_usage(args, worker_usage_csv)
            except Exception as e: # pragma: no cover
                helpers.log_error(f"Error: {e}")
                sys.exit(3)
        else: # pragma: no cover
            helpers.log_error(f"File '{worker_usage_csv}' does not exist.")
            sys.exit(19)

if __name__ == "__main__":
    main()
