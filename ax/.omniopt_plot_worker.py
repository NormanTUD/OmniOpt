# DESCRIPTION: Plot number of workers over time
# EXPECTED FILES: worker_usage.csv
# TEST_OUTPUT_MUST_CONTAIN: Requested Number of Workers
# TEST_OUTPUT_MUST_CONTAIN: Number of Current Workers
# TEST_OUTPUT_MUST_CONTAIN: Worker Usage Plot

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

import re
import traceback
import sys
from datetime import datetime
import argparse
import pandas as pd
import matplotlib.pyplot as plt

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

def plot_worker_usage(args, pd_csv):
    try:
        data = pd.read_csv(pd_csv, names=['time', 'num_parallel_jobs', 'nr_current_workers', 'percentage'])

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
            _path = os.path.dirname(args.save_to_file)
            if _path:
                os.makedirs(_path, exist_ok=True)

            plt.savefig(args.save_to_file)
        else:
            if not args.no_plt_show:
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

    parser.add_argument('--alpha', type=float, help='Transparency of plot bars (useless here)', default=0.5)
    parser.add_argument('--no_legend', help='Disables legend (useless here)', action='store_true', default=False)

    parser.add_argument('--min', type=float, help='Minimum value for result filtering (useless here)')
    parser.add_argument('--max', type=float, help='Maximum value for result filtering (useless here)')
    parser.add_argument('--darkmode', help='Enable darktheme (useless here)', action='store_true', default=False)
    parser.add_argument('--bins', type=int, help='Number of bins for distribution of results (useless here)', default=10)
    parser.add_argument('--bubblesize', type=int, help='Size of the bubbles (useless here)', default=7)
    parser.add_argument('--delete_temp', help='Delete temp files (useless here)', action='store_true', default=False)
    parser.add_argument('--merge_with_previous_runs', action='append', nargs='+', help="Run-Dirs to be merged with (useless here)", default=[])
    parser.add_argument('--exclude_params', action='append', nargs='+', help="Params to be ignored (useless here)", default=[])
    parser.add_argument('--single', help='Print plot to command line (useless here)', action='store_true', default=False)
    parser.add_argument('--no_plt_show', help='Disable showing the plot', action='store_true', default=False)
    args = parser.parse_args()

    if args.debug:
        print(f"Debug mode enabled. Run directory: {args.run_dir}")

    if args.run_dir:
        worker_usage_csv = os.path.join(args.run_dir, "worker_usage.csv")
        if os.path.exists(worker_usage_csv):
            try:
                plot_worker_usage(args, worker_usage_csv)
            except Exception as e:
                log_error(f"Error: {e}")
                sys.exit(3)
        else:
            log_error(f"File '{worker_usage_csv}' does not exist.")
            sys.exit(19)

if __name__ == "__main__":
    main()
