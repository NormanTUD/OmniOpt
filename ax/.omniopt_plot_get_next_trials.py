# DESCRIPTION: Plot get_next_trials got/requested
# EXPECTED FILES: get_next_trials.csv
# TEST_OUTPUT_MUST_CONTAIN: Trials Usage Plot
# TEST_OUTPUT_MUST_CONTAIN: Count
# TEST_OUTPUT_MUST_CONTAIN: Time
# TEST_OUTPUT_MUST_CONTAIN: Got
# TEST_OUTPUT_MUST_CONTAIN: Requested

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

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

import re
import traceback
import sys
from datetime import datetime
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint

def dier(msg):
    pprint(msg)
    sys.exit(10)

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

def parse_log_file(log_file_path):
    try:
        data = pd.read_csv(log_file_path, header=None, names=['time', 'got', 'requested'])

        def is_valid_time_format(time_string):
            try:
                datetime.strptime(time_string, '%Y-%m-%d %H:%M:%S')
                return True
            except ValueError:
                return False

        valid_time_mask = data['time'].apply(is_valid_time_format)
        if not valid_time_mask.all():
            if not os.environ.get("NO_NO_RESULT_ERROR"):
                log_error("Some rows have invalid time format and will be removed.")
        data = data[valid_time_mask]

        data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')

        # Sort data by time
        data = data.sort_values(by='time')

        return data
    except FileNotFoundError:
        log_error(f"File '{log_file_path}' not found.")
        raise
    except AssertionError as e:
        log_error(str(e))
        raise
    except Exception as e:
        log_error(f"An unexpected error occurred: {e}")
        print(traceback.format_exc(), file=sys.stderr)
        raise

def plot_trial_usage(args, log_file_path):
    try:
        data = parse_log_file(log_file_path)

        plt.figure(figsize=(12, 6))

        # Plot 'got'
        plt.plot(data['time'], data['got'], label='Got', color='blue')

        # Plot 'requested'
        plt.plot(data['time'], data['requested'], label='Requested', color='orange')

        plt.xlabel('Time')
        plt.ylabel('Count')
        plt.title('Trials Usage Plot')
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
    except Exception as e:
        log_error(f"An error occurred while plotting: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Plot trial usage from log file')
    parser.add_argument('--run_dir', type=str, help='Directory containing log file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    parser.add_argument('--save_to_file', type=str, help='Save the plot to the specified file', default=None)
    parser.add_argument('--no_plt_show', help='Disable showing the plot', action='store_true', default=False)
    args = parser.parse_args()

    if args.debug:
        print(f"Debug mode enabled. Run directory: {args.run_dir}")

    if args.run_dir:
        log_file_path = os.path.join(args.run_dir, "get_next_trials.csv")
        if os.path.exists(log_file_path):
            try:
                plot_trial_usage(args, log_file_path)
            except Exception as e:
                log_error(f"Error: {e}")
                sys.exit(3)
        else:
            log_error(f"File '{log_file_path}' does not exist.")

if __name__ == "__main__":
    main()
