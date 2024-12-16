# TEST_OUTPUT_MUST_CONTAIN: Got
# TEST_OUTPUT_MUST_CONTAIN: Requested

import argparse
import importlib.util
import os
import signal
import sys
import traceback
from typing import Any, Union

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

signal.signal(signal.SIGINT, signal.SIG_DFL)

def parse_log_file(args: Any, log_file_path: str) -> Union[pd.DataFrame, None]:
    try:
        data = pd.read_csv(log_file_path, header=None, names=['time', 'got', 'requested'])

        valid_time_mask = data['time'].apply(helpers.is_valid_time_format)
        if not valid_time_mask.all():
            if not os.environ.get("NO_NO_RESULT_ERROR"): # pragma: no cover
                helpers.log_error("Some rows have invalid time format and will be removed.")
        data = data[valid_time_mask]

        if "time" not in data:
            if not os.environ.get("NO_NO_RESULT_ERROR"): # pragma: no cover
                print("time could not be found in data")
            sys.exit(19)

        data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')

        # Sort data by time
        data = data.sort_values(by='time')

        return data
    except FileNotFoundError: # pragma: no cover
        helpers.log_error(f"File '{log_file_path}' not found.")
        raise
    except AssertionError as e: # pragma: no cover
        helpers.log_error(str(e))
        raise
    except UnicodeDecodeError: # pragma: no cover
        if not os.environ.get("PLOT_TESTS"):
            print(f"{args.run_dir}/results.csv seems to be invalid utf8.")
        sys.exit(7)
    except Exception as e: # pragma: no cover
        helpers.log_error(f"An unexpected error occurred: {e}")
        print(traceback.format_exc(), file=sys.stderr)
        raise

    return None # pragma: no cover

def plot_trial_usage(args: Any, log_file_path: str) -> None:
    try:
        data = parse_log_file(args, log_file_path)

        plt.figure(figsize=(12, 6))

        # Plot 'got'
        if data is not None:
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
                fig = plt.figure(1)
                helpers.save_to_file(fig, args, plt)
            else: # pragma: no cover
                if not args.no_plt_show:
                    plt.show()
        else: # pragma: no cover
            helpers.log_error("Failed to get job data")
            sys.exit(8)
    except Exception as e: # pragma: no cover
        helpers.log_error(f"An error occurred while plotting: {e}")
        raise

def main() -> None:
    parser = argparse.ArgumentParser(description='Plot trial usage from log file')
    parser.add_argument('--run_dir', type=str, help='Directory containing log file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    parser.add_argument('--save_to_file', type=str, help='Save the plot to the specified file', default=None)
    parser.add_argument('--no_plt_show', help='Disable showing the plot', action='store_true', default=False)
    args = parser.parse_args()

    if args.debug: # pragma: no cover
        print(f"Debug mode enabled. Run directory: {args.run_dir}")

    if args.run_dir:
        log_file_path = os.path.join(args.run_dir, "get_next_trials.csv")
        if os.path.exists(log_file_path):
            try:
                plot_trial_usage(args, log_file_path)
            except Exception as e: # pragma: no cover
                helpers.log_error(f"main Error: {e}")
                sys.exit(3)
        else: # pragma: no cover
            helpers.log_error(f"File '{log_file_path}' does not exist.")

if __name__ == "__main__":
    main()
