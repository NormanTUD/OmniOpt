# DESCRIPTION: Plot time and exit code infos
# EXPECTED FILES: job_infos.csv
# TEST_OUTPUT_MUST_CONTAIN: Run Time Distribution
# TEST_OUTPUT_MUST_CONTAIN: Run Time by Hostname
# TEST_OUTPUT_MUST_CONTAIN: Distribution of Run Time
# TEST_OUTPUT_MUST_CONTAIN: Result over Time

import argparse
import importlib.util
import os
import signal
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tzlocal import get_localzone

signal.signal(signal.SIGINT, signal.SIG_DFL)

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

def main() -> None:
    parser = argparse.ArgumentParser(description='Plot worker usage from CSV file')
    parser.add_argument('--run_dir', type=str, help='Directory containing worker usage CSV file')

    parser.add_argument('--save_to_file', type=str, help='Save the plot to the specified file', default=None)

    parser.add_argument('--bins', type=int, help='Number of bins for distribution of results (useless here)', default=10)
    parser.add_argument('--no_plt_show', help='Disable showing the plot', action='store_true', default=False)
    args = parser.parse_args()

    _job_infos_csv = f'{args.run_dir}/job_infos.csv'

    if not os.path.exists(_job_infos_csv): # pragma: no cover
        print(f"Error: {_job_infos_csv} not found")
        sys.exit(1)

    df = None

    try:
        df = pd.read_csv(_job_infos_csv)
    except pd.errors.EmptyDataError:
        if not os.environ.get("NO_NO_RESULT_ERROR"): # pragma: no cover
            print(f"Could not find values in file {_job_infos_csv}")
        sys.exit(19)
    except UnicodeDecodeError:
        if not os.environ.get("PLOT_TESTS"): # pragma: no cover
            print(f"{args.run_dir}/results.csv seems to be invalid utf8.")
        sys.exit(7)
    df = df.sort_values(by='exit_code')

    fig, axes = plt.subplots(2, 2, figsize=(20, 30))

    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    if "run_time" not in df:
        if not os.environ.get("NO_NO_RESULT_ERROR"): # pragma: no cover
            print("Error: run_time not in df. Probably the job_infos.csv file is corrupted.")
        sys.exit(2)

    axes[0, 0].hist(df['run_time'], bins=args.bins)
    axes[0, 0].set_title('Distribution of Run Time')
    axes[0, 0].set_xlabel('Run Time')
    axes[0, 0].set_ylabel(f'Number of jobs in this runtime ({args.bins} bins)')

    local_tz = get_localzone()

    df['start_time'] = pd.to_datetime(df['start_time'], unit='s', utc=True).dt.tz_convert(local_tz)
    df['end_time'] = pd.to_datetime(df['end_time'], unit='s', utc=True).dt.tz_convert(local_tz)

    df['start_time'] = df['start_time'].apply(lambda x: datetime.utcfromtimestamp(int(float(x))).strftime('%Y-%m-%d %H:%M:%S') if helpers.looks_like_number(x) else x)
    df['end_time'] = df['start_time'].apply(lambda x: datetime.utcfromtimestamp(int(float(x))).strftime('%Y-%m-%d %H:%M:%S') if helpers.looks_like_number(x) else x)

    sns.scatterplot(data=df, x='start_time', y='result', marker='o', label='Start Time', ax=axes[0, 1])
    sns.scatterplot(data=df, x='end_time', y='result', marker='x', label='End Time', ax=axes[0, 1])

    axes[0, 1].set_title('Result over Time')

    df["exit_code"] = [str(int(x)) for x in df["exit_code"]]

    sns.violinplot(data=df, x='exit_code', y='run_time', ax=axes[1, 0])
    axes[1, 0].set_title('Run Time Distribution by Exit Code')

    sns.boxplot(data=df, x='hostname', y='run_time', ax=axes[1, 1])
    axes[1, 1].set_title('Run Time by Hostname')

    if args.save_to_file:
        helpers.save_to_file(fig, args, plt)
    else: # pragma: no cover
        window_title = f'Times and exit codes for {args.run_dir}'
        if fig is not None and fig.canvas is not None and fig.canvas.manager is not None:
            fig.canvas.manager.set_window_title(window_title)
            if not args.no_plt_show:
                plt.show()

if __name__ == "__main__":
    main()
