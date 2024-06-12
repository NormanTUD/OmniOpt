# DESCRIPTION: Plot time and exit code infos
# EXPECTED FILES: job_infos.csv
# TEST_OUTPUT_MUST_CONTAIN: Run Time Distribution
# TEST_OUTPUT_MUST_CONTAIN: Run Time by Hostname
# TEST_OUTPUT_MUST_CONTAIN: Distribution of Run Time
# TEST_OUTPUT_MUST_CONTAIN: Result over Time

import pandas as pd
import argparse
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint
import pytz
from tzlocal import get_localzone
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

def dier(msg):
    pprint(msg)
    sys.exit(1)

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

    _job_infos_csv = f'{args.run_dir}/job_infos.csv'

    if not os.path.exists(_job_infos_csv):
        print(f"Error: {_job_infos_csv} not found")
        sys.exit(1)

    df = pd.read_csv(_job_infos_csv)
    df = df.sort_values(by='exit_code')

    fig, axes = plt.subplots(2, 2, figsize=(20, 30))

    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    axes[0, 0].hist(df['run_time'], bins=args.bins)
    axes[0, 0].set_title('Distribution of Run Time')
    axes[0, 0].set_xlabel('Run Time')
    axes[0, 0].set_ylabel(f'Number of jobs in this runtime ({args.bins} bins)')

    local_tz = get_localzone()

    df['start_time'] = pd.to_datetime(df['start_time'], unit='s', utc=True).dt.tz_convert(local_tz)
    df['end_time'] = pd.to_datetime(df['end_time'], unit='s', utc=True).dt.tz_convert(local_tz)

    df['start_time'] = df['start_time'].apply(lambda x: datetime.utcfromtimestamp(int(float(x))).strftime('%Y-%m-%d %H:%M:%S') if looks_like_number(x) else x)
    df['end_time'] = df['start_time'].apply(lambda x: datetime.utcfromtimestamp(int(float(x))).strftime('%Y-%m-%d %H:%M:%S') if looks_like_number(x) else x)

    sns.scatterplot(data=df, x='start_time', y='result', marker='o', label='Start Time', ax=axes[0, 1])
    sns.scatterplot(data=df, x='end_time', y='result', marker='x', label='End Time', ax=axes[0, 1])
    axes[0, 1].legend()
    axes[0, 1].set_title('Result over Time')

    df["exit_code"] = [str(int(x)) for x in df["exit_code"]]

    sns.violinplot(data=df, x='exit_code', y='run_time', ax=axes[1, 0])
    axes[1, 0].set_title('Run Time Distribution by Exit Code')

    sns.boxplot(data=df, x='hostname', y='run_time', ax=axes[1, 1])
    axes[1, 1].set_title('Run Time by Hostname')

    if args.save_to_file:
        _path = os.path.dirname(args.save_to_file)
        if _path:
            os.makedirs(_path, exist_ok=True)

        plt.savefig(args.save_to_file)
    else:
        window_title = f'Times and exit codes for {args.run_dir}'
        fig.canvas.manager.set_window_title(window_title)
        if not args.no_plt_show:
            plt.show()

if __name__ == "__main__":
    main()
