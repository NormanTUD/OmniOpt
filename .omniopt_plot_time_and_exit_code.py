# DESCRIPTION: Plot time and exit code infos
# TITLE: Plot time and exit code infos
# FULL_DESCRIPTION: <p>This graph has 4 subgraphs that show different information regarding the job runtime, it's results and it's exit codes.</p><ul><li><i>Distribution of Run Time</i>: This shows you how many jobs had which runtime. The <i>y</i>-Axis shows you the number of jobs in one specific time-bin, while the <i>x</i>-axis shows you the number of seconds that the jobs in those bins ran.</li><li><i>Result over Time</i>: This shows you a distribution of results and when they were started and the results attained, so you can find out how long jobs took and how well their results were. </li><li><i>Run Time Distribution by Exit Code</i>: Every job as an exit code and a run time, and this shows you a violin plot of the runtimes and exit-code distribution of a job. It may be helpful when larger jobs fail to find out how long they need until they fail.</li><li><i>Run Time by Hostname</i>: Shows a boxplot of runtime by each hostname where it ran on. Useful to detect nodes that may execute code slower than other codes or to find out which nodes larger models were scheduled to.</li></ul>
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
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tzlocal import get_localzone

from beartype import beartype

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
else:
    raise ImportError(f"Could not load module from {helpers_file}")

parser = argparse.ArgumentParser(description='Plot worker usage from CSV file')
parser.add_argument('--run_dir', type=str, help='Directory containing worker usage CSV file')
parser.add_argument('--save_to_file', type=str, help='Save the plot to the specified file', default=None)
parser.add_argument('--bins', type=int, help='Number of bins for distribution of results (useless here)', default=10)
parser.add_argument('--no_plt_show', help='Disable showing the plot', action='store_true', default=False)
args = parser.parse_args()

@beartype
def load_from_csv(filepath: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found")
        sys.exit(1)

    try:
        return pd.read_csv(filepath)
    except pd.errors.EmptyDataError:
        handle_error(f"Could not find values in file {filepath}", 19)
    except UnicodeDecodeError:
        handle_error(f"{filepath} seems to be invalid utf8.", 7)

    return None

@beartype
def handle_error(errmsg: str, exit_code: int) -> None:
    if not os.environ.get("NO_NO_RESULT_ERROR"):
        print(errmsg)
    sys.exit(exit_code)

@beartype
def validate_dataframe(df: pd.DataFrame) -> None:
    if "run_time" not in df:
        if not os.environ.get("NO_NO_RESULT_ERROR"):
            print("Error: run_time not in df. Probably the job_infos.csv file is corrupted.")
        sys.exit(2)

@beartype
def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(by='exit_code')

    local_tz = get_localzone()
    df['start_time'] = pd.to_datetime(df['start_time'], unit='s', utc=True).dt.tz_convert(local_tz)
    df['end_time'] = pd.to_datetime(df['end_time'], unit='s', utc=True).dt.tz_convert(local_tz)

    df['start_time'] = df['start_time'].apply(format_timestamp)
    df['end_time'] = df['end_time'].apply(format_timestamp)

    df["exit_code"] = df["exit_code"].astype(int).astype(str)

    return df

@beartype
def format_timestamp(value: object) -> str:
    if helpers.looks_like_number(str(value)):
        int_val = int(str(value))
        return datetime.utcfromtimestamp(int_val).strftime('%Y-%m-%d %H:%M:%S')

    return str(value)

@beartype
def plot_histogram(df: pd.DataFrame, axes: plt.Axes, bins: int) -> None:
    axes.hist(df['run_time'], bins=bins)
    axes.set_title('Distribution of Run Time')
    axes.set_xlabel('Run Time')
    axes.set_ylabel(f'Number of jobs in this runtime ({bins} bins)')

@beartype
def plot_time_scatter(df: pd.DataFrame, axes: plt.Axes) -> None:
    res_col_name = helpers.get_result_name_or_default_from_csv_file_path(args.run_dir + "/results.csv")
    sns.scatterplot(data=df, x='start_time', y=res_col_name, marker='o', label='Start Time', ax=axes)
    sns.scatterplot(data=df, x='end_time', y=res_col_name, marker='x', label='End Time', ax=axes)
    axes.set_title('Result over Time')

@beartype
def plot_violinplot(df: pd.DataFrame, axes: plt.Axes) -> None:
    sns.violinplot(data=df, x='exit_code', y='run_time', ax=axes)
    axes.set_title('Run Time Distribution by Exit Code')

@beartype
def plot_boxplot(df: pd.DataFrame, axes: plt.Axes) -> None:
    sns.boxplot(data=df, x='hostname', y='run_time', ax=axes)
    axes.set_title('Run Time by Hostname')

@beartype
def create_plots(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(20, 30))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    plot_histogram(df, axes[0, 0], args.bins)
    plot_time_scatter(df, axes[0, 1])
    plot_violinplot(df, axes[1, 0])
    plot_boxplot(df, axes[1, 1])

    return fig

@beartype
def handle_output(fig: plt.Figure) -> None:
    if args.save_to_file:
        helpers.save_to_file(fig, args, plt)
    else:
        window_title = f'Times and exit codes for {args.run_dir}'
        if fig.canvas.manager is not None:
            fig.canvas.manager.set_window_title(window_title)
            if not args.no_plt_show:
                plt.show()

@beartype
def main() -> None:
    _job_infos_csv: str = f'{args.run_dir}/job_infos.csv'
    df: Optional[pd.DataFrame] = load_from_csv(_job_infos_csv)

    if df is not None:
        validate_dataframe(df)
        df = preprocess_dataframe(df)

        fig = create_plots(df)
        handle_output(fig)
    else:
        print("df was empty")

if __name__ == "__main__":
    main()
