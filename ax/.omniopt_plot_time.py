# DESCRIPTION: Plot time infos
# EXPECTED FILES: job_infos.csv

import argparse
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint

def dier(msg):
    pprint(msg)
    sys.exit(1)

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
    parser.add_argument('--print_to_command_line', help='Print plot to command line (useless here)', action='store_true', default=False)
    parser.add_argument('--exclude_params', action='append', nargs='+', help="Params to be ignored (useless here)", default=[])
    parser.add_argument('--single', help='Print plot to command line (useless here)', action='store_true', default=False)
    args = parser.parse_args()

    _job_infos_csv = f'{args.run_dir}/job_infos.csv'

    if not os.path.exists(_job_infos_csv):
        print(f"Error: {_job_infos_csv} not found")
        sys.exit(1)

    df = pd.read_csv(_job_infos_csv)

    fig, axes = plt.subplots(2, 2, figsize=(20, 30))

    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    """
    sns.scatterplot(data=df, x='int_param', y='run_time', hue='exit_code', ax=axes[0, 0])
    axes[0, 0].set_title('Run Time vs. int_param')

    sns.scatterplot(data=df, x='float_param', y='run_time', hue='exit_code', ax=axes[0, 1])
    axes[0, 1].set_title('Run Time vs. float_param')

    sns.scatterplot(data=df, x='choice_param', y='run_time', hue='exit_code', ax=axes[1, 0])
    axes[1, 0].set_title('Run Time vs. choice_param')

    sns.scatterplot(data=df, x='int_param_two', y='run_time', hue='exit_code', ax=axes[1, 1])
    axes[1, 1].set_title('Run Time vs. int_param_two')
    """

    axes[0, 0].hist(df['run_time'], bins=30)
    axes[0, 0].set_title('Distribution of Run Time')
    axes[0, 0].set_xlabel('Run Time')
    axes[0, 0].set_ylabel('Frequency')

    sns.scatterplot(data=df, x='start_time', y='result', marker='o', label='Start Time', ax=axes[0, 1])
    sns.scatterplot(data=df, x='end_time', y='result', marker='x', label='End Time', ax=axes[0, 1])
    axes[0, 1].legend()
    axes[0, 1].set_title('Result over Time')

    #sns.countplot(data=df, x='exit_code', ax=axes[1, 0])
    #axes[1, 0].set_title('Count of Exit Codes')

    #avg_run_time = df.groupby('exit_code')['run_time'].mean().reset_index()
    #sns.barplot(data=avg_run_time, x='exit_code', y='run_time', ax=axes[1, 0])
    #axes[1, 0].set_title('Average Run Time by Exit Code')

    sns.violinplot(data=df, x='exit_code', y='run_time', ax=axes[1, 0])
    axes[1, 0].set_title('Run Time Distribution by Exit Code')


    sns.boxplot(data=df, x='hostname', y='run_time', ax=axes[1, 1])
    axes[1, 1].set_title('Run Time by Hostname')


    """
    sns.boxplot(data=df, x='exit_code', y='int_param', ax=axes[0, 1])
    axes[0, 1].set_title('int_param by Exit Code')

    sns.boxplot(data=df, x='exit_code', y='float_param', ax=axes[1, 0])
    axes[1, 0].set_title('float_param by Exit Code')

    sns.boxplot(data=df, x='exit_code', y='choice_param', ax=axes[1, 1])
    axes[1, 1].set_title('choice_param by Exit Code')
    """

    plt.show()

if __name__ == "__main__":
    main()
