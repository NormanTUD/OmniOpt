# DESCRIPTION: Kernel-Density estimation plot
# EXPECTED FILES: pd.csv

import numpy as np
import math
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Plotting tool for analyzing trial data.')
    parser.add_argument('--run_dir', type=str, help='Path to a run dir', required=True)
    parser.add_argument('--bins', type=int, help='Number of bins for distribution of results', default=10)
    parser.add_argument('--plot_type', action='append', nargs='+', help="Params to be ignored", default=[])
    parser.add_argument('--alpha', type=float, help='Transparency of plot bars', default=0.5)
    parser.add_argument('--no_legend', help='Disables legend', action='store_true', default=False)
    parser.add_argument('--debug', help='Enable debug', action='store_true', default=False)
    parser.add_argument('--save_to_file', type=str, help='Save the plot to the specified file', default=None)
    parser.add_argument('--darkmode', help='Enable darktheme (useless here)', action='store_true', default=False)
    parser.add_argument('--bubblesize', type=int, help='Size of the bubbles (useless here)', default=7)
    parser.add_argument('--merge_with_previous_runs', action='append', nargs='+', help="Run-Dirs to be merged with (useless here)", default=[])
    parser.add_argument('--delete_temp', help='Delete temp files (useless here)', action='store_true', default=False)
    parser.add_argument('--result_column', type=str, help='Name of the result column (useless here)', default="result")
    parser.add_argument('--single', help='Print plot to command line (useless here)', action='store_true', default=False)
    parser.add_argument('--print_to_command_line', help='Print plot to command line (useless here)', action='store_true', default=False)
    parser.add_argument('--min', type=float, help='Minimum value for result filtering (useless here)')
    parser.add_argument('--max', type=float, help='Maximum value for result filtering (useless here)')
    parser.add_argument('--exclude_params', action='append', nargs='+', help="Params to be ignored (useless here)", default=[])
    return parser.parse_args()

def plot_histograms(dataframe, save_to_file=None):
    exclude_columns = ['trial_index', 'arm_name', 'trial_status', 'generation_method', 'result']
    numeric_columns = [col for col in dataframe.select_dtypes(include=['float64', 'int64']).columns if col not in exclude_columns]

    num_plots = len(numeric_columns)
    num_rows = 1
    num_cols = num_plots

    if num_plots > 1:
        num_rows = int(num_plots ** 0.5)
        num_cols = int(math.ceil(num_plots / num_rows))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))
    axes = axes.flatten()

    for i, col in enumerate(numeric_columns):
        ax = axes[i]
        values = dataframe[col]
        result_values = dataframe['result']
        bin_edges = np.linspace(result_values.min(), result_values.max(), args.bins + 1)  # Divide the range into 10 equal bins
        colormap = plt.get_cmap('RdYlGn_r')

        for j in range(args.bins):
            color = colormap(j / 9)  # Calculate color based on colormap
            bin_mask = (result_values >= bin_edges[j]) & (result_values <= bin_edges[j+1])
            bin_range = f'{bin_edges[j]:.2f}-{bin_edges[j+1]:.2f}'
            ax.hist(values[bin_mask], bins=args.bins, alpha=args.alpha, color=color, label=f'{bin_range}')

        ax.set_title(f'Histogram for {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        if not args.no_legend:
            ax.legend(loc='upper right')

    # Hide any unused subplots
    for j in range(num_plots, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()

    if save_to_file:
        plt.savefig(save_to_file)
    else:
        plt.show()

def update_graph():
    pd_csv = args.run_dir + "/pd.csv"
    try:
        dataframe = pd.read_csv(pd_csv)
        plot_histograms(dataframe, args.save_to_file)
    except FileNotFoundError:
        logging.error("File not found: %s", pd_csv)
    except Exception as exception:
        logging.error("An unexpected error occurred: %s", str(exception))

if __name__ == "__main__":
    setup_logging()
    args = parse_arguments()

    if not args.alpha:
        logging.error("--alpha cannot be left unset.")
        sys.exit(2)

    if args.alpha > 1 or args.alpha < 0:
        logging.error("--alpha must between 0 and 1")
        sys.exit(3)

    if not os.path.exists(args.run_dir):
        logging.error("Specified --run_dir does not exist")
        sys.exit(1)

    update_graph()

