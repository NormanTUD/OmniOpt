import os
import sys
import signal
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import logging
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

root = None

def _quit():
    global root
    if root:
        root.quit()
        root.destroy()

signal.signal(signal.SIGINT, signal.SIG_DFL)

def plot_boxplot(dataframe, axis):
    axis.clear()
    sns.boxplot(x='generation_method', y='result', data=dataframe, ax=axis)
    axis.set_title('Results by Generation Method')
    axis.set_xlabel('Generation Method')
    axis.set_ylabel('Result')

def plot_barplot(dataframe, axis):
    axis.clear()
    sns.countplot(x='trial_status', data=dataframe, ax=axis)
    axis.set_title('Distribution of Trial Status')
    axis.set_xlabel('Trial Status')
    axis.set_ylabel('Count')

def plot_correlation_matrix(dataframe, axis):
    axis.clear()
    exclude_columns = ['trial_index', 'arm_name', 'trial_status', 'generation_method']
    numeric_columns = dataframe.select_dtypes(include=['float64', 'int64']).columns
    numeric_columns = [col for col in numeric_columns if col not in exclude_columns]
    correlation_matrix = dataframe[numeric_columns].corr()

    # Remove existing legend if it exists
    legend = axis.get_legend()
    if legend:
        legend.remove()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=axis, cbar=False)
    axis.set_title('Correlation Matrix')

def plot_distribution_by_generation(dataframe, axis):
    axis.clear()
    histogram = sns.histplot(data=dataframe, x='result', hue='generation_method', multiple="stack", kde=True, bins=args.bins, ax=axis)
    for patch in histogram.patches:
        patch.set_alpha(0.5)
    axis.set_title('Distribution of Results by Generation Method')
    axis.set_xlabel('Result')
    axis.set_ylabel('Frequency')

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Plotting tool for analyzing trial data.')
    parser.add_argument('--min', type=float, help='Minimum value for result filtering')
    parser.add_argument('--max', type=float, help='Maximum value for result filtering')
    parser.add_argument('--save_to_file', nargs='?', const='plot', type=str, help='Path to save the plot(s)')
    parser.add_argument('--exclude_params', action='append', nargs='+', help="Params to be ignored", default=[])
    parser.add_argument('--run_dir', type=str, help='Path to a CSV file', required=True)
    parser.add_argument('--result_column', type=str, help='Name of the result column', default="result")
    parser.add_argument('--delete_temp', help='Delete temp files', action='store_true', default=False)
    parser.add_argument('--darkmode', help='Enable darktheme', action='store_true', default=False)
    parser.add_argument('--print_to_command_line', help='Print plot to command line', action='store_true', default=False)
    parser.add_argument('--single', help='Print plot to command line', action='store_true', default=False)
    parser.add_argument('--bubblesize', type=int, help='Size of the bubbles', default=7)
    parser.add_argument('--merge_with_previous_runs', action='append', nargs='+', help="Run-Dirs to be merged with", default=[])
    parser.add_argument('--plot_type', action='append', nargs='+', help="Params to be ignored", default=[])
    parser.add_argument('--bins', type=int, help='Number of bins for distribution of results', default=10)

    parser.add_argument('--debug', help='Enable debug', action='store_true', default=False)

    return parser.parse_args()

def filter_data(dataframe, min_value=None, max_value=None):
    if min_value is not None:
        dataframe = dataframe[dataframe['result'] >= min_value]
    if max_value is not None:
        dataframe = dataframe[dataframe['result'] <= max_value]
    return dataframe

def update_graph():
    try:
        dataframe = pd.read_csv(pd_csv)

        if args.min is not None or args.max is not None:
            dataframe = filter_data(dataframe, args.min, args.max)

        if dataframe.empty:
            logging.warning("DataFrame is empty after filtering.")
            return

        """
        for axis in axes.flatten():
            axis.clear()
            if axis.get_legend() is not None:
                print(f"removing {axis.get_legend()} legend")
                axis.get_legend().remove()
        """

        plot_boxplot(dataframe, axes[0, 0])
        plot_barplot(dataframe, axes[0, 1])
        plot_correlation_matrix(dataframe, axes[1, 0])
        plot_distribution_by_generation(dataframe, axes[1, 1])

        """
        for axis in axes.flatten():
            if axis.get_legend() is not None:
                print(f"removing {axis.get_legend()} legend B")
                axis.get_legend().remove()
        """

        fig.canvas.draw()
    except FileNotFoundError:
        logging.error("File not found: %s", pd_csv)
    except Exception as exception:
        logging.error("An unexpected error occurred: %s", str(exception))

def on_key_press(event):
    if event.keysym == 'F5':
        update_graph()

if __name__ == "__main__":
    setup_logging()
    args = parse_arguments()

    if not args.run_dir:
        logging.error("--run_dir not specified")
        sys.exit(33)

    if not os.path.exists(args.run_dir):
        logging.error("Specified --run_dir does not exist")
        sys.exit(34)

    pd_csv = os.path.join(args.run_dir, "pd.csv")
    if not os.path.exists(pd_csv):
        logging.error(f"{pd_csv} could not be found")
        sys.exit(35)

    root = tk.Tk()
    root.protocol("WM_DELETE_WINDOW", _quit)
    root.title("General info for run " + args.run_dir)
    root.geometry("800x800")

    main_frame = ttk.Frame(root, padding=10)
    main_frame.pack(fill=tk.BOTH, expand=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.subplots_adjust(left=0.071, bottom=0.07, right=0.983, top=0.926, wspace=0.167, hspace=0.276)

    canvas = FigureCanvasTkAgg(fig, master=main_frame)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    button_frame = ttk.Frame(main_frame)
    button_frame.pack(side=tk.BOTTOM, fill=tk.X)

    update_button = ttk.Button(button_frame, text="Update Graph", command=update_graph)
    update_button.pack(side=tk.LEFT, padx=5, pady=0)

    quit_button = ttk.Button(button_frame, text="Quit", command=root.quit)
    quit_button.pack(side=tk.RIGHT, padx=5, pady=0)

    root.bind('<KeyPress>', on_key_press)

    update_graph()
    root.mainloop()
