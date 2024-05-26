import numpy as np
import math
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

signal.signal(signal.SIGINT, signal.SIG_DFL)

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Plotting tool for analyzing trial data.')
    parser.add_argument('--run_dir', type=str, help='Path to a CSV file', required=True)
    return parser.parse_args()

def plot_histograms(dataframe, main_frame):
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
        bin_edges = np.linspace(result_values.min(), result_values.max(), 11)  # Divide the range into 10 equal bins
        colormap = plt.cm.get_cmap('RdYlGn_r')  # Reverse RdYlGn colormap

        for j in range(10):
            color = colormap(j / 9)  # Calculate color based on colormap
            bin_mask = (result_values >= bin_edges[j]) & (result_values <= bin_edges[j+1])
            bin_range = f'{bin_edges[j]:.2f}-{bin_edges[j+1]:.2f}'
            ax.hist(values[bin_mask], bins=10, alpha=0.7, color=color, label=f'{bin_range}')

        ax.set_title(f'Histogram for {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        ax.legend(loc='upper right')

    # Hide any unused subplots
    for j in range(num_plots, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=main_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

def update_graph():
    pd_csv = args.run_dir + "/pd.csv"
    try:
        dataframe = pd.read_csv(pd_csv)

        main_frame = ttk.Frame(root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        plot_histograms(dataframe, main_frame)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        canvas = FigureCanvasTkAgg(plt.figure(), master=main_frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas.get_tk_widget().config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=canvas.get_tk_widget().yview)

        root.mainloop()
    except FileNotFoundError:
        logging.error("File not found: %s", pd_csv)
    except Exception as exception:
        logging.error("An unexpected error occurred: %s", str(exception))

if __name__ == "__main__":
    setup_logging()
    args = parse_arguments()

    if not os.path.exists(args.run_dir):
        logging.error("Specified --run_dir does not exist")
        sys.exit(1)

    root = tk.Tk()
    root.title(f"KDE Plot for {args.run_dir}")
    root.geometry("800x600")

    update_graph()

