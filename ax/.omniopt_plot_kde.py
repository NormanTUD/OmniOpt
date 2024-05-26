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

def plot_kde_plots(dataframe, main_frame):
    exclude_columns = ['trial_index', 'arm_name', 'trial_status', 'generation_method', 'result']
    numeric_columns = [col for col in dataframe.select_dtypes(include=['float64', 'int64']).columns if col not in exclude_columns]
    
    for col in numeric_columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.kdeplot(data=dataframe, x=col, ax=ax, fill=True)
        ax.set_title(f'Kernel Density Estimation for {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Density')
        
        canvas = FigureCanvasTkAgg(fig, master=main_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

def update_graph():
    pd_csv = args.run_dir + "/pd.csv"
    try:
        dataframe = pd.read_csv(pd_csv)

        main_frame = ttk.Frame(root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        plot_kde_plots(dataframe, main_frame)

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

