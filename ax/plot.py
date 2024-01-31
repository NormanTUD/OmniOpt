import sys
import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

# Parse command line arguments
parser = argparse.ArgumentParser(description='Path to CSV file that should be plotted.')
parser.add_argument('--run_dir', type=str, help='Path to a CSV file', required=True)
parser.add_argument('--save_to_file', type=str, help='Save the plot to the specified file', default=None)
args = parser.parse_args()

# Check if the specified directory exists
if not os.path.exists(args.run_dir):
    print(f'The folder {args.run_dir} does not exist.')
    sys.exit(1)

pd_csv = "pd.csv"

# Check if the specified CSV file exists
csv_file_path = os.path.join(args.run_dir, pd_csv)
if not os.path.exists(csv_file_path):
    print(f'The file {csv_file_path} does not exist.')
    sys.exit(1)

# Check if the "--save_to_file" parameter is provided
save_to_file = args.save_to_file is not None

# Load the DataFrame from the CSV file
df = pd.read_csv(csv_file_path)

# Remove specified columns
columns_to_remove = ['trial_index', 'arm_name', 'trial_status', 'generation_method']
df_filtered = df.drop(columns=columns_to_remove)

# Ensure 'result' is not x or y
if 'result' in df_filtered.columns:
    df_filtered = df_filtered.drop(columns='result')

# Create combinations of parameters
parameter_combinations = list(combinations(df_filtered.columns, 2))

# Check if there are non-empty graphs to display
non_empty_graphs = [param_comb for param_comb in parameter_combinations if df_filtered[param_comb[0]].notna().any() and df_filtered[param_comb[1]].notna().any()]

if not non_empty_graphs:
    print('No non-empty graphs to display.')
    sys.exit(1)

# Matplotlib figure creation
num_subplots = len(non_empty_graphs)
num_rows = 2  # Number of rows in the plot
num_cols = (num_subplots + num_rows - 1) // num_rows  # Calculate the number of columns

fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 7))

# Color assignment for points based on 'result'
result_column = 'result'
colors = df[result_column]
if args.run_dir + "/maximize" in os.listdir(args.run_dir):
    colors = -colors  # Negate colors for maximum result
norm = plt.Normalize(colors.min(), colors.max())
cmap = plt.cm.viridis

# Loop over non-empty combinations and create 2D plots
for i, (param1, param2) in enumerate(non_empty_graphs):
    row = i // num_cols
    col = i % num_cols
    scatter = axs[row, col].scatter(df_filtered[param1], df_filtered[param2], c=colors, cmap=cmap, norm=norm)
    axs[row, col].set_xlabel(param1)
    axs[row, col].set_ylabel(param2)

# Color bar addition
cbar = fig.colorbar(scatter, ax=axs, orientation='vertical', fraction=0.02, pad=0.1)
cbar.set_label('Result', rotation=270, labelpad=15)

# Add title with parameters and result
result_column_values = df[result_column]
extreme_index = result_column_values.idxmax() if args.run_dir + "/maximize" in os.listdir(args.run_dir) else result_column_values.idxmin()
extreme_values = df_filtered.loc[extreme_index].to_dict()

title = "Minimum"
if args.run_dir + "/maximize" in os.listdir(args.run_dir):
    title = "Maximum"

title += " of f("
title += ', '.join([f"{key} = {value}" for key, value in extreme_values.items()])
title += f") at {result_column_values[extreme_index]}"

# Set the title for the figure
fig.suptitle(title)

# Show the plot or save it to a file based on the command line argument
if save_to_file:
    plt.savefig(args.save_to_file)
else:
    plt.show()
