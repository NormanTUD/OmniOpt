import sys
import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

parser = argparse.ArgumentParser(description='Path to CSV file that should be plotted.')
parser.add_argument('--csv_file', type=str, help='Path to a CSV file', required=True)
parser.add_argument('--maximum', action='store_true', help='Display maximum result (default: minimum)')
args = parser.parse_args()

if not os.path.exists(args.csv_file):
    print(f'Die Datei {args.csv_file} existiert nicht.')
    sys.exit(1)

# Load the DataFrame from the CSV file
df = pd.read_csv(args.csv_file)

# Remove specified columns
columns_to_remove = ['trial_index', 'arm_name', 'trial_status', 'generation_method']
df_filtered = df.drop(columns=columns_to_remove)

# Ensure 'result' is not x or y
if 'result' in df_filtered.columns:
    df_filtered = df_filtered.drop(columns='result')

# Create combinations of parameters
parameter_combinations = list(combinations(df_filtered.columns, 2))

# Matplotlib figure creation
num_subplots = len(parameter_combinations)
num_rows = 2  # Number of rows in the plot
num_cols = (num_subplots + num_rows - 1) // num_rows  # Calculate the number of columns

fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 7))

# Color assignment for points based on 'result'
result_column = 'result'
colors = df[result_column]
if args.maximum:
    colors = -colors  # Negate colors for maximum result
norm = plt.Normalize(colors.min(), colors.max())
cmap = plt.cm.viridis

# Loop over combinations and create 2D plots
for i, (param1, param2) in enumerate(parameter_combinations):
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
extreme_index = result_column_values.idxmax() if args.maximum else result_column_values.idxmin()
extreme_values = df_filtered.loc[extreme_index].to_dict()

title = "Minimum"
if args.maximum:
    title = "Maximum"

title += " of f("
title += ', '.join([f"{key} = {value}" for key, value in extreme_values.items()])
title += f") at {result_column_values[extreme_index]}"
fig.suptitle(title)

plt.show()
