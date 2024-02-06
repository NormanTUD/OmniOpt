import sys
import os
import argparse

try:
    from rich.pretty import pprint
except ModuleNotFoundError:
    from pprint import pprint

def dier(msg):
    pprint(msg)
    sys.exit(10)

import importlib.util 
spec = importlib.util.spec_from_file_location(
    name="helpers",
    location=".helpers.py",
)
my_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(my_module)

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    from itertools import combinations
except ModuleNotFoundError as e:
    print(f"Error: {e}")
    sys.exit(0)

# Get shell variables or use default values
BUBBLESIZEINPX = int(os.environ.get('BUBBLESIZEINPX', 10))
SCIENTIFICNOTATION = int(os.environ.get('SCIENTIFICNOTATION', 2))

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Path to CSV file that should be plotted.')

    parser.add_argument('--run_dir', type=str, help='Path to a CSV file', required=True)
    parser.add_argument('--save_to_file', type=str, help='Save the plot to the specified file', default=None)
    parser.add_argument('--max', type=float, help='Maximum value', default=None)
    parser.add_argument('--min', type=float, help='Minimum value', default=None)
    parser.add_argument('--result_column', type=str, help='Name of the result column', default="result")

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
        sys.exit(3)

    # Check if the "--save_to_file" parameter is provided
    save_to_file = args.save_to_file is not None

    # Load the DataFrame from the CSV file
    df = None
    try:
        df = pd.read_csv(csv_file_path)
    except pd.errors.EmptyDataError:
        print(f"{csv_file_path} has no lines to parse.")
        sys.exit(5)
    except pd.errors.ParserError as e:
        print(f"{csv_file_path} is invalid CSV. Parsing error: {str(e).rstrip()}")
        sys.exit(6)

    try:
        negative_rows_to_remove = df[df[args.result_column].astype(str) == '-1e+59'].index
        positive_rows_to_remove = df[df[args.result_column].astype(str) == '1e+59'].index

        # Entferne die Zeilen mit den spezifischen Werten
        df.drop(negative_rows_to_remove, inplace=True)
        df.drop(positive_rows_to_remove, inplace=True)
    except KeyError:
        print(f"column named `{args.result_column}` could not be found in {csv_file_path}.")
        sys.exit(6)

    # Remove specified columns
    all_columns_to_remove = ['trial_index', 'arm_name', 'trial_status', 'generation_method']
    columns_to_remove = []
    existing_columns = df.columns.values.tolist()

    for col in existing_columns:
        if col in all_columns_to_remove:
            columns_to_remove.append(col)

    df_filtered = df.drop(columns=columns_to_remove)

    if args.result_column in df_filtered.columns:
        df_filtered = df_filtered.drop(columns=args.result_column)

    num_entries = len(df_filtered)

    if num_entries is None or num_entries == 0:
        sys.exit(4)

    # Create combinations of parameters
    parameter_combinations = list(combinations(df_filtered.columns, 2))

    # Check if there are non-empty graphs to display
    non_empty_graphs = [param_comb for param_comb in parameter_combinations if df_filtered[param_comb[0]].notna().any() and df_filtered[param_comb[1]].notna().any()]

    if not non_empty_graphs:
        print('No non-empty graphs to display.')
        sys.exit(2)

    if args.min is not None:
        df[args.result_column] = df[args.result_column].clip(lower=args.min)
    if args.max is not None:
        df[args.result_column] = df[args.result_column].clip(upper=args.max)

    # Matplotlib figure creation
    num_subplots = len(non_empty_graphs)
    num_rows = 2  # Number of rows in the plot
    num_cols = (num_subplots + num_rows - 1) // num_rows  # Calculate the number of columns

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 7))

    try:
        colors = df[args.result_column]
    except KeyError as e:
        if str(e) == "'" + args.result_column + "'":
            print(f"Could not find any results in {csv_file_path}")
            sys.exit(3)
        else:
            print(f"Key-Error: {e}")
            sys.exit(4)


    if args.run_dir + "/maximize" in os.listdir(args.run_dir):
        colors = -colors  # Negate colors for maximum result
    norm = plt.Normalize(colors.min(), colors.max())
    cmap = plt.cm.viridis

    # Loop over non-empty combinations and create 2D plots
    for i, (param1, param2) in enumerate(non_empty_graphs):
        row = i // num_cols
        col = i % num_cols
        scatter = axs[row, col].scatter(df_filtered[param1], df_filtered[param2], c=colors, cmap=cmap, norm=norm, s=BUBBLESIZEINPX)
        axs[row, col].set_xlabel(param1)
        axs[row, col].set_ylabel(param2)

    # Color bar addition
    cbar = fig.colorbar(scatter, ax=axs, orientation='vertical', fraction=0.02, pad=0.1)
    cbar.set_label(args.result_column, rotation=270, labelpad=15)

    # Add title with parameters and result
    result_column_values = df[args.result_column]
    extreme_index = result_column_values.idxmax() if args.run_dir + "/maximize" in os.listdir(args.run_dir) else result_column_values.idxmin()
    extreme_values = df_filtered.loc[extreme_index].to_dict()

    title = "Minimum"
    if args.run_dir + "/maximize" in os.listdir(args.run_dir):
        title = "Maximum"

    title += " of f("
    title += ', '.join([f"{key} = {value}" for key, value in extreme_values.items()])
    title += f") at {result_column_values[extreme_index]}"

    title += f"\nNumber of evaluations shown: {num_entries}"

    if args.min is not None:
        title += f", minimum value: {args.min}"

    if args.max is not None:
        title += f", maximum value: {args.max}"

    # Set the title for the figure
    fig.suptitle(title)

    # Show the plot or save it to a file based on the command line argument
    if save_to_file:
        plt.savefig(args.save_to_file)
    else:
        plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        sys.exit(0)
