val_if_nothing_found = 99999999999999999999999999999999999999999999999999999999999
NO_RESULT = "{:.0e}".format(val_if_nothing_found)

import sys
import os
import argparse
import math

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

console = None

try:
    from rich.pretty import pprint
    from rich.console import Console
    console = Console(force_terminal=True, force_interactive=True)
except ModuleNotFoundError:
    from pprint import pprint

def dier(msg):
    pprint(msg)
    sys.exit(9)

import importlib.util
spec = importlib.util.spec_from_file_location(
    name="helpers",
    location=".helpers.py",
)
my_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(my_module)

try:
    with console.status("[bold green]Importing re") as status:
        import re
    with console.status("[bold green]Importing pandas") as status:
        import pandas as pd

    with console.status("[bold green]Importing matplotlib") as status:
        import matplotlib
        import matplotlib.pyplot as plt

    with console.status("[bold green]Importing combinations from itertools") as status:
        from itertools import combinations
except ModuleNotFoundError as e:
    print(f"Error: {e}")
    sys.exit(0)

# Get shell variables or use default values
BUBBLESIZEINPX = int(os.environ.get('BUBBLESIZEINPX', 10))
SCIENTIFICNOTATION = int(os.environ.get('SCIENTIFICNOTATION', 2))

def to_int_when_possible (val):
    if type(val) == int or (type(val) == float and val.is_integer()) or (type(val) == str and val.isdigit()):
        return int(val)
    if type(val) == str and re.match(r'^-?\d+(?:\.\d+)$', val) is None:
        return val
    return '{:f}'.format(val)

def main():
    print("DONELOADING")
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Path to CSV file that should be plotted.', prog="plot")

    parser.add_argument('--run_dir', type=str, help='Path to a CSV file', required=True)
    parser.add_argument('--save_to_file', type=str, help='Save the plot to the specified file', default=None)
    parser.add_argument('--max', type=float, help='Maximum value', default=None)
    parser.add_argument('--min', type=float, help='Minimum value', default=None)
    parser.add_argument('--result_column', type=str, help='Name of the result column', default="result")
    parser.add_argument('--debug', help='Enable debugging', action='store_true', default=False)

    args = parser.parse_args()

    if args.min and args.max:
        if args.min > args.max:
            tmp = args.max
            args.max = args.min
            args.min = tmp
        elif args.min == args.max:
            print("Max and min value are the same. May result in empty data")

    result_column = os.getenv("OO_RESULT_COLUMN_NAME", args.result_column)

    if not args.save_to_file:
        matplotlib.use('TkAgg')

    # Check if the specified directory exists
    if not os.path.exists(args.run_dir):
        print(f'The folder {args.run_dir} does not exist.')
        sys.exit(1)

    pd_csv = "pd.csv"

    # Check if the specified CSV file exists
    csv_file_path = os.path.join(args.run_dir, pd_csv)
    if not os.path.isdir(args.run_dir):
        print(f"The path {args.run_dir} does not point to a folder. Must be a folder.")
        sys.exit(11)


    if not os.path.exists(csv_file_path):
        print(f'The file {csv_file_path} does not exist.')
        sys.exit(10)

    # Load the DataFrame from the CSV file
    df = None
    nr_of_items_before_filtering = 0
    try:
        df = pd.read_csv(csv_file_path, index_col=0)

        nr_of_items_before_filtering = len(df)

        if args.min is not None:
            f = df[result_column] >= args.min
            df.where(f, inplace=True)
        if args.max is not None:
            f = df[result_column] <= args.max
            df.where(f, inplace=True)
        df.dropna(subset=[result_column], inplace=True)
    except pd.errors.EmptyDataError:
        print(f"{csv_file_path} has no lines to parse.")
        sys.exit(5)
    except pd.errors.ParserError as e:
        print(f"{csv_file_path} is invalid CSV. Parsing error: {str(e).rstrip()}")
        sys.exit(12)
    except UnicodeDecodeError:
        print(f"{csv_file_path} does not seem to be a text-file or it has invalid UTF8 encoding.")
        sys.exit(7)

    nr_of_items_before_removing_NO_RESULT = len(df)

    try:
        negative_rows_to_remove = df[df[result_column].astype(str) == '-' + NO_RESULT].index
        positive_rows_to_remove = df[df[result_column].astype(str) == NO_RESULT].index

        # Entferne die Zeilen mit den spezifischen Werten
        df.drop(negative_rows_to_remove, inplace=True)
        df.drop(positive_rows_to_remove, inplace=True)
    except KeyError:
        print(f"column named `{result_column}` could not be found in {csv_file_path}.")
        sys.exit(6)

    # Remove specified columns
    all_columns_to_remove = ['trial_index', 'arm_name', 'trial_status', 'generation_method']
    columns_to_remove = []
    existing_columns = df.columns.values.tolist()

    for col in existing_columns:
        if col in all_columns_to_remove:
            columns_to_remove.append(col)


    df_filtered = df.drop(columns=columns_to_remove)

    #if result_column in df_filtered.columns:
    #    df_filtered = df_filtered.drop(columns=result_column)

    num_entries = len(df_filtered)

    if num_entries is None or num_entries == 0:
        if nr_of_items_before_filtering:
            if args.min and not args.max:
                print(f"Using --min filtered out all results")
            elif not args.min and args.max:
                print(f"Using --max filtered out all results")
            elif args.min and args.max:
                print(f"Using --min and --max filtered out all results")
            else:
                print(f"For some reason, there were values in the beginning but not after filtering")
        else:
            if nr_of_items_before_removing_NO_RESULT:
                print(f"All result entries are {NO_RESULT} (the value meaning execution failed).")
            else:
                print(f"No applicable values could be found in {csv_file_path}.")
        sys.exit(4)

    # Create combinations of parameters
    r = 2

    if len(list(df_filtered.columns)) == 1:
        r = 1

    parameter_combinations = list(combinations(df_filtered.columns, r))

    if len(parameter_combinations) == 1:
        param = parameter_combinations[0][0]
        if df_filtered[param].notna().any():
            non_empty_graphs = [(param,)]
        else:
            non_empty_graphs = []
    else:
        non_empty_graphs = [param_comb for param_comb in parameter_combinations if df_filtered[param_comb[0]].notna().any() and df_filtered[param_comb[1]].notna().any()]


    if not non_empty_graphs:
        print('No non-empty graphs to display.')
        sys.exit(2)

    num_subplots = len(non_empty_graphs)

    # Calculate optimal number of rows and columns for subplot layout
    num_cols = math.ceil(math.sqrt(num_subplots))
    num_rows = math.ceil(num_subplots / num_cols)

    # Matplotlib figure creation with adjusted subplot layout
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15*num_cols, 7*num_rows))

    try:
        colors = df[result_column]
    except KeyError as e:
        if str(e) == "'" + result_column + "'":
            print(f"Could not find any results in {csv_file_path}")
            sys.exit(3)
        else:
            print(f"Key-Error: {e}")
            sys.exit(8)


    if args.run_dir + "/maximize" in os.listdir(args.run_dir):
        colors = -colors  # Negate colors for maximum result

    norm = None
    try:
        norm = plt.Normalize(colors.min(), colors.max())
    except:
        print(f"Wrong values in {csv_file_path}")
        sys.exit(16)

    cmap = plt.cm.viridis

    # Loop über non-empty combinations und Erstellung von 2D-Plots
    if num_subplots == 1:
        if len(non_empty_graphs[0]) == 1:
            ax = axs  # Use the single axis
            _range = range(len(df_filtered))
            _data = df_filtered

            _data = _data[:].values

            _x = []
            _y = []

            for l in _data:
                _x.append(l[0])
                _y.append(l[1])

            scatter = ax.scatter(_x, _y, c=colors, cmap=cmap, norm=norm, s=BUBBLESIZEINPX)
            ax.set_xlabel(non_empty_graphs[0][0])
            ax.set_ylabel(result_column)

            # Farbgebung und Legende für das einzelne Scatterplot
            cbar = fig.colorbar(scatter, ax=ax, orientation='vertical', fraction=0.02, pad=0.1)
            cbar.set_label(result_column, rotation=270, labelpad=15)
        else:
            scatter = axs.scatter(df_filtered[non_empty_graphs[0][0]], df_filtered[non_empty_graphs[0][1]], c=colors, cmap=cmap, norm=norm, s=BUBBLESIZEINPX)
            axs.set_xlabel(non_empty_graphs[0][0])
            axs.set_ylabel(non_empty_graphs[0][1])
            # Farbgebung und Legende für das einzelne Scatterplot
            cbar = fig.colorbar(scatter, ax=axs, orientation='vertical', fraction=0.02, pad=0.1)
            cbar.set_label(result_column, rotation=270, labelpad=15)
    else:
        for i, (param1, param2) in enumerate(non_empty_graphs):
            row = i // num_cols
            col = i % num_cols
            try:
                scatter = axs[row, col].scatter(df_filtered[param1], df_filtered[param2], c=colors, cmap=cmap, norm=norm, s=BUBBLESIZEINPX)
                axs[row, col].set_xlabel(param1)
                axs[row, col].set_ylabel(param2)
            except Exception as e:
                print(str(e))
                sys.exit(17)

        for i in range(len(parameter_combinations), num_rows*num_cols):
            row = i // num_cols
            col = i % num_cols
            axs[row, col].set_visible(False)

        # Color bar addition für mehrere Subplots
        cbar = fig.colorbar(scatter, ax=axs, orientation='vertical', fraction=0.02, pad=0.1)
        cbar.set_label(result_column, rotation=270, labelpad=15)

    for i in range(len(parameter_combinations), num_rows*num_cols):
        row = i // num_cols
        col = i % num_cols
        axs[row, col].set_visible(False)

    # Add title with parameters and result
    result_column_values = df[result_column]
    filtered_data = list(filter(lambda x: not math.isnan(x), result_column_values.tolist()))
    number_of_non_nan_results = len(filtered_data)

    if number_of_non_nan_results == 0:
        print(f"No values were found. Every evaluation found in {csv_file_path} evaluated to NaN.")
        sys.exit(11)

    #extreme_index = result_column_values.idxmax() if args.run_dir + "/maximize" in os.listdir(args.run_dir) else result_column_values.idxmin()
    extreme_index = result_column_values.idxmin()
    if os.path.exists(args.run_dir + "/maximize"):
        extreme_index = result_column_values.idxmax()

    extreme_values = df_filtered.loc[extreme_index].to_dict()

    title = "Minimum"
    if args.run_dir + "/maximize" in os.listdir(args.run_dir):
        title = "Maximum"

    extreme_values_items = extreme_values.items()

    filtered_extreme_values_items = {}

    title_values = []

    for l in extreme_values_items:
        if not args.result_column in l:
            key = l[0]
            value = to_int_when_possible(l[1])
            title_values.append(f"{key} = {value}")

    #title_values = [f"{key} = {value}" for key, value in filtered_extreme_values_items]

    title += " of f("
    title += ', '.join(title_values)
    title += f") = {to_int_when_possible(result_column_values[extreme_index])}"

    title += f"\nNumber of evaluations shown: {num_entries}"

    if args.min is not None:
        title += f", show min = {to_int_when_possible(args.min)}"

    if args.max is not None:
        title += f", show max = {to_int_when_possible(args.max)}"

    # Set the title for the figure
    fig.suptitle(title)

    left  = 0.125
    right = 0.9
    bottom = 0.12
    top = 0.9
    wspace = 0.2
    hspace = 0.3

    fig.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=wspace)

    # Show the plot or save it to a file based on the command line argument
    if args.save_to_file:
        plt.savefig(args.save_to_file)
    else:
        plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        sys.exit(0)
