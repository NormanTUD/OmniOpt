import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from itertools import combinations
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

def print_debug(message):
    global args
    if args.debug:
        print("DEBUG: ", end="")
        print(message)

def to_int_when_possible(value):
    print_debug("to_int_when_possible")
    if isinstance(value, (int, float)) or (isinstance(value, str) and value.isdigit()):
        return int(value)
    if isinstance(value, str):
        try:
            value = float(value)
            return '{:.{}f}'.format(value, len(str(value).split('.')[1])).rstrip('0').rstrip('.')
        except:
            return value
    return value

def set_margins(fig):
    print_debug("set_margins()")
    fig.subplots_adjust(left=0.06, bottom=0.1, right=0.96, top=0.95, wspace=0.3, hspace=0.3)

def contains_strings(series):
    return series.apply(lambda x: isinstance(x, str)).any()

def get_data(csv_file_path, result_column, minimum, maximum):
    print_debug("get_data")
    try:
        dataframe = pd.read_csv(csv_file_path, index_col=0)
        if minimum is not None:
            dataframe = dataframe[dataframe[result_column] >= minimum]
        if maximum is not None:
            dataframe = dataframe[dataframe[result_column] <= maximum]
        if result_column not in dataframe:
            print(f"There was no {result_column} in {csv_file_path}.")
            sys.exit(10)
        dataframe.dropna(subset=[result_column], inplace=True)
        columns_with_strings = [col for col in dataframe.columns if contains_strings(dataframe[col])]
        dataframe = dataframe.drop(columns=columns_with_strings)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(5)
    return dataframe

def get_args():
    parser = argparse.ArgumentParser(description='Plot optimization runs.', prog="plot")
    parser.add_argument('--run_dir', type=str, help='Path to a CSV file', required=True)
    parser.add_argument('--save_to_file', type=str, help='Save the plot to the specified file', default=None)
    parser.add_argument('--max', type=float, help='Maximum value', default=None)
    parser.add_argument('--min', type=float, help='Minimum value', default=None)
    parser.add_argument('--result_column', type=str, help='Name of the result column', default="result")
    parser.add_argument('--debug', help='Enable debug', action='store_true', default=False)
    return parser.parse_args()

def plot_contour(ax, df, param1, param2, result_column):
    print_debug("plot_contour")
    x = df[param1].values
    y = df[param2].values
    z = df[result_column].values

    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    xi, yi = np.meshgrid(xi, yi)
    #zi = griddata((x, y), z, (xi, yi), method='cubic')
    zi = griddata((x, y), z, (xi, yi), method='linear')

    count, xedges, yedges = np.histogram2d(x, y, bins=5)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    ax.imshow(count.T, extent=extent, origin='lower', cmap='Blues', alpha=0.6)

    contour = ax.contour(xi, yi, zi, 15, linewidths=0.5, colors='k')
    contourf = ax.contourf(xi, yi, zi, 15, cmap='RdBu_r')

    ax.set_xlabel(param1)
    ax.set_ylabel(param2)
    plt.colorbar(contourf, ax=ax, label=result_column)

def handle_sigint(signal, frame):
    print("Interrupted! Closing the plot.")
    plt.close('all')
    sys.exit(0)

def main():
    global args
    args = get_args()

    csv_file_path = os.path.join(args.run_dir, "results.csv")
    df = get_data(csv_file_path, args.result_column, args.min, args.max)
    if df is None:
        sys.exit(1)

    parameters = df.columns.tolist()
    parameters = [p for p in parameters if p not in ['trial_index', 'arm_name', 'trial_status', 'generation_method', args.result_column]]

    if len(parameters) < 2:
        print("Not enough parameters to plot.")
        sys.exit(1)

    param_combinations = list(combinations(parameters, 2))
    n_combinations = len(param_combinations)
    n_cols = 2
    n_rows = (n_combinations + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6 * n_rows))
    axes = axes.flatten()

    for idx, (param1, param2) in enumerate(param_combinations):
        plot_contour(axes[idx], df, param1, param2, args.result_column)
    
    for ax in axes[n_combinations:]:
        fig.delaxes(ax)

    set_margins(fig)
    fig.suptitle("Parameter Combinations Plot", fontsize=16)

    if args.save_to_file:
        plt.savefig(args.save_to_file)
        print(f"Plot saved to {args.save_to_file}")
    else:
        plt.show()

if __name__ == "__main__":
    main()

