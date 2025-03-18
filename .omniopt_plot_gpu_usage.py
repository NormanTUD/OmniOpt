# DESCRIPTION: Plot GPU usage over time on different hosts
# TITLE: Plot GPU usage over time on different hosts
# FULL_DESCRIPTION: <p>Shows the workload of different GPUs on all nodes that jobs of an evaluation has run on over time.</p>
# EXPECTED FILES: gpu_usage_
# TEST_OUTPUT_MUST_CONTAIN: GPU Usage Over Time
# TEST_OUTPUT_MUST_CONTAIN: pci.bus_id
# TEST_OUTPUT_MUST_CONTAIN: GPU Usage
# TEST_OUTPUT_MUST_CONTAIN: Time

import argparse
import importlib.util
import os
import signal
import sys

import matplotlib.pyplot as plt
import pandas as pd

from beartype import beartype

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--run_dir', type=str, help='Directory where to search for CSV files')

parser.add_argument('--no_legend', help='Disables legend (useless here)', action='store_true', default=False)
parser.add_argument('--save_to_file', nargs='?', const='plot', type=str, help='Path to save the plot(s)')
parser.add_argument('--no_plt_show', help='Disable showing the plot', action='store_true', default=False)

args = None
fig = None

script_dir = os.path.dirname(os.path.realpath(__file__))
helpers_file = f"{script_dir}/.helpers.py"
spec = importlib.util.spec_from_file_location(
    name="helpers",
    location=helpers_file,
)
if spec is not None and spec.loader is not None:
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
else:
    raise ImportError(f"Could not load module from {helpers_file}")

signal.signal(signal.SIGINT, signal.SIG_DFL)

@beartype
def get_names_array() -> list:
    return [
        "timestamp",
        "name",
        "pci.bus_id",
        "driver_version",
        "pstate",
        "pcie.link.gen.max",
        "pcie.link.gen.current",
        "temperature.gpu",
        "utilization.gpu [%]",
        "utilization.memory [%]",
        "memory.total [MiB]",
        "memory.free [MiB]",
        "memory.used [MiB]"
    ]

@beartype
def plot_gpu_usage(run_dir: str) -> None:
    global fig

    gpu_data = []
    num_plots = 0
    plot_rows = 2
    plot_cols = 2  # standard number of columns for subplot grid
    _paths = []
    gpu_data_len = 0

    for root, _, files in os.walk(run_dir):
        for file in files:
            if file.startswith("gpu_usage_") and file.endswith(".csv"):
                file_path = os.path.join(root, file)
                _paths.append(file_path)
                df = pd.read_csv(file_path, names=get_names_array())

                df = df.dropna()
                gpu_data.append(df)
                num_plots += 1
                gpu_data_len += len(df)

    if len(_paths) == 0:
        helpers.print_if_not_plot_tests_and_exit(f"No gpu_usage_*.csv files could be found in {run_dir}", 10)

    if not gpu_data:
        helpers.print_if_not_plot_tests_and_exit("No GPU usage data found.", 44)

    if gpu_data_len < 1:
        helpers.print_if_not_plot_tests_and_exit(f"No valid GPU usage data found (len = {gpu_data_len}).", 19)

    plot_cols = min(num_plots, plot_cols)  # Adjusting number of columns based on available plots
    plot_rows = (num_plots + plot_cols - 1) // plot_cols  # Calculating number of rows based on columns

    fig, axs = plt.subplots(plot_rows, plot_cols, figsize=(10, 5 * plot_rows))
    if num_plots > 1:
        axs = axs.flatten()  # Flatten the axs array to handle both 1D and 2D subplots

    for i, df in enumerate(gpu_data):
        _ax = axs
        try:
            _ax = axs[i]
        except Exception:
            pass
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y/%m/%d %H:%M:%S.%f', errors='coerce')
        df = df.sort_values(by='timestamp')

        grouped_data = df.groupby('pci.bus_id')
        for bus_id, group in grouped_data:
            group['utilization.gpu [%]'] = pd.to_numeric(group['utilization.gpu [%]'].str.replace('%', ''), errors='coerce')
            group = group.dropna(subset=['timestamp', 'utilization.gpu [%]'])
            _ax.scatter(group['timestamp'], group['utilization.gpu [%]'], label=f'pci.bus_id: {bus_id}')

        _ax.set_xlabel('Time')
        _ax.set_ylabel('GPU Usage (%)')
        _ax.set_title(f'GPU Usage Over Time - {os.path.basename(_paths[i])}')
        if args:
            if not args.no_legend:
                _ax.legend(loc='upper right')

    # Hide empty subplots
    for j in range(num_plots, plot_rows * plot_cols):
        axs[j].axis('off')

    plt.subplots_adjust(bottom=0.086, hspace=0.35)

    save_to_file_or_show_canvas()

@beartype
def save_to_file_or_show_canvas() -> None:
    if args:
        if args.save_to_file:
            helpers.save_to_file(fig, args, plt)
        else:
            fig.canvas.manager.set_window_title("GPU-Usage: " + str(args.run_dir))
            if not args.no_plt_show:
                plt.show()

if __name__ == "__main__":
    args = parser.parse_args()
    try:
        plot_gpu_usage(args.run_dir)
    except UnicodeDecodeError:
        if not os.environ.get("PLOT_TESTS"):
            print(f"{args.run_dir}/results.csv seems to be invalid utf8.")
        sys.exit(7)
