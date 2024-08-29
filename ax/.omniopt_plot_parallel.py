import os
import sys
import importlib.util
import logging
import signal
import traceback
import argparse
import pandas as pd
import plotly.express as px

signal.signal(signal.SIGINT, signal.SIG_DFL)

script_dir = os.path.dirname(os.path.realpath(__file__))
helpers_file = f"{script_dir}/.helpers.py"
spec = importlib.util.spec_from_file_location(
    name="helpers",
    location=helpers_file,
)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Plotting tool for analyzing trial data.')
    parser.add_argument('--min', type=float, help='Minimum value for result filtering')
    parser.add_argument('--max', type=float, help='Maximum value for result filtering')
    parser.add_argument('--save_to_file', nargs='?', const='plot.html', type=str, help='Path to save the plot(s)')
    parser.add_argument('--run_dir', type=str, help='Path to a CSV file', required=True)
    parser.add_argument('--no_plt_show', help='Disable showing the plot', action='store_true', default=False)
    return parser.parse_args()

def filter_data(dataframe, min_value=None, max_value=None):
    try:
        if min_value is not None:
            dataframe = dataframe[dataframe['result'] >= min_value]
        if max_value is not None:
            dataframe = dataframe[dataframe['result'] <= max_value]
    except KeyError:
        print_if_not_plot_tests_and_exit(f"{args.run_dir}/results.csv seems to have no results column.", 19)
    return dataframe

def plot_parallel_coordinates(dataframe, save_to_file=None):
    if "result" not in dataframe:
        if not os.environ.get("NO_NO_RESULT_ERROR"):
            print("General: Result column not found in dataframe. That may mean that the job had no valid runs")
        sys.exit(169)

    # Create a Parallel Coordinates plot
    try:
        fig = px.parallel_coordinates(dataframe, color='result', 
                                      labels={col: col for col in dataframe.columns},
                                      title='Parallel Coordinates Plot',
                                      color_continuous_scale=px.colors.sequential.Viridis)
        fig.update_layout(autosize=True, margin=dict(l=0, r=0, t=40, b=0))
        if save_to_file:
            fig.write_html(save_to_file)
        else:
            if not args.no_plt_show:
                fig.show()
    except Exception as e:
        print(f"Error generating plot: {e}")
        sys.exit(199)

def print_if_not_plot_tests_and_exit(msg, exit_code):
    if not os.environ.get("PLOT_TESTS"):
        print(msg)
    if exit_code is not None:
        sys.exit(exit_code)

def print_traceback():
    tb = traceback.format_exc()
    print(tb)

def update_graph():
    try:
        dataframe = None

        try:
            dataframe = pd.read_csv(args.run_dir + "/results.csv")
        except pd.errors.EmptyDataError:
            print_if_not_plot_tests_and_exit(f"{args.run_dir}/results.csv seems to be empty.", 19)
        except UnicodeDecodeError:
            print_if_not_plot_tests_and_exit(f"{args.run_dir}/results.csv seems to be invalid utf8.", 7)

        if args.min is not None or args.max is not None:
            dataframe = filter_data(dataframe, args.min, args.max)

        if dataframe.empty:
            print_if_not_plot_tests_and_exit("No applicable values could be found.", None)
            return

        if args.save_to_file:
            _path = os.path.dirname(args.save_to_file)
            if _path:
                os.makedirs(_path, exist_ok=True)

        plot_parallel_coordinates(dataframe, args.save_to_file)

    except FileNotFoundError:
        logging.error("File not found: %s", args.run_dir + "/results.csv")
    except Exception as exception:
        logging.error("An unexpected error occurred: %s", str(exception))
        print_traceback()

if __name__ == "__main__":
    setup_logging()
    args = parse_arguments()

    if not os.path.exists(args.run_dir):
        logging.error("Specified --run_dir does not exist")
        sys.exit(1)

    update_graph()
