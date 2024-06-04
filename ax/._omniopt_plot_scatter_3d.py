import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import argparse
import traceback
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)


NO_RESULT = '0'
LOG_FILE = "error_log.txt"

def assert_condition(condition, error_text):
    assert condition, error_text

def log_error(error_text):
    with open(LOG_FILE, "a") as f:
        f.write(error_text + "\n")

def to_int_when_possible(x):
    try:
        return int(x)
    except ValueError:
        return x

def get_data(csv_file_path, _min=None, _max=None):
    try:
        assert_condition(os.path.exists(csv_file_path), f"{csv_file_path} does not exist.")
        df = pd.read_csv(csv_file_path)
        assert_condition("result" in df.columns, f"result not found in CSV columns.")
        df = df.dropna(subset=["result"])
        if _min is not None and _max is not None:
            df = df[(df["result"] >= _min) & (df["result"] <= _max)]
        return df
    except AssertionError as e:
        log_error(str(e))
        traceback.print_exc()
        raise e
    except Exception as e:
        log_error(f"An unexpected error occurred: {str(e)}")
        traceback.print_exc()
        raise e

def check_if_results_are_empty(result_series):
    try:
        assert_condition(not result_series.empty, "Result column is empty.")
    except AssertionError as e:
        log_error(str(e))
        traceback.print_exc()
        raise e

def create_scatter_plot(df):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(df['x'], df['y'], df['z'], c=df["result"], cmap='RdYlGn', alpha=0.8)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_title('Scatter plot of X, Y, Z')

    plt.tight_layout()
    plt.show()

def main(run_dir, _min, _max):
    csv_file_path = f"{run_dir}/pd.csv"

    try:
        df = get_data(csv_file_path, _min, _max)
        check_if_results_are_empty(df["result"])
        create_scatter_plot(df)
    except AssertionError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='3D Scatter Plot for all combinations of three parameters')
    parser.add_argument('--run_dir', type=str, help='Path to the CSV file containing the data')
    parser.add_argument('--min', dest='_min', type=float, help='Minimum result value to include in the plot')
    parser.add_argument('--max', dest='_max', type=float, help='Maximum result value to include in the plot')
    args = parser.parse_args()

    main(args.run_dir, args._min, args._max)

