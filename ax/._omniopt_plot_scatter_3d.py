import sys
from pprint import pprint
import pandas as pd
import numpy as np
import os
import argparse
import traceback

def dier(msg):
    pprint(msg)
    sys.exit(1)

def assert_condition(condition, error_text):
    assert condition, error_text

def to_int_when_possible(x):
    try:
        return int(x)
    except ValueError:
        return x

def get_data(csv_file_path, _min=None, _max=None):
    try:
        assert_condition(os.path.exists(csv_file_path), f"{csv_file_path} does not exist.")
        df = pd.read_csv(csv_file_path)
        pprint(df)
        assert_condition("result" in df.columns, f"result not found in CSV columns.")
        df = df.dropna(subset=["result"])
        if _min is not None and _max is not None:
            df = df[(df["result"] >= _min) & (df["result"] <= _max)]
        return df
    except AssertionError as e:
        traceback.print_exc()
        raise e
    except Exception as e:
        traceback.print_exc()
        raise e

def check_if_results_are_empty(result_series):
    try:
        assert_condition(not result_series.empty, "Result column is empty.")
    except AssertionError as e:
        traceback.print_exc()
        raise e

import itertools
import plotly.graph_objs as go

def create_scatter_plots(df):
    columns = df.columns.tolist()
    columns.remove("result")  # Ignoring the result column
    permutations = list(itertools.permutations(columns, 3))

    for perm in permutations:
        # Convert strings in the "result" column to floats
        result_values = df['result'].str.replace(',', '').astype(float)
        
        fig = go.Figure(data=[go.Scatter3d(
            x=df[perm[0]],
            y=df[perm[1]],
            z=df[perm[2]],
            mode='markers',
            marker=dict(
                size=5,
                color=result_values,  # Use converted result values
                colorscale='RdYlGn',
                opacity=0.8
            )
        )])
        fig.update_layout(
            scene=dict(
                xaxis_title=perm[0],
                yaxis_title=perm[1],
                zaxis_title=perm[2],
                title=f'Scatter plot of {perm[0]}, {perm[1]}, {perm[2]}'
            )
        )
        fig.show()

def create_scatter_plots(df):
    columns = df.columns.tolist()
    columns.remove("result")  # Ignoring the result column
    permutations = list(itertools.permutations(columns, 3))

    for perm in permutations:
        # Convert strings in the "result" column to floats
        result_values = df['result'].str.replace(',', '').astype(float)
        
        fig = go.Figure(data=[go.Scatter3d(
            x=df[perm[0]],
            y=df[perm[1]],
            z=df[perm[2]],
            mode='markers',
            marker=dict(
                size=5,
                color=result_values,  # Use converted result values
                colorscale='RdYlGn',
                opacity=0.8
            )
        )])
        fig.update_layout(
            scene=dict(
                xaxis_title=perm[0],
                yaxis_title=perm[1],
                zaxis_title=perm[2]
            ),
            title=f'Scatter plot of {perm[0]}, {perm[1]}, {perm[2]}'  # Set the title here
        )
        fig.show()

def create_scatter_plots(df):
    columns = df.columns.tolist()
    columns.remove("result")  # Ignoring the result column
    permutations = list(itertools.permutations(columns, 3))

    data = []
    for perm in permutations:
        # Convert strings in the "result" column to floats
        result_values = df['result'].str.replace(',', '').astype(float)

        trace = go.Scatter3d(
            x=df[perm[0]],
            y=df[perm[1]],
            z=df[perm[2]],
            mode='markers',
            marker=dict(
                size=5,
                color=result_values,  # Use converted result values
                colorscale='RdYlGn',
                opacity=0.8
            ),
            name=f'{perm[0]}, {perm[1]}, {perm[2]}'
        )
        data.append(trace)

    layout = go.Layout(
        scene=dict(
            xaxis=dict(title=columns[0]),
            yaxis=dict(title=columns[1]),
            zaxis=dict(title=columns[2])
        ),
        title='3D Scatter Plot of Parameter Combinations'
    )

    fig = go.Figure(data=data, layout=layout)
    fig.show()

def create_scatter_plots(df):
    columns = df.columns.tolist()
    columns.remove("result")  # Ignoring the result column
    columns.remove("trial_index")  # Ignoring trial_index column
    columns.remove("arm_name")  # Ignoring arm_name column
    columns.remove("trial_status")  # Ignoring trial_status column
    columns.remove("generation_method")  # Ignoring generation_method column

    permutations = list(itertools.permutations(columns, 3))

    data = []
    for perm in permutations:
        # Convert strings in the "result" column to floats
        result_values = df['result'].str.replace(',', '').astype(float)

        trace = go.Scatter3d(
            x=df[perm[0]],
            y=df[perm[1]],
            z=df[perm[2]],
            mode='markers',
            marker=dict(
                size=5,
                color=result_values,  # Use converted result values
                colorscale='RdYlGn',
                opacity=0.8
            ),
            name=f'{perm[0]}, {perm[1]}, {perm[2]}'
        )
        data.append(trace)

    layout = go.Layout(
        scene=dict(
            xaxis=dict(title=columns[0]),
            yaxis=dict(title=columns[1]),
            zaxis=dict(title=columns[2])
        ),
        title='3D Scatter Plot of Parameter Combinations'
    )

    fig = go.Figure(data=data, layout=layout)
    fig.show()

def create_scatter_plots(df):
    columns = df.columns.tolist()
    columns.remove("result")  # Ignoring the result column
    columns.remove("trial_index")  # Ignoring trial_index column
    columns.remove("arm_name")  # Ignoring arm_name column
    columns.remove("trial_status")  # Ignoring trial_status column
    columns.remove("generation_method")  # Ignoring generation_method column

    permutations = list(itertools.permutations(columns, 3))

    num_plots = len(permutations)
    num_cols = 2
    num_rows = num_plots // num_cols + (1 if num_plots % num_cols > 0 else 0)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 6 * num_rows))

    for i, perm in enumerate(permutations):
        if num_plots > 1:
            ax = axes[i // num_cols, i % num_cols]
        else:
            ax = axes
        # Convert strings in the "result" column to floats
        result_values = df['result'].str.replace(',', '').astype(float)
        ax.scatter(df[perm[0]], df[perm[1]], df[perm[2]], c=result_values, cmap='RdYlGn', alpha=0.8)
        ax.set_xlabel(perm[0])
        ax.set_ylabel(perm[1])
        ax.set_zlabel(perm[2])
        ax.set_title(f'Scatter plot of {perm[0]}, {perm[1]}, {perm[2]}')

    plt.tight_layout()
    plt.show()


def main(run_dir, _min, _max):
    csv_file_path = f"{run_dir}/results.csv"

    try:
        df = get_data(csv_file_path, _min, _max)
        check_if_results_are_empty(df["result"])
        create_scatter_plots(df)
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

