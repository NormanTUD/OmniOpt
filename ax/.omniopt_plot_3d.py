# DESCRIPTION: 3d plot
# EXPECTED FILES: results.csv

import sys
from pprint import pprint
def dier (msg):
    pprint(msg)
    sys.exit(1)
import argparse
import pandas as pd
import itertools
import pyvista as pv

def main():
    parser = argparse.ArgumentParser(description='3D Scatter Plot from CSV')
    parser.add_argument('--run_dir', type=str, required=True, help='Directory containing the CSV file')
    parser.add_argument('--no_plt_show', help='Disable showing the plot', action='store_true', default=False)
    args = parser.parse_args()

    if args.no_plt_show:
        print("Cannot use 3d plot without showing plots. Exiting")
        sys.exit(1)

    csv_file_path = f"{args.run_dir}/results.csv"
    try:
        dataframe = None

        try:
            dataframe = pd.read_csv(args.run_dir + "/results.csv")
        except pd.errors.EmptyDataError:
            print(f"{args.run_dir}/results.csv seems to be empty.")
            sys.exit(19)

        # Columns to ignore
        ignore_columns = ['trial_index', 'arm_name', 'trial_status', 'generation_method', 'result']
        dynamic_columns = [col for col in dataframe.columns if col not in ignore_columns]

        # Generate all permutations of 3 columns
        column_permutations = list(itertools.combinations(dynamic_columns, 3))

        # Create a plotter with the appropriate shape
        num_plots = len(column_permutations)
        plotter_shape = (num_plots // 2 + num_plots % 2, 2)  # Create a shape that fits all plots
        try:
            plotter = pv.Plotter(shape=plotter_shape)
        except ValueError as e:
            print(f"Error: {e} This may happen when your results.csv has no result column or you don't have at least 3 numeric columns.")
            sys.exit(12)

        plotted = 0
        for index, (col1, col2, col3) in enumerate(column_permutations):
            row, col = divmod(index, 2)
            plotter.subplot(row, col)

            points = dataframe[[col1, col2, col3]].values
            scalars = dataframe['result']

            labels = dict(xlabel=col1, ylabel=col2, zlabel=col3)

            try:
                plotter.add_mesh(pv.PolyData(points),
                                 scalars=scalars,
                                 point_size=10,
                                 render_points_as_spheres=True,
                                 cmap="coolwarm",  # Colormap ranging from blue to red
                                 scalar_bar_args={'title': 'Result'})

                plotter.show_grid()
                plotter.add_axes(interactive=True, **labels)
                plotter.add_scalar_bar(title='Result')
                plotted += 1
            except TypeError as e:
                print(f"Cannot plot {col1}, {col2}, {col3}")

        if plotted:
            plotter.show()
        else:
            print(f"Did not plot anything")
            sys.exit(42)
    except FileNotFoundError:
        print(f"results.csv cannot be found under {args.run_dir}")
        sys.exit(45)

if __name__ == "__main__":
    main()

