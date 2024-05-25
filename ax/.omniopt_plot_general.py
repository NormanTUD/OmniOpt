import sys

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import argparse
    import logging
except ModuleNotFoundError as e:
    print(f"Warning: {e}")
    print("Please make sure to install the required modules.")
    sys.exit(1)

def plot_boxplot(df, ax):
    sns.boxplot(x='generation_method', y='result', data=df, ax=ax)
    ax.set_title('Results by Generation Method')
    ax.set_xlabel('Generation Method')
    ax.set_ylabel('Result')

def plot_barplot(df, ax):
    sns.countplot(x='trial_status', data=df, ax=ax)
    ax.set_title('Distribution of Trial Status')
    ax.set_xlabel('Trial Status')
    ax.set_ylabel('Count')

def plot_correlation_matrix(df, ax):
    exclude_cols = ['trial_index', 'arm_name', 'trial_status', 'generation_method']
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title('Correlation Matrix')

def plot_distribution_by_generation(df, ax):
    hist = sns.histplot(data=df, x='result', hue='generation_method', multiple="stack", kde=True, bins=20, ax=ax)
    for patch in hist.patches:
        patch.set_alpha(0.5)
    ax.set_title('Distribution of Results by Generation Method')
    ax.set_xlabel('Result')
    ax.set_ylabel('Frequency')

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Plotting tool for analyzing trial data.')
    parser.add_argument('--min', type=float, help='Minimum value for result filtering')
    parser.add_argument('--max', type=float, help='Maximum value for result filtering')
    parser.add_argument('--save_to_file', nargs='?', const='plot', type=str, help='Path to save the plot(s)')

    parser.add_argument('--exclude_params', action='append', nargs='+', help="Params to be ignored", default=[])
    parser.add_argument('--run_dir', type=str, help='Path to a CSV file', required=True)
    parser.add_argument('--result_column', type=str, help='Name of the result column', default="result")
    parser.add_argument('--debug', help='Enable debugging', action='store_true', default=False)
    parser.add_argument('--delete_temp', help='Delete temp files', action='store_true', default=False)
    parser.add_argument('--darkmode', help='Enable darktheme', action='store_true', default=False)
    parser.add_argument('--print_to_command_line', help='Print plot to command line', action='store_true', default=False)
    parser.add_argument('--single', help='Print plot to command line', action='store_true', default=False)
    parser.add_argument('--bubblesize', type=int, help='Size of the bubbles', default=7)
    parser.add_argument('--merge_with_previous_runs', action='append', nargs='+', help="Run-Dirs to be merged with", default=[])

    parser.add_argument('--plot_type', action='append', nargs='+', help="Params to be ignored", default=[])

    return parser.parse_args()

def filter_data(df, min_value=None, max_value=None):
    if min_value is not None:
        df = df[df['result'] >= min_value]
    if max_value is not None:
        df = df[df['result'] <= max_value]
    return df

if __name__ == "__main__":
    setup_logging()

    args = parse_arguments()

    try:
        df = pd.read_csv(args.run_dir + "/pd.csv")

        if args.min is not None or args.max is not None:
            df = filter_data(df, args.min, args.max)

        if df.empty:
            logging.warning("DataFrame is empty after filtering.")
            sys.exit()

        fig, axes = plt.subplots(2, 2, figsize=(10, 10), gridspec_kw={'left': 0.071, 'bottom': 0.07, 'right': 0.983, 'top': 0.976, 'wspace': 0.167, 'hspace': 0.68})

        plot_boxplot(df, axes[0, 0])
        plot_barplot(df, axes[0, 1])
        plot_correlation_matrix(df, axes[1, 0])
        plot_distribution_by_generation(df, axes[1, 1])

        plt.tight_layout()

        if args.save_to_file:
            plt.savefig(args.save_to_file)
        else:
            plt.show()

    except FileNotFoundError:
        logging.error("File not found: %s", args.run_dir + "/pd.csv")
    except Exception as e:
        logging.error("An unexpected error occurred: %s", str(e))
