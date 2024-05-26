import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_worker_usage(pd_csv):
    try:
        data = pd.read_csv(pd_csv)
        plt.plot(data['time'], data['num_parallel_jobs'], label='Number of Parallel Jobs')
        plt.plot(data['time'], data['nr_current_workers'], label='Number of Current Workers')
        plt.xlabel('Time')
        plt.ylabel('Count')
        plt.title('Worker Usage Plot')
        plt.legend()
        plt.show()
    except FileNotFoundError:
        print(f"File '{pd_csv}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description='Plot worker usage from CSV file')
    parser.add_argument('--run_dir', type=str, help='Directory containing worker usage CSV file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    if args.debug:
        print(f"Debug mode enabled. Run directory: {args.run_dir}")

    if args.run_dir:
        pd_csv = os.path.join(args.run_dir, "worker_usage.csv")
        if os.path.exists(pd_csv):
            plot_worker_usage(pd_csv)
        else:
            print(f"File '{pd_csv}' does not exist.")

if __name__ == "__main__":
    main()
