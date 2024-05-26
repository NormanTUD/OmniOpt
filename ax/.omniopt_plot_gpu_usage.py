import sys
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint

def dier(msg):
    pprint(msg)
    sys.exit(1)

def remove_duplicate_headers(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Standard-Headerzeile
        header_line = "timestamp, name, pci.bus_id, driver_version, pstate, pcie.link.gen.max, pcie.link.gen.current, temperature.gpu, utilization.gpu [%], utilization.memory [%], memory.total [MiB], memory.free [MiB], memory.used [MiB]\n"

        return [line.rstrip() for line in lines if not line.startswith("timestamp")]  # Alle Headerzeilen entfernen

    except Exception as e:
        warn(f"Error occurred while removing duplicate headers: {e}")
        return None




def plot_gpu_usage(run_dir):
    gpu_data = []
    for root, dirs, files in os.walk(run_dir):
        for file in files:
            if file.startswith("gpu_usage_") and file.endswith(".csv"):
                file_path = os.path.join(root, file)
                no_headers = remove_duplicate_headers(file_path)
                _tmp_file = file_path + "_tmp"
                with open(_tmp_file, 'w') as file:
                    file.writelines(no_headers)
                # Einlesen der Daten mit expliziter Angabe der Spaltennamen
                df = pd.read_csv(_tmp_file,
                                 names=["timestamp", "name", "pci.bus_id", "driver_version", "pstate",
                                        "pcie.link.gen.max", "pcie.link.gen.current", "temperature.gpu",
                                        "utilization.gpu [%]", "utilization.memory [%]", "memory.total [MiB]",
                                        "memory.free [MiB]", "memory.used [MiB]"])
                gpu_data.append(df)

                os.unlink(_tmp_file)

    if not gpu_data:
        print("No GPU usage data found.")
        return

    dier(gpu_data)
    all_gpu_data = pd.concat(gpu_data, ignore_index=True)
    all_gpu_data['timestamp'] = pd.to_datetime(all_gpu_data['timestamp'], format='%Y/%m/%d %H:%M:%S.%f', errors='coerce')
    all_gpu_data = all_gpu_data.dropna(subset=['timestamp'])  # Entfernen von ungültigen Zeitstempeln
    all_gpu_data.plot(x='timestamp', y='utilization.gpu [%]', kind='line')
    plt.xlabel('Time')
    plt.ylabel('GPU Usage (%)')
    plt.title('GPU Usage Over Time')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--run_dir', type=str, help='Directory where to search for CSV files')
    parser.add_argument('--plot_type', type=str, default="irgendwas", help='Type of plot (ignored)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (ignored)')
    args = parser.parse_args()

    plot_gpu_usage(args.run_dir)

