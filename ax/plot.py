import sys
import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

parser = argparse.ArgumentParser(description='Path to CSV file that should be plotted.')
parser.add_argument('--csv_file', type=str, help='Path to a CSV file', required=True)
args = parser.parse_args()

if not os.path.exists(args.csv_file):
    print(f'Die Datei {args.csv_file} existiert nicht.')
    sys.exit(1)

# Lade das DataFrame aus der CSV-Datei
df = pd.read_csv(args.csv_file)

# Spalten entfernen
columns_to_remove = ['trial_index', 'arm_name', 'trial_status', 'generation_method']
df_filtered = df.drop(columns=columns_to_remove)

# Sicherstellen, dass 'result' nicht x oder y ist
if 'result' in df_filtered.columns:
    df_filtered = df_filtered.drop(columns='result')

# Kombinationen der Parameter erstellen
parameter_combinations = list(combinations(df_filtered.columns, 2))

# Matplotlib-Figure erstellen
num_subplots = len(parameter_combinations)
num_rows = 2  # Anzahl der Zeilen in der Darstellung
num_cols = (num_subplots + num_rows - 1) // num_rows  # Berechne die Anzahl der Spalten

fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 7))

# Farbzuweisung für Punkte basierend auf 'result'
colors = df['result']
norm = plt.Normalize(colors.min(), colors.max())
cmap = plt.cm.viridis

# Schleife über die Kombinationen und Erstellung der 2D-Plots
for i, (param1, param2) in enumerate(parameter_combinations):
    row = i // num_cols
    col = i % num_cols
    scatter = axs[row, col].scatter(df_filtered[param1], df_filtered[param2], c=colors, cmap=cmap, norm=norm)
    axs[row, col].set_xlabel(param1)
    axs[row, col].set_ylabel(param2)

# Farbskala hinzufügen
cbar = fig.colorbar(scatter, ax=axs, orientation='vertical', fraction=0.02, pad=0.1)
cbar.set_label('Result', rotation=270, labelpad=15)

plt.show()

