import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

# Annahme: df ist dein DataFrame
# Ersetze dies durch deinen tatsächlichen DataFrame-Namen

df = pd.DataFrame({
    'trial_index': [0, 1, 2],
    'arm_name': ['0_0', '1_0', '2_0'],
    'trial_status': ['COMPLETED', 'COMPLETED', 'COMPLETED'],
    'generation_method': ['Sobol', 'Sobol', 'Sobol'],
    'result': [12.145566, -765.380083, -4.341217],
    'x': [-236.854434, -770.380083, 1],
    'y': [123123.0, 5.0, 1000],
    'z': [102.0, 102.0, 4]
})

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

