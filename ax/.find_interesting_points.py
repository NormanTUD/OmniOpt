import pandas as pd

def is_near_boundary(value, min_value, max_value, threshold=0.1):
    """
    Überprüft, ob ein Wert nahe an einem der Ränder seiner Parametergrenzen liegt.
    
    :param value: Der zu überprüfende Wert
    :param min_value: Der minimale Grenzwert
    :param max_value: Der maximale Grenzwert
    :param threshold: Der Prozentsatz der Nähe zum Rand (Standard ist 10%)
    :return: True, wenn der Wert nahe am Rand ist, sonst False
    """
    range_value = max_value - min_value
    return abs(value - min_value) < threshold * range_value or abs(value - max_value) < threshold * range_value

def find_promising_points(csv_file):
    """
    Findet vielversprechende Punkte aus einer CSV-Datei, die hyperparametrische Suchergebnisse enthält.
    
    :param csv_file: Der Pfad zur CSV-Datei
    """
    # CSV-Datei einlesen
    data = pd.read_csv(csv_file)
    
    # Ergebnisse sortieren (niedrigste Werte zuerst)
    sorted_data = data.sort_values(by='result')
    
    # Bestimmen der Parametergrenzen
    int_param_min = sorted_data['int_param'].min()
    int_param_max = sorted_data['int_param'].max()
    float_param_min = sorted_data['float_param'].min()
    float_param_max = sorted_data['float_param'].max()
    int_param_two_min = sorted_data['int_param_two'].min()
    int_param_two_max = sorted_data['int_param_two'].max()

    # Vielversprechende Punkte finden
    promising_points = []
    for index, row in sorted_data.iterrows():
        if (is_near_boundary(row['int_param'], int_param_min, int_param_max) or
            is_near_boundary(row['float_param'], float_param_min, float_param_max) or
            is_near_boundary(row['int_param_two'], int_param_two_min, int_param_two_max)):
            promising_points.append(row)
    
    # Ergebnisse ausgeben
    for point in promising_points:
        print(point)

# Beispielausführung
csv_file = 'runs/custom_run/114/pd.csv'  # Ersetzen Sie diesen Pfad durch den tatsächlichen Pfad zu Ihrer CSV-Datei
find_promising_points(csv_file)

