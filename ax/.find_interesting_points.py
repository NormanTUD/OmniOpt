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

def find_promising_bubbles(csv_file, result_threshold):
    """
    Findet vielversprechende Punkte (grüne Bubbles) am Rand des Parameterraums.
    
    :param csv_file: Der Pfad zur CSV-Datei
    :param result_threshold: Schwellenwert für die Klassifizierung von guten (grünen) Punkten
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
    promising_bubbles = set()
    probability_dict = {}
    
    for index, row in sorted_data.iterrows():
        if row['result'] > result_threshold:
            continue
        
        if is_near_boundary(row['int_param'], int_param_min, int_param_max):
            direction = 'negative' if abs(row['int_param'] - int_param_min) < abs(row['int_param'] - int_param_max) else 'positive'
            probability = calculate_probability(row['int_param'], int_param_min, int_param_max)
            key = ('int_param', direction)
            promising_bubbles.add(key)
            probability_dict[key] = probability_dict.get(key, []) + [probability]
        
        if is_near_boundary(row['float_param'], float_param_min, float_param_max):
            direction = 'negative' if abs(row['float_param'] - float_param_min) < abs(row['float_param'] - float_param_max) else 'positive'
            probability = calculate_probability(row['float_param'], float_param_min, float_param_max)
            key = ('float_param', direction)
            promising_bubbles.add(key)
            probability_dict[key] = probability_dict.get(key, []) + [probability]
        
        if is_near_boundary(row['int_param_two'], int_param_two_min, int_param_two_max):
            direction = 'negative' if abs(row['int_param_two'] - int_param_two_min) < abs(row['int_param_two'] - int_param_two_max) else 'positive'
            probability = calculate_probability(row['int_param_two'], int_param_two_min, int_param_two_max)
            key = ('int_param_two', direction)
            promising_bubbles.add(key)
            probability_dict[key] = probability_dict.get(key, []) + [probability]
    
    # Ergebnisse ausgeben
    for point in promising_bubbles:
        param, direction = point
        average_probability = sum(probability_dict[point]) / len(probability_dict[point])
        print(f"Es wäre gut, den Parameter '{param}' ins {direction} zu erweitern (Wahrscheinlichkeit, dass es klappt: {average_probability:.2f}%)")

def calculate_probability(value, min_value, max_value):
    """
    Berechnet die Wahrscheinlichkeit basierend auf der relativen Nähe des Werts zu den Grenzen.
    
    :param value: Der zu überprüfende Wert
    :param min_value: Der minimale Grenzwert
    :param max_value: Der maximale Grenzwert
    :return: Wahrscheinlichkeit, dass eine Erweiterung sinnvoll ist
    """
    range_value = max_value - min_value
    if abs(value - min_value) < abs(value - max_value):
        distance_to_boundary = abs(value - min_value)
    else:
        distance_to_boundary = abs(value - max_value)
        
    # Je näher am Rand, desto höher die Wahrscheinlichkeit
    probability = (1 - (distance_to_boundary / range_value)) * 100
    return round(probability, 2)

# Beispielausführung
csv_file = 'runs/custom_run/0/pd.csv'  # Ersetzen Sie diesen Pfad durch den tatsächlichen Pfad zu Ihrer CSV-Datei
result_threshold = -2000  # Schwellenwert für gute Ergebnisse (anpassen nach Bedarf)
find_promising_bubbles(csv_file, result_threshold)

