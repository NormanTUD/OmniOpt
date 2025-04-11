import sys
import os
import json
import random

def generate_random_value(parameter):
    if parameter['parameter_type'] == 'RANGE':
        range_min, range_max = parameter['range']
        if parameter['type'] == 'INT':
            return random.randint(range_min, range_max)
        elif parameter['type'] == 'FLOAT':
            return random.uniform(range_min, range_max)
    elif parameter['parameter_type'] == 'CHOICE':
        values = parameter['values']
        if parameter['type'] == 'INT':
            return random.choice(values)
        elif parameter['type'] == 'STRING':
            return random.choice(values)
        else:
            return random.choice(values)  # For FLOAT or other types, choose one value

    elif parameter['parameter_type'] == 'FIXED':
        return parameter['value']

def generate_random_point(parameters):
    point = {}
    for param_name, param_data in parameters.items():
        point[param_name] = generate_random_value(param_data)
    return point

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <path>")
        sys.exit(1)

    path = sys.argv[1]
    json_file_path = os.path.join(path, 'input.json')
    results_file_path = os.path.join(path, 'results.json')

    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {json_file_path} not found.")
        sys.exit(2)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON in {json_file_path}.")
        sys.exit(3)

    # Generate random point within parameter bounds
    random_point = generate_random_point(data)

    # Write results to results.json
    with open(results_file_path, 'w') as f:
        json.dump(random_point, f, indent=4)

if __name__ == "__main__":
    main()
