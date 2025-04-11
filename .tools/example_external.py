import sys
import os
import json
import random

def generate_random_value(parameter):
    try:
        if parameter['parameter_type'] == 'RANGE':
            range_min, range_max = parameter['range']
            if parameter['type'] == 'INT':
                return random.randint(range_min, range_max)

            if parameter['type'] == 'FLOAT':
                return random.uniform(range_min, range_max)
        elif parameter['parameter_type'] == 'CHOICE':
            values = parameter['values']
            if parameter['type'] == 'INT':
                return random.choice(values)

            if parameter['type'] == 'STRING':
                return random.choice(values)

            return random.choice(values)
        elif parameter['parameter_type'] == 'FIXED':
            return parameter['value']
    except KeyError as e:
        print(f"KeyError: Missing {e} in parameter")
        sys.exit(4)

    return None

def generate_random_point(parameters):
    point = {}
    for param_data in parameters.items():
        p = param_data[1]

        if p:
            if not isinstance(p, list):
                p_keys = list(p.keys())

                for param_name in p_keys:
                    this_param = param_data[1][param_name]
                    point[param_name] = generate_random_value(this_param)
    return point

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <path>")
        sys.exit(1)

    path = sys.argv[1]
    json_file_path = os.path.join(path, 'input.json')
    results_file_path = os.path.join(path, 'results.json')

    try:
        with open(json_file_path, mode='r', encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {json_file_path} not found.")
        sys.exit(2)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON in {json_file_path}.")
        sys.exit(3)

    random_point = generate_random_point(data)

    with open(results_file_path, mode='w', encoding="utf-8") as f:
        json.dump({"parameters": random_point}, f, indent=4)

if __name__ == "__main__":
    main()
