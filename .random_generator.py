import sys
import os
import json
import random
from typing import Union, Optional

def check_constraint(constraint: str, params: dict) -> bool:
    return eval(constraint, {}, params)

def constraints_not_ok(constraints: list, point: dict) -> bool:
    if not constraints or constraints is None or len(constraints) == 0:
        return True

    for constraint in constraints:
        if not check_constraint(constraint, point):
            return True

    return False

def generate_random_value(parameter: dict) -> Optional[Union[int, float, str]]:
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

def generate_random_point(data: dict) -> dict:
    constraints = data["constraints"]
    point: dict = {}

    param_data = data["parameters"]

    i = 0

    if len(constraints):
        while not point or constraints_not_ok(constraints, point):
            for param_name in list(param_data.keys()):
                point[param_name] = generate_random_value(param_data[param_name])

            if i > 100: # if after 100 trials nothing was found, stop trying
                break

            i = i + 1
    else:
        for param_name in list(param_data.keys()):
            point[param_name] = generate_random_value(param_data[param_name])

    return point

def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python script.py <path>")
        sys.exit(1)

    path = sys.argv[1]

    if not os.path.isdir(path):
        print(f"Error: The path '{path}' is not a valid folder.")
        sys.exit(2)

    json_file_path = os.path.join(path, 'input.json')
    results_file_path = os.path.join(path, 'results.json')

    try:
        with open(json_file_path, mode='r', encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {json_file_path} not found.")
        sys.exit(3)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON in {json_file_path}.")
        sys.exit(4)

    random_point = generate_random_point(data)

    with open(results_file_path, mode='w', encoding="utf-8") as f:
        json.dump({"parameters": random_point}, f, indent=4)

if __name__ == "__main__":
    main()
