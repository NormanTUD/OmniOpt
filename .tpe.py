import sys
import os
import json
import logging
try:
    import optuna
except ModuleNotFoundError:
    print("Optuna not found. Cannot continue.")
    sys.exit(1)

logging.getLogger("optuna").setLevel(logging.WARNING)

def check_constraint(constraint: str, params: dict) -> bool:
    return eval(constraint, {}, params)

def constraints_not_ok(constraints: list, point: dict) -> bool:
    if not constraints or constraints is None or len(constraints) == 0:
        return True

    for constraint in constraints:
        if not check_constraint(constraint, point):
            return True

    return False

def tpe_suggest_point(trial: optuna.Trial, parameters: dict) -> dict:
    point = {}
    for param_name, param in parameters.items():
        ptype = param['parameter_type']
        pvaltype = param['type']

        try:
            if ptype == 'RANGE':
                rmin, rmax = param['range']
                if pvaltype == 'INT':
                    point[param_name] = trial.suggest_int(param_name, rmin, rmax)
                elif pvaltype == 'FLOAT':
                    point[param_name] = trial.suggest_float(param_name, rmin, rmax)
                else:
                    raise ValueError(f"Unsupported type {pvaltype} for RANGE")

            elif ptype == 'CHOICE':
                values = param['values']
                point[param_name] = trial.suggest_categorical(param_name, values)

            elif ptype == 'FIXED':
                point[param_name] = param['value']

            else:
                raise ValueError(f"Unknown parameter_type {ptype}")
        except KeyboardInterrupt:
            print("You pressed CTRL-c.")
            sys.exit(1)

    return point

def generate_tpe_point(data: dict, max_trials: int = 100) -> dict:
    parameters = data["parameters"]
    constraints = data.get("constraints", [])
    seed = data.get("seed", None)
    trials_data = data.get("trials", [])

    def objective(trial: optuna.Trial):
        point = tpe_suggest_point(trial, parameters)
        if not constraints_not_ok(constraints, point):
            return 1e6
        return 0

    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=seed),
        direction="minimize"
    )

    for trial_entry in trials_data:
        if len(trial_entry) != 2:
            continue  # Ungültig
        param_dict, result_dict = trial_entry[0], trial_entry[1]

        # Dynamisch den einzigen Ergebnis-Key ermitteln
        if not result_dict or len(result_dict) != 1:
            continue  # ungültiger Result-Eintrag

        result_key = next(iter(result_dict))
        result_value = result_dict[result_key]

        trial_params = {}
        trial_distributions = {}

        for name, p in parameters.items():
            if p["parameter_type"] == "FIXED":
                continue
            value = param_dict[name]
            if p["parameter_type"] == "RANGE":
                if p["type"] == "INT":
                    dist = optuna.distributions.IntUniformDistribution(p["range"][0], p["range"][1])
                elif p["type"] == "FLOAT":
                    dist = optuna.distributions.UniformDistribution(p["range"][0], p["range"][1])
                else:
                    continue
            elif p["parameter_type"] == "CHOICE":
                dist = optuna.distributions.CategoricalDistribution(p["values"])
            else:
                continue

            trial_params[name] = value
            trial_distributions[name] = dist

        study.add_trial(
            create_trial(
                params=trial_params,
                distributions=trial_distributions,
                value=result_value
            )
        )

    study.optimize(objective, n_trials=max_trials)

    best_point = (
        study.best_params
        if study.best_trial.value < 1e6
        else tpe_suggest_point(study.best_trial, parameters)
    )

    return best_point

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

    random_point = generate_tpe_point(data)

    with open(results_file_path, mode='w', encoding="utf-8") as f:
        json.dump({"parameters": random_point}, f, indent=4)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("You pressed CTRL-c.")
        sys.exit(1)
