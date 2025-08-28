import sys
import os
import json
import logging
from typing import Optional, Any

try:
    import optuna
    from optuna.trial import create_trial

    from optuna.distributions import (
        BaseDistribution,
        IntUniformDistribution,
        FloatDistribution,        # Optuna ≥3.6
    )
except ModuleNotFoundError:
    print("Optuna not found. Cannot continue.")
    sys.exit(1)

try:
    from beartype import beartype
except ModuleNotFoundError:
    print("beartype not found. Cannot continue.")
    sys.exit(1)

logging.getLogger("optuna").setLevel(logging.WARNING)

@beartype
def check_constraint(constraint: str, params: dict) -> bool:
    return eval(constraint, {}, params)

@beartype
def constraints_not_ok(constraints: list, point: dict) -> bool:
    if not constraints or constraints is None or len(constraints) == 0:
        return True

    for constraint in constraints:
        if not check_constraint(constraint, point):
            return True

    return False

@beartype
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
                    point[param_name] = trial.suggest_float(param_name, rmin, rmax) # type: ignore[assignment]
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

@beartype
def generate_tpe_point(data: dict, max_trials: int = 100) -> dict:
    parameters = data["parameters"]
    constraints = data.get("constraints", [])
    seed = data.get("seed", None)
    trials_data = data.get("trials", [])
    objectives = data.get("objectives", {})

    direction, result_key = parse_objectives(objectives)
    study = create_study_with_seed(seed, direction)

    for trial_entry in trials_data:
        add_existing_trial_to_study(study, trial_entry, parameters, result_key)

    study.optimize(lambda trial: wrapped_objective(trial, parameters, constraints, direction), n_trials=max_trials)

    return get_best_or_new_point(study, parameters, direction)

@beartype
def parse_objectives(objectives: dict) -> tuple[str, str]:
    if len(objectives) != 1:
        raise ValueError("Only single-objective optimization is supported.")
    result_key, result_goal = next(iter(objectives.items()))
    if result_goal.lower() not in ("min", "max"):
        raise ValueError(f"Unsupported objective direction: {result_goal}")
    direction = "maximize" if result_goal.lower() == "max" else "minimize"
    return direction, result_key

@beartype
def create_study_with_seed(seed: Optional[int], direction: str) -> optuna.study.study.Study:
    return optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=seed),
        direction=direction
    )

@beartype
def wrapped_objective(trial: optuna.Trial, parameters: dict, constraints: list, direction: str) -> float:
    point = tpe_suggest_point(trial, parameters)
    if not constraints_not_ok(constraints, point):
        return 1e6 if direction == "minimize" else -1e6
    return 0.0

@beartype
def add_existing_trial_to_study(study: optuna.study.study.Study, trial_entry: list, parameters: dict, result_key: str) -> None:
    if len(trial_entry) != 2:
        return
    param_dict, result_dict = trial_entry

    if not result_dict or result_key not in result_dict:
        return

    if not all(k in param_dict for k in parameters):
        return

    final_value = result_dict[result_key]

    trial_params: dict[str, object] = {}
    trial_distributions: dict[str, BaseDistribution] = {}   # 👈 explicit & correct

    for name, p in parameters.items():
        value = param_dict[name]

        if p["parameter_type"] == "FIXED":
            trial_params[name] = value
            continue

        dist: BaseDistribution
        if p["parameter_type"] == "RANGE":
            if p["type"] == "INT":
                dist = IntUniformDistribution(p["range"][0], p["range"][1])
            elif p["type"] == "FLOAT":
                dist = FloatDistribution(p["range"][0], p["range"][1])
            else:
                continue
        elif p["parameter_type"] == "CHOICE":
            dist = optuna.distributions.CategoricalDistribution(p["values"])
        else:
            continue

        trial_params[name] = value
        trial_distributions[name] = dist      # keys are str, values are BaseDistribution

    study.add_trial(
        create_trial(
            params=trial_params,
            distributions=trial_distributions,  # ✅ mypy is happy now
            value=final_value
        )
    )

@beartype
def get_best_or_new_point(study: Any, parameters: dict, direction: str) -> dict:
    best_trial_value = study.best_trial.value
    if best_trial_value is not None:
        if (direction == "minimize" and best_trial_value < 1e6) or \
           (direction == "maximize" and best_trial_value > -1e6):
            return study.best_params
    return tpe_suggest_point(study.best_trial, parameters)

@beartype
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
