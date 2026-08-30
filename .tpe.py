"""Tree-structured Parzen Estimator (TPE) using hyperopt.

This is the new TPE generator for OmniOpt's ``EXTERNAL_GENERATOR`` interface.
It reads the same ``input.json`` that the previous Optuna-based implementation
(``.optuna_tpe.py``) read, and writes a ``results.json`` in the same format.

Behavioral contract (kept identical to the Optuna version so the external
generator stays a drop-in replacement):

  * Read ``input.json`` with keys ``parameters``, ``constraints``, ``seed``,
    ``trials`` and ``objectives``.
  * Replay the recorded trials so hyperopt's TPE model has history to work with.
  * Run TPE for a fixed number of additional iterations.
  * Write ``{"parameters": <next point>}`` to ``results.json``.

The objective is the same constraint-aware dummy used by the Optuna version:
valid points return loss 0.0, constraint-violating points return 1e6 (or -1e6
for ``maximize``). That keeps the optimizer focused on respecting constraints
even when the true objective values are not available to the generator.

Usage:
    python3 .tpe.py <path>
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

try:
    from hyperopt import fmin, hp, tpe, Trials, STATUS_OK
except ModuleNotFoundError:
    print("hyperopt not found. Cannot continue.")
    sys.exit(1)

try:
    import numpy as np
except ModuleNotFoundError:
    print("numpy not found. Cannot continue.")
    sys.exit(1)

try:
    from beartype import beartype
except ModuleNotFoundError:
    print("beartype not found. Cannot continue.")
    sys.exit(1)


@beartype
def build_space(parameters: dict) -> dict:
    """Translate the OmniOpt parameter dict into a hyperopt search space."""
    space: dict = {}
    for name, p in parameters.items():
        ptype = p["parameter_type"]
        if ptype == "RANGE":
            lo, hi = p["range"]
            lo_f, hi_f = float(lo), float(hi)
            if p["type"] == "INT":
                space[name] = hp.quniform(name, lo_f, hi_f, 1)
            elif p["type"] == "FLOAT":
                space[name] = hp.uniform(name, lo_f, hi_f)
            else:
                raise ValueError(f"Unsupported RANGE type {p['type']} for {name}")
        elif ptype == "CHOICE":
            space[name] = hp.choice(name, list(p["values"]))
        elif ptype == "FIXED":
            # hp.choice with a single option keeps the parameter fixed
            space[name] = hp.choice(name, [p["value"]])
        else:
            raise ValueError(f"Unknown parameter_type {ptype} for {name}")
    return space


@beartype
def check_constraint(constraint: str, params: dict) -> bool:
    return bool(eval(constraint, {}, params))  # pylint: disable=eval-used


@beartype
def constraint_violated(constraints: list, point: dict) -> bool:
    if not constraints:
        return False
    return any(not check_constraint(c, point) for c in constraints)


@beartype
def parse_objectives(objectives: dict) -> tuple[str, str]:
    if len(objectives) != 1:
        raise ValueError("Only single-objective optimization is supported.")
    result_key, result_goal = next(iter(objectives.items()))
    if result_goal.lower() not in ("min", "max"):
        raise ValueError(f"Unsupported objective direction: {result_goal}")
    direction = "minimize" if result_goal.lower() == "min" else "maximize"
    return direction, result_key


@beartype
def extract_known_trials(trials_data: list, result_key: str) -> list:
    """Pull the (param_dict, loss) pairs out of the OmniOpt trial format.

    OmniOpt writes ``trials`` as a 2-element list: ``[list_of_param_dicts,
    list_of_result_dicts]``. We pair them up by position.
    """
    out: list = []
    if not isinstance(trials_data, list) or len(trials_data) != 2:
        return out
    params_list, results_list = trials_data
    if not isinstance(params_list, list) or not isinstance(results_list, list):
        return out
    for param_dict, result_dict in zip(params_list, results_list):
        if not isinstance(param_dict, dict) or not isinstance(result_dict, dict):
            continue
        if result_key not in result_dict:
            continue
        out.append((param_dict, float(result_dict[result_key])))
    return out


@beartype
def _dummy_loss(params: dict, constraints: list, penalty: float) -> dict:
    if constraint_violated(constraints, params):
        return {"loss": penalty, "status": STATUS_OK}
    return {"loss": 0.0, "status": STATUS_OK}


@beartype
def make_objective(known_trials: list, constraints: list, direction: str) -> Callable[[dict], dict]:
    """Build the hyperopt objective.

    The first ``len(known_trials)`` invocations return the recorded loss for
    each historic trial, in order, so hyperopt can build its TPE model from
    real history. Once the replay is exhausted, the objective returns the
    constraint-aware dummy loss used by the Optuna version.
    """
    penalty = 1e6 if direction == "minimize" else -1e6
    iter_known = iter(known_trials)
    state = {"replay_done": False}

    def objective(params: dict) -> dict:
        if not state["replay_done"]:
            try:
                _, recorded_loss = next(iter_known)
            except StopIteration:
                state["replay_done"] = True
                return _dummy_loss(params, constraints, penalty)
            return {"loss": recorded_loss, "status": STATUS_OK}
        return _dummy_loss(params, constraints, penalty)

    return objective


@beartype
def _decode_trial(vals: dict, parameters: dict) -> dict:
    """Convert hyperopt's trial result dictionary into OmniOpt's parameter dict."""
    point: dict = {}
    for name, raw in vals.items():
        values = list(raw)
        value = values[0] if values else None
        if value is None:
            point[name] = None
            continue
        if name in parameters:
            p = parameters[name]
            if p["parameter_type"] == "RANGE" and p["type"] == "INT":
                value = int(round(float(value)))
            elif p["parameter_type"] == "RANGE" and p["type"] == "FLOAT":
                value = float(value)
            elif p["parameter_type"] == "CHOICE":
                value = list(p["values"])[int(value)]
            elif p["parameter_type"] == "FIXED":
                value = p["value"]
        point[name] = value
    return point


@beartype
def generate_tpe_point(data: dict, extra_iters: int = 50) -> dict:
    """Top-level entry point: read input, run TPE, return the next point."""
    parameters = data["parameters"]
    constraints = data.get("constraints", []) or []
    seed = data.get("seed", None)
    trials_data = data.get("trials", []) or []
    objectives = data.get("objectives", {})

    direction, result_key = parse_objectives(objectives)
    space = build_space(parameters)
    known = extract_known_trials(trials_data, result_key)
    objective = make_objective(known, constraints, direction)

    trials = Trials()
    max_evals = max(len(known) + extra_iters, 1)
    rng = np.random.default_rng(seed)

    fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        trials=trials,
        max_evals=max_evals,
        verbose=False,
        rstate=rng,
    )

    # Prefer the last TPE-suggested point (after replay) so we return a fresh
    # suggestion. Fall back to best_trial if every suggested point was filtered
    # out by constraints.
    new_suggestions = list(trials.trials[len(known):])
    if new_suggestions:
        chosen = new_suggestions[-1]["misc"]["vals"]
    else:
        chosen = trials.best_trial["misc"]["vals"]

    return _decode_trial(chosen, parameters)


@beartype
def _resolve_extra_iters() -> int:
    """Allow tests to override the TPE iteration count via env var."""
    raw = os.environ.get("OMNIOPT_TPE_EXTRA_ITERS")
    if raw is None:
        return 50
    try:
        return max(int(raw), 0)
    except ValueError:
        print(f"Warning: OMNIOPT_TPE_EXTRA_ITERS={raw!r} is not an int, ignoring.")
        return 50


@beartype
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python3 .tpe.py <path>")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.isdir(path):
        print(f"Error: The path '{path}' is not a valid folder.")
        sys.exit(2)

    work = Path(path)
    json_file_path = work / "input.json"
    results_file_path = work / "results.json"

    try:
        with json_file_path.open(mode="r", encoding="utf-8") as fh:
            data: Any = json.load(fh)
    except FileNotFoundError:
        print(f"Error: {json_file_path} not found.")
        sys.exit(3)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON in {json_file_path}.")
        sys.exit(4)

    next_point = generate_tpe_point(data, extra_iters=_resolve_extra_iters())

    with results_file_path.open(mode="w", encoding="utf-8") as fh:
        json.dump({"parameters": next_point}, fh, indent=4)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("You pressed CTRL-c.")
        sys.exit(1)
