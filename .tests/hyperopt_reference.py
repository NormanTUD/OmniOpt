#!/usr/bin/env python3
"""Hyperopt reference implementation for comparing OmniOpt's TPE (.tpe.py).

This script reads the same input.json format that OmniOpt's `.tpe.py` reads,
reconstructs the search space with hyperopt, replays the existing trials into
hyperopt's Trials object, then runs TPE for a fixed number of additional
iterations and writes the suggested point to results.json.

It mirrors the *behaviour* of OmniOpt's `.tpe.py` as closely as possible: the
objective function returns 0.0 for constraint-satisfying points and 1e6 (or
-1e6 for maximize) for constraint-violating points, exactly like
``wrapped_objective`` in ``.tpe.py``. This makes the comparison apples-to-
apples: both optimizers get the same information and try to do the same thing.

Usage:
    python3 hyperopt_reference.py <path>

Where ``<path>`` contains an ``input.json`` file. The script writes
``results.json`` next to it.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from hyperopt import fmin, hp, tpe, Trials, STATUS_OK  # noqa: F401


def build_space(parameters: dict) -> dict:
    """Build a hyperopt search space from the OmniOpt parameter definition."""
    space: dict = {}
    for name, p in parameters.items():
        ptype = p["parameter_type"]
        if ptype == "RANGE":
            lo, hi = p["range"]
            if p["type"] == "INT":
                # quniform with q=1 gives int-valued samples
                space[name] = hp.quniform(name, float(lo), float(hi), 1)
            elif p["type"] == "FLOAT":
                space[name] = hp.uniform(name, float(lo), float(hi))
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


def check_constraint(constraint: str, params: dict) -> bool:
    return bool(eval(constraint, {}, params))  # pylint: disable=eval-used


def constraint_violated(constraints, point: dict) -> bool:
    if not constraints:
        return False
    return any(not check_constraint(c, point) for c in constraints)


def replay_objective_factory(known_trials, constraints, direction):
    """Build an objective that mirrors OmniOpt's ``wrapped_objective``.

    The first ``len(known_trials)`` invocations replay the recorded losses so
    hyperopt can build its TPE model from real history. Subsequent invocations
    return the constraint-aware dummy loss used by OmniOpt's wrapper.
    """
    iter_known = iter(known_trials)
    penalty = 1e6 if direction == "minimize" else -1e6
    replay_done = [False]

    def objective(params):
        if not replay_done[0]:
            try:
                _, recorded_loss = next(iter_known)
            except StopIteration:
                replay_done[0] = True
                return _dummy_loss(params, constraints, penalty)
            return {"loss": recorded_loss, "status": STATUS_OK}
        return _dummy_loss(params, constraints, penalty)

    return objective


def _dummy_loss(params, constraints, penalty):
    if constraint_violated(constraints, params):
        return {"loss": penalty, "status": STATUS_OK}
    return {"loss": 0.0, "status": STATUS_OK}


def parse_objectives(objectives: dict) -> str:
    if len(objectives) != 1:
        raise ValueError("Only single-objective optimization is supported.")
    _, goal = next(iter(objectives.items()))
    goal = goal.lower()
    if goal not in ("min", "max"):
        raise ValueError(f"Unsupported objective direction: {goal}")
    return "minimize" if goal == "min" else "maximize"


def extract_known_trials(trials_data, result_key):
    out = []
    for entry in trials_data:
        if not isinstance(entry, list) or len(entry) != 2:
            continue
        param_dict, result_dict = entry
        if not isinstance(result_dict, dict) or result_key not in result_dict:
            continue
        out.append((param_dict, float(result_dict[result_key])))
    return out


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python3 hyperopt_reference.py <path>", file=sys.stderr)
        return 1

    path = Path(sys.argv[1])
    if not path.is_dir():
        print(f"Error: {path} is not a directory.", file=sys.stderr)
        return 2

    input_path = path / "input.json"
    results_path = path / "results.json"

    with input_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    parameters = data["parameters"]
    constraints = data.get("constraints", []) or []
    objectives = data.get("objectives", {})
    trials_data = data.get("trials", []) or []

    direction = parse_objectives(objectives)
    result_key = next(iter(objectives))

    space = build_space(parameters)
    known = extract_known_trials(trials_data, result_key)

    trials = Trials()
    objective = replay_objective_factory(known, constraints, direction)

    # We need max_evals to cover both the replay and the new TPE-suggested trials.
    extra_iters = int(os.environ.get("HYPEROPT_EXTRA_ITERS", "50"))
    max_evals = len(known) + extra_iters

    fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        trials=trials,
        max_evals=max(max_evals, 1),
        verbose=False,
        rstate=__import__("numpy").random.default_rng(42),
    )

    # Pick the last TPE-suggested point (after replay) so we compare a fresh
    # suggestion, not one of the seeded ones.
    new_suggestions = [t for t in trials.trials[len(known):]]
    if not new_suggestions:
        # If everything was filtered out by constraints, fall back to best.
        best_trial = trials.best_trial
        chosen_params = best_trial["misc"]["vals"]
    else:
        last = new_suggestions[-1]
        chosen_params = last["misc"]["vals"]

    point = {}
    for name, raw in chosen_params.items():
        vals = list(raw)
        value = vals[0] if vals else None
        # Cast back to the declared type (hyperopt stores quniform as float).
        if name in parameters:
            p = parameters[name]
            if p["parameter_type"] == "RANGE" and p["type"] == "INT":
                value = int(round(float(value)))
            elif p["parameter_type"] == "CHOICE":
                option_list = p["values"]
                value = option_list[int(value)]
            elif p["parameter_type"] == "FIXED":
                value = p["value"]
        point[name] = value

    with results_path.open("w", encoding="utf-8") as fh:
        json.dump({"parameters": point}, fh, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
