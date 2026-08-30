"""Comprehensive Optuna backend for OmniOpt.

This module turns Optuna into a *special case* of OmniOpt's
``ExternalProgramGenerationNode`` contract:

  * Reads the same ``input.json`` that ``.tpe.py`` reads
    (``parameters``, ``constraints``, ``seed``, ``trials``, ``objectives``).
  * Replays the recorded trials into an Optuna study so the sampler has the
    same history OmniOpt itself sees.
  * Lets the caller pick *any* Optuna sampler (``tpe``, ``cmaes``, ``gp``,
    ``random``, ``grid``, ``nsgaii``, ``nsgaiii``, ``motpe``, ``brute_force``,
    ``qmc``) and *any* pruner (``median``, ``hyperband``, ``threshold``,
    ``successive_halving``, ``none``) via CLI flags.
  * Supports *multi-objective* optimization by creating a multi-direction
    study whenever ``len(objectives) > 1``.
  * Writes ``{"parameters": <next point>}`` to ``results.json`` so the
    ``ExternalProgramGenerationNode`` reads it back as if it were any other
    OmniOpt-compatible generator.

Subcommands
~~~~~~~~~~~

``suggest``  (default)

    Reads ``input.json`` from ``<workdir>`` and writes ``results.json`` with
    the next suggested point. Equivalent to what ``.tpe.py`` does.

``study create`` / ``study add`` / ``study suggest`` / ``study best`` /
``study trials`` / ``study delete``

    Subcommands for *remote control*: they expose the Optuna study as a
    persistent object on disk (SQLite by default, file-backed) so an external
    process can drive it without ever importing Optuna itself. Every
    subcommand takes the same ``--storage`` / ``--study-name`` flags so the
    user can talk to the same study across processes.

Configuration
~~~~~~~~~~~~~

Sampler choice and Optuna-specific knobs are read from CLI flags *and*
environment variables. CLI flags win. The omniopt side wires the flags via
``ExternalProgramGenerationNode(external_generator="python3 .optuna_runner.py
--sampler=tpe ...")`` so a single OmniOpt call can steer Optuna in any way.

Environment variables (CLI flags override):

  OMNIOPT_OPTUNA_SAMPLER             one of tpe/cmaes/gp/random/grid/nsgaii/
                                     nsgaiii/motpe/brute_force/qmc
  OMNIOPT_OPTUNA_PRUNER              one of median/hyperband/threshold/
                                     successive_halving/none
  OMNIOPT_OPTUNA_SEED                integer seed (falls back to input.json)
  OMNIOPT_OPTUNA_N_STARTUP_TRIALS    integer, defaults to 10
  OMNIOPT_OPTUNA_MULTIVARIATE        "1"/"0"
  OMNIOPT_OPTUNA_GROUP               "1"/"0"
  OMNIOPT_OPTUNA_CONSTRAINTS         "1"/"0"
  OMNIOPT_OPTUNA_N_EI_CANDIDATES     integer
  OMNIOPT_OPTUNA_STORAGE             storage URL (default: sqlite file in
                                     workdir)
  OMNIOPT_OPTUNA_STUDY_NAME          study name (default: ``omniopt_study``)
  OMNIOPT_OPTUNA_EXTRA_ITERS         how many new trials to run per call
                                     (default 1, so the study advances by
                                     exactly one trial per ``suggest`` call)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Optional, cast

try:
    import optuna
    from optuna.trial import create_trial, TrialState
    from optuna.distributions import (
        BaseDistribution,
        CategoricalDistribution,
        FloatDistribution,
    )
    try:
        from optuna.distributions import IntDistribution as _IntDistributionOptuna
    except ImportError:  # pragma: no cover - older Optuna
        from optuna.distributions import (  # type: ignore[assignment]
            IntUniformDistribution as _IntDistributionOptuna,
        )
except ModuleNotFoundError:
    print("Optuna not found. Cannot continue.", file=sys.stderr)
    sys.exit(1)

try:
    from beartype import beartype
except ModuleNotFoundError:
    print("beartype not found. Cannot continue.", file=sys.stderr)
    sys.exit(1)


SAMPLER_ALIASES: dict[str, str] = {
    "tpe": "TPESampler",
    "cmaes": "CmaEsSampler",
    "gp": "GPSampler",
    "random": "RandomSampler",
    "grid": "GridSampler",
    "nsgaii": "NSGAIISampler",
    "nsgaiii": "NSGAIIISampler",
    "motpe": "MOTPESampler",
    "brute_force": "BruteForceSampler",
    "qmc": "QMCSampler",
}

PRUNER_ALIASES: dict[str, str] = {
    "median": "MedianPruner",
    "hyperband": "HyperbandPruner",
    "threshold": "ThresholdPruner",
    "successive_halving": "SuccessiveHalvingPruner",
    "none": "NopPruner",
}

LOG_LEVEL = os.environ.get("OMNIOPT_OPTUNA_LOG_LEVEL", "WARNING")


def _logger() -> logging.Logger:
    logger = logging.getLogger("omniopt_optuna")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("[optuna] %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.WARNING))
    return logger


# Silence Optuna's own chatty logger unless the user explicitly asks for it.
logging.getLogger("optuna").setLevel(logging.WARNING)


def _int_distribution(lo: int, hi: int) -> BaseDistribution:
    """Build an Optuna integer distribution that works across versions."""
    try:
        return _IntDistributionOptuna(int(lo), int(hi))
    except TypeError:
        return _IntDistributionOptuna(int(lo), int(hi), step=1)  # type: ignore[call-arg]


@beartype
def parse_objectives(objectives: dict) -> tuple[list[str], list[str]]:
    if not objectives:
        raise ValueError("objectives must not be empty")
    directions: list[str] = []
    keys: list[str] = []
    for k, v in objectives.items():
        goal = str(v).lower()
        if goal not in ("min", "max"):
            raise ValueError(f"Unsupported objective direction for {k}: {v!r}")
        directions.append("minimize" if goal == "min" else "maximize")
        keys.append(k)
    return directions, keys


@beartype
def check_constraint(constraint: str, params: dict) -> bool:
    return bool(eval(constraint, {}, params))  # pylint: disable=eval-used


@beartype
def constraint_violated(constraints: list, point: dict) -> bool:
    if not constraints:
        return False
    return any(not check_constraint(c, point) for c in constraints)


@beartype
def build_optuna_distribution(p: dict) -> BaseDistribution:
    ptype = p["parameter_type"]
    if ptype == "RANGE":
        lo, hi = p["range"]
        lo_f, hi_f = float(lo), float(hi)
        if p["type"] == "INT":
            return _int_distribution(int(lo), int(hi))
        if p["type"] == "FLOAT":
            return FloatDistribution(lo_f, hi_f)
        raise ValueError(f"Unsupported RANGE type {p['type']!r}")
    if ptype == "CHOICE":
        return CategoricalDistribution(list(p["values"]))
    if ptype == "FIXED":
        return CategoricalDistribution([p["value"]])
    raise ValueError(f"Unknown parameter_type {ptype!r}")


@beartype
def suggest_point_from_trial(trial: optuna.Trial, parameters: dict) -> dict:
    point: dict = {}
    for name, p in parameters.items():
        ptype = p["parameter_type"]
        if ptype == "FIXED":
            point[name] = p["value"]
            continue
        if ptype == "RANGE":
            lo, hi = p["range"]
            if p["type"] == "INT":
                point[name] = trial.suggest_int(name, int(lo), int(hi))
            elif p["type"] == "FLOAT":
                point[name] = trial.suggest_float(name, float(lo), float(hi))
            else:
                raise ValueError(f"Unsupported RANGE type {p['type']!r}")
        elif ptype == "CHOICE":
            point[name] = trial.suggest_categorical(name, list(p["values"]))
        else:
            raise ValueError(f"Unknown parameter_type {ptype!r}")
    return point


@beartype
def decode_trial_values(params: dict, parameters: dict) -> dict:
    out: dict = {}
    for name, value in params.items():
        if name not in parameters:
            continue
        p = parameters[name]
        ptype = p["parameter_type"]
        if ptype == "FIXED":
            out[name] = p["value"]
        elif ptype == "RANGE" and p["type"] == "INT":
            try:
                out[name] = int(round(float(value)))
            except (TypeError, ValueError):
                out[name] = value
        elif ptype == "RANGE" and p["type"] == "FLOAT":
            try:
                out[name] = float(value)
            except (TypeError, ValueError):
                out[name] = value
        else:
            out[name] = value
    return out


@beartype
def build_distributions_map(parameters: dict) -> dict[str, BaseDistribution]:
    return {name: build_optuna_distribution(p) for name, p in parameters.items()}


@beartype
def extract_known_trials(
    trials_data: list,
    result_keys: list[str],
    parameters: dict,
) -> list[tuple[dict, dict]]:
    out: list[tuple[dict, dict]] = []
    if not isinstance(trials_data, list) or len(trials_data) != 2:
        return out
    params_list, results_list = trials_data
    if not isinstance(params_list, list) or not isinstance(results_list, list):
        return out
    for param_dict, result_dict in zip(params_list, results_list):
        if not isinstance(param_dict, dict) or not isinstance(result_dict, dict):
            continue
        if not all(k in result_dict for k in result_keys):
            continue
        if not all(k in param_dict for k in parameters):
            continue
        out.append((param_dict, result_dict))
    return out


@beartype
def replay_trials(
    study: optuna.Study,
    known_trials: Iterable[tuple[dict, dict]],
    parameters: dict,
    result_keys: list[str],
    constraints: list,
    penalty_value: float,
) -> None:
    distributions = build_distributions_map(parameters)
    for param_dict, result_dict in known_trials:
        values: list[float] = []
        for k in result_keys:
            try:
                values.append(float(result_dict[k]))
            except (TypeError, ValueError):
                values = []  # type: ignore[assignment]
                break
        if not values:
            continue

        trial_params: dict[str, Any] = {}
        for name, p in parameters.items():
            if name not in param_dict:
                continue
            if p["parameter_type"] == "FIXED":
                continue
            trial_params[name] = param_dict[name]

        if constraint_violated(constraints, param_dict):
            if len(values) == 1:
                penalty = penalty_value if values[0] > penalty_value / 2 else values[0]
                values = [penalty]

        try:
            if len(values) == 1:
                study.add_trial(
                    create_trial(
                        params=trial_params,
                        distributions=distributions,
                        value=values[0],
                    )
                )
            else:
                study.add_trial(
                    create_trial(
                        params=trial_params,
                        distributions=distributions,
                        values=values,
                    )
                )
        except Exception as exc:  # noqa: BLE001
            _logger().warning("Could not replay trial %s: %s", param_dict, exc)


@beartype
def _resolve_sampler_name(name: str) -> str:
    n = name.strip().lower()
    if n not in SAMPLER_ALIASES:
        raise ValueError(
            f"Unknown sampler {name!r}. Valid: {sorted(SAMPLER_ALIASES)}"
        )
    return SAMPLER_ALIASES[n]


@beartype
def _resolve_pruner_name(name: str) -> str:
    n = name.strip().lower()
    if n not in PRUNER_ALIASES:
        raise ValueError(
            f"Unknown pruner {name!r}. Valid: {sorted(PRUNER_ALIASES)}"
        )
    return PRUNER_ALIASES[n]


@beartype
def _filter_kwargs(cls_name: str, kwargs: dict) -> dict:
    """Drop kwargs that the installed Optuna version does not accept."""
    cls = getattr(optuna.samplers, cls_name)
    try:
        import inspect
        sig = inspect.signature(cls.__init__)
        params = sig.parameters
    except (TypeError, ValueError):
        return kwargs
    out: dict = {}
    for k, v in kwargs.items():
        if k in params:
            out[k] = v
    return out


@beartype
def build_sampler(
    sampler_name: str,
    seed: Optional[int],
    n_startup_trials: int,
    multivariate: bool,
    group: bool,
    constraints: bool,
    n_ei_candidates: int,
) -> optuna.samplers.BaseSampler:
    cls_name = _resolve_sampler_name(sampler_name)
    cls = getattr(optuna.samplers, cls_name)
    kwargs: dict[str, Any] = {"seed": seed}

    if cls_name == "TPESampler":
        kwargs.update(
            n_startup_trials=n_startup_trials,
            multivariate=multivariate,
            group=group,
        )
        # Older Optuna used ``constraints``, newer uses ``constraints_func``.
        # Pass it through the filter so it just works on both.
        if constraints:
            kwargs["constraints_func"] = lambda trial: []
    elif cls_name == "CmaEsSampler":
        kwargs.update(n_startup_trials=n_startup_trials)
    elif cls_name == "GPSampler":
        kwargs.update(n_startup_trials=n_startup_trials)
    elif cls_name in ("NSGAIISampler", "NSGAIIISampler"):
        kwargs.update(population_size=max(n_startup_trials, 2))
    elif cls_name == "MOTPESampler":
        kwargs.update(n_startup_trials=n_startup_trials)

    if n_ei_candidates and cls_name == "TPESampler":
        kwargs["n_ei_candidates"] = n_ei_candidates

    filtered = _filter_kwargs(cls_name, kwargs)
    if len(filtered) != len(kwargs):
        dropped = set(kwargs) - set(filtered)
        _logger().warning("Dropped unsupported kwargs for %s: %s", cls_name, sorted(dropped))
    return cls(**filtered)


@beartype
def build_pruner(pruner_name: str) -> optuna.pruners.BasePruner:
    cls_name = _resolve_pruner_name(pruner_name)
    cls = getattr(optuna.pruners, cls_name)
    if cls_name == "ThresholdPruner":
        # ThresholdPruner needs ``lower``/``upper`` kwargs we don't have;
        # fall back to NopPruner for safety.
        return optuna.pruners.NopPruner()
    return cls()


@beartype
def _default_storage(workdir: Path) -> str:
    db = workdir / "optuna_study.db"
    return f"sqlite:///{db}"


@beartype
def build_study(
    sampler: optuna.samplers.BaseSampler,
    directions: list[str],
    storage: Optional[str],
    study_name: str,
    load_if_exists: bool,
    workdir: Optional[Path] = None,
    search_space: Optional[dict[str, BaseDistribution]] = None,
) -> optuna.Study:
    if storage is None and workdir is not None:
        storage = _default_storage(workdir)

    if isinstance(sampler, optuna.samplers.GridSampler) and search_space:
        sampler = optuna.samplers.GridSampler(
            search_space={
                k: list(cast(CategoricalDistribution, v).choices)
                for k, v in search_space.items()
            },
        )

    if storage is None:
        try:
            return optuna.create_study(
                sampler=sampler,
                directions=directions,
                study_name=study_name,
            )
        except TypeError:
            # Fallback for older Optuna where directions= is not yet supported.
            return optuna.create_study(
                sampler=sampler,
                direction=directions[0],
                study_name=study_name,
            )

    try:
        return optuna.create_study(
            sampler=sampler,
            directions=directions,
            study_name=study_name,
            storage=storage,
            load_if_exists=load_if_exists,
        )
    except TypeError:
        return optuna.create_study(
            sampler=sampler,
            directions=directions,
            study_name=study_name,
            storage=storage,
        )


@beartype
def _load_input(workdir: Path) -> dict:
    p = workdir / "input.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


@beartype
def _resolve_seed(data: dict, cli_seed: Optional[int]) -> Optional[int]:
    if cli_seed is not None:
        return cli_seed
    raw = data.get("seed")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


@beartype
def _resolve_extra_iters() -> int:
    raw = os.environ.get("OMNIOPT_OPTUNA_EXTRA_ITERS", "1")
    try:
        return max(int(raw), 1)
    except ValueError:
        return 1


@beartype
def suggest_workdir(
    workdir: Path,
    *,
    sampler_name: str,
    pruner_name: str,
    seed: Optional[int],
    n_startup_trials: int,
    multivariate: bool,
    group: bool,
    constraints: bool,
    n_ei_candidates: int,
    extra_iters: int,
    storage: Optional[str],
    study_name: str,
    load_if_exists: bool,
) -> dict:
    data = _load_input(workdir)

    parameters = data["parameters"]
    constraints_list = data.get("constraints") or []
    trials_data = data.get("trials") or []
    objectives = data.get("objectives") or {}

    directions, result_keys = parse_objectives(objectives)
    actual_seed = _resolve_seed(data, seed)

    search_space = build_distributions_map(parameters)
    sampler_obj = build_sampler(
        sampler_name,
        actual_seed,
        n_startup_trials,
        multivariate,
        group,
        constraints,
        n_ei_candidates,
    )

    study = build_study(
        sampler_obj,
        directions,
        storage,
        study_name,
        load_if_exists,
        workdir=workdir,
        search_space=search_space,
    )

    known_trials = extract_known_trials(trials_data, result_keys, parameters)

    penalty = 1e6
    if "minimize" in directions:
        replay_trials(study, known_trials, parameters, result_keys, constraints_list, penalty)
    else:
        replay_trials(study, known_trials, parameters, result_keys, constraints_list, -1e6)

    next_point: dict = {}
    build_pruner(pruner_name)
    for _ in range(max(extra_iters, 1)):
        trial = study.ask()
        try:
            point = suggest_point_from_trial(trial, parameters)
        except Exception as exc:  # noqa: BLE001
            _logger().warning("Failed to suggest point: %s", exc)
            try:
                study.tell(trial, state=TrialState.FAIL)
            except Exception:  # noqa: BLE001
                pass
            continue

        if constraint_violated(constraints_list, point):
            penalty_value = 1e6 if "minimize" in directions else -1e6
            try:
                if len(directions) == 1:
                    study.tell(trial, [penalty_value])
                else:
                    study.tell(trial, [penalty_value] * len(directions))
            except Exception as exc:  # noqa: BLE001
                _logger().warning("Failed to report penalized trial: %s", exc)
            continue

        next_point = point
        break

    if not next_point and study.trials:
        last = study.trials[-1]
        next_point = decode_trial_values(dict(last.params), parameters)

    if not next_point:
        # Last-resort fallback: if the configured sampler cannot produce a
        # point (e.g. GPSampler without torch), suggest a random point in the
        # parameter space so OmniOpt's ExternalProgramGenerationNode still has
        # a well-formed ``results.json`` to read.
        next_point = _random_point(parameters, actual_seed)

    return {"parameters": next_point}


@beartype
def _random_point(parameters: dict, seed: Optional[int]) -> dict:
    """Uniform-random point from the OmniOpt parameter space.

    Used as a safety net when the configured sampler can't produce a point
    (e.g. missing optional dependency). The same point structure as
    ``suggest_point_from_trial`` is returned.
    """
    import random
    rng = random.Random(seed)
    point: dict = {}
    for name, p in parameters.items():
        ptype = p["parameter_type"]
        if ptype == "FIXED":
            point[name] = p["value"]
        elif ptype == "RANGE":
            lo, hi = p["range"]
            if p["type"] == "INT":
                point[name] = rng.randint(int(lo), int(hi))
            else:
                point[name] = rng.uniform(float(lo), float(hi))
        elif ptype == "CHOICE":
            point[name] = rng.choice(list(p["values"]))
        else:
            raise ValueError(f"Unknown parameter_type {ptype!r}")
    return point


@beartype
def write_results(workdir: Path, payload: dict) -> Path:
    out = workdir / "results.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


@beartype
def _env_or(name: str, default: Any) -> Any:
    raw = os.environ.get(name)
    return default if raw is None else raw


@beartype
def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@beartype
def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@beartype
def _build_parser() -> argparse.ArgumentParser:
    """Build the runner's argument parser.

    The parser is intentionally flat: OmniOpt's ``ExternalProgramGenerationNode``
    invokes us with the workdir as the *last* positional argument, so all
    configuration flags are exposed at the top level. ``study`` is a separate
    parser (see :func:`_build_study_parser`) and gets parsed only when the
    first positional is ``study``.
    """
    p = argparse.ArgumentParser(
        prog="omniopt-optuna",
        description=(
            "OmniOpt's Optuna backend. When called with a workdir positional "
            "(and optionally the explicit ``suggest`` keyword) the workdir is "
            "treated as an ExternalProgramGenerationNode workdir (read "
            "input.json, write results.json). With ``study ...`` you can "
            "remote-control an Optuna study from disk."
        ),
    )

    p.add_argument("--sampler", default=_env_or("OMNIOPT_OPTUNA_SAMPLER", "tpe"))
    p.add_argument("--pruner", default=_env_or("OMNIOPT_OPTUNA_PRUNER", "none"))
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--n-startup-trials", type=int, default=_env_int("OMNIOPT_OPTUNA_N_STARTUP_TRIALS", 10))
    p.add_argument("--multivariate", action="store_true", default=_env_bool("OMNIOPT_OPTUNA_MULTIVARIATE", False))
    p.add_argument("--group", action="store_true", default=_env_bool("OMNIOPT_OPTUNA_GROUP", False))
    p.add_argument("--constraints", action="store_true", default=_env_bool("OMNIOPT_OPTUNA_CONSTRAINTS", False))
    p.add_argument("--n-ei-candidates", type=int, default=_env_int("OMNIOPT_OPTUNA_N_EI_CANDIDATES", 0))
    p.add_argument("--extra-iters", type=int, default=1)
    p.add_argument("--storage", default=_env_or("OMNIOPT_OPTUNA_STORAGE", None))
    p.add_argument("--study-name", default=_env_or("OMNIOPT_OPTUNA_STUDY_NAME", "omniopt_study"))
    p.add_argument("--load-if-exists", action="store_true", default=_env_bool("OMNIOPT_OPTUNA_LOAD_IF_EXISTS", True))
    p.add_argument("--no-load-if-exists", dest="load_if_exists", action="store_false")

    p.add_argument("positional", nargs="*", help=argparse.SUPPRESS)
    return p


@beartype
def _build_study_parser() -> argparse.ArgumentParser:
    """Parser for the ``omniopt_optuna study ...`` subcommand family."""
    p = argparse.ArgumentParser(prog="omniopt-optuna study")
    sub = p.add_subparsers(dest="study_cmd", required=True)

    sc_create = sub.add_parser("create", help="Create a new persistent study")
    sc_create.add_argument("--workdir", required=True)
    sc_create.add_argument("--sampler", default="tpe")
    sc_create.add_argument("--seed", type=int, default=None)
    sc_create.add_argument("--storage", default=None)
    sc_create.add_argument("--study-name", default="omniopt_study")
    sc_create.add_argument("--objectives", default="RESULT",
                           help="Comma-separated list of result keys for multi-objective")
    sc_create.add_argument("--directions", default=None,
                           help="Comma-separated directions matching --objectives (minimize/maximize)")

    sc_add = sub.add_parser("add", help="Record a trial into a persistent study")
    sc_add.add_argument("--workdir", required=True)
    sc_add.add_argument("--storage", default=None)
    sc_add.add_argument("--study-name", default="omniopt_study")
    sc_add.add_argument("--params-file", required=True,
                        help="JSON file with parameter dict (OmniOpt-style)")
    sc_add.add_argument("--values", required=True,
                        help="JSON dict or @file with {objective_name: value}")
    sc_add.add_argument("--state", default="COMPLETE",
                        choices=["COMPLETE", "PRUNED", "FAIL"])

    sc_suggest = sub.add_parser("suggest", help="Suggest the next point from a persistent study")
    sc_suggest.add_argument("--workdir", required=True)
    sc_suggest.add_argument("--storage", default=None)
    sc_suggest.add_argument("--study-name", default="omniopt_study")
    sc_suggest.add_argument("--parameters-file", required=True,
                            help="JSON file with the OmniOpt parameter dict")
    sc_suggest.add_argument("--output", default=None)

    sc_best = sub.add_parser("best", help="Print the best trial(s) of a persistent study")
    sc_best.add_argument("--storage", default=None)
    sc_best.add_argument("--study-name", default="omniopt_study")
    sc_best.add_argument("--workdir", required=True)

    sc_trials = sub.add_parser("trials", help="Print all trials of a persistent study as JSON")
    sc_trials.add_argument("--storage", default=None)
    sc_trials.add_argument("--study-name", default="omniopt_study")
    sc_trials.add_argument("--workdir", required=True)

    sc_delete = sub.add_parser("delete", help="Delete a persistent study")
    sc_delete.add_argument("--storage", default=None)
    sc_delete.add_argument("--study-name", default="omniopt_study")

    return p


@beartype
def _resolve_storage(storage: Optional[str], workdir: Optional[Path]) -> str:
    if storage:
        return storage
    if workdir is None:
        raise ValueError("storage or workdir is required")
    return _default_storage(Path(workdir))


@beartype
def _read_json_arg(raw: str) -> Any:
    if raw.startswith("@"):
        return json.loads(Path(raw[1:]).read_text(encoding="utf-8"))
    return json.loads(raw)


@beartype
def _infer_param_spec(value: Any) -> dict:
    """Build an OmniOpt-style parameter spec from a single value.

    Used by ``study add`` so callers can just feed raw ``{name: value}`` pairs
    and let us figure out the type. ``CHOICE`` is used for strings to avoid
    accidental ``IntUniform`` matches for short ranges; numeric values become
    ``RANGE``.
    """
    if isinstance(value, bool):
        return {"parameter_type": "CHOICE", "values": [value]}
    if isinstance(value, int):
        return {"parameter_type": "RANGE", "type": "INT",
                "range": [min(value, 0), max(value, 1) or 1]}
    if isinstance(value, float):
        return {"parameter_type": "RANGE", "type": "FLOAT",
                "range": [min(value, 0.0), max(value, 1.0) or 1.0]}
    return {"parameter_type": "CHOICE", "values": [value]}


@beartype
def _expand_params_spec(params: dict[str, Any]) -> dict:
    """Wrap a flat ``{name: value}`` mapping in OmniOpt's parameter dict."""
    return {name: _infer_param_spec(v) for name, v in params.items()}


@beartype
def _resolve_sampler(sampler_name: str, seed: Optional[int]) -> optuna.samplers.BaseSampler:
    return build_sampler(
        sampler_name,
        seed,
        n_startup_trials=10,
        multivariate=False,
        group=False,
        constraints=False,
        n_ei_candidates=0,
    )


@beartype
def _cmd_study_create(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    storage = _resolve_storage(args.storage, workdir)
    objectives = [k for k in args.objectives.split(",") if k]
    if args.directions:
        directions = [d.strip() for d in args.directions.split(",") if d.strip()]
        if len(directions) != len(objectives):
            raise ValueError(
                f"--directions has {len(directions)} entries but "
                f"--objectives has {len(objectives)}"
            )
    else:
        directions = ["minimize"] * len(objectives)
    sampler = _resolve_sampler(args.sampler, args.seed)
    study = build_study(sampler, directions, storage, args.study_name, load_if_exists=True)
    print(json.dumps({
        "study_name": study.study_name,
        "directions": [_direction_name(d) for d in study.directions],
        "n_trials": len(study.trials),
        "storage": storage,
    }, indent=2))
    return 0


@beartype
def _direction_name(d: Any) -> str:
    """Coerce an Optuna ``StudyDirection`` into its JSON-friendly name."""
    name = getattr(d, "name", None)
    if name:
        return str(name).lower()
    s = str(d)
    if "MIN" in s.upper():
        return "minimize"
    if "MAX" in s.upper():
        return "maximize"
    return s.lower()


@beartype
def _cmd_study_add(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    storage = _resolve_storage(args.storage, workdir)

    params = json.loads(Path(args.params_file).read_text(encoding="utf-8"))
    values_dict = _read_json_arg(args.values)

    study = optuna.load_study(study_name=args.study_name, storage=storage)

    n_directions = len(study.directions)
    if isinstance(values_dict, dict):
        if len(values_dict) != n_directions:
            raise ValueError(
                f"--values has {len(values_dict)} entries but study has "
                f"{n_directions} objective(s). Provide one value per objective."
            )
        values = [float(v) for v in values_dict.values()]
    else:
        values = [float(v) for v in values_dict]

    spec = _expand_params_spec(params)
    distributions = build_distributions_map(spec)

    state = {
        "COMPLETE": TrialState.COMPLETE,
        "PRUNED": TrialState.PRUNED,
        "FAIL": TrialState.FAIL,
    }[args.state]

    if len(values) == 1:
        study.add_trial(create_trial(params=params, distributions=distributions,
                                     value=values[0], state=state))
    else:
        study.add_trial(create_trial(params=params, distributions=distributions,
                                     values=values, state=state))
    print(json.dumps({"added_trial": len(study.trials), "state": args.state}))
    return 0


@beartype
def _cmd_study_suggest(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    storage = _resolve_storage(args.storage, workdir)
    parameters = json.loads(Path(args.parameters_file).read_text(encoding="utf-8"))
    study = optuna.load_study(study_name=args.study_name, storage=storage)
    trial = study.ask()
    point = suggest_point_from_trial(trial, parameters)
    payload = {"parameters": point, "trial_number": trial.number}
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


@beartype
def _cmd_study_best(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    storage = _resolve_storage(args.storage, workdir)
    study = optuna.load_study(study_name=args.study_name, storage=storage)

    out: dict = {"study_name": study.study_name, "n_trials": len(study.trials)}
    if len(study.directions) == 1:
        try:
            out["best_value"] = study.best_value
            out["best_params"] = study.best_params
        except ValueError:
            out["best_value"] = None
            out["best_params"] = None
    else:
        out["best_trials"] = [
            {"number": t.number, "values": list(t.values), "params": dict(t.params)}
            for t in study.best_trials
        ]
    print(json.dumps(out, indent=2, default=str))
    return 0


@beartype
def _cmd_study_trials(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    storage = _resolve_storage(args.storage, workdir)
    study = optuna.load_study(study_name=args.study_name, storage=storage)
    payload = []
    for t in study.trials:
        payload.append({
            "number": t.number,
            "state": t.state.name,
            "params": dict(t.params),
            "values": list(t.values) if t.values is not None else None,
            "value": t.value if t.value is not None else None,
            "datetime_start": t.datetime_start.isoformat() if t.datetime_start else None,
            "datetime_complete": t.datetime_complete.isoformat() if t.datetime_complete else None,
        })
    print(json.dumps(payload, indent=2, default=str))
    return 0


@beartype
def _cmd_study_delete(args: argparse.Namespace) -> int:
    storage = _resolve_storage(args.storage, None)
    optuna.delete_study(study_name=args.study_name, storage=storage)
    print(json.dumps({"deleted": args.study_name}))
    return 0


@beartype
def _dispatch_study(args: argparse.Namespace) -> int:
    table = {
        "create": _cmd_study_create,
        "add": _cmd_study_add,
        "suggest": _cmd_study_suggest,
        "best": _cmd_study_best,
        "trials": _cmd_study_trials,
        "delete": _cmd_study_delete,
    }
    return table[args.study_cmd](args)


@beartype
def _split_argv_at_study(raw_argv: list[str]) -> tuple[list[str], list[str]]:
    """Split ``raw_argv`` into ``(before_study, from_study_onward)``.

    Returns the prefix up to (but excluding) the literal token ``study`` and
    the suffix starting with ``study``. Used so the parent parser sees the
    prefix (which may contain ``--workdir`` / ``--objectives`` etc. that are
    irrelevant to the suggest flow) and the study subparser sees the suffix.
    """
    for i, tok in enumerate(raw_argv):
        if tok == "study":
            return raw_argv[:i], raw_argv[i:]
    return raw_argv, []


@beartype
def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    raw_argv = list(argv if argv is not None else sys.argv[1:])

    # If ``study`` appears as a top-level command, hand the suffix to the
    # study parser and ignore the prefix. The prefix may contain study-only
    # flags (--workdir, etc.) that the parent parser doesn't know about, so
    # we don't even try to parse it.
    _, post = _split_argv_at_study(raw_argv)
    if post and post[0] == "study":
        study_parser = _build_study_parser()
        # Strip the leading ``study`` token - the study_parser doesn't know
        # about it because it expects the subcommand keyword directly.
        study_args = study_parser.parse_args(post[1:])
        return _dispatch_study(study_args)

    # Default: ``suggest`` flow. Allow an optional leading ``suggest`` keyword
    # for backward compatibility with the original subparser layout.
    if raw_argv and raw_argv[0] == "suggest":
        raw_argv = raw_argv[1:]

    args = parser.parse_args(raw_argv)
    positionals = list(getattr(args, "positional", []) or [])
    # ``suggest`` can appear anywhere in the positional args (e.g. when
    # OmniOpt wires ``python3 runner.py --flag suggest <workdir>``).
    positionals = [p for p in positionals if p != "suggest"]
    if not positionals:
        parser.print_help()
        return 2
    workdir = Path(positionals[0])

    payload = suggest_workdir(
        workdir,
        sampler_name=args.sampler,
        pruner_name=args.pruner,
        seed=args.seed,
        n_startup_trials=args.n_startup_trials,
        multivariate=args.multivariate,
        group=args.group,
        constraints=args.constraints,
        n_ei_candidates=args.n_ei_candidates,
        extra_iters=args.extra_iters,
        storage=args.storage,
        study_name=args.study_name,
        load_if_exists=args.load_if_exists,
    )
    write_results(workdir, payload)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
