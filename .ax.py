import logging
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

_BACKEND_AVAILABLE: bool = False
_BACKEND_IMPORT_ERROR: Optional[str] = None

try:
    import ax
    from ax.adapter.registry import Cont_X_trans, Y_trans, Generators
    from ax.core.arm import Arm
    from ax.core.base_trial import TrialStatus as _AxTrialStatus
    from ax.core.data import Data
    from ax.core.experiment import Experiment
    from ax.core.generator_run import GeneratorRun
    from ax.core.objective import MultiObjective as _AxMultiObjective
    from ax.core.optimization_config import OptimizationConfig
    from ax.core.parameter import (
        ChoiceParameter as _AxChoiceParameter,
        FixedParameter as _AxFixedParameter,
        ParameterType as _AxParameterType,
        RangeParameter as _AxRangeParameter,
    )
    from ax.core.types import TParameterization
    from ax.core import Metric as _AxMetric
    from ax.exceptions import core as _ax_exc_core
    from ax.exceptions import generation_strategy as _ax_exc_gs
    from ax.generation_strategy.external_generation_node import ExternalGenerationNode
    from ax.generation_strategy.generator_spec import GeneratorSpec
    from ax.generation_strategy.generation_node import GenerationNode
    from ax.generation_strategy.generation_strategy import GenerationStrategy
    from ax.generation_strategy.transition_criterion import MinTrials
    from ax.service.ax_client import AxClient, ObjectiveProperties
    from ax.storage.json_store.registry import CORE_DECODER_REGISTRY
    import botorch
    from botorch import exceptions as _botorch_exc
    _BACKEND_AVAILABLE = True
except ModuleNotFoundError as exc:
    _BACKEND_IMPORT_ERROR = str(exc)
    print(f"\033[93m.ax.py: backend module not available yet: {exc}\033[0m")


SUPPORTED_LOGGER_NAMES: Tuple[str, ...] = ("ax.adapter.base",)


ExternalGenerationNodeBase: Any = None
TParameterizationAlias: Any = None


def get_ext_node_name_kw() -> str:
    """Return the constructor kwarg name ExternalGenerationNode accepts.

    Older backend releases use ``node_name``; newer ones use ``name``.
    """
    return _EXT_NODE_NAME_KW


def get_node_name_kw() -> str:
    return _NODE_NAME_KW


RangeParameterType: Any = None
FixedParameterType: Any = None
ChoiceParameterType: Any = None
ParameterTypeEnum: Any = None


def is_range_parameter(p: Any) -> bool:
    if not _BACKEND_AVAILABLE:
        return False
    return isinstance(p, _AxRangeParameter)


def is_fixed_parameter(p: Any) -> bool:
    if not _BACKEND_AVAILABLE:
        return False
    return isinstance(p, _AxFixedParameter)


def is_choice_parameter(p: Any) -> bool:
    if not _BACKEND_AVAILABLE:
        return False
    return isinstance(p, _AxChoiceParameter)


def get_choice_values_typed(p: Any) -> List[Any]:
    """Return ``p.values`` cast to the parameter's declared value type."""
    raw = list(p.values)
    if p.parameter_type == _AxParameterType.INT:
        return [int(v) for v in raw]
    if p.parameter_type == _AxParameterType.FLOAT:
        return [float(v) for v in raw]
    return [str(v) for v in raw]


def get_range_lower_upper(p: Any) -> Tuple[Union[int, float], Union[int, float]]:
    if p.parameter_type == _AxParameterType.INT:
        return int(p.lower), int(p.upper)
    return float(p.lower), float(p.upper)


def is_int_parameter(p: Any) -> bool:
    return p.parameter_type == _AxParameterType.INT


def is_float_parameter(p: Any) -> bool:
    return p.parameter_type == _AxParameterType.FLOAT


def is_string_parameter(p: Any) -> bool:
    return p.parameter_type == _AxParameterType.STRING


class TrialStatus:
    COMPLETED: Any = None
    ABANDONED: Any = None
    FAILED: Any = None
    RUNNING: Any = None
    CANDIDATE: Any = None
    STAGED: Any = None
    DISPATCHED: Any = None
    EARLY_STOPPED: Any = None


class ParameterValueType:
    INT: Any = None
    FLOAT: Any = None
    STRING: Any = None
    BOOL: Any = None


class SearchSpaceExhausted(Exception):
    pass


class GenerationStrategyRepeatedPoints(Exception):
    pass


class MaxParallelismReachedException(Exception):
    pass


class GenerationStrategyMisconfiguredException(Exception):
    pass


class UserInputError(Exception):
    pass


class UnsupportedError(Exception):
    pass


class DataRequiredError(Exception):
    pass


class BotorchInputDataError(Exception):
    pass


class BotorchModelFittingError(Exception):
    pass


class MultiObjective:
    pass


def init() -> None:
    """Idempotently bring the backend online and install all compatibility shims.

    OmniOpt calls this exactly once at startup.  All other public functions
    in this module assume ``init()`` has already been invoked.
    """
    global ExternalGenerationNodeBase, TParameterizationAlias
    global RangeParameterType, FixedParameterType, ChoiceParameterType, ParameterTypeEnum
    if not _BACKEND_AVAILABLE:
        raise RuntimeError(
            f"Cannot init .ax.py: backend module not available ({_BACKEND_IMPORT_ERROR})"
        )
    ExternalGenerationNodeBase = ExternalGenerationNode
    TParameterizationAlias = TParameterization
    RangeParameterType = _AxRangeParameter
    FixedParameterType = _AxFixedParameter
    ChoiceParameterType = _AxChoiceParameter
    ParameterTypeEnum = _AxParameterType
    TrialStatus.COMPLETED = _AxTrialStatus.COMPLETED
    TrialStatus.ABANDONED = _AxTrialStatus.ABANDONED
    TrialStatus.FAILED = _AxTrialStatus.FAILED
    TrialStatus.RUNNING = _AxTrialStatus.RUNNING
    TrialStatus.CANDIDATE = _AxTrialStatus.CANDIDATE
    TrialStatus.STAGED = _AxTrialStatus.STAGED
    TrialStatus.DISPATCHED = _AxTrialStatus.DISPATCHED
    TrialStatus.EARLY_STOPPED = _AxTrialStatus.EARLY_STOPPED
    ParameterValueType.INT = _AxParameterType.INT
    ParameterValueType.FLOAT = _AxParameterType.FLOAT
    ParameterValueType.STRING = _AxParameterType.STRING
    ParameterValueType.BOOL = _AxParameterType.BOOL
    install_legacy_choice_parameter_decoder()
    install_legacy_objective_decoder()
    install_legacy_trial_decoder()
    detect_generation_node_kwarg_name()


_NODE_NAME_KW: str = "name"
_EXT_NODE_NAME_KW: str = "name"


def detect_generation_node_kwarg_name() -> None:
    """Detect whether the installed backend wants ``name=`` or ``node_name=``.

    Older backend releases use ``node_name=`` on GenerationNode /
    ExternalGenerationNode; newer ones renamed it to ``name=``.  Capture
    the right kwarg name exactly once so :func:`create_generation_node`
    and :func:`create_external_generation_node` always build the
    constructor call correctly.
    """
    global _NODE_NAME_KW, _EXT_NODE_NAME_KW
    import inspect as _inspect
    _gn_params = list(_inspect.signature(GenerationNode.__init__).parameters.keys())
    if "name" in _gn_params:
        _NODE_NAME_KW = "name"
    elif "node_name" in _gn_params:
        _NODE_NAME_KW = "node_name"
    else:
        _NODE_NAME_KW = "name"

    _ext_params = list(_inspect.signature(ExternalGenerationNode.__init__).parameters.keys())
    if "name" in _ext_params:
        _EXT_NODE_NAME_KW = "name"
    elif "node_name" in _ext_params:
        _EXT_NODE_NAME_KW = "node_name"
    else:
        _EXT_NODE_NAME_KW = "name"


def disable_internal_loggers(level: int = logging.CRITICAL) -> None:
    """Silence chatty backend loggers (best-effort)."""
    try:
        from ax.utils.common.logger import disable_loggers as _ax_disable
        _ax_disable(names=list(SUPPORTED_LOGGER_NAMES), level=level)
    except Exception:
        pass


def set_rng_seed(seed: Optional[int]) -> None:
    """Set the RNG seed used by the backend for reproducibility."""
    from ax.utils.common.random import set_rng_seed as _set
    _set(int(seed) if seed is not None else 0)


_decoder_registry: Any = None


def get_decoder_registry() -> Dict[str, Any]:
    global _decoder_registry
    if _decoder_registry is None and _BACKEND_AVAILABLE:
        _decoder_registry = CORE_DECODER_REGISTRY
    return _decoder_registry


def register_decoder(name: str, decoder: Any) -> None:
    reg = get_decoder_registry()
    if reg is not None:
        reg[name] = decoder


def install_legacy_choice_parameter_decoder() -> None:
    """Older backend releases reject newer ``ChoiceParameter`` kwargs.

    State files produced by newer versions include ``sort_values`` /
    ``log_scale`` which older code raises on.  Replace the registered
    decoder with one that silently drops them.
    """
    try:
        import inspect as _inspect
        _accepted = set(_inspect.signature(_AxChoiceParameter.__init__).parameters.keys())
    except Exception:
        return

    for _extra in ("sort_values", "log_scale"):
        if _extra not in _accepted:
            try:
                _orig = _decoder_registry["ChoiceParameter"]
            except KeyError:
                return

            def _safe_choiceparam(*_args: Any, _orig: Any = _orig, **_kwargs: Any) -> Any:
                _kwargs.pop("sort_values", None)
                _kwargs.pop("log_scale", None)
                return _orig(*_args, **_kwargs)

            _safe_choiceparam.__name__ = getattr(_orig, "__name__", "ChoiceParameter")
            _decoder_registry["ChoiceParameter"] = _safe_choiceparam
            break


def _parse_linear_inequality(_inequality: str) -> Optional[Tuple[Dict[str, float], float]]:
    import re as _re

    _ineq = _inequality.strip().replace(" ", "")
    _m = _re.match(r"^(.+?)(<=|>=|<|>|=)(.+)$", _ineq)
    if not _m:
        return None
    _lhs, _op, _rhs = _m.group(1), _m.group(2), _m.group(3)

    try:
        _bound = float(_rhs)
    except ValueError:
        return None

    if _op in (">=", ">"):
        _bound = -_bound
        _lhs = _lhs.replace("+", "|+|").replace("-", "|-|").replace("|+|", "-").replace("|-|", "+")
        if _lhs.startswith("+"):
            _lhs = _lhs[1:]
        elif _lhs.startswith("-"):
            _lhs = "-" + _lhs[1:]
        else:
            _lhs = "-" + _lhs

    _constraint_dict: Dict[str, float] = {}
    for _term in _re.split(r"(?=[+-])", _lhs):
        if not _term:
            continue
        _term_m = _re.match(
            r"^([+-]?)(?:\*?)([+-]?[\d.eE]+)?(?:\*?)([A-Za-z_][A-Za-z_0-9]*)$",
            _term,
        )
        if _term_m is None:
            _term_m = _re.match(r"^([+-]?[\d.eE]+)\*([A-Za-z_][A-Za-z_0-9]*)$", _term)
            if _term_m is None:
                return None
            _sign, _coef, _name = "+", _term_m.group(1), _term_m.group(2)
        else:
            _sign = _term_m.group(1) or "+"
            _coef = _term_m.group(2)
            _name = _term_m.group(3)
        if _coef is None:
            _coef_val: float = 1.0
        else:
            try:
                _coef_val = float(_coef)
            except ValueError:
                return None
        if _sign == "-":
            _coef_val = -_coef_val
        _constraint_dict[_name] = _constraint_dict.get(_name, 0.0) + _coef_val

    if not _constraint_dict:
        return None
    return _constraint_dict, _bound


def install_legacy_objective_decoder() -> None:
    """Backwards-compat shim for older ``OptimizationConfig`` JSON layouts.

    Older state files store objectives as ``{"expression": "-X"}``; the
    current backend wants ``{"metric": {...}, "minimize": bool}``.
    Wrap the decoders so loading either layout works transparently.
    """
    try:
        import sys as _sys
        import inspect as _inspect
        from ax.storage.json_store import decoder as _decoder_mod
        from ax.storage.json_store.decoder import objective_from_json as _orig_obj
    except Exception:
        return

    _optcfg_params = set(_inspect.signature(OptimizationConfig.__init__).parameters.keys())
    _experiment_params = set(_inspect.signature(Experiment.__init__).parameters.keys())

    def _safe_objective_from_json(object_json: Any, **_kwargs: Any) -> Any:
        if isinstance(object_json, dict) \
                and "expression" in object_json \
                and "metric" not in object_json:
            expression = str(object_json.get("expression", "")).strip()
            minimize: Optional[bool]
            if expression.startswith("-") or expression.startswith("(-"):
                minimize = True
            elif expression.startswith("+"):
                minimize = False
            else:
                minimize = None

            metric_name: Optional[str] = None
            tracking_metrics = _kwargs.get("tracking_metrics") or []
            for tm in tracking_metrics:
                if isinstance(tm, dict):
                    name = tm.get("name")
                    if name and name in expression:
                        metric_name = name
                        break

            if metric_name is None:
                import re as _re
                _m = _re.match(r"\(*\s*[+-]?\s*([a-zA-Z_][a-zA-Z_0-9]*)", expression)
                if _m:
                    metric_name = _m.group(1)

            if metric_name is not None:
                lower_is_better = minimize if minimize is not None else True
                object_json = {
                    "metric": {
                        "__type": "Metric",
                        "name": metric_name,
                        "lower_is_better": lower_is_better,
                        "properties": {},
                        "signature_override": None,
                    },
                    "minimize": lower_is_better,
                }
        return _orig_obj(object_json=object_json, **_kwargs)

    _safe_objective_from_json.__name__ = getattr(_orig_obj, "__name__", "objective_from_json")
    _decoder_mod.objective_from_json = _safe_objective_from_json  # type: ignore[assignment]

    _orig_obj_from_json = _decoder_mod.object_from_json

    def _safe_object_from_json(object_json: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(object_json, dict):
            _t = object_json.get("__type")
            if _t == "OptimizationConfig":
                for _extra in list(object_json.keys()):
                    if _extra not in _optcfg_params and _extra != "__type":
                        object_json.pop(_extra, None)
            elif _t == "Experiment":
                for _extra in list(object_json.keys()):
                    if _extra not in _experiment_params and _extra != "__type":
                        object_json.pop(_extra, None)
            elif _t == "Data":
                try:
                    import json as _json
                    _df = object_json.get("df")
                    if isinstance(_df, dict) \
                            and _df.get("__type") == "DataFrame" \
                            and "value" in _df:
                        _decoded = _json.loads(_df["value"])
                        if isinstance(_decoded, dict) and "metric_signature" in _decoded:
                            _decoded.pop("metric_signature")
                            _df["value"] = _json.dumps(_decoded)
                except Exception:
                    pass
            elif _t == "ParameterConstraint":
                _ineq = object_json.get("inequality")
                if isinstance(_ineq, str):
                    _parsed = _parse_linear_inequality(_ineq)
                    if _parsed is not None:
                        _constraint_dict, _bound = _parsed
                        object_json.pop("inequality", None)
                        object_json["constraint_dict"] = _constraint_dict
                        object_json["bound"] = _bound
        return _orig_obj_from_json(object_json, *args, **kwargs)

    _safe_object_from_json.__name__ = getattr(_orig_obj_from_json, "__name__", "object_from_json")
    _decoder_mod.object_from_json = _safe_object_from_json

    _orig_experiment_from_json = _decoder_mod.experiment_from_json

    _experiment_reserved_keys = {"__type", "time_created", "trials", "experiment_type", "data_by_trial"}

    def _safe_experiment_from_json(object_json: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(object_json, dict):
            for _extra in list(object_json.keys()):
                if _extra not in _experiment_params \
                        and _extra not in _experiment_reserved_keys:
                    object_json.pop(_extra, None)
        return _orig_experiment_from_json(object_json, *args, **kwargs)

    _safe_experiment_from_json.__name__ = getattr(
        _orig_experiment_from_json, "__name__", "experiment_from_json"
    )
    _decoder_mod.experiment_from_json = _safe_experiment_from_json

    _orig_generator_run_from_json = _decoder_mod.generator_run_from_json
    _generator_run_params = set(_inspect.signature(GeneratorRun.__init__).parameters.keys())
    _generator_run_reserved = {
        "time_created",
        "generator_run_type",
        "index",
        "objective_thresholds",
    }

    def _safe_generator_run_from_json(object_json: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(object_json, dict):
            object_json.setdefault("index", 0)
            object_json.setdefault("generator_run_type", None)
            for _extra in list(object_json.keys()):
                if _extra not in _generator_run_params \
                        and _extra not in _generator_run_reserved:
                    object_json.pop(_extra, None)
        return _orig_generator_run_from_json(object_json, *args, **kwargs)

    _safe_generator_run_from_json.__name__ = getattr(
        _orig_generator_run_from_json, "__name__", "generator_run_from_json"
    )
    _decoder_mod.generator_run_from_json = _safe_generator_run_from_json

    _decoders_mod: Any = None  # type: ignore[no-redef]
    _trial_fj: Any = None  # type: ignore[no-redef]
    _batch_trial_fj: Any = None  # type: ignore[no-redef]
    try:
        from ax.storage.json_store import decoders as _decoders_mod  # type: ignore[no-redef]
        from ax.storage.json_store.decoders import batch_trial_from_json as _batch_trial_fj  # type: ignore[no-redef]
        from ax.storage.json_store.decoders import trial_from_json as _trial_fj  # type: ignore[no-redef]
    except Exception:
        pass

    _trial_named_params = (
        set(_inspect.signature(_trial_fj).parameters.keys()) if _trial_fj is not None else set()
    )
    _batch_trial_named_params = (
        set(_inspect.signature(_batch_trial_fj).parameters.keys())
        if _batch_trial_fj is not None
        else set()
    )

    _trial_required_defaults: Dict[str, Any] = {
        "abandoned_reason": None,
        "failed_reason": None,
        "run_metadata": None,
        "stop_metadata": None,
        "generation_step_index": None,
        "properties": None,
        "lifecycle_stage": None,
        "status_quo_weight_override": 1.0,
        "abandoned_arms_metadata": {},
        "status_reason": None,
    }

    def _make_safe_trial_fj(orig: Any, named_params: Any) -> Any:
        if orig is None:
            return None
        _defaults = {
            _k: _v for _k, _v in _trial_required_defaults.items() if _k in named_params
        }
        _unwanted = {_k for _k in _trial_required_defaults if _k not in named_params}

        def _wrapper(*_args: Any, **_kwargs: Any) -> Any:
            for _k, _v in _defaults.items():
                _kwargs.setdefault(_k, _v)
            for _k in _unwanted:
                _kwargs.pop(_k, None)
            return orig(*_args, **_kwargs)

        _wrapper.__name__ = getattr(orig, "__name__", "trial_from_json")
        return _wrapper

    if _decoders_mod is not None and _trial_fj is not None:
        _safe_trial_fj = _make_safe_trial_fj(_trial_fj, _trial_named_params)
        _decoders_mod.trial_from_json = _safe_trial_fj  # type: ignore[attr-defined]
        if "ax.storage.json_store.decoder" in _sys.modules:
            _sys.modules["ax.storage.json_store.decoder"].trial_from_json = _safe_trial_fj  # type: ignore[attr-defined]

    if _decoders_mod is not None and _batch_trial_fj is not None:
        _safe_batch_trial_fj = _make_safe_trial_fj(
            _batch_trial_fj, _batch_trial_named_params
        )
        _decoders_mod.batch_trial_from_json = _safe_batch_trial_fj  # type: ignore[attr-defined]
        if "ax.storage.json_store.decoder" in _sys.modules:
            _sys.modules["ax.storage.json_store.decoder"].batch_trial_from_json = _safe_batch_trial_fj  # type: ignore[attr-defined]


install_legacy_trial_decoder = install_legacy_objective_decoder


def _translate_exception(exc: BaseException) -> BaseException:
    if isinstance(exc, _ax_exc_core.SearchSpaceExhausted):
        return SearchSpaceExhausted(str(exc))
    if isinstance(exc, _ax_exc_gs.GenerationStrategyRepeatedPoints):
        return GenerationStrategyRepeatedPoints(str(exc))
    if isinstance(exc, _ax_exc_gs.MaxParallelismReachedException):
        return MaxParallelismReachedException(str(exc))
    if isinstance(exc, _ax_exc_gs.GenerationStrategyMisconfiguredException):
        return GenerationStrategyMisconfiguredException(str(exc))
    if isinstance(exc, _ax_exc_core.UserInputError):
        return UserInputError(str(exc))
    if isinstance(exc, _ax_exc_core.UnsupportedError):
        return UnsupportedError(str(exc))
    if isinstance(exc, _ax_exc_core.DataRequiredError):
        return DataRequiredError(str(exc))
    if isinstance(exc, _botorch_exc.errors.InputDataError):
        return BotorchInputDataError(str(exc))
    if isinstance(exc, _botorch_exc.errors.ModelFittingError):
        return BotorchModelFittingError(str(exc))
    return exc


def create_client(
    *,
    verbose_logging: bool = False,
    enforce_sequential_optimization: bool = False,
    generation_strategy: Optional[Any] = None,
    random_seed: Optional[int] = None,
) -> Any:
    kwargs: Dict[str, Any] = {
        "verbose_logging": verbose_logging,
        "enforce_sequential_optimization": enforce_sequential_optimization,
    }
    if generation_strategy is not None:
        kwargs["generation_strategy"] = generation_strategy
    if random_seed is not None:
        kwargs["random_seed"] = int(random_seed)
    return AxClient(**kwargs)


def load_client_from_json_file(path: str) -> Any:
    return AxClient.load_from_json_file(path)


def save_client_to_json_file(client: Any, path: str) -> None:
    client.save_to_json_file(path)


def save_client_to_database(client: Any) -> None:
    from ax.storage.sqa_store.save import save_experiment as _save_exp
    _save_exp(client.experiment)


def client_to_json_snapshot(client: Any) -> Optional[Dict[str, Any]]:
    return client.to_json_snapshot()


def client_get_trial(client: Any, trial_index: int) -> Any:
    return client.get_trial(trial_index)


def client_get_trials_dataframe(client: Any) -> Any:
    return client.get_trials_data_frame()


def client_experiment(client: Any) -> Any:
    return client.experiment


def client_fetch_data(client: Any) -> Any:
    client.experiment.fetch_data()
    return client_get_trials_dataframe(client)


def client_metric_names(client: Any) -> List[str]:
    return list(client.metric_names)


def client_get_next_trial(client: Any) -> Tuple[Dict[str, Any], int]:
    parameters, trial_index = client.get_next_trial()
    return parameters, int(trial_index)


def client_complete_trial(
    client: Any,
    trial_index: int,
    raw_data: Union[List[Any], Dict[str, Any]],
) -> None:
    try:
        client.complete_trial(trial_index=trial_index, raw_data=raw_data)
    except BaseException as exc:
        raise _translate_exception(exc) from exc


def client_attach_trial(client: Any, arm_params: Dict[str, Any]) -> Tuple[Any, int]:
    try:
        result = client.attach_trial(arm_params)
    except BaseException as exc:
        raise _translate_exception(exc) from exc
    # Newer Ax releases return ``(parameterization, trial_index)``.
    # Older releases returned a ``Trial`` object carrying ``.index``.
    if isinstance(result, tuple) and len(result) == 2:
        parameters, trial_index = result
        return parameters, int(trial_index)
    new_trial = result
    return new_trial, int(getattr(new_trial, "index"))


def client_log_trial_failure(client: Any, trial_index: int) -> None:
    client.log_trial_failure(trial_index=trial_index)


def client_create_experiment(
    client: Any,
    *,
    name: str,
    parameters: Any,
    objectives: Dict[str, Any],
    parameter_constraints: Optional[Sequence[str]] = None,
    choose_generation_strategy_kwargs: Optional[Dict[str, Any]] = None,
    extra_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    kwargs: Dict[str, Any] = {
        "name": name,
        "parameters": parameters,
        "objectives": objectives,
    }
    if parameter_constraints:
        kwargs["parameter_constraints"] = list(parameter_constraints)
    if choose_generation_strategy_kwargs:
        kwargs["choose_generation_strategy_kwargs"] = dict(choose_generation_strategy_kwargs)
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    client.create_experiment(**kwargs)


def add_tracking_metrics(client: Any, metric_names: Sequence[str]) -> None:
    existing = set(client_metric_names(client))
    new_metrics = [_AxMetric(name=k) for k in metric_names if k not in existing]
    if new_metrics:
        client.experiment.add_tracking_metrics(new_metrics)


def search_space_parameter_names(client: Any) -> List[str]:
    return list(client.experiment.search_space.parameters.keys())


def optimization_config(client: Any) -> Any:
    return client.experiment.optimization_config


def experiment_num_trials(client: Any) -> int:
    return int(client.experiment.num_trials)


def experiment_new_trial_from_arm(
    client: Any,
    arm: Any,
    generation_node_name: str,
) -> Tuple[Any, int]:
    generator_run = GeneratorRun(
        arms=[arm],
        generation_node_name=generation_node_name,
    )
    trial = client.experiment.new_trial(generator_run)
    return trial, int(trial.index)


def experiment_get_trial_by_index(client: Any, trial_idx: int) -> Any:
    return client.experiment.trials.get(trial_idx)


def mark_trial_running(trial: Any, *, no_runner_required: bool = True) -> None:
    trial.mark_running(no_runner_required=no_runner_required)


def mark_trial_abandoned(trial: Any, reason: str) -> None:
    trial.mark_abandoned(reason)


def get_trial_arm(trial: Any) -> Any:
    return trial.arms[0]


def get_trial_arm_parameters(trial: Any) -> Dict[str, Any]:
    return dict(trial.arms[0].parameters)


def get_trial_arm_name(trial: Any) -> str:
    return str(trial.arms[0].name)


def get_trial_index(trial: Any) -> int:
    return int(trial.index)


def get_trial_status(trial: Any) -> Any:
    return trial.status


def is_trial_completed(trial: Any) -> bool:
    return trial.status == TrialStatus.COMPLETED


def create_objective(minimize: bool) -> Any:
    return ObjectiveProperties(minimize=bool(minimize))


def create_tracking_metric(name: str) -> Any:
    return _AxMetric(name=name)


def is_multi_objective(experiment: Any) -> bool:
    cfg = experiment.optimization_config
    if cfg is None:
        return False
    if isinstance(cfg.objective, _AxMultiObjective):
        return True
    return bool(getattr(cfg, "is_moo_problem", False))


def get_objective_metric_names(objective: Any) -> List[str]:
    """Return metric names from an Objective/MultiObjective in a backend-agnostic way."""
    names_attr = getattr(objective, "metric_names", None)
    if names_attr is not None:
        return list(names_attr)
    inner = getattr(objective, "objectives", None)
    if inner:
        out: List[str] = []
        for o in inner:
            out.extend(get_objective_metric_names(o))
        return out
    return []


def get_objective_is_minimize(objective: Any) -> Optional[bool]:
    """Return whether ``objective`` minimizes (None if unknown)."""
    if hasattr(objective, "minimize") and isinstance(getattr(objective, "minimize"), bool):
        return bool(objective.minimize)
    inner = getattr(objective, "objectives", None)
    if inner:
        first = inner[0]
        return get_objective_is_minimize(first)
    weights = getattr(objective, "metric_weights", None)
    if isinstance(weights, list) and weights:
        entry = weights[0]
        if isinstance(entry, (tuple, list)) and len(entry) == 2:
            w = entry[1]
            if isinstance(w, (int, float)):
                return w < 0
    return None


def list_objectives(client_or_experiment: Any) -> List[Tuple[str, bool]]:
    """Return ``[(metric_name, minimize), ...]`` for the active experiment.

    Works whether ``client_or_experiment`` is an AxClient or an Experiment
    and whether the underlying objective is single- or multi-objective.
    """
    if hasattr(client_or_experiment, "experiment"):
        experiment = client_or_experiment.experiment
    else:
        experiment = client_or_experiment

    cfg = getattr(experiment, "optimization_config", None)
    if cfg is None:
        return []

    if is_multi_objective(experiment):
        objective = getattr(cfg, "objective", None)
        if objective is None:
            return []

        names = get_objective_metric_names(objective)
        weights = getattr(objective, "metric_weights", None) or []
        weight_map: Dict[str, float] = {}
        if isinstance(weights, list):
            for entry in weights:
                if isinstance(entry, (tuple, list)) and len(entry) == 2:
                    weight_map[entry[0]] = entry[1]
        out: List[Tuple[str, bool]] = []
        for name in names:
            w = weight_map.get(name, -1.0)
            minimize = (not isinstance(w, (int, float))) or w < 0
            out.append((name, minimize))
        return out

    objective = cfg.objective
    if objective is None:
        return []
    names = get_objective_metric_names(objective)
    minimize_opt: Optional[bool] = get_objective_is_minimize(objective)
    is_minimize: bool = True if minimize_opt is None else minimize_opt
    if not names:
        return []
    return [(names[0], is_minimize)]


def create_arm(parameters: Dict[str, Any], name: Optional[str] = None) -> Any:
    if name is None:
        return Arm(parameters=dict(parameters))
    return Arm(parameters=dict(parameters), name=name)


def create_generator_run(arm: Any, generation_node_name: str) -> Any:
    return GeneratorRun(arms=[arm], generation_node_name=generation_node_name)


def get_arm_parameters(arm: Any) -> Dict[str, Any]:
    return dict(arm.parameters)


def get_arm_name(arm: Any) -> str:
    return str(arm.name)


def create_generation_strategy(
    *,
    name: Optional[str] = None,
    nodes: Sequence[Any],
) -> Any:
    kwargs: Dict[str, Any] = {"nodes": list(nodes)}
    if name is not None:
        kwargs["name"] = name
    try:
        return GenerationStrategy(**kwargs)
    except BaseException as exc:
        raise _translate_exception(exc) from exc


def generation_strategy_gen(
    gs: Any,
    *,
    experiment: Any,
    n: int = 1,
) -> Any:
    try:
        return gs.gen(experiment=experiment, n=n)
    except BaseException as exc:
        raise _translate_exception(exc) from exc


def generation_strategy_current_node_name(gs: Any) -> str:
    return str(gs.current_node_name)


def create_generation_node(
    *,
    name: str,
    generator_specs: Sequence[Any],
    should_deduplicate: bool = True,
    transition_criteria: Optional[Sequence[Any]] = None,
) -> Any:
    kwargs: Dict[str, Any] = {
        _NODE_NAME_KW: name,
        "generator_specs": list(generator_specs),
        "should_deduplicate": should_deduplicate,
    }
    if transition_criteria:
        kwargs["transition_criteria"] = list(transition_criteria)
    return GenerationNode(**kwargs)


def create_external_generation_node(
    *,
    external_generator: str,
    name: str,
) -> Any:
    # pylint: disable-next=abstract-class-instantiated
    return ExternalGenerationNode(  # type: ignore[abstract, call-arg]
        external_generator=external_generator,  # type: ignore[call-arg]
        **{_EXT_NODE_NAME_KW: name},  # type: ignore[arg-type]
    )


def create_generator_spec(
    generator: Any,
    *,
    model_kwargs: Optional[Dict[str, Any]] = None,
    model_gen_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    kwargs: Dict[str, Any] = {}
    if model_kwargs:
        kwargs["model_kwargs"] = dict(model_kwargs)
    if model_gen_kwargs:
        kwargs["model_gen_kwargs"] = dict(model_gen_kwargs)
    return GeneratorSpec(generator, **kwargs)


def create_min_trials_transition(
    *,
    threshold: int,
    transition_to: Optional[str],
    count_only_trials_with_data: bool = True,
) -> Any:
    return MinTrials(
        threshold=int(threshold),
        transition_to=str(transition_to) if transition_to is not None else "",
        count_only_trials_with_data=count_only_trials_with_data,
    )


def list_supported_model_names() -> List[str]:
    return list(Generators.__members__.keys())


def select_model(name: Optional[str]) -> Any:
    if not name:
        return Generators.BOTORCH_MODULAR
    upper = str(name).upper()
    members = Generators.__members__
    if upper in members:
        return members[upper]
    return Generators.BOTORCH_MODULAR


def get_transforms(transforms_name: Optional[str]) -> Dict[str, Any]:
    if transforms_name == "Cont_X_trans_Y_trans":
        return {"transforms": Cont_X_trans + Y_trans}
    if transforms_name == "Cont_X_trans":
        return {"transforms": Cont_X_trans}
    return {}


def create_range_parameter(
    *,
    name: str,
    lower: float,
    upper: float,
    value_type: str = "FLOAT",
    log_scale: bool = False,
) -> Any:
    if value_type.upper() == "INT":
        return _AxRangeParameter(
            name=name,
            parameter_type=_AxParameterType.INT,
            lower=int(lower),
            upper=int(upper),
            log_scale=bool(log_scale),
        )
    return _AxRangeParameter(
        name=name,
        parameter_type=_AxParameterType.FLOAT,
        lower=float(lower),
        upper=float(upper),
        log_scale=bool(log_scale),
    )


def create_fixed_parameter(*, name: str, value: Any) -> Any:
    if isinstance(value, bool):
        return _AxFixedParameter(name=name, value=value, parameter_type=_AxParameterType.BOOL)
    if isinstance(value, int):
        return _AxFixedParameter(name=name, value=value, parameter_type=_AxParameterType.INT)
    if isinstance(value, float):
        return _AxFixedParameter(name=name, value=value, parameter_type=_AxParameterType.FLOAT)
    return _AxFixedParameter(name=name, value=str(value), parameter_type=_AxParameterType.STRING)


def create_choice_parameter(
    *,
    name: str,
    values: Sequence[Any],
    value_type: str = "STRING",
    is_ordered: bool = False,
) -> Any:
    if value_type.upper() == "INT":
        return _AxChoiceParameter(
            name=name,
            parameter_type=_AxParameterType.INT,
            values=[int(v) for v in values],
            is_ordered=bool(is_ordered),
        )
    if value_type.upper() == "FLOAT":
        return _AxChoiceParameter(
            name=name,
            parameter_type=_AxParameterType.FLOAT,
            values=[float(v) for v in values],
            is_ordered=bool(is_ordered),
        )
    return _AxChoiceParameter(
        name=name,
        parameter_type=_AxParameterType.STRING,
        values=[str(v) for v in values],
        is_ordered=bool(is_ordered),
    )


def get_range_lower(p: Any) -> Any:
    return p.lower


def get_range_upper(p: Any) -> Any:
    return p.upper


def get_range_value_type(p: Any) -> str:
    return p.parameter_type


def get_fixed_value(p: Any) -> Any:
    return p.value


def get_choice_values(p: Any) -> List[Any]:
    return get_choice_values_typed(p)


def get_parameter_value_type(p: Any) -> str:
    return str(p.parameter_type)


def get_parameter_value_type_name(param_type: Any) -> str:
    if param_type == _AxParameterType.INT:
        return "INT"
    if param_type == _AxParameterType.FLOAT:
        return "FLOAT"
    if param_type == _AxParameterType.STRING:
        return "STRING"
    return "<UNKNOWN>"


def cast_to_parameter_type(p: Any, value: Any) -> Any:
    if isinstance(p, _AxRangeParameter):
        if p.parameter_type == _AxParameterType.INT:
            return int(round(float(value)))
        return float(value)
    if isinstance(p, _AxChoiceParameter):
        if p.parameter_type == _AxParameterType.INT:
            return int(round(float(value)))
        if p.parameter_type == _AxParameterType.FLOAT:
            return float(value)
        return str(value)
    return value


def init_storage_engine(*, url: Optional[str] = None) -> None:
    """Initialize the SQLAlchemy storage backend (idempotent)."""
    try:
        from ax.storage.sqa_store.db import (
            create_all_tables,
            get_engine,
            init_engine_and_session_factory,
        )
    except Exception:
        return

    if url is not None:
        init_engine_and_session_factory(duckdb_engine_url=url)
    else:
        try:
            init_engine_and_session_factory(duckdb_engine_url=None)
        except Exception:
            pass

    try:
        engine = get_engine()
        if engine is not None:
            create_all_tables(engine)
    except Exception:
        pass


def save_generation_strategy_to_database(gs: Any) -> None:
    from ax.storage.sqa_store.save import save_generation_strategy as _save_gs
    _save_gs(gs)


def load_experiment_from_json(path: str) -> Any:
    from ax.storage.json_store.load import load_experiment as _load
    return _load(path)


def save_experiment_to_json(experiment: Any, path: str) -> None:
    from ax.storage.json_store.save import save_experiment as _save
    _save(experiment, path)


def init_database_engine() -> None:
    try:
        from ax.storage.sqa_store.db import (
            create_all_tables,
            get_engine,
            init_engine_and_session_factory,
        )
        init_engine_and_session_factory(duckdb_engine_url=None)
        engine = get_engine()
        if engine is not None:
            create_all_tables(engine)
    except Exception:
        pass


def get_botorch() -> Any:
    """Return the underlying BoTorch module (only for callers that genuinely
    need BoTorch-specific symbols; new code should avoid this)."""
    return botorch


def get_ax() -> Any:
    """Return the underlying Ax module (only for callers that genuinely need
    Ax-specific symbols; new code should avoid this)."""
    return ax


class RandomForestGenerationNode(ExternalGenerationNode if _BACKEND_AVAILABLE else object):  # type: ignore[misc]  # pylint: disable=useless-object-inheritance
    """Backend-agnostic RandomForest-based external generation node.

    Trains an internal RandomForestRegressor on completed trials and
    selects the candidate with the best predicted objective value.  The
    surrounding framework sees it as a regular ExternalGenerationNode.
    """

    def __init__(
        self: Any,
        regressor_options: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        num_samples: int = 1,
    ) -> None:
        import time as _time
        from sklearn.ensemble import RandomForestRegressor

        t_init_start = _time.monotonic()
        super().__init__(**{_EXT_NODE_NAME_KW: "RANDOMFOREST"})
        self.num_samples: int = int(num_samples)
        self.seed: int = int(seed) if seed is not None else 0
        self.regressor: Any = RandomForestRegressor(
            **(regressor_options or {}),
            random_state=self.seed if seed is not None else None,
        )
        self.parameters: Optional[Dict[str, Any]] = None
        self.minimize: Optional[bool] = None
        self.fit_time_since_gen: float = _time.monotonic() - t_init_start

    def update_generator_state(self: Any, experiment: "Experiment", data: "Data") -> None:
        import numpy as np

        search_space = experiment.search_space
        parameter_names = list(search_space.parameters.keys())
        if experiment.optimization_config is None:
            return
        try:
            metric_names = list(experiment.optimization_config.metric_names)
        except AttributeError:
            return

        completed_trials = [
            trial
            for trial in experiment.trials.values()
            if trial.status == TrialStatus.COMPLETED
        ]
        num_completed_trials = len(completed_trials)

        x = np.zeros([num_completed_trials, len(parameter_names)])
        y = np.zeros([num_completed_trials, 1])

        for t_idx, trial in enumerate(completed_trials):
            trial_parameters = trial.arms[0].parameters
            x[t_idx, :] = np.array([trial_parameters[p] for p in parameter_names])
            trial_df = data.df[data.df["trial_index"] == trial.index]
            y[t_idx, 0] = trial_df[trial_df["metric_name"] == metric_names[0]]["mean"].item()

        self.regressor.fit(x, y)
        self.parameters = search_space.parameters

        if isinstance(experiment.optimization_config.objective, _AxMultiObjective):
            self.minimize = experiment.optimization_config.objective.minimize
        else:
            self.minimize = experiment.optimization_config.objective.minimize

    def get_next_candidate(
        self: Any,
        pending_parameters: List["TParameterization"],  # pylint: disable=unused-argument
    ) -> "TParameterization":
        import numpy as np

        if self.parameters is None:
            raise RuntimeError(
                "Parameters are not initialized. Call update_generator_state first."
            )

        ranged_parameters, fixed_values, choice_parameters = self._separate_parameters()
        reverse_choice_map = self._build_reverse_choice_map(choice_parameters)
        ranged_samples = self._generate_ranged_samples(ranged_parameters)
        all_samples = self._build_all_samples(
            ranged_parameters, ranged_samples, fixed_values, choice_parameters
        )

        x_pred = self._build_prediction_matrix(all_samples)
        y_pred = self.regressor.predict(x_pred)

        sorted_indices = (
            np.argsort(y_pred)
            if self.minimize
            else np.argsort(-np.array(y_pred))
        )

        for idx in sorted_indices:
            candidate = all_samples[idx]
            if self._is_within_constraints(list(candidate.values())):
                self._format_best_sample(candidate, reverse_choice_map)
                return candidate

        raise RuntimeError("No valid candidate found within constraints.")

    def _is_within_constraints(self: Any, params_list: list) -> bool:
        if self.experiment.search_space.parameter_constraints:  # pylint: disable=no-member
            param_names = list(self.parameters.keys())
            params = dict(zip(param_names, params_list))

            for constraint in self.experiment.search_space.parameter_constraints:  # pylint: disable=no-member
                if not constraint.check(params):
                    return False

        return True

    def _separate_parameters(self: Any) -> Tuple[list, Dict[str, Any], Dict[str, Any]]:
        ranged_parameters: list = []
        fixed_values: Dict[str, Any] = {}
        choice_parameters: Dict[str, Any] = {}

        for name, param in self.parameters.items():
            if isinstance(param, _AxRangeParameter):
                ranged_parameters.append((name, param.lower, param.upper))
            elif isinstance(param, _AxFixedParameter):
                fixed_values[name] = str(param.value)
            elif isinstance(param, _AxChoiceParameter):
                choice_values = list(param.values)
                choice_value_map = {value: idx for idx, value in enumerate(choice_values)}
                choice_parameters[name] = choice_value_map

        return ranged_parameters, fixed_values, choice_parameters

    def _build_reverse_choice_map(self: Any, choice_parameters: Dict[str, Any]) -> Dict[int, Any]:
        choice_value_map: Dict[Any, int] = {}
        for _, param in choice_parameters.items():
            for value, idx in param.items():
                choice_value_map[value] = idx
        return {idx: value for value, idx in choice_value_map.items()}

    def _generate_ranged_samples(self: Any, ranged_parameters: list) -> Any:
        import numpy as np

        ranged_bounds = np.array([[low, high] for _, low, high in ranged_parameters])
        unit_samples = np.random.random_sample([self.num_samples, len(ranged_bounds)])
        return ranged_bounds[:, 0] + (ranged_bounds[:, 1] - ranged_bounds[:, 0]) * unit_samples

    def _build_all_samples(
        self: Any,
        ranged_parameters: list,
        ranged_samples: Any,
        fixed_values: Dict[str, Any],
        choice_parameters: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        all_samples: List[Dict[str, Any]] = []
        for sample_idx in range(self.num_samples):
            sample = self._build_single_sample(
                sample_idx,
                ranged_parameters,
                ranged_samples,
                fixed_values,
                choice_parameters,
            )
            all_samples.append(sample)
        return all_samples

    def _build_single_sample(
        self: Any,
        sample_idx: int,
        ranged_parameters: list,
        ranged_samples: Any,
        fixed_values: Dict[str, Any],
        choice_parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        import numpy as np

        sample: Dict[str, Any] = {}

        for dim, (name, _, _) in enumerate(ranged_parameters):
            value = ranged_samples[sample_idx, dim].item()
            param = self.parameters.get(name)
            value = self._cast_value(param, name, value)
            sample[name] = value

        for name, val in fixed_values.items():
            val_str = str(int(val)) if float(val).is_integer() else str(float(val))
            sample[name] = val_str

        for name, param in choice_parameters.items():
            param_values_array = list(param.keys())
            choice_index = np.random.choice(param_values_array)

            if self.parameters[name].parameter_type == _AxParameterType.FLOAT:
                sample[name] = float(param[int(choice_index)])
            elif self.parameters[name].parameter_type == _AxParameterType.INT:
                sample[name] = int(round(param[int(choice_index)]))
            elif self.parameters[name].parameter_type == _AxParameterType.STRING:
                value = param[choice_index]
                if isinstance(value, str):
                    sample[name] = value
                else:
                    sample[name] = str(int(value)) if float(value).is_integer() else str(float(value))

        return sample

    def _cast_value(self: Any, param: Any, name: Any, value: Any) -> Union[int, float]:
        if isinstance(param, _AxRangeParameter) and param.parameter_type == "INT":
            return int(round(value))
        if isinstance(param, _AxRangeParameter) and param.parameter_type == "FLOAT":
            return float(value)
        return self._try_convert_to_float(value, name)

    def _try_convert_to_float(self: Any, value: Any, name: str) -> float:
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(
                f"Parameter '{name}' has a non-numeric value: {value}"
            ) from exc

    def _build_prediction_matrix(self: Any, all_samples: List[Dict[str, Any]]) -> Any:
        import numpy as np

        x_pred = np.zeros([self.num_samples, len(self.parameters)])
        for sample_idx, sample in enumerate(all_samples):
            for dim, name in enumerate(self.parameters.keys()):
                x_pred[sample_idx, dim] = sample[name]
        return x_pred

    def _format_best_sample(
        self: Any,
        best_sample: "TParameterization",
        reverse_choice_map: Dict[int, Any],
    ) -> None:
        for name in list(best_sample.keys()):
            param = self.parameters.get(name)
            best_sample_by_name = best_sample[name]

            if isinstance(param, _AxRangeParameter) and param.parameter_type == _AxParameterType.INT:
                if best_sample_by_name is not None:
                    best_sample[name] = int(round(float(best_sample_by_name)))
            elif isinstance(param, _AxChoiceParameter):
                if best_sample_by_name is not None:
                    best_sample[name] = str(reverse_choice_map.get(int(best_sample_by_name)))


if _BACKEND_AVAILABLE:
    register_decoder("RandomForestGenerationNode", RandomForestGenerationNode)


_INTERACTIVE_CLI_PROMPT_SPECIALS_DEFAULT: Dict[str, Any] = {}


class InteractiveCLIGenerationNode(ExternalGenerationNode if _BACKEND_AVAILABLE else object):  # type: ignore[misc]  # pylint: disable=useless-object-inheritance
    """Backend-agnostic interactive-CLI generation node.

    Instead of spawning a subprocess, this node asks the user on the
    command line (via *rich*) for the next candidate hyperparameter
    set.  Prompts come pre-filled with sensible defaults derived from
    the parameter definition:

    * ``RangeParameter`` (INT/FLOAT) → midpoint (cast to int for INT)
    * ``ChoiceParameter``           → first element of ``param.values``
    * ``FixedParameter``            → its fixed value (prompt skipped)

    The user can press *Enter* to accept the default or type a new
    value (validated & cast automatically).
    """

    def __init__(
        self: Any,
        seed: Optional[int] = None,
        prompt_specials: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(**{_EXT_NODE_NAME_KW: "INTERACTIVE_CLI"})
        self.seed: int = int(seed) if seed is not None else 0
        self.parameters: Optional[Dict[str, Any]] = None
        self.minimize: Optional[bool] = None
        self.data: Optional[Any] = None
        self.constraints: Optional[Any] = None
        self.prompt_specials: Dict[str, Any] = dict(
            prompt_specials or _INTERACTIVE_CLI_PROMPT_SPECIALS_DEFAULT
        )

    def update_generator_state(self: Any, experiment: "Experiment", data: "Data") -> None:
        self.parameters = experiment.search_space.parameters
        if experiment.optimization_config is None:
            self.minimize = None
        else:
            obj = experiment.optimization_config.objective
            self.minimize = getattr(obj, "minimize", None)
        self.data = data
        self.constraints = experiment.search_space.parameter_constraints

    def get_next_candidate(
        self: Any,
        pending_parameters: List["TParameterization"],
    ) -> "TParameterization":
        if self.parameters is None:
            raise RuntimeError(
                "Parameters are not initialized. Call update_generator_state first."
            )

        from rich.console import Console
        from rich.prompt import Prompt

        console = Console()
        candidate: Dict[str, Any] = {}

        for name, param in self.parameters.items():
            default = self._default_for(name, param)

            if isinstance(param, _AxFixedParameter):
                candidate[name] = param.value
                continue

            value = Prompt.ask(
                f"[bold]{name}[/bold]",
                default=str(default),
                console=console,
            )
            try:
                candidate[name] = self._cast(param, value)
            except Exception as exc:
                console.print(f"[red]Invalid value '{value}' for {name}: {exc}[/red]")

        if self.constraints:
            params_list = list(candidate.values())
            param_names = list(self.parameters.keys())
            for constraint in self.constraints:
                params = dict(zip(param_names, params_list))
                if not constraint.check(params):
                    console.print(
                        f"[red]Constraint violated by {candidate}. Try again.[/red]"
                    )
                    return self.get_next_candidate(pending_parameters)

        return candidate

    def _default_for(self: Any, name: str, param: Any) -> Any:
        special = self.prompt_specials.get(name)
        if special is not None:
            if special == "min":
                if isinstance(param, _AxRangeParameter):
                    return param.lower
            elif special == "max":
                if isinstance(param, _AxRangeParameter):
                    return param.upper
            else:
                return special

        if isinstance(param, _AxRangeParameter):
            mid = (float(param.lower) + float(param.upper)) / 2.0
            if param.parameter_type == _AxParameterType.INT:
                return int(round(mid))
            return mid
        if isinstance(param, _AxChoiceParameter):
            return list(param.values)[0]
        if isinstance(param, _AxFixedParameter):
            return param.value
        return None

    def _cast(self: Any, param: Any, value: str) -> Any:
        if isinstance(param, _AxRangeParameter):
            if param.parameter_type == _AxParameterType.INT:
                return int(round(float(value)))
            return float(value)
        if isinstance(param, _AxChoiceParameter):
            if param.parameter_type == _AxParameterType.INT:
                return int(round(float(value)))
            if param.parameter_type == _AxParameterType.FLOAT:
                return float(value)
            return str(value)
        return value


register_decoder("InteractiveCLIGenerationNode", InteractiveCLIGenerationNode)


def get_experiment_data(experiment: Any) -> Any:
    return Data(experiment.fetch_data()) if _BACKEND_AVAILABLE else None


def fetch_data_df(client_or_experiment: Any) -> Any:
    """Return a fresh trials DataFrame, fetching data first if needed."""
    if hasattr(client_or_experiment, "fetch_data"):
        try:
            client_or_experiment.fetch_data()
        except Exception:
            pass
    if hasattr(client_or_experiment, "get_trials_data_frame"):
        return client_or_experiment.get_trials_data_frame()
    return None


def clear_module_cache() -> None:
    """Drop ``ax``/``botorch`` modules from :mod:`sys.modules`.

    Useful for tests that re-import different versions of the backend.
    """
    for _name in [k for k in sys.modules if k == "ax" or k.startswith("ax.") or k.startswith("botorch")]:
        sys.modules.pop(_name, None)


if __name__ == "__main__":
    try:
        init()
        print(".ax.py loaded successfully.")
        print(f"Backend Ax: {ax.__version__ if hasattr(ax, '__version__') else 'unknown'}")
        print(f"Backend BoTorch: {botorch.__version__ if hasattr(botorch, '__version__') else 'unknown'}")
        print(f"Supported models: {', '.join(list_supported_model_names())}")
    except RuntimeError as exc:
        print(f"Failed to initialize .ax.py: {exc}")
