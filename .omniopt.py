#!/bin/env python3

#from mayhemmonkey import MayhemMonkey
#mayhemmonkey = MayhemMonkey()
#mayhemmonkey.set_function_fail_after_count("open", 201)
#mayhemmonkey.set_function_error_rate("open", 0.1)
#mayhemmonkey.set_function_group_error_rate(["io", "math"], 0.8)
#mayhemmonkey.install_faulty()

import sys
import os
import signal
import pickle
import re
import math
import time
import random
import tempfile
import threading
import copy
import functools
from collections import Counter, defaultdict
import types
from typing import TypeVar, Callable, Any
import traceback
import inspect
import tracemalloc
import resource
from urllib.parse import urlencode

import psutil

FORCE_EXIT: bool = False

def force_exit(signal_number: Any, frame: Any) -> Any:
    global FORCE_EXIT

    if 'print_debug' in globals():
        print_debug(f"force_exit(signal_number = {signal_number}, frame = {frame})")

    print("")
    if FORCE_EXIT:
        print("Exiting now")
        os._exit(0)
    else:
        print("Shutting down, this may take a while. Press CTRL-c again to force exit now.")
        FORCE_EXIT = True
        sys.exit(0)

signal.signal(signal.SIGINT, force_exit)

F = TypeVar("F", bound=Callable[..., object])

_has_run_once = False
_current_live_share_future = None

last_progress_bar_refresh_time = 0.0
MIN_REFRESH_INTERVAL = 1.0

_last_count_time = 0
_last_count_result: tuple[int, str] = (0, "")

_total_time = 0.0
_func_times: defaultdict = defaultdict(float)
_func_mem: defaultdict = defaultdict(float)
_func_call_paths: defaultdict = defaultdict(Counter)
_last_mem: defaultdict = defaultdict(float)
_leak_threshold_mb = 10.0
generation_strategy_names: list = []
default_max_range_difference: int = 1000000

_function_name_cache: dict = {}

experiment_parameters: dict | None = None
arms_by_name_for_deduplication: dict = {}
initialized_storage: bool = False
prepared_setting_to_custom: bool = False
whole_start_time: float = time.time()
last_progress_bar_desc: str = ""
job_submit_durations: list[float] = []
job_submit_nrs: list[int] = []
log_gen_times: list[float] = []
log_nr_gen_jobs: list[int] = []
generation_strategy_human_readable: str = ""
oo_call: str = "./omniopt"
progress_bar_length: int = 0

if os.environ.get("CUSTOM_VIRTUAL_ENV") == "1":
    oo_call = "omniopt"

shown_run_live_share_command: bool = False
ci_env: bool = os.getenv("CI", "false").lower() == "true"
original_print = print
overwritten_to_random: bool = False

valid_occ_types: list = ["geometric", "euclid", "signed_harmonic", "signed_minkowski", "weighted_euclid", "composite"]
joined_valid_occ_types: str = ", ".join(valid_occ_types)

SUPPORTED_MODELS: list = ["SOBOL", "FACTORIAL", "SAASBO", "BOTORCH_MODULAR", "UNIFORM", "BO_MIXED", "RANDOMFOREST", "EXTERNAL_GENERATOR", "PSEUDORANDOM", "TPE"]
joined_supported_models: str = ", ".join(SUPPORTED_MODELS)

special_col_names: list = ["arm_name", "generation_method", "trial_index", "trial_status", "generation_node", "idxs", "start_time", "end_time", "run_time", "exit_code", "program_string", "signal", "hostname", "submit_time", "queue_time", "metric_name", "mean", "sem", "worker_generator_uuid"]

IGNORABLE_COLUMNS: list = ["start_time", "end_time", "hostname", "signal", "exit_code", "run_time", "program_string"] + special_col_names

uncontinuable_models: list = ["RANDOMFOREST", "EXTERNAL_GENERATOR", "TPE", "PSEUDORANDOM", "HUMAN_INTERVENTION_MINIMUM"]

post_generation_constraints: list = []
abandoned_trial_indices: list = []
global_param_names: list = []

figlet_loaded: bool = False

try:
    from rich.console import Console

    from rich.panel import Panel
    from rich.text import Text

    terminal_width = 150

    try:
        terminal_width = os.get_terminal_size().columns
    except OSError:
        pass

    console = Console(
        force_interactive=True,
        soft_wrap=True,
        color_system="256",
        force_terminal=not ci_env,
        width=max(200, terminal_width)
    )

    def spinner(text: str) -> Any:
        return console.status(f"[bold green]{text}", speed=0.2, refresh_per_second=6)

    with spinner("Importing logging..."):
        import logging
        logging.basicConfig(level=logging.CRITICAL)

    with spinner("Importing warnings..."):
        import warnings

        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            module="ax.adapter.best_model_selector"
        )

        warnings.filterwarnings(
            "ignore",
            message="Ax currently requires a sqlalchemy version below 2.0.*",
        )

        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="coroutine 'start_logging_daemon' was never awaited"
        )

    with spinner("Importing argparse..."):
        import argparse

    with spinner("Importing datetime..."):
        import datetime

    with spinner("Importing dataclass..."):
        from dataclasses import dataclass

    with spinner("Importing socket..."):
        import socket

    with spinner("Importing stat..."):
        import stat

    with spinner("Importing pwd..."):
        import pwd

    with spinner("Importing base64..."):
        import base64

    with spinner("Importing json..."):
        import json

    with spinner("Importing yaml..."):
        import yaml

    with spinner("Importing toml..."):
        import toml

    with spinner("Importing csv..."):
        import csv

    with spinner("Importing ast..."):
        import ast

    with spinner("Importing rich.table..."):
        from rich.table import Table

    with spinner("Importing rich print..."):
        from rich import print

    with spinner("Importing rich.pretty..."):
        from rich.pretty import pprint

    with spinner("Importing rich.prompt..."):
        from rich.prompt import Prompt, FloatPrompt, IntPrompt

    with spinner("Importing types.FunctionType..."):
        from types import FunctionType

    with spinner("Importing typing..."):
        from typing import Pattern, Optional, Tuple, cast, Union, TextIO, List, Dict, Type

    with spinner("Importing ThreadPoolExecutor..."):
        from concurrent.futures import ThreadPoolExecutor, as_completed

    with spinner("Importing submitit.LocalExecutor..."):
        from submitit import LocalExecutor, AutoExecutor

    with spinner("Importing submitit.Job..."):
        from submitit import Job

    with spinner("Importing importlib.util..."):
        import importlib.util

    with spinner("Importing platform..."):
        import platform

    with spinner("Importing inspect frame info..."):
        from inspect import currentframe, getframeinfo

    with spinner("Importing pathlib.Path..."):
        from pathlib import Path

    with spinner("Importing uuid..."):
        import uuid

    #with spinner("Importing qrcode..."):
    #    import qrcode

    with spinner("Importing cowsay..."):
        import cowsay

    with spinner("Importing shutil..."):
        import shutil

    with spinner("Importing itertools.combinations..."):
        from itertools import combinations

    with spinner("Importing os.listdir..."):
        from os import listdir

    with spinner("Importing os.path..."):
        from os.path import isfile, join

    with spinner("Importing PIL.Image..."):
        from PIL import Image

    with spinner("Importing sixel..."):
        import sixel

    with spinner("Importing subprocess..."):
        import subprocess

    with spinner("Importing tqdm..."):
        from tqdm import tqdm

    with spinner("Importing beartype..."):
        from beartype import beartype

    with spinner("Importing statistics..."):
        import statistics

    with spinner("Trying to import pyfiglet..."):
        try:
            from pyfiglet import Figlet
            figlet_loaded = True
        except ModuleNotFoundError:
            figlet_loaded = False
except ModuleNotFoundError as e:
    print(f"Some of the base modules could not be loaded. Most probably that means you have not loaded or installed the virtualenv properly. Error: {e}")
    print("Exit-Code: 4")
    sys.exit(4)
except ImportError as e:
    print(f"Error loading modules: {e}\nThis may be caused by forgetting to 'module load' the right python version or missing the python virtual environment.")
    sys.exit(4)
except KeyboardInterrupt:
    print("You pressed CTRL-C while modules were loading.")
    sys.exit(17)

def collect_runtime_stats() -> dict:
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()

    # RLIMIT_NOFILE
    if hasattr(resource, "RLIMIT_NOFILE"):
        try:
            ulimit_nofile = resource.getrlimit(resource.RLIMIT_NOFILE)
        except Exception:
            ulimit_nofile = (0, 0)
    else:
        # Windows fallback: psutil provides open file limit approximately
        ulimit_nofile = (process.num_fds() if hasattr(process, "num_fds") else 0, 0)

    # RLIMIT_AS
    if hasattr(resource, "RLIMIT_AS"):
        try:
            ulimit_as = resource.getrlimit(resource.RLIMIT_AS)
        except Exception:
            ulimit_as = (0, 0)
    else:
        # Windows fallback: set to total virtual memory
        ulimit_as = (psutil.virtual_memory().total, psutil.virtual_memory().total)

    return {
        "rss_MB": mem_info.rss / (1024 * 1024),
        "vms_MB": mem_info.vms / (1024 * 1024),
        "threads": threading.active_count(),
        "open_files": len(process.open_files()),
        "ulimit_nofile": ulimit_nofile,
        "ulimit_as": ulimit_as,
        "cpu_percent": process.cpu_percent(interval=0.05),
    }

def show_func_name_wrapper(func: F) -> F:
    @functools.wraps(func)
    def wrapper(*func_args: Any, **kwargs: Any) -> Any:
        print(f"==== {func.__name__} START ====")
        result = func(*func_args, **kwargs)
        print(f"==== {func.__name__} END   ====")

        return result

    return wrapper  # type: ignore

def log_time_and_memory_wrapper(func: F) -> F:
    @functools.wraps(func)
    def wrapper(*func_args: Any, **kwargs: Any) -> Any:
        process = psutil.Process()
        mem_before = process.memory_info().rss / (1024 * 1024)

        tracemalloc.start()
        start = time.perf_counter()
        result = func(*func_args, **kwargs)
        elapsed = time.perf_counter() - start

        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        mem_after = process.memory_info().rss / (1024 * 1024)
        mem_diff = mem_after - mem_before
        mem_peak_mb = peak / (1024 * 1024)

        if elapsed >= 0.05:
            _record_stats(func.__name__, elapsed, mem_diff, mem_after, mem_peak_mb)

        _check_memory_leak(func.__name__, mem_peak_mb)

        return result

    return wrapper  # type: ignore

def _record_stats(func_name: str, elapsed: float, mem_diff: float, mem_after: float, mem_peak: float) -> None:
    global _total_time

    current_total = _total_time
    current_func_total = _func_times[func_name]
    simulated_total = current_total + elapsed
    simulated_func_total = current_func_total + elapsed
    percent_if_added = (simulated_func_total / simulated_total) * 100 if simulated_total else 100

    if percent_if_added >= 1.0:
        _func_times[func_name] = simulated_func_total
        _func_mem[func_name] += mem_diff
        _total_time = simulated_total

        stack = traceback.extract_stack()[:-1]
        short_stack = [f"{f.filename.split('/')[-1]}:{f.lineno} in {f.name}" for f in stack[-5:]]
        call_path_str = " -> ".join(short_stack)
        _func_call_paths[func_name][call_path_str] += 1

        print(
            f"Function '{func_name}' took {elapsed:.4f}s "
            f"(total {percent_if_added:.1f}% of tracked time)"
        )
        print(
            f"Memory before: {mem_after - mem_diff:.2f} MB, after: {mem_after:.2f} MB, "
            f"diff: {mem_diff:+.2f} MB, peak during call: {mem_peak:.2f} MB"
        )

        # NEU: Runtime Stats
        runtime_stats = collect_runtime_stats()
        print("=== Runtime Stats ===")
        print(f"RSS: {runtime_stats['rss_MB']:.2f} MB, VMS: {runtime_stats['vms_MB']:.2f} MB")
        print(f"Threads: {runtime_stats['threads']}, Open files: {runtime_stats['open_files']}")
        print(f"ulimit nofile: {runtime_stats['ulimit_nofile']}, ulimit as: {runtime_stats['ulimit_as']}")
        print(f"CPU %: {runtime_stats['cpu_percent']:.1f}")

        print(f"!!! '{func_name}' added {percent_if_added:.1f}% of the total runtime. !!!")
        _print_time_and_memory_functions_wrapper_stats()

def _print_time_and_memory_functions_wrapper_stats() -> None:
    if _total_time == 0:
        return

    print("=== Time Stats ===")
    items_time = sorted(_func_times.items(), key=lambda x: -x[1])
    for i, (name, t) in enumerate(items_time, 1):
        percent_total = t / _total_time * 100
        print(f"{i}. {name}: {t:.4f}s ({percent_total:.1f}%)")

    print("\n=== Memory Usage Stats (Top 10 by diff) ===")
    items_mem = sorted(_func_mem.items(), key=lambda x: -abs(x[1]))
    for i, (name, mem) in enumerate(items_mem[:10], 1):
        print(f"{i}. {name}: {mem:+.2f} MB total change")

    print("\n=== Top 10 slowest call origins ===")
    for name, _ in items_time[:10]:
        print(f"\n{name}:")
        for call_path, count in _func_call_paths[name].most_common(3):
            print(f"  {count}×  {call_path}")
    print("==================")

def _check_memory_leak(func_name: str, current_mem: float) -> None:
    last_mem = _last_mem[func_name]
    if current_mem - last_mem > _leak_threshold_mb:
        print(f"⚠ Possible memory leak detected in '{func_name}': +{current_mem - last_mem:.2f} MB since last call")
    _last_mem[func_name] = current_mem

def fool_linter(*fool_linter_args: Any) -> Any:
    return fool_linter_args

def makedirs(p: str) -> bool:
    if not os.path.exists(p):
        try:
            os.makedirs(p, exist_ok=True)
        except Exception as ee:
            print(f"Failed to create >{p}<. Error: {ee}")

    if os.path.exists(p):
        return True

    return False

YELLOW: str = "\033[93m"
RESET: str = "\033[0m"

uuid_regex: Pattern = re.compile(r"^[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-4[a-fA-F0-9]{3}-[89aAbB][a-fA-F0-9]{3}-[a-fA-F0-9]{12}$")

worker_generator_uuid: str = str(uuid.uuid4())

new_uuid: str = str(uuid.uuid4())
run_uuid: str = os.getenv("RUN_UUID", new_uuid)

if not uuid_regex.match(run_uuid):
    print(f"{YELLOW}WARNING: The provided RUN_UUID is not a valid UUID. Using new UUID {new_uuid} instead.{RESET}")
    run_uuid = new_uuid

JOBS_FINISHED: int = 0
RESULTS_CSV_FILENAME: str = "results.csv"
WORKER_PERCENTAGE_USAGE: list = []
END_PROGRAM_RAN: bool = False
ALREADY_SHOWN_WORKER_USAGE_OVER_TIME: bool = False
ax_client = None
CURRENT_RUN_FOLDER: str = ""
RESULT_CSV_FILE: str = ""
SHOWN_END_TABLE: bool = False
max_eval: int = 1
random_steps: int = 1
progress_bar: Optional[tqdm] = None
error_8_saved: List[str] = []

def get_current_run_folder(name: Optional[str] = None) -> str:
    if name is not None:
        return f"{CURRENT_RUN_FOLDER}/{name}"

    return CURRENT_RUN_FOLDER

def get_state_file_name(name) -> str:
    state_files_folder = f"{get_current_run_folder()}/state_files/"
    makedirs(state_files_folder)

    return f"{state_files_folder}/{name}"

script_dir = os.path.dirname(os.path.realpath(__file__))

try:
    with spinner("Importing helpers..."):
        helpers_file: str = f"{script_dir}/.helpers.py"
        spec = importlib.util.spec_from_file_location(
            name="helpers",
            location=helpers_file,
        )
        if spec is not None and spec.loader is not None:
            helpers = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(helpers)
        else:
            raise ImportError(f"Could not load module from {helpers_file}")

        dier: FunctionType = helpers.dier
        is_equal: FunctionType = helpers.is_equal
        is_not_equal: FunctionType = helpers.is_not_equal
except KeyboardInterrupt:
    print("You pressed CTRL-c while importing the helpers file")
    sys.exit(0)

ORCHESTRATE_TODO: dict = {}

class SignalUSR (Exception):
    pass

class SignalINT (Exception):
    pass

class SignalTERM (Exception):
    pass

class SignalCONT (Exception):
    pass

def is_slurm_job() -> bool:
    if os.environ.get('SLURM_JOB_ID') is not None:
        return True
    return False

def _sleep(t: Union[float, int]) -> None:
    if args is not None and not args.no_sleep:
        try:
            time.sleep(t)
        except KeyboardInterrupt:
            pass

LOG_DIR: str = "logs"
makedirs(LOG_DIR)

log_uuid_dir = f"{LOG_DIR}/{run_uuid}"
logfile: str = f'{log_uuid_dir}_log'
logfile_bare: str = f'{log_uuid_dir}_log_bare'
logfile_nr_workers: str = f'{log_uuid_dir}_nr_workers'
logfile_progressbar: str = f'{log_uuid_dir}_progressbar'
logfile_worker_creation_logs: str = f'{log_uuid_dir}_worker_creation_logs'
logfile_trial_index_to_param_logs: str = f'{log_uuid_dir}_trial_index_to_param_logs'
LOGFILE_DEBUG_GET_NEXT_TRIALS: Union[str, None] = None

def print_red(text: str) -> None:
    helpers.print_color("red", text)

    print_debug(text)

    if get_current_run_folder():
        try:
            with open(get_current_run_folder("oo_errors.txt"), mode="a", encoding="utf-8") as myfile:
                myfile.write(text + "\n\n")
        except (OSError, FileNotFoundError) as e:
            helpers.print_color("red", f"Error: {e}. This may mean that the {get_current_run_folder()} was deleted during the run. Could not write '{text} to {get_current_run_folder()}/oo_errors.txt'")
            sys.exit(99)

def _debug(msg: str, _lvl: int = 0, eee: Union[None, str, Exception] = None) -> None:
    if _lvl > 3:
        original_print(f"Cannot write _debug, error: {eee}")
        print("Exit-Code: 193")
        sys.exit(193)

    try:
        with open(logfile, mode='a', encoding="utf-8") as f:
            original_print(msg, file=f)
    except FileNotFoundError:
        print_red("It seems like the run's folder was deleted during the run. Cannot continue.")
        sys.exit(99)
    except Exception as e:
        original_print(f"_debug: Error trying to write log file: {e}")

        _debug(msg, _lvl + 1, e)

def _get_debug_json(time_str: str, msg: str) -> str:
    function_stack = []
    try:
        frame = inspect.currentframe().f_back  # skip _get_debug_json
        while frame:
            func_name = _function_name_cache.get(frame.f_code)
            if func_name is None:
                func_name = frame.f_code.co_name
                _function_name_cache[frame.f_code] = func_name

            if func_name not in ("<module>", "print_debug", "wrapper"):
                function_stack.append({
                    "function": func_name,
                    "line_number": frame.f_lineno
                })

            frame = frame.f_back
    except (SignalUSR, SignalINT, SignalCONT):
        print_red("\n⚠ You pressed CTRL-C. This is ignored in _get_debug_json.")

    return json.dumps(
        {"function_stack": function_stack, "time": time_str, "msg": msg},
        separators=(",", ":")  # no pretty indent → smaller, faster
    ).replace('\r', '').replace('\n', '')

def print_stack_paths() -> None:
    stack = inspect.stack()[1:]  # skip current frame
    stack.reverse()

    last_filename = None
    for depth, frame_info in enumerate(stack):
        filename = frame_info.filename
        lineno = frame_info.lineno
        func_name = frame_info.function

        if func_name in ["<module>", "print_debug"]:
            continue

        if filename != last_filename:
            print(filename)
            last_filename = filename
            indent = ""
        else:
            indent = " " * 4 * depth

        print(f"{indent}↳ {func_name}:{lineno}")

def print_debug(msg: str) -> None:
    time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    stack = traceback.extract_stack()[:-1]
    stack_funcs = [frame.name for frame in stack]

    if "args" in globals() and args and hasattr(args, "debug_stack_regex") and args.debug_stack_regex:
        matched = any(any(re.match(regex, func) for regex in args.debug_stack_regex) for func in stack_funcs)
        if matched:
            print(f"DEBUG: {msg}")
            print_stack_paths()

    stack_trace_element = _get_debug_json(time_str, msg)
    _debug(stack_trace_element)

    try:
        with open(logfile_bare, mode='a', encoding="utf-8") as f:
            original_print(msg, file=f)
    except FileNotFoundError:
        print_red("It seems like the run's folder was deleted during the run. Cannot continue.")
        sys.exit(99)
    except Exception as e:
        original_print(f"_debug: Error trying to write log file: {e}")

def human_time_when_larger_than_a_min(seconds: Union[int, float]) -> str:
    total_seconds = int(seconds)

    if total_seconds < 60:
        return ""

    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if secs or not parts:
        parts.append(f"{secs}s")
    return f"({''.join(parts)})"

def my_exit(_code: int = 0) -> None:
    tb = traceback.format_exc()

    try:
        print_debug(f"Exiting with error code {_code}. Traceback: {tb}")
    except NameError:
        print(f"Exiting with error code {_code}. Traceback: {tb}")

    try:
        if (is_slurm_job() and not args.force_local_execution) and not (args.show_sixel_scatter or args.show_sixel_general or args.show_sixel_trial_index_result):
            _sleep(2)
        else:
            time.sleep(2)
    except KeyboardInterrupt:
        pass

    exit_code_string = f"\nExit-Code: {_code}"

    print(exit_code_string)
    print_debug(exit_code_string)

    whole_end_time: float = time.time()
    whole_run_time = round(whole_end_time - whole_start_time)
    human_time = human_time_when_larger_than_a_min(whole_run_time)

    print(f"Wallclock-Runtime: {whole_run_time} seconds {human_time}")

    if is_skip_search() and os.getenv("SKIP_SEARCH_EXIT_CODE"):
        skip_search_exit_code = os.getenv("SKIP_SEARCH_EXIT_CODE")

        try:
            sys.exit(int(skip_search_exit_code))
        except ValueError:
            print(f"Trying to look for SKIP_SEARCH_EXIT_CODE failed. Exiting with original exit code {_code}")
            sys.exit(_code)

    sys.exit(_code)

def print_green(text: str) -> None:
    helpers.print_color("green", text)

    print_debug(text)

def print_yellow(text: str) -> None:
    helpers.print_color("yellow", f"{text}")

    print_debug(text)

def get_min_max_from_file(continue_path: str, n: int, _default_min_max: str) -> str:
    path = f"{continue_path}/result_min_max.txt"

    if not os.path.exists(path):
        print_yellow(f"File '{path}' not found, will use {_default_min_max}")
        return _default_min_max

    with open(path, encoding="utf-8", mode='r') as file:
        lines = file.read().splitlines()

    line = lines[n] if 0 <= n < len(lines) else ""

    if line in {"min", "max"}:
        return line

    print_yellow(f"Line {n} did not contain min/max, will be set to {_default_min_max}")
    return _default_min_max

def set_max_eval(new_max_eval: int) -> None:
    global max_eval

    print_debug(f"set_max_eval({new_max_eval})")

    max_eval = new_max_eval

def set_random_steps(new_steps: int) -> None:
    global random_steps

    print_debug(f"Setting random_steps from {random_steps} to {new_steps}")

    random_steps = new_steps

_DEFAULT_SPECIALS: Dict[str, Any] = {
    "epochs": 1,
    "epoch": 1,
    "steps": 1,
    "batchsize": 1,
    "batchsz": 1,
    "bs": 1,
    "lr": "min",
    "learning_rate": "min",
}

class ConfigLoader:
    runtime_debug: bool
    show_func_name: bool
    debug_stack_regex: str
    number_of_generators: int
    disable_previous_job_constraint: bool
    save_to_database: bool
    dependency: str
    run_tests_that_fail_on_taurus: bool
    num_random_steps: int
    verbose: bool
    disable_tqdm: bool
    slurm_use_srun: bool
    reservation: Optional[str]
    account: Optional[str]
    exclude: Optional[str]
    show_sixel_trial_index_result: bool
    num_parallel_jobs: int
    force_local_execution: bool
    occ_type: str
    raise_in_eval: bool
    maximize: bool
    show_sixel_general: bool
    show_sixel_scatter: bool
    gpus: int
    num_cpus_main_job: Optional[int]
    model: Optional[str]
    live_share: bool
    experiment_name: str
    show_worker_percentage_table_at_end: bool
    abbreviate_job_names: bool
    verbose_tqdm: bool
    tests: bool
    max_eval: int
    run_program: Optional[List[str]]
    orchestrator_file: Optional[str]
    run_dir: str
    ui_url: Optional[str]
    nodes_per_job: int
    seed: int
    cpus_per_task: int
    parameter: Optional[List[str]]
    experiment_constraints: Optional[List[str]]
    main_process_gb: int
    worker_timeout: int
    slurm_signal_delay_s: int
    gridsearch: bool
    auto_exclude_defective_hosts: bool
    debug: bool
    debug_stack_trace_regex: Optional[str]
    num_restarts: int
    raw_samples: int
    show_generate_time_table: bool
    max_attempts_for_generation: int
    dont_warm_start_refitting: bool
    refit_on_cv: bool
    fit_out_of_design: bool
    fit_abandoned: bool
    no_sleep: bool
    username: Optional[str]
    max_nr_of_zero_results: int
    run_program_once: str
    mem_gb: int
    flame_graph: bool
    continue_previous_job: Optional[str]
    calculate_pareto_front_of_job: Optional[List[str]]
    revert_to_random_when_seemingly_exhausted: bool
    minkowski_p: float
    partition: str
    signed_weighted_euclidean_weights: str
    external_generator: Optional[str]
    generation_strategy: Optional[str]
    root_venv_dir: str
    follow: bool
    show_generation_and_submission_sixel: bool
    n_estimators_randomforest: int
    max_num_of_parallel_sruns: int
    checkout_to_latest_tested_version: bool
    load_data_from_existing_jobs: List[str]
    worker_generator_path: str
    time: str
    share_password: Optional[str]
    prettyprint: bool
    generate_all_jobs_at_once: bool
    result_names: Optional[List[str]]
    verbose_break_run_search_table: bool
    send_anonymized_usage_stats: bool
    max_failed_jobs: Optional[int]
    max_abandoned_retrial: int
    show_ram_every_n_seconds: int
    config_toml: Optional[str]
    config_json: Optional[str]
    config_yaml: Optional[str]
    workdir: str
    db_url: Optional[str]
    dont_jit_compile: bool
    no_normalize_y: bool
    transforms: List[str]
    no_transform_inputs: bool
    occ: bool
    force_choice_for_ranges: bool
    dryrun: bool
    range_max_difference: int
    skip_search: bool
    just_return_defaults: bool
    run_mode: str

    def __init__(self: Any, _parsing_arguments_loader: Any) -> None:
        self.parser = argparse.ArgumentParser(
            prog="omniopt",
            description='A hyperparameter optimizer for slurm-based HPC-systems',
            epilog=f"Example:\n\n{oo_call} --partition=alpha --experiment_name=neural_network ..."
        )

        self._parsing_arguments_loader = _parsing_arguments_loader

        self.parser.add_argument('--config_yaml', help='YAML configuration file', type=str, default=None)
        self.parser.add_argument('--config_toml', help='TOML configuration file', type=str, default=None)
        self.parser.add_argument('--config_json', help='JSON configuration file', type=str, default=None)

        self.add_arguments()

    def add_arguments(self: Any) -> None:
        required = self.parser.add_argument_group('Required arguments', 'These options have to be set')
        required_but_choice = self.parser.add_argument_group('Required arguments that allow a choice', 'Of these arguments, one has to be set to continue')
        optional = self.parser.add_argument_group('Optional', 'These options are optional')
        speed = self.parser.add_argument_group('Speed', 'These options are for speeding up the Process of Starting Processes or Generating new Points')
        slurm = self.parser.add_argument_group('SLURM', 'Parameters related to SLURM')
        installing = self.parser.add_argument_group('Installing', 'Parameters related to installing')
        debug = self.parser.add_argument_group('Debug', 'These options are mainly useful for debugging')

        required.add_argument('--num_random_steps', help='Number of random (SOBOL) steps to start with', type=int, default=20)
        required.add_argument('--max_eval', help='Maximum number of evaluations', type=int)
        required.add_argument('--run_program', action='append', nargs='+', help='A program that should be run. Use, for example, $x for the parameter named x', type=str)
        required.add_argument('--experiment_name', help='Name of the experiment', type=str)
        required.add_argument('--mem_gb', help='Amount of RAM for each worker in GB (default: 1GB)', type=int, default=1)

        required_but_choice.add_argument('--parameter', action='append', nargs='+', help='Experiment parameters in the formats (options in round brackets are optional): <NAME> range <LOWER BOUND> <UPPER BOUND> (<INT, FLOAT>, log_scale: True/False, default: false>) -- OR -- <NAME> fixed <VALUE> -- OR -- <NAME> choice <Comma-separated list of values>', default=None)
        required_but_choice.add_argument('--continue_previous_job', help='Continue from a previous checkpoint, use run-dir as argument', type=str, default=None)

        optional.add_argument('--experiment_constraints', action='append', nargs='+', help='Constraints for parameters. Example: x + y <= 2.0. Convert them to base64', type=str)
        optional.add_argument('--run_dir', help='Directory, in which runs should be saved. Default: runs', default='runs', type=str)
        optional.add_argument('--seed', help='Seed for random number generator', type=int)
        optional.add_argument('--verbose_tqdm', help='Show verbose TQDM messages', action='store_true', default=False)
        optional.add_argument('--model', help=f'Use special models for nonrandom steps. Valid models are: {joined_supported_models}', type=str, default=None)
        optional.add_argument('--gridsearch', help='Enable gridsearch', action='store_true', default=False)
        optional.add_argument('--occ', help='Use optimization with combined criteria (OCC)', action='store_true', default=False)
        optional.add_argument('--show_sixel_scatter', help='Show sixel graphics of scatter plots in the end', action='store_true', default=False)
        optional.add_argument('--show_sixel_general', help='Show sixel graphics of general plots in the end', action='store_true', default=False)
        optional.add_argument('--show_sixel_trial_index_result', help='Show sixel graphics of trial index in the end', action='store_true', default=False)
        optional.add_argument('--follow', help='Automatically follow log file of sbatch', action='store_true', default=False)
        optional.add_argument('--send_anonymized_usage_stats', help='Send anonymized usage stats', action='store_true', default=False)
        optional.add_argument('--ui_url', help='Site from which the OO-run was called', default=None, type=str)
        optional.add_argument('--root_venv_dir', help=f'Where to install your modules to ($root_venv_dir/.omniax_..., default: {Path.home()})', default=Path.home(), type=str)
        optional.add_argument('--exclude', help='A comma separated list of values of excluded nodes (taurusi8009,taurusi8010)', default=None, type=str)
        optional.add_argument('--main_process_gb', help='Amount of RAM for the main process in GB (default: 8GB)', type=int, default=8)
        optional.add_argument('--max_nr_of_zero_results', help='Max. nr of successive zero results by the generator before the search space is seen as exhausted', type=int, default=10)
        optional.add_argument('--abbreviate_job_names', help='Abbreviate pending job names (r = running, p = pending, u = unknown, c = cancelling)', action='store_true', default=False)
        optional.add_argument('--orchestrator_file', help='An orchestrator file', default=None, type=str)
        optional.add_argument('--checkout_to_latest_tested_version', help='Automatically checkout to latest version that was tested in the CI pipeline', action='store_true', default=False)
        optional.add_argument('--live_share', help='Automatically live-share the current optimization run automatically', action='store_true', default=False)
        optional.add_argument('--disable_tqdm', help='Disables the TQDM progress bar', action='store_true', default=False)
        optional.add_argument('--disable_previous_job_constraint', help='For continued jobs: Disable getting the constraint of the previous job that is about to be continued', action='store_true', default=False)
        optional.add_argument('--workdir', help='Working directory', default='', type=str)
        optional.add_argument('--occ_type', help=f'Optimization-with-combined-criteria-type (valid types are {joined_valid_occ_types})', type=str, default='euclid')
        optional.add_argument('--result_names', nargs='+', default=[], help='Name of hyperparameters. Example --result_names result1=max result2=min result3. Default: RESULT=min')
        optional.add_argument('--minkowski_p', help='Minkowski order of distance (default: 2), needs to be larger than 0', type=float, default=2)
        optional.add_argument('--signed_weighted_euclidean_weights', help='A comma-separated list of values for the signed weighted Euclidean distance. Needs to be equal to the number of results. Else, default will be 1', default='', type=str)
        optional.add_argument('--generation_strategy', help='A string containing the generation_strategy. Example: SOBOL=10,BOTORCH_MODULAR=10,SOBOL=10. Cannot use --model EXTERNAL_GENERATOR, TPE, RANDOMFOREST or PSEUDORANDOM', type=str, default=None)
        optional.add_argument('--generate_all_jobs_at_once', help='Generate all jobs at once rather than to create them and start them as soon as possible', action='store_true', default=False)
        optional.add_argument('--revert_to_random_when_seemingly_exhausted', help='Generate random steps instead of systematic steps when the search space is (seemingly) exhausted', action='store_true', default=False)
        optional.add_argument('--load_data_from_existing_jobs', type=str, nargs='*', default=[], help='List of job data to load from existing jobs')
        optional.add_argument('--n_estimators_randomforest', help='The number of trees in the forest for RANDOMFOREST (default: 100)', type=int, default=100)
        optional.add_argument('--max_attempts_for_generation', help='Max. number of attempts for generating sets of new points (default: 20)', type=int, default=20)
        optional.add_argument('--external_generator', help='Programm call for an external generator', type=str, default=None)
        optional.add_argument('--username', help='A username for live share', default=None, type=str)
        optional.add_argument('--max_failed_jobs', help='Maximum number of failed jobs before the search is cancelled. Is defaulted to the value of --max_eval', default=None, type=int)
        optional.add_argument('--num_cpus_main_job', help='Number of CPUs for the main job', default=None, type=int)
        optional.add_argument('--calculate_pareto_front_of_job', help='This can be used to calculate a Pareto-front for a multi-objective job that previously has results, but has been cancelled, and has no Pareto-front (yet)', type=str, nargs='+', default=[])
        optional.add_argument('--show_generate_time_table', help='Generate a table at the end, showing how much time was spent trying to generate new points', action='store_true', default=False)
        optional.add_argument('--force_choice_for_ranges', help='Force float ranges to be converted to choice', action='store_true', default=False)
        optional.add_argument('--max_abandoned_retrial', help='Maximum number retrials to get when a job is abandoned post-generation', default=20, type=int)
        optional.add_argument('--share_password', help='Use this as a password for share. Default is none.', default=None, type=str)
        optional.add_argument('--dryrun', help='Try to do a dry run, i.e. a run for very short running jobs to test the installation of OmniOpt2 and check if environment stuff and paths and so on works properly', action='store_true', default=False)
        optional.add_argument('--db_url', type=str, default=None, help='Database URL (e.g., mysql+pymysql://user:pass@host/db), disables sqlite3 storage')
        optional.add_argument('--run_program_once', type=str, help='Path to a setup script that will run once before the main program starts.')
        optional.add_argument('--worker_generator_path', type=str, help='Path of the run folder where this script should plug itself in as a worker points generator')
        optional.add_argument('--save_to_database', help='Save all entries into a sqlite3 database', action='store_true', default=False)
        optional.add_argument('--range_max_difference', help=f'Max. difference for range, default is {default_max_range_difference}', default=default_max_range_difference, type=int)
        optional.add_argument('--skip_search', help='Skips the actual search, uses exit code 0 if not the environment variable SKIP_SEARCH_EXIT_CODE is set', action='store_true', default=False)

        speed.add_argument('--dont_warm_start_refitting', help='Do not keep Model weights, thus, refit for every generator (may be more accurate, but slower)', action='store_true', default=False)
        speed.add_argument('--refit_on_cv', help='Refit on Cross-Validation (helps in accuracy, but makes generating new points slower)', action='store_true', default=False)
        speed.add_argument('--fit_out_of_design', help='Ignore data points outside of the design while creating new points', action='store_true', default=False)
        speed.add_argument('--fit_abandoned', help='Do not ignore abandoned data points while creating new points', action='store_true', default=False)
        speed.add_argument('--dont_jit_compile', help='Disable JIT-compiling the model', action='store_true', default=False)
        speed.add_argument('--num_restarts', help='num_restarts option for optimizer_options', type=int, default=20)
        speed.add_argument('--raw_samples', help='raw_samples option for optimizer_options', type=int, default=1024)
        speed.add_argument('--max_num_of_parallel_sruns', help='Maximal number of parallel sruns', type=int, default=16)
        speed.add_argument('--no_transform_inputs', help='Disable input transformations', action='store_true', default=False)
        speed.add_argument('--no_normalize_y', help='Disable target normalization', action='store_true', default=False)
        speed.add_argument('--transforms', nargs='*', choices=['Cont_X_trans', 'Cont_X_trans_Y_trans'], default=[], help='Enable input/target transformations (choose one or both: Cont_X_trans, Cont_X_trans_Y_trans)')
        speed.add_argument('--number_of_generators', help='Number of generator main scripts, only works with Slurm', type=int, default=1)

        slurm.add_argument('--num_parallel_jobs', help='Number of parallel SLURM jobs (default: 20)', type=int, default=20)
        slurm.add_argument('--worker_timeout', help='Timeout for SLURM jobs (i.e. for each single point to be optimized)', type=int, default=30)
        slurm.add_argument('--slurm_use_srun', help='Using srun instead of sbatch', action='store_true', default=False)
        slurm.add_argument('--time', help='Time for the main job', default='', type=str)
        slurm.add_argument('--partition', help='Partition to be run on', default='', type=str)
        slurm.add_argument('--reservation', help='Reservation', default=None, type=str)
        slurm.add_argument('--force_local_execution', help='Forces local execution even when SLURM is available', action='store_true', default=False)
        slurm.add_argument('--slurm_signal_delay_s', help='When the workers end, they get a signal so your program can react to it. Default is 0, but set it to any number of seconds you wish your program to be able to react to USR1', type=int, default=0)
        slurm.add_argument('--nodes_per_job', help='Number of nodes per job due to the new alpha restriction', type=int, default=1)
        slurm.add_argument('--cpus_per_task', help='CPUs per task', type=int, default=1)
        slurm.add_argument('--account', help='Account to be used for SLURM', type=str, default=None)
        slurm.add_argument('--gpus', help='Number of GPUs per worker', type=int, default=0)
        #slurm.add_ argument('--tasks_per_node', help='ntasks', type=int, default=1)
        slurm.add_argument('--dependency', type=str, help='Allows slurm-dependencies, like --dependency=afterok:<slurm id> or --dependency:after:<slurm_id> or --dependency=singleton, the latter one allows to let only run job running as long as they have the same job name, and --dependency=omniopt_singleton, which allows only one OmniOpt job to be running and puts all running once into the dependency string automatically')

        installing.add_argument('--run_mode', help='Either local or docker', default='local', type=str)

        debug.add_argument('--verbose', help='Verbose logging', action='store_true', default=False)
        debug.add_argument('--verbose_break_run_search_table', help='Verbose logging for break_run_search', action='store_true', default=False)
        debug.add_argument('--debug', help='Enable debugging', action='store_true', default=False)
        debug.add_argument('--flame_graph', help='Enable flame-graphing. Makes everything slower, but creates a flame graph', action='store_true', default=False)
        debug.add_argument('--no_sleep', help='Disables sleeping for fast job generation (not to be used on HPC)', action='store_true', default=False)
        debug.add_argument('--tests', help='Run simple internal tests', action='store_true', default=False)
        debug.add_argument('--show_worker_percentage_table_at_end', help='Show a table of percentage of usage of max worker over time', action='store_true', default=False)
        debug.add_argument('--auto_exclude_defective_hosts', help='Run a Test if you can allocate a GPU on each node and if not, exclude it since the GPU driver seems to be broken somehow', action='store_true', default=False)
        debug.add_argument('--run_tests_that_fail_on_taurus', help='Run tests on Taurus that usually fail', action='store_true', default=False)
        debug.add_argument('--raise_in_eval', help='Raise a signal in eval (only useful for debugging and testing)', action='store_true', default=False)
        debug.add_argument('--show_ram_every_n_seconds', help='Show RAM usage every n seconds (0 = disabled)', type=int, default=0)
        debug.add_argument('--show_generation_and_submission_sixel', help='Show sixel plots for generation and submission times', action='store_true', default=False)
        debug.add_argument('--just_return_defaults', help='Just return defaults in dryrun', action='store_true', default=False)
        debug.add_argument('--prettyprint', help='Shows stdout and stderr in a pretty printed format', action='store_true', default=False)
        debug.add_argument('--runtime_debug', help='Logs which functions use most of the time', action='store_true', default=False)
        debug.add_argument('--debug_stack_regex', help='Only print debug messages if call stack matches any regex', type=str, default='')
        debug.add_argument('--debug_stack_trace_regex', help='Show compact call stack with arrows if any function in stack matches regex', type=str, default=None)

        debug.add_argument('--show_func_name', help='Show func name before each execution and when it is done', action='store_true', default=False)

    def load_config(self: Any, config_path: str, file_format: str) -> dict:
        if not os.path.isfile(config_path):
            self._parsing_arguments_loader.stop()
            print("Exit-Code: 5")
            sys.exit(5)

        with open(config_path, mode='r', encoding="utf-8") as file:
            try:
                if file_format == 'yaml':
                    return yaml.safe_load(file)

                if file_format == 'toml':
                    return toml.load(file)

                if file_format == 'json':
                    return json.load(file)
            except (Exception, json.decoder.JSONDecodeError) as e:
                print_red(f"Error parsing {file_format} file '{config_path}'")
                print_debug(f"Error parsing {file_format} file {config_path}: {e}")
                self._parsing_arguments_loader.stop()
                print("Exit-Code: 5")
                sys.exit(5)

        return {}

    def validate_and_convert(self: Any, config: dict, arg_defaults: dict) -> dict:
        """
        Validates the config data and converts them to the right types based on argparse defaults.
        Warns about unknown or unused parameters.
        """
        converted_config = {}
        for key, value in config.items():
            if key in arg_defaults:
                default_value = arg_defaults[key]
                if default_value is not None:
                    expected_type = type(default_value)
                else:
                    expected_type = type(value)

                try:
                    converted_config[key] = expected_type(value)
                except (ValueError, TypeError):
                    print(f"Warning: Cannot convert '{key}' to {expected_type.__name__}. Using default value.")
            else:
                print(f"Warning: Unknown config parameter '{key}' found in the config file and ignored.")

        return converted_config

    def merge_args_with_config(self: Any, config: Any, cli_args: Any) -> argparse.Namespace:
        """ Merge CLI args with config file args (CLI takes precedence) """
        arg_defaults = {arg.dest: arg.default for arg in self.parser._actions if arg.default is not argparse.SUPPRESS}

        validated_config = self.validate_and_convert(config, arg_defaults)

        for key, _ in vars(cli_args).items():
            if key in validated_config:
                setattr(cli_args, key, validated_config[key])

        return cli_args

    def parse_arguments(self: Any) -> argparse.Namespace:
        _args = self.parser.parse_args()

        config = {}

        yaml_and_toml = _args.config_yaml and _args.config_toml
        yaml_and_json = _args.config_yaml and _args.config_json
        json_and_toml = _args.config_json and _args.config_toml

        if yaml_and_toml or yaml_and_json or json_and_toml:
            print("Error: Cannot use YAML, JSON and TOML configuration files simultaneously.]")
            print("Exit-Code: 5")

        if _args.config_yaml:
            config = self.load_config(_args.config_yaml, 'yaml')

        elif _args.config_toml:
            config = self.load_config(_args.config_toml, 'toml')

        elif _args.config_json:
            config = self.load_config(_args.config_json, 'json')

        _args = self.merge_args_with_config(config, _args)

        if _args.dryrun:
            print_yellow("--dryrun activated. This job will try to run only one job which should be running quickly.")

            print_yellow("Setting max_eval to 1, ignoring your settings")
            set_max_eval(1)

            print_yellow("Setting random steps to 0")
            set_random_steps(0)

            print_yellow("Using generation strategy HUMAN_INTERVENTION_MINIMUM")
            set_global_gs_to_HUMAN_INTERVENTION_MINIMUM()

            print_yellow("Setting --force_local_execution to disable SLURM")
            _args.force_local_execution = True

            print_yellow("Disabling TQDM")
            _args.disable_tqdm = True

            print_yellow("Enabling pretty-print")
            _args.prettyprint = True

            if _args.live_share:
                print_yellow("Disabling live-share")
                _args.live_share = False

        return _args

def start_worker_generators() -> None:
    load_existing_data_for_worker_generation_path()

    with spinner("Starting generator workers"):
        if args.worker_generator_path:
            return

        num_workers = max(0, args.number_of_generators - 1)

        if shutil.which("sbatch") is None:
            if num_workers > 1:
                print_yellow("No sbatch, cannot start multiple generation workers")
            return

        omniopt_path = os.path.join(script_dir, "omniopt")
        if not os.path.isfile(omniopt_path):
            print_yellow(f"Cannot find omniopt script at {omniopt_path}")
            return

        def filter_args(_args: list, exclude_params: list) -> list:
            return [arg for arg in _args if all(not arg.startswith(excl) for excl in exclude_params)]

        exclude_params = ["--generate_all_jobs_at_once"]
        filtered_args = filter_args(sys.argv[1:], exclude_params)

        base_command = ["bash", omniopt_path] + filtered_args
        worker_arg = f"--worker_generator_path={get_current_run_folder()}"

        clean_env = copy.deepcopy(os.environ)
        slurm_keys = [key for key in clean_env if key.upper().startswith("SLURM_")]
        for key in slurm_keys:
            del clean_env[key]

        for i in range(num_workers):
            try:
                batch_script = f"""#!/bin/bash
#SBATCH -J worker_generator_{run_uuid}

{" ".join(base_command + [worker_arg])}
"""

                cmd = ["sbatch", "-N", "1"]
                if args.gpus:
                    cmd = cmd + ["--gres", f"gpu:{args.gpus}"]
                result = subprocess.run(
                    cmd,
                    input=batch_script,
                    env=clean_env,
                    check=False,
                    capture_output=True,
                    text=True
                )

                if result.returncode != 0:
                    print_yellow(f"Failed to start worker {i + 1}: {result.stderr.strip()}")
                else:
                    print(f"Started worker {i + 1} via sbatch: {result.stdout.strip()}")

            except Exception as e:
                print_yellow(f"Error starting worker {i + 1}: {e}")

    return

def set_global_gs_to_HUMAN_INTERVENTION_MINIMUM() -> None:
    global prepared_setting_to_custom

    if not prepared_setting_to_custom:
        prepared_setting_to_custom = True
        return

    global global_gs

    node = InteractiveCLIGenerationNode()

    global_gs = GenerationStrategy(
        name="HUMAN_INTERVENTION_MINIMUM",
        nodes=[node]
    )

with spinner("Parsing arguments...") as parsing_arguments_loader:
    loader = ConfigLoader(parsing_arguments_loader)
    args = loader.parse_arguments()

def is_skip_search() -> bool:
    if args.skip_search:
        return True

    if os.getenv("SKIP_SEARCH"):
        return True

    return False

original_result_names = args.result_names

if args.seed is not None:
    with spinner("Importing ax random seed..."):
        from ax.utils.common.random import set_rng_seed

    set_rng_seed(args.seed)

def _fatal_error(message: str, code: int) -> None:
    print_red(message)
    my_exit(code)

if args.max_eval is None and args.generation_strategy is None and args.continue_previous_job is None and (not args.calculate_pareto_front_of_job or len(args.calculate_pareto_front_of_job) == 0):
    _fatal_error("Either --max_eval or --generation_strategy must be set.", 104)

arg_result_names = []
arg_result_min_or_max = []

if len(args.result_names) == 0:
    args.result_names = ["RESULT=min"]

for _rn in args.result_names:
    _key = ""
    _min_or_max = ""

    __default_min_max = "min"

    if "=" in _rn:
        _key, _min_or_max = _rn.split('=', 1)
    else:
        _key = _rn
        _min_or_max = __default_min_max

    if _min_or_max not in ["min", "max"]:
        if _min_or_max:
            print_yellow(f"Value for determining whether to minimize or maximize was neither 'min' nor 'max' for key '{_key}', but '{_min_or_max}'. It will be set to the default, which is '{__default_min_max}' instead.")
        _min_or_max = __default_min_max

    if _key in arg_result_names:
        console.print(f"[red]The --result_names option '{_key}' was specified multiple times![/]")
        sys.exit(50)

    if not re.fullmatch(r'^[a-zA-Z0-9_]+$', _key):
        console.print(f"[red]The --result_names option '{_key}' contains invalid characters! Must be one of a-z, A-Z, 0-9 or _[/]")
        sys.exit(50)

    arg_result_names.append(_key)
    arg_result_min_or_max.append(_min_or_max)

if len(arg_result_names) > 20:
    print_yellow(f"There are {len(arg_result_names)} result_names. This is probably too much.")

if args.continue_previous_job is not None:
    look_for_result_names_file = f"{args.continue_previous_job}/result_names.txt"
    print_debug(f"--continue was set. Trying to figure out if there is a results file in {look_for_result_names_file} and, if so, trying to load it...")

    found_result_names = []

    if os.path.exists(look_for_result_names_file):
        try:
            with open(look_for_result_names_file, 'r', encoding='utf-8') as _file:
                _content = _file.read()
                found_result_names = _content.split('\n')

                if found_result_names and found_result_names[-1] == '':
                    found_result_names.pop()
        except FileNotFoundError:
            print_red(f"Error: The file at '{look_for_result_names_file}' was not found.")
        except IOError as e:
            print_red(f"Error reading file '{look_for_result_names_file}': {e}")
    else:
        print_yellow(f"{look_for_result_names_file} not found!")

    found_result_min_max = []
    default_min_max = "min"

    for _n in range(len(found_result_names)):
        min_max = get_min_max_from_file(args.continue_previous_job, _n, default_min_max)

        found_result_min_max.append(min_max)

    arg_result_names = found_result_names
    arg_result_min_or_max = found_result_min_max

    path_to_external_generator_file = os.path.join(args.continue_previous_job, "state_files", "external_generator")
    if os.path.exists(path_to_external_generator_file) and args.external_generator is None:
        with open(path_to_external_generator_file, encoding="utf-8", mode="r") as ext_gen_f:
            args.external_generator = ext_gen_f.readline().strip()

    path_to_force_choice_for_ranges = os.path.join(args.continue_previous_job, "state_files", "force_choice_for_ranges")
    if os.path.exists(path_to_force_choice_for_ranges):
        args.force_choice_for_ranges = True

try:
    with spinner("Importing torch...") as status:
        import torch
    with spinner("Importing numpy...") as status:
        import numpy as np
    with spinner("Importing ax..."):
        import ax

    with spinner("Importing ax.core.generator_run..."):
        from ax.core.generator_run import GeneratorRun

    with spinner("Importing Cont_X_trans and Y_trans from ax.adapter.registry..."):
        from ax.adapter.registry import Cont_X_trans, Y_trans

    with spinner("Importing ax.core.arm..."):
        from ax.core.arm import Arm

    with spinner("Importing ax.core.objective..."):
        from ax.core.objective import MultiObjective

    with spinner("Importing ax.core.Metric..."):
        from ax.core import Metric

    with spinner("Importing ax.exceptions.core..."):
        import ax.exceptions.core

    with spinner("Importing ax.exceptions.generation_strategy..."):
        import ax.exceptions.generation_strategy

    with spinner("Importing CORE_DECODER_REGISTRY..."):
        from ax.storage.json_store.registry import CORE_DECODER_REGISTRY

    #try:
    with spinner("Trying ax.generation_strategy.generation_node..."):
        import ax.generation_strategy.generation_node

    with spinner("Importing GenerationStep, GenerationStrategy from generation_strategy..."):
        from ax.generation_strategy.generation_strategy import GenerationStep, GenerationStrategy

    with spinner("Importing GenerationNode from generation_node..."):
        from ax.generation_strategy.generation_node import GenerationNode

    with spinner("Importing ExternalGenerationNode..."):
        from ax.generation_strategy.external_generation_node import ExternalGenerationNode

    with spinner("Importing MaxTrials..."):
        from ax.generation_strategy.transition_criterion import MaxTrials

    with spinner("Importing GeneratorSpec..."):
        from ax.generation_strategy.generator_spec import GeneratorSpec

    #except Exception:
    #    with spinner("Fallback: Importing ax.generation_strategy.generation_node..."):
    #        import ax.generation_strategy.generation_node

    #    with spinner("Fallback: Importing GenerationStep, GenerationStrategy from ax.generation_strategy..."):
    #        from ax.generation_strategy.generation_node import GenerationNode, GenerationStep

    #    with spinner("Fallback: Importing ExternalGenerationNode..."):
    #        from ax.generation_strategy.external_generation_node import ExternalGenerationNode

    #    with spinner("Fallback: Importing MaxTrials..."):
    #        from ax.generation_strategy.transition_criterion import MaxTrials

    with spinner("Importing Models from ax.generation_strategy.registry..."):
        from ax.adapter.registry import Models

    with spinner("Importing get_pending_observation_features..."):
        from ax.core.utils import get_pending_observation_features

    with spinner("Importing load_experiment..."):
        from ax.storage.json_store.load import load_experiment

    with spinner("Importing save_experiment..."):
        from ax.storage.json_store.save import save_experiment

    with spinner("Importing save_experiment_to_db..."):
        from ax.storage.sqa_store.save import save_experiment as save_experiment_to_db, save_generation_strategy

    with spinner("Importing TrialStatus..."):
        from ax.core.base_trial import TrialStatus

    with spinner("Importing Data..."):
        from ax.core.data import Data

    with spinner("Importing Experiment..."):
        from ax.core.experiment import Experiment

    with spinner("Importing parameter types..."):
        from ax.core.parameter import RangeParameter, FixedParameter, ChoiceParameter, ParameterType

    with spinner("Importing TParameterization..."):
        from ax.core.types import TParameterization

    with spinner("Importing pandas..."):
        import pandas as pd

    with spinner("Importing AxClient and ObjectiveProperties..."):
        from ax.service.ax_client import AxClient, ObjectiveProperties

    with spinner("Importing RandomForestRegressor..."):
        from sklearn.ensemble import RandomForestRegressor

    with spinner("Importing botorch...") as status:
        import botorch
    with spinner("Importing submitit...") as status:
        import submitit
        from submitit import DebugJob, LocalJob, SlurmJob
except ModuleNotFoundError as ee:
    original_print(f"Base modules could not be loaded: {ee}")
    my_exit(31)
except SignalINT:
    print("\n⚠ Signal INT was detected. Exiting with 128 + 2.")
    my_exit(130)
except SignalUSR:
    print("\n⚠ Signal USR was detected. Exiting with 128 + 10.")
    my_exit(138)
except SignalCONT:
    print("\n⚠ Signal CONT was detected. Exiting with 128 + 18.")
    my_exit(146)
except KeyboardInterrupt:
    print("\n⚠ You pressed CTRL+C. Program execution halted while loading modules.")
    my_exit(0)
except AttributeError:
    print(f"\n⚠ This error means that your virtual environment is probably outdated. Try removing the virtual environment under '{os.getenv('VENV_DIR')}' and re-install your environment.")
    my_exit(7)
except FileNotFoundError as e:
    print(f"\n⚠ Error {e}. This probably means that your hard disk is full")
    my_exit(92)
except ImportError as e:
    print(f"Failed to load module: {e}")
    my_exit(93)

with spinner("Importing ax logger...") as status:
    from ax.utils.common.logger import disable_loggers

with spinner("Importing SQL-Storage-Stuff...") as status:
    from ax.storage.sqa_store.db import init_engine_and_session_factory, get_engine, create_all_tables

    disable_loggers(names=["ax.adapter.base"], level=logging.CRITICAL)

decoder_registry = CORE_DECODER_REGISTRY

NVIDIA_SMI_LOGS_BASE = None
global_gs: Optional[GenerationStrategy] = None

@dataclass(init=False)
class RandomForestGenerationNode(ExternalGenerationNode):
    def __init__(self: Any, regressor_options: Dict[str, Any] = {}, seed: Optional[int] = None, num_samples: int = 1) -> None:
        print_debug("Initializing RandomForestGenerationNode...")
        t_init_start = time.monotonic()
        super().__init__(node_name="RANDOMFOREST")
        self.num_samples: int = num_samples
        self.seed: int = seed

        self.regressor: RandomForestRegressor = RandomForestRegressor(
            **regressor_options,
            random_state=self.seed if self.seed is not None else None
        )

        self.parameters: Optional[Dict[str, Any]] = None
        self.minimize: Optional[bool] = None
        self.fit_time_since_gen: float = time.monotonic() - t_init_start

        fool_linter(self.fit_time_since_gen)

        print_debug("Initialized RandomForestGenerationNode")

    def update_generator_state(self: Any, experiment: Experiment, data: Data) -> None:
        search_space = experiment.search_space
        parameter_names = list(search_space.parameters.keys())
        metric_names = list(experiment.optimization_config.metrics.keys())

        completed_trials = [
            trial for trial in experiment.trials.values() if trial.status == TrialStatus.COMPLETED
        ]
        num_completed_trials = len(completed_trials)

        x = np.zeros([num_completed_trials, len(parameter_names)])
        y = np.zeros([num_completed_trials, 1])

        for t_idx, trial in enumerate(completed_trials):
            trial_parameters = trial.arm.parameters
            x[t_idx, :] = np.array([trial_parameters[p] for p in parameter_names])
            trial_df = data.df[data.df["trial_index"] == trial.index]
            y[t_idx, 0] = trial_df[trial_df["metric_name"] == metric_names[0]]["mean"].item()

        self.regressor.fit(x, y)
        self.parameters = search_space.parameters

        if isinstance(experiment.optimization_config.objective, MultiObjective):
            for moo in experiment.optimization_config.objective.objectives:
                self.minimize.append(moo.minimize)
        else:
            self.minimize = experiment.optimization_config.objective.minimize

    def get_next_candidate(self: Any, pending_parameters: List[TParameterization]) -> TParameterization:
        if self.parameters is None:
            raise RuntimeError("Parameters are not initialized. Call update_generator_state first.")

        ranged_parameters, fixed_values, choice_parameters = self._separate_parameters()
        reverse_choice_map = self._build_reverse_choice_map(choice_parameters)
        ranged_samples = self._generate_ranged_samples(ranged_parameters)
        all_samples = self._build_all_samples(ranged_parameters, ranged_samples, fixed_values, choice_parameters)

        x_pred = self._build_prediction_matrix(all_samples)
        y_pred = self.regressor.predict(x_pred)

        sorted_indices = np.argsort(y_pred) if self.minimize else np.argsort(-np.array(y_pred))

        for idx in sorted_indices:
            candidate = all_samples[idx]
            if self._is_within_constraints(list(candidate.values())):
                self._format_best_sample(candidate, reverse_choice_map)
                return candidate

        raise RuntimeError("No valid candidate found within constraints.")

    def _is_within_constraints(self: Any, params_list: list) -> bool:
        if self.experiment.search_space.parameter_constraints:
            param_names = list(self.parameters.keys())
            params = dict(zip(param_names, params_list))

            for constraint in self.experiment.search_space.parameter_constraints:
                if not constraint.check(params):
                    return False

                return True

        return True

    def _separate_parameters(self: Any) -> tuple[list, dict, dict]:
        ranged_parameters = []
        fixed_values = {}
        choice_parameters = {}

        for name, param in self.parameters.items():
            if isinstance(param, RangeParameter):
                ranged_parameters.append((name, param.lower, param.upper))
            elif isinstance(param, FixedParameter):
                fixed_values[name] = str(param.value)
            elif isinstance(param, ChoiceParameter):
                choice_values = param.values
                choice_value_map = {value: idx for idx, value in enumerate(choice_values)}
                choice_parameters[name] = choice_value_map

        return ranged_parameters, fixed_values, choice_parameters

    def _build_reverse_choice_map(self: Any, choice_parameters: dict) -> dict:
        choice_value_map = {}
        for _, param in choice_parameters.items():
            for value, idx in param.items():
                choice_value_map[value] = idx
        return {idx: value for value, idx in choice_value_map.items()}

    def _generate_ranged_samples(self: Any, ranged_parameters: list) -> np.ndarray:
        ranged_bounds = np.array([[low, high] for _, low, high in ranged_parameters])
        unit_samples = np.random.random_sample([self.num_samples, len(ranged_bounds)])
        return ranged_bounds[:, 0] + (ranged_bounds[:, 1] - ranged_bounds[:, 0]) * unit_samples

    def _build_all_samples(self: Any, ranged_parameters: list, ranged_samples: np.ndarray, fixed_values: dict, choice_parameters: dict) -> list:
        all_samples = []
        for sample_idx in range(self.num_samples):
            sample = self._build_single_sample(sample_idx, ranged_parameters, ranged_samples, fixed_values, choice_parameters)
            all_samples.append(sample)
        return all_samples

    def _build_single_sample(self: Any, sample_idx: int, ranged_parameters: list, ranged_samples: np.ndarray, fixed_values: dict, choice_parameters: dict) -> dict:
        sample = {}

        for dim, (name, _, _) in enumerate(ranged_parameters):
            value = ranged_samples[sample_idx, dim].item()
            param = self.parameters.get(name)
            value = self._cast_value(param, name, value)
            sample[name] = value

        for name, val in fixed_values.items():
            val = str(int(val)) if float(val).is_integer() else str(float(val))
            sample[name] = val

        for name, param in choice_parameters.items():
            param_values_array = list(param.keys())

            choice_index = np.random.choice(param_values_array)

            if self.parameters[name].parameter_type == ParameterType.FLOAT:
                sample[name] = float(param[int(choice_index)])
            elif self.parameters[name].parameter_type == ParameterType.INT:
                sample[name] = int(round(param[int(choice_index)]))
            elif self.parameters[name].parameter_type == ParameterType.STRING:
                value = param[choice_index]
                if isinstance(value, str):
                    sample[name] = value
                else:
                    sample[name] = str(int(value)) if float(value).is_integer() else str(float(value))

        return sample

    def _cast_value(self: Any, param: Any, name: Any, value: Any) -> Union[int, float]:
        if isinstance(param, RangeParameter) and param.parameter_type == "INT":
            return int(round(value))
        if isinstance(param, RangeParameter) and param.parameter_type == "FLOAT":
            return float(value)

        return self._try_convert_to_float(value, name)

    def _try_convert_to_float(self: Any, value: Any, name: str) -> float:
        try:
            return float(value)
        except ValueError as e:
            raise ValueError(f"Parameter '{name}' has a non-numeric value: {value}") from e

    def _build_prediction_matrix(self: Any, all_samples: list) -> np.ndarray:
        x_pred = np.zeros([self.num_samples, len(self.parameters)])
        for sample_idx, sample in enumerate(all_samples):
            for dim, name in enumerate(self.parameters.keys()):
                x_pred[sample_idx, dim] = sample[name]
        return x_pred

    def _format_best_sample(self: Any, best_sample: TParameterization, reverse_choice_map: dict) -> None:
        for name in best_sample.keys():
            param = self.parameters.get(name)
            if isinstance(param, RangeParameter) and param.parameter_type == ParameterType.INT:
                best_sample[name] = int(round(best_sample[name]))
            elif isinstance(param, ChoiceParameter):
                best_sample[name] = str(reverse_choice_map.get(int(best_sample[name])))

decoder_registry["RandomForestGenerationNode"] = RandomForestGenerationNode

def warn_if_param_outside_of_valid_params(param: dict, _res: Any, keyname: str) -> None:
    if param["parameter_type"] == "RANGE":
        _min = param["range"][0]
        _max = param["range"][1]

        if not _min <= _res <= _max:
            print_yellow(f"The result by the external generator for the axis '{keyname}' (RANGE) is outside of the range of min {_min}/max {_max}: {_res}")
    elif param["parameter_type"] == "CHOICE":
        if _res not in param["values"]:
            joined_res = ', '.join(param["values"])
            print_yellow(f"The result by the external generator for the axis '{keyname}' (CHOICE) is not in the valid results {joined_res}: {_res}")
    elif param["parameter_type"] == "FIXED":
        if _res != param["value"]:
            print_yellow(f"The result by the external generator for the axis '{keyname}' (FIXED) is not the specified fixed value '{param['value']}' {_res}")

@dataclass(init=False)
class InteractiveCLIGenerationNode(ExternalGenerationNode):
    """
    A GenerationNode that queries the user on the command line (via *rich*)
    for the next candidate hyper‑parameter set instead of spawning an external
    program.  All prompts come pre‑filled with sensible defaults:

    • If the parameter name matches a key in `_DEFAULT_SPECIALS`, the associated
      value is used:
        – literal ``"min"`` → lower bound of the RangeParameter
        – any other literal  → taken verbatim

    • Otherwise:
        – RangeParameter(INT/FLOAT) ⇒ midpoint (cast to int for INT)
        – ChoiceParameter          ⇒ first element of ``param.values``
        – FixedParameter           ⇒ its fixed value (prompt is skipped)

    The user can simply press *Enter* to accept the default or type a new
    value (validated & casted to the correct type automatically).
    """
    seed: int
    parameters: Optional[Dict[str, Any]]
    minimize: Optional[bool]
    data: Optional[Any]
    constraints: Optional[Any]
    fit_time_since_gen: float

    def __init__(
        self: Any,
        node_name: str = "INTERACTIVE_GENERATOR",
    ) -> None:
        t0 = time.monotonic()
        super().__init__(node_name=node_name)
        self.parameters = None
        self.minimize = None
        self.data = None
        self.constraints = None
        self.seed = int(time.time())  # deterministic seeds are pointless here
        self.fit_time_since_gen = time.monotonic() - t0

    def update_generator_state(self: Any, experiment: Any, data: Any) -> None:
        self.data = data
        search_space = experiment.search_space
        self.parameters = search_space.parameters
        self.constraints = search_space.parameter_constraints

    @staticmethod
    def _ptype_to_str(param_type: Any) -> str:
        return {
            ParameterType.INT: "INT",
            ParameterType.FLOAT: "FLOAT",
            ParameterType.STRING: "STRING",
        }.get(param_type, "<UNKNOWN>")

    def _default_for_param(self: Any, name: str, param: Any) -> Any:
        # 1. explicit override
        if name.lower() in _DEFAULT_SPECIALS:
            override = _DEFAULT_SPECIALS[name.lower()]
            if override == "max" and isinstance(param, RangeParameter):
                return param.upper
            if override == "min" and isinstance(param, RangeParameter):
                return param.lower
            return override

        # 2. generic rules
        if isinstance(param, FixedParameter):
            return param.value
        if isinstance(param, RangeParameter):
            mid = (param.lower + param.upper) / 2
            return int(mid) if param.parameter_type == ParameterType.INT else mid
        if isinstance(param, ChoiceParameter):
            return param.values[0]

        # fall back
        return None

    def _ask_user(self: Any, name: str, param: Any, default: Any) -> Any:
        if args.just_return_defaults:
            print_yellow(f"Returning defaults for '{name}' in dry-mode with --just_return_defaults")
            return default

        if not console.is_terminal:
            print_red(f"Cannot prompt for {name!r}: no interactive terminal attached.")
            return default

        prompt_msg = f"{name} ({self._ptype_to_str(param.parameter_type)})"

        if isinstance(param, FixedParameter):
            try:
                return self._handle_fixed(param)
            except Exception as e:
                print_red(f"Error #1: {e}")

        elif isinstance(param, ChoiceParameter):
            try:
                return self._handle_choice(param, default, prompt_msg)
            except Exception as e:
                print_red(f"Error #2: {e}")

        elif isinstance(param, RangeParameter):
            try:
                return self._handle_range(param, default, prompt_msg)
            except Exception as e:
                print_red(f"Error #3: {e}")

        return self._handle_fallback(prompt_msg, default, param)

    def _handle_fixed(self, param: Any) -> Any:
        return param.value

    def _handle_choice(self, param: Any, default: Any, prompt_msg: str) -> Any:
        choices_str = ", ".join(f"{v}" for v in param.values)
        console.print(f"{prompt_msg} choices → {choices_str}")
        user_val = Prompt.ask("Pick choice", default=str(default))
        return param.values[int(user_val)] if user_val.isdigit() else user_val

    def _handle_range(self, param: Any, default: Any, prompt_msg: str) -> Any:
        low, high = param.lower, param.upper
        console.print(f"{prompt_msg} range → [{low}, {high}]")

        if param.parameter_type == ParameterType.FLOAT:
            user_val = FloatPrompt.ask("Enter float", default=str(default))
            try:
                val = float(user_val)
            except ValueError:
                val = default
        else:
            user_val = IntPrompt.ask("Enter int", default=str(default))
            try:
                val = int(user_val)
            except ValueError:
                val = default

        return min(max(val, low), high)

    def _handle_fallback(self, prompt_msg: str, default: Any, param: Any) -> Any:
        print_red(f"Unknown type detected: {param}")
        return Prompt.ask(prompt_msg, default=str(default))

    def get_next_candidate(
        self: Any,
        pending_parameters: List[TParameterization],
    ) -> Dict[str, Any]:
        """
        Build the next candidate by querying the user for **each** parameter.
        Raises RuntimeError if `update_generator_state` has not been called.
        """
        if self.parameters is None:
            raise RuntimeError(
                "Parameters are not initialized – call update_generator_state() first."
            )

        console.rule("[bold magenta]Enter values for evaluation point, or press enter to accept the default[/]")

        candidate: Dict[str, Any] = {}
        for name, param in self.parameters.items():
            default_val = self._default_for_param(name, param)
            value = self._ask_user(name, param, default_val)
            candidate[name] = value

        # ── simple constraint check (optional) ──────────────────────────

        if self.constraints:
            console.rule("[bold magenta]Checking constraints[/]")
            violations = [
                c
                for c in self.constraints
                if not c.check(candidate)  # Ax Constraint objects support .check
            ]
            if violations:
                console.print(
                    "[red]WARNING:[/] The candidate violates "
                    f"{len(violations)} constraint(s): {violations}"
                )

        # show summary table
        tbl = Table(title="Chosen Hyperparameters", show_lines=True)
        tbl.add_column("Name", style="cyan", no_wrap=True)
        tbl.add_column("Value", style="green")
        for k, v in candidate.items():
            tbl.add_row(k, str(v))
        console.print(tbl)

        console.rule()
        return candidate

@dataclass(init=False)
class ExternalProgramGenerationNode(ExternalGenerationNode):
    def __init__(self: Any, external_generator: str = args.external_generator, node_name: str = "EXTERNAL_GENERATOR") -> None:
        print_debug("Initializing ExternalProgramGenerationNode...")
        t_init_start = time.monotonic()
        super().__init__(node_name=node_name)
        self.seed: int = args.seed
        self.external_generator: str = decode_if_base64(external_generator)
        self.constraints = None
        self.data = None

        self.parameters: Optional[Dict[str, Any]] = None
        self.minimize: Optional[bool] = None
        self.fit_time_since_gen: float = time.monotonic() - t_init_start

        print_debug(f"Initialized ExternalProgramGenerationNode in {self.fit_time_since_gen:.4f} seconds")

    def update_generator_state(self: Any, experiment: Any, data: Any) -> None:
        print_debug("Updating generator state...")

        self.data = data

        search_space = experiment.search_space
        self.parameters = search_space.parameters

        print_debug("Generator state updated successfully.")

    def _parameter_type_to_string(self: Any, param_type: Any) -> str:
        if param_type == ParameterType.INT:
            return "INT"

        if param_type == ParameterType.FLOAT:
            return "FLOAT"

        if param_type == ParameterType.STRING:
            return "STRING"

        _fatal_error(f"Unknown data type {param_type}", 33)

        return ""

    def _serialize_parameters(self: Any, params: dict) -> dict:
        serialized = {}
        for key in params.keys():
            param = params[key]
            param_name = param.name
            if isinstance(param, FixedParameter):
                serialized[param_name] = {
                        "parameter_type": "FIXED",
                        "type": self._parameter_type_to_string(param.parameter_type),
                        "value": param.value
                }
            elif isinstance(param, RangeParameter):
                serialized[param_name] = {
                        "parameter_type": "RANGE",
                        "type": self._parameter_type_to_string(param.parameter_type),
                        "range": [param.lower, param.upper]
                }
            elif isinstance(param, ChoiceParameter):
                serialized[param_name] = {
                        "parameter_type": "CHOICE",
                        "type": self._parameter_type_to_string(param.parameter_type),
                        "values": param.values
                }
            else:
                _fatal_error(f"Unknown parameter type: {param}", 15)

        return serialized

    def _serialize_constraints(self: Any, constraints: Optional[list]) -> Optional[list]:
        parsed_constraints = []
        if constraints and len(constraints):
            for constraint in constraints:
                representation = str(constraint)
                equation = representation[representation.find('(') + 1:representation.rfind(')')]

                parsed_constraints.append(equation)

        return parsed_constraints

    def get_and_create_temp_dir(self: Any) -> str:
        temp_dir_counter = 0
        temp_dir = os.path.join(get_current_run_folder(), "external_generator_tmp", str(temp_dir_counter))
        while os.path.isdir(temp_dir):
            temp_dir_counter = temp_dir_counter + 1
            temp_dir = os.path.join(get_current_run_folder(), "external_generator_tmp", str(temp_dir_counter))

        os.makedirs(temp_dir, exist_ok=True)

        print_debug(f"Created temporary directory: {temp_dir}")

        return temp_dir

    def get_next_candidate(self: Any, pending_parameters: List[TParameterization]) -> Any:
        if self.parameters is None:
            raise RuntimeError("Parameters are not initialized. Call update_generator_state first.")

        print_debug("Getting next candidate...")

        try:
            temp_dir = self.get_and_create_temp_dir()

            if self.experiment.search_space.parameter_constraints:
                self.constraints = self.experiment.search_space.parameter_constraints

            serialized_params = self._serialize_parameters(self.parameters)

            current_trials = parse_csv(RESULT_CSV_FILE)

            this_objectives = {}

            for i in range(len(arg_result_names)):
                this_objectives[arg_result_names[i]] = arg_result_min_or_max[i]

            input_json = {
                "parameters": serialized_params,
                "constraints": self._serialize_constraints(self.constraints),
                "seed": self.seed,
                "trials": current_trials,
                "objectives": this_objectives
            }

            input_path = os.path.join(temp_dir, "input.json")

            with open(input_path, mode="w", encoding="utf-8") as f:
                json.dump(input_json, f, indent=4)

            print_debug(f"Saved inputs.json to {input_path}")

            run_this_program = self.external_generator.replace('\n', '').split() + [temp_dir]

            subprocess.run(run_this_program, check=True)

            print_debug(f"Executed external program: {' '.join(run_this_program)}")

            results_path = os.path.join(temp_dir, "results.json")
            if not os.path.exists(results_path):
                raise FileNotFoundError(f"Missing results.json in {results_path}")

            with open(results_path, mode="r", encoding="utf-8") as f:
                results = json.load(f)
            print_debug(f"Loaded results.json from {results_path}")

            if "parameters" not in results:
                raise ValueError(f"Invalid results format in {results_path}")

            for keyname in serialized_params.keys():
                if keyname not in results["parameters"]:
                    raise ValueError(f"Missing {keyname} from JSON file {results_path}")

                to_type = serialized_params[keyname]["type"]

                _res = results["parameters"][keyname]

                try:
                    if to_type == "STRING":
                        results["parameters"][keyname] = str(_res)
                    elif to_type == "FLOAT":
                        results["parameters"][keyname] = float(_res)
                    elif to_type == "INT":
                        results["parameters"][keyname] = int(_res)
                except ValueError:
                    failed_res = results["parameters"][keyname]
                    print_red(f"Failed to convert '{keyname}' to {to_type}. Value: {failed_res}")

                warn_if_param_outside_of_valid_params(serialized_params[keyname], _res, keyname)

            candidate = results["parameters"]
            print_debug(f"Found new candidate: {candidate}")
            return candidate

        except Exception as e:
            raise RuntimeError(f"Error getting next candidate: {e}") from e

decoder_registry["ExternalProgramGenerationNode"] = ExternalProgramGenerationNode

def append_and_read(file: str, nr: int = 0, recursion: int = 0) -> int:
    try:
        with open(file, mode='a+', encoding="utf-8") as f:
            f.seek(0)
            nr_lines = len(f.readlines())

            if nr == 1:
                f.write('1\n')

        return nr_lines

    except FileNotFoundError as e:
        original_print(f"File not found: {e}")
    except (SignalUSR, SignalINT, SignalCONT):
        if recursion:
            print_red("Recursion error in append_and_read.")
            sys.exit(199)
        append_and_read(file, nr, recursion + 1)
    except OSError as e:
        print_red(f"OSError: {e}. This may happen on unstable file systems.")
        sys.exit(199)
    except Exception as e:
        print(f"Error editing the file: {e}")

    return 0

def run_live_share_command(force: bool = False) -> Tuple[str, str]:
    global shown_run_live_share_command

    if get_current_run_folder():
        try:
            _user = get_username()

            _command = f"bash {script_dir}/omniopt_share {get_current_run_folder()} --update --username={_user} --no_color"

            if force:
                _command = f"{_command} --force"

            if args.share_password:
                _command = f"{_command} --password='{args.share_password}'"

            if not shown_run_live_share_command:
                print_debug(f"run_live_share_command: {_command}")
                shown_run_live_share_command = True

            result = subprocess.run(_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            return str(result.stdout), str(result.stderr)
        except subprocess.CalledProcessError as e:
            if e.stderr:
                original_print(f"run_live_share_command: command failed with error: {e}, stderr: {e.stderr}")
            else:
                original_print(f"run_live_share_command: command failed with error: {e}")
            return "", str(e.stderr)
        except Exception as e:
            print(f"run_live_share_command: An error occurred: {e}")

    return "", ""

def force_live_share() -> bool:
    if args.live_share:
        return live_share(True)

    return False

def live_share(force: bool = False, text_and_qr: bool = False) -> bool:
    if not get_current_run_folder():
        print(f"live_share: get_current_run_folder was empty or false: {get_current_run_folder()}")
        return False

    if not args.live_share or not get_current_run_folder():
        return False

    stdout, stderr = run_live_share_command(force)

    if text_and_qr:
        if stderr:
            print_green(stderr)
        else:
            print_red("This call should have shown the CURL, but didnt. Stderr: {stderr}, stdout: {stdout}")
    if stdout:
        print_debug(f"live_share stdout: {stdout}")

    return True

def init_live_share() -> bool:
    with spinner("Initializing live share..."):
        ret = live_share(True, True)

        return ret

async def start_periodic_live_share() -> None:
    if args.live_share and not os.environ.get("CI"):
        while True:
            live_share(force=False)
            time.sleep(30)

def init_storage(db_url: str) -> None:
    init_engine_and_session_factory(url=db_url, force_init=True)
    engine = get_engine()
    create_all_tables(engine)

def try_saving_to_db() -> None:
    try:
        global initialized_storage

        db_url = f"sqlite:////{get_current_run_folder()}/database.db"

        if args.db_url:
            db_url = args.db_url

        if not initialized_storage:
            init_storage(db_url)

            initialized_storage = True

        if ax_client is not None:
            save_experiment_to_db(ax_client.experiment)
        else:
            print_red("ax_client was not defined in try_saving_to_db")
            my_exit(101)
        save_generation_strategy(global_gs)
    except Exception as e:
        print_debug(f"Failed trying to save sqlite3-DB: {e}")

def merge_with_job_infos(df: pd.DataFrame) -> pd.DataFrame:
    job_infos_path = os.path.join(get_current_run_folder(), "job_infos.csv")
    if not os.path.exists(job_infos_path):
        return df

    job_df = pd.read_csv(job_infos_path)

    if 'trial_index' not in df.columns or 'trial_index' not in job_df.columns:
        raise ValueError("Both DataFrames must contain a 'trial_index' column.")

    job_df_filtered = job_df[job_df['trial_index'].isin(df['trial_index'])]

    new_cols = [col for col in job_df_filtered.columns if col != 'trial_index' and col not in df.columns]

    job_df_reduced = job_df_filtered[['trial_index'] + new_cols]

    merged = pd.merge(df, job_df_reduced, on='trial_index', how='left')

    old_cols = [col for col in df.columns if col != 'trial_index']

    new_order = ['trial_index'] + new_cols + old_cols

    merged = merged[new_order]

    return merged

def save_results_csv() -> Optional[str]:
    if args.dryrun:
        return None

    pd_csv, pd_json = get_results_paths()

    save_experiment_state()

    if not ax_client:
        return None

    save_checkpoint()

    try:
        df = fetch_and_prepare_trials()
        write_csv(df, pd_csv)
        write_json_snapshot(pd_json)
        save_experiment_to_file()

        if should_save_to_database():
            try_saving_to_db()
        elif args.save_to_database:
            print_debug(f"Model {args.model} is an uncontinuable model, so it will not be saved to a DB")

    except (SignalUSR, SignalCONT, SignalINT) as e:
        raise type(e)(str(e)) from e
    except Exception as e:
        print_red(f"While saving all trials as a pandas-dataframe-csv, an error occurred: {e}")

    return pd_csv

def get_results_paths() -> tuple[str, str]:
    return (get_current_run_folder(RESULTS_CSV_FILENAME), get_state_file_name('pd.json'))

def fetch_and_prepare_trials() -> pd.DataFrame:
    ax_client.experiment.fetch_data()
    df = ax_client.get_trials_data_frame()

    #print("========================")
    #print("BEFORE merge_with_job_infos:")
    #print(df["generation_node"])
    df = merge_with_job_infos(df)
    #print("AFTER merge_with_job_infos:")
    #print(df["generation_node"])

    return df

def write_csv(df: pd.DataFrame, path: str) -> None:
    try:
        df = df.sort_values(by=["trial_index"], kind="stable").reset_index(drop=True)
    except KeyError:
        pass
    df.to_csv(path, index=False, float_format="%.30f")

def write_json_snapshot(path: str) -> None:
    json_snapshot = ax_client.to_json_snapshot()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_snapshot, f, indent=4)

def save_experiment_to_file() -> None:
    save_experiment(
        ax_client.experiment,
        get_state_file_name("ax_client.experiment.json")
    )

def should_save_to_database() -> bool:
    return args.model not in uncontinuable_models and args.save_to_database

def add_to_phase_counter(phase: str, nr: int = 0, run_folder: str = "") -> int:
    if run_folder == "":
        run_folder = get_current_run_folder()
    return append_and_read(f'{run_folder}/state_files/phase_{phase}_steps', nr)

if args.model and str(args.model).upper() not in SUPPORTED_MODELS:
    print(f"Unsupported model {args.model}. Cannot continue. Valid models are {joined_supported_models}")
    my_exit(203)

if isinstance(args.num_parallel_jobs, int) or helpers.looks_like_int(args.num_parallel_jobs):
    num_parallel_jobs = int(args.num_parallel_jobs)

if num_parallel_jobs <= 0:
    _fatal_error(f"--num_parallel_jobs must be 1 or larger, is {num_parallel_jobs}", 106)

class SearchSpaceExhausted (Exception):
    pass

NR_INSERTED_JOBS: int = 0
submitit_executor: Union[LocalExecutor, AutoExecutor, None] = None

NR_OF_0_RESULTS: int = 0

orchestrator = None

def print_logo() -> None:
    if os.environ.get('NO_OO_LOGO') is not None:
        return

    if random.choice([True, False]):
        sprueche = [
            "Tuning hyperparameters like a caffeinated monk with a Rubik's Cube.",
            "Finds minima faster than a squirrel on espresso.",
            "More focused than a cat watching a laser pointer.",
            "Exploring parameter space like a ninja in a maze.",
            "Makes grid search look like it's still on dial-up.",
            "Fine-tuning like a boss!",
            "Finding the needle in the hyper haystack!",
            "Hyperparameters? Nailed it!",
            "Optimizing with style!",
            "Dialing in the magic numbers.",
            "Turning knobs since day one!",
            "When in doubt, optimize!",
            "Tuning like a maestro!",
            "In search of the perfect fit.",
            "Hyper-sanity check complete!",
            "Taking parameters to the next level.",
            "Cracking the code of perfect parameters!",
            "Turning dials like a DJ!",
            "In pursuit of the ultimate accuracy!",
            "May the optimal values be with you.",
            "Tuning up for success!",
            "Animals are friends, not food!",
            "Hyperparam magic, just add data!",
            "Unlocking the secrets of the grid.",
            "Tuning: because close enough isn't good enough.",
            "When it clicks, it sticks!",
            "Adjusting the dials, one click at a time.",
            "Finding the sweet spot in the matrix.",
            "Like a hyperparameter whisperer.",
            "Cooking up some optimization!",
            "Because defaults are for amateurs.",
            "Maximizing the model mojo!",
            "Hyperparameter alchemy in action!",
            "Precision tuning, no shortcuts.",
            "Climbing the hyperparameter mountain... Montana Sacra style!",
            "better than OmniOpt1!",
            "Optimizing like it's the Matrix, but I am the One.",
            "Not all who wander are lost... just doing a random search.",
            "Grid search? Please, I’m doing ballet through parameter space.",
            "Hyperparameter tuning: part science, part sorcery.",
            "Channeling my inner Gandalf: ‘You shall not pass... without fine-tuning!’",
            "Inception-level optimization: going deeper with every layer.",
            "Hyperparameter quest: It's dangerous to go alone, take this!",
            "Tuning like a Jedi: Feel the force of the optimal values.",
            "Welcome to the Hyperparameter Games: May the odds be ever in your favor!",
            "Like Neo, dodging suboptimal hyperparameters in slow motion.",
            "Hyperparameters: The Hitchcock thriller of machine learning.",
            "Dialing in hyperparameters like a classic noir detective.",
            "It’s a hyperparameter life – every tweak counts!",
            "As timeless as Metropolis, but with better optimization.",
            "Adjusting parameters with the precision of a laser-guided squirrel.",
            "Tuning hyperparameters with the finesse of a cat trying not to knock over the vase.",
            "Optimizing parameters with the flair of a magician pulling rabbits out of hats.",
            "Optimizing like a koala climbing a tree—slowly but surely reaching the perfect spot.",
            "Tuning so deep, even Lovecraft would be scared!",
            "Dialing in parameters like Homer Simpson at an all-you-can-eat buffet - endless tweaks!",
            "Optimizing like Schrödinger’s cat—until you look, it's both perfect and terrible.",
            "Hyperparameter tuning: the art of making educated guesses look scientific!",
            "Cranking the dials like a mad scientist - IT’S ALIIIIVE!",
            "Tuning like a pirate - arr, where be the optimal values?",
            "Hyperparameter tuning: the extreme sport of machine learning!",
            "Fine-tuning on a quantum level – Schrödinger’s hyperparameter.",
            "Like an alchemist searching for the Philosopher’s Stone.",
            "The fractal of hyperparameters: The deeper you go, the more you see.",
            "Adjusting parameters as if it were a sacred ritual.",
            "Machine, data, parameters – the holy trinity of truth.",
            "A trip through the hyperspace of the parameter landscape.",
            "A small tweak, a big effect – the butterfly principle of tuning.",
            "Like a neural synapse becoming self-aware.",
            "The Montana Sacra of optimization – only the enlightened reach the peak.",
            "Fine-tuning on the frequency of reality.",
            "Hyperparameters: Where science and magic shake hands.",
            "Open the third eye of optimization – the truth is in the numbers.",
            "Walking the fine line between overfit and underfit like a tightrope artist.",
            "This is not madness... this is hyperparameter tuning!",
            "Hyperparameter tuning: The philosopher’s stone of deep learning.",
            "Dancing on the edge of chaos – welcome to the tuning dimension.",
            "Like Borges' infinite library, but every book is a different model configuration."
        ]

        spruch = random.choice(sprueche)

        _cn = [
            'cow',
            'daemon',
            'dragon',
            'fox',
            'ghostbusters',
            'kitty',
            'milk',
            'pig',
            'stegosaurus',
            'stimpy',
            'trex',
            'turtle',
            'tux'
        ]

        char = random.choice(_cn)

        cowsay.char_funcs[char](f"OmniOpt2 - {spruch}")
    else:
        if figlet_loaded:
            fonts = [
                "slant",
                "big",
                "doom",
                "larry3d",
                "starwars",
                "colossal",
                "avatar",
                "pebbles",
                "script",
                "stop",
                "banner3",
                "nancyj",
                "poison"
            ]

            f = Figlet(font=random.choice(fonts))
            original_print(f.renderText('OmniOpt2'))
        else:
            original_print('OmniOpt2')

process = None
try:
    process = psutil.Process(os.getpid())
except Exception as e:
    print(f"Error trying to get process: {e}")

global_vars: dict = {}
global_vars_jobs_lock = threading.Lock()

VAL_IF_NOTHING_FOUND: int = 99999999999999999999999999999999999999999999999999999999999
NO_RESULT: str = "{:.0e}".format(VAL_IF_NOTHING_FOUND)

global_vars["jobs"] = []
global_vars["_time"] = None
global_vars["mem_gb"] = None
global_vars["num_parallel_jobs"] = None
global_vars["parameter_names"] = []

main_pid = os.getpid()

def set_nr_inserted_jobs(new_nr_inserted_jobs: int) -> None:
    global NR_INSERTED_JOBS

    print_debug(f"set_nr_inserted_jobs({new_nr_inserted_jobs})")

    NR_INSERTED_JOBS = new_nr_inserted_jobs

def write_worker_usage() -> None:
    if len(WORKER_PERCENTAGE_USAGE):
        csv_filename = get_current_run_folder('worker_usage.csv')

        csv_columns = ['time', 'num_parallel_jobs', 'nr_current_workers', 'percentage']

        with open(csv_filename, mode='w', encoding="utf-8", newline='') as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            for row in WORKER_PERCENTAGE_USAGE:
                csv_writer.writerow(row)
    else:
        if is_slurm_job():
            print_debug("WORKER_PERCENTAGE_USAGE seems to be empty. Not writing worker_usage.csv")

def log_system_usage() -> None:
    if not get_current_run_folder():
        return

    ram_cpu_csv_file_path = os.path.join(get_current_run_folder(), "cpu_ram_usage.csv")

    makedirs(os.path.dirname(ram_cpu_csv_file_path))

    file_exists = os.path.isfile(ram_cpu_csv_file_path)

    with open(ram_cpu_csv_file_path, mode='a', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)

        current_time = int(time.time())

        if process is not None:
            mem_proc = process.memory_info()

            if mem_proc is not None:
                ram_usage_mb = mem_proc.rss / (1024 * 1024)
                cpu_usage_percent = psutil.cpu_percent(percpu=False)

                if ram_usage_mb > 0 and cpu_usage_percent > 0:
                    if not file_exists:
                        writer.writerow(["timestamp", "ram_usage_mb", "cpu_usage_percent"])

                    writer.writerow([current_time, ram_usage_mb, cpu_usage_percent])

def write_process_info() -> None:
    try:
        log_system_usage()
    except Exception as e:
        print_debug(f"Error retrieving process information: {str(e)}")

def log_nr_of_workers() -> None:
    try:
        write_process_info()
    except Exception as e:
        print_debug(f"log_nr_of_workers: failed to write_process_info: {e}")
        return None

    if "jobs" not in global_vars:
        print_debug("log_nr_of_workers: Could not find jobs in global_vars")
        return None

    nr_current_workers, nr_current_workers_errmsg = count_jobs_in_squeue()

    if not nr_current_workers or nr_current_workers_errmsg:
        if nr_current_workers_errmsg:
            print_debug(f"log_nr_of_workers: {nr_current_workers_errmsg}")
        else:
            print_debug("log_nr_of_workers: nr_current_workers is False")
        return None

    try:
        with open(logfile_nr_workers, mode='a+', encoding="utf-8") as f:
            f.write(str(nr_current_workers) + "\n")
    except FileNotFoundError:
        _fatal_error(f"It seems like the folder for writing {logfile_nr_workers} was deleted during the run. Cannot continue.", 99)
    except OSError as e:
        _fatal_error(f"Tried writing log_nr_of_workers to file {logfile_nr_workers}, but failed with error: {e}. This may mean that the file system you are running on is instable. OmniOpt2 probably cannot do anything about it.", 199)

    return None

def log_data() -> None:
    try:
        log_worker_numbers()
    except Exception as e:
        print_debug(f"Error in log_worker_numbers: {e}")

    if "write_worker_usage" in globals():
        try:
            write_worker_usage()
        except Exception:
            pass

    if "write_process_info" in globals():
        try:
            write_process_info()
        except Exception as e:
            print_debug(f"Error in write_process_info: {e}")

    if "log_nr_of_workers" in globals():
        try:
            log_nr_of_workers()
        except Exception as e:
            print_debug(f"Error in log_nr_of_workers: {e}")

def get_line_info() -> Any:
    try:
        stack = inspect.stack()
        if len(stack) < 2:
            return ("<no caller>", ":", -1, ":", "<unknown>")

        frame_info = stack[1]

        try:
            filename = str(frame_info.filename)
        except Exception as e:
            filename = f"<filename error: {e}>"

        try:
            lineno = int(frame_info.lineno)
        except Exception as e:
            lineno = -1

            print_red(f"Error in get_line_info: {e}, using lineno = -1")

        try:
            function = str(frame_info.function)
        except Exception as e:
            function = f"<function error: {e}>"

        return (filename, ":", lineno, ":", function)

    except Exception as e:
        return ("<exception in get_line_info>", ":", -1, ":", str(e))

def print_image_to_cli(image_path: str, width: int) -> bool:
    print("")

    if not supports_sixel():
        print("Cannot print sixel in this environment.")
        return False

    try:
        image = Image.open(image_path)
        original_width, original_height = image.size

        height = int((original_height / original_width) * width)

        sixel_converter = sixel.converter.SixelConverter(image_path, w=width, h=height)

        sixel_converter.write(sys.stdout)

        print("")

        _sleep(2)

        return True
    except Exception as e:
        print_debug(
            f"Error converting and resizing image: "
            f"{str(e)}, width: {width}, image_path: {image_path}"
        )

    return False

def log_message_to_file(_logfile: Union[str, None], message: str, _lvl: int = 0, eee: Union[None, str, Exception] = None) -> None:
    if not _logfile:
        return None

    if _lvl > 3:
        original_print(f"Cannot write _debug, error: {eee}")
        return None

    try:
        with open(_logfile, mode='a', encoding="utf-8") as f:
            #original_print(f"========= {time.time()} =========", file=f)
            original_print(message, file=f)
    except FileNotFoundError:
        print_red("It seems like the run's folder was deleted during the run. Cannot continue.")
        sys.exit(99)
    except Exception as e:
        original_print(f"Error trying to write log file: {e}")
        log_message_to_file(_logfile, message, _lvl + 1, e)

    return None

def _log_trial_index_to_param(trial_index: dict, _lvl: int = 0, eee: Union[None, str, Exception] = None) -> None:
    log_message_to_file(logfile_trial_index_to_param_logs, str(trial_index), _lvl, str(eee))

def _debug_worker_creation(msg: str, _lvl: int = 0, eee: Union[None, str, Exception] = None) -> None:
    log_message_to_file(logfile_worker_creation_logs, msg, _lvl, str(eee))

def append_to_nvidia_smi_logs(_file: str, _host: str, result: str, _lvl: int = 0, eee: Union[None, str, Exception] = None) -> None:
    log_message_to_file(_file, result, _lvl, str(eee))

def _debug_progressbar(msg: str, _lvl: int = 0, eee: Union[None, str, Exception] = None) -> None:
    log_message_to_file(logfile_progressbar, msg, _lvl, str(eee))

def decode_if_base64(input_str: str) -> str:
    try:
        decoded_bytes = base64.b64decode(input_str)
        decoded_str = decoded_bytes.decode('utf-8')
        return decoded_str
    except Exception:
        return input_str

def get_file_as_string(f: str) -> str:
    datafile: str = ""
    if not os.path.exists(f):
        print_debug(f"{f} not found!")
        return ""

    with open(f, encoding="utf-8") as _f:
        _df = _f.read()

        if isinstance(_df, str):
            datafile = _df
        else:
            datafile = "\n".join(_df)

    return "".join(datafile)

global_vars["joined_run_program"] = ""

if not args.continue_previous_job:
    if args.run_program:
        if isinstance(args.run_program, list):
            global_vars["joined_run_program"] = " ".join(args.run_program[0])
        else:
            global_vars["joined_run_program"] = args.run_program

        global_vars["joined_run_program"] = decode_if_base64(global_vars["joined_run_program"])
else:
    prev_job_folder = args.continue_previous_job
    prev_job_file = f"{prev_job_folder}/state_files/joined_run_program"
    if os.path.exists(prev_job_file):
        global_vars["joined_run_program"] = get_file_as_string(prev_job_file)
    else:
        _fatal_error(f"The previous job file {prev_job_file} could not be found. You may forgot to add the run number at the end.", 44)

if not args.tests and len(global_vars["joined_run_program"]) == 0 and not args.calculate_pareto_front_of_job:
    _fatal_error("--run_program was empty", 19)

global_vars["experiment_name"] = args.experiment_name

def load_global_vars(_file: str) -> None:
    global global_vars

    if not os.path.exists(_file):
        _fatal_error(f"You've tried to continue a non-existing job: {_file}", 44)
    try:
        with open(_file, encoding="utf-8") as f:
            global_vars = json.load(f)
    except Exception as e:
        _fatal_error(f"Error while loading old global_vars: {e}, trying to load {_file}", 44)

def load_or_exit(filepath: str, error_msg: str, exit_code: int) -> None:
    if not os.path.exists(filepath):
        _fatal_error(error_msg, exit_code)

def get_file_content_or_exit(filepath: str, error_msg: str, exit_code: int) -> str:
    load_or_exit(filepath, error_msg, exit_code)
    return get_file_as_string(filepath).strip()

def check_param_or_exit(param: Any, error_msg: str, exit_code: int) -> None:
    if param is None:
        _fatal_error(error_msg, exit_code)

def check_continue_previous_job(continue_previous_job: Optional[str]) -> dict:
    if continue_previous_job:
        load_global_vars(f"{continue_previous_job}/state_files/global_vars.json")

        if not global_vars.get("experiment_name"):
            exp_name_file = f"{continue_previous_job}/experiment_name"
            global_vars["experiment_name"] = get_file_content_or_exit(
                exp_name_file,
                f"{exp_name_file} not found, and no --experiment_name given. Cannot continue.",
                19
            )
    return global_vars

def check_required_parameters(_args: Any) -> None:
    check_param_or_exit(
        _args.parameter or _args.continue_previous_job or args.calculate_pareto_front_of_job is None or len(args.calculate_pareto_front_of_job) == 0,
        "Either --parameter, --calculate_pareto_front_of_job or --continue_previous_job is required. Both were not found.",
        19
    )
    check_param_or_exit(
        _args.run_program or _args.continue_previous_job or args.calculate_pareto_front_of_job is None or len(args.calculate_pareto_front_of_job) == 0,
        "--run_program or --calculate_pareto_front_of_job needs to be defined when --continue_previous_job is not set",
        19
    )
    check_param_or_exit(
        global_vars.get("experiment_name") or _args.continue_previous_job or args.calculate_pareto_front_of_job is None or len(args.calculate_pareto_front_of_job) == 0,
        "--experiment_name or --calculate_pareto_front_of_job needs to be defined when --continue_previous_job is not set",
        19
    )

def load_time_or_exit(_args: Any) -> None:
    if _args.time:
        global_vars["_time"] = _args.time
    elif _args.continue_previous_job:
        time_file = f"{_args.continue_previous_job}/state_files/time"
        time_content = get_file_content_or_exit(time_file, f"neither --time nor file {time_file} found", 19).rstrip()
        time_content = time_content.replace("\n", "").replace(" ", "")

        if time_content.isdigit():
            global_vars["_time"] = int(time_content)
            print_yellow(f"Using old run's --time: {global_vars['_time']}")
        else:
            print_yellow(f"Time-setting: The contents of {time_file} do not contain a single number")
    else:
        if len(args.calculate_pareto_front_of_job) == 0:
            _fatal_error("Missing --time parameter. Cannot continue.", 19)

def load_mem_gb_or_exit(_args: Any) -> Optional[int]:
    if _args.mem_gb:
        return int(_args.mem_gb)

    if _args.continue_previous_job:
        mem_gb_file = f"{_args.continue_previous_job}/state_files/mem_gb"
        mem_gb_content = get_file_content_or_exit(mem_gb_file, f"neither --mem_gb nor file {mem_gb_file} found", 19)
        if mem_gb_content.isdigit():
            mem_gb = int(mem_gb_content)
            print_yellow(f"Using old run's --mem_gb: {mem_gb}")
            return mem_gb

        print_yellow(f"mem_gb-setting: The contents of {mem_gb_file} do not contain a single number")
        return None

    _fatal_error("--mem_gb needs to be set", 19)

    return None

def load_gpus_or_exit(_args: Any) -> Optional[int]:
    if _args.continue_previous_job and not _args.gpus:
        gpus_file = f"{_args.continue_previous_job}/state_files/gpus"
        gpus_content = get_file_content_or_exit(gpus_file, f"neither --gpus nor file {gpus_file} found", 19)
        if gpus_content.isdigit():
            gpus = int(gpus_content)
            print_yellow(f"Using old run's --gpus: {gpus}")
            return gpus

        print_yellow(f"--gpus: The contents of {gpus_file} do not contain a single number")
    return _args.gpus

def load_max_eval_or_exit(_args: Any) -> None:
    if _args.max_eval:
        set_max_eval(_args.max_eval)
        if _args.max_eval <= 0:
            _fatal_error("--max_eval must be larger than 0", 19)
    elif _args.continue_previous_job:
        max_eval_file = f"{_args.continue_previous_job}/state_files/max_eval"
        max_eval_content = get_file_content_or_exit(max_eval_file, f"neither --max_eval nor file {max_eval_file} found", 19)
        if max_eval_content.isdigit():
            set_max_eval(int(max_eval_content))
            print_yellow(f"Using old run's --max_eval: {max_eval_content}")
        else:
            print_yellow(f"max_eval-setting: The contents of {max_eval_file} do not contain a single number")
    else:
        if len(args.calculate_pareto_front_of_job) == 0:
            print_yellow("--max_eval needs to be set")

try:
    if not args.tests:
        global_vars = check_continue_previous_job(args.continue_previous_job)
        check_required_parameters(args)
        load_time_or_exit(args)

        loaded_mem_gb = load_mem_gb_or_exit(args)

        if loaded_mem_gb:
            args.mem_gb = loaded_mem_gb
            global_vars["mem_gb"] = args.mem_gb

        loaded_gpus = load_gpus_or_exit(args)

        if loaded_gpus:
            args.gpus = loaded_gpus
            global_vars["gpus"] = args.gpus

        load_max_eval_or_exit(args)
except KeyboardInterrupt:
    print("\n⚠ You pressed CTRL+C. Program execution halted while loading modules.")
    my_exit(0)

def print_debug_get_next_trials(got: int, requested: int, _line: int) -> None:
    time_str: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg: str = f"{time_str}, {got}, {requested}"

    log_message_to_file(LOGFILE_DEBUG_GET_NEXT_TRIALS, msg, 0, "")

def print_debug_progressbar(msg: str) -> None:
    time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"{time_str} ({worker_generator_uuid}): {msg}"

    _debug_progressbar(msg)

def get_process_info(pid: Any) -> str:
    try:
        proc: Optional[psutil.Process] = psutil.Process(pid)
        hierarchy: list[str] = []

        while proc is not None:
            try:
                pid_str = f"[PID {proc.pid}] {proc.name()} - {proc.cmdline()}"
                hierarchy.append(pid_str)
                proc = proc.parent()
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                hierarchy.append(f"[PID ?] <Process info unavailable: {e}>")
                break

        return "\n  → ".join(hierarchy[::-1])

    except psutil.NoSuchProcess:
        return f"Process with PID {pid} no longer exists."

    except psutil.AccessDenied:
        return f"Access denied when trying to retrieve information for PID {pid}."

    except Exception as e:
        return f"An unexpected error occurred while retrieving process info: {str(e)}"

def receive_usr_signal(signum: int, stack: Any) -> None:
    """Handle SIGUSR1 signal."""
    siginfo = signal.sigwaitinfo({signum})
    fool_linter(stack)
    print_yellow(f"\nReceived SIGUSR1 ({signum}) from PID {siginfo.si_pid}, sent by UID {siginfo.si_uid}\n")

    process_info = get_process_info(siginfo.si_pid)
    print_yellow(f"Process info for PID {siginfo.si_pid}:\n  → {process_info}\n")

    raise SignalUSR(f"USR1-signal received ({signum})")

def receive_usr_signal_term(signum: int, stack: Any) -> None:
    """Handle SIGTERM signal (termination)."""
    siginfo = signal.sigwaitinfo({signum})
    fool_linter(stack)
    print_yellow(f"\nReceived SIGTERM ({signum}) from PID {siginfo.si_pid}, sent by UID {siginfo.si_uid}\n")

    process_info = get_process_info(siginfo.si_pid)
    print_yellow(f"Process info for PID {siginfo.si_pid}:\n  → {process_info}\n")

    raise SignalTERM(f"TERM-signal received ({signum})")

def receive_signal_cont(signum: int, stack: Any) -> None:
    """Handle SIGCONT signal (continue)."""
    siginfo = signal.sigwaitinfo({signum})
    fool_linter(stack)
    print_yellow(f"\nReceived SIGCONT ({signum}) from PID {siginfo.si_pid}, sent by UID {siginfo.si_uid}\n")

    process_info = get_process_info(siginfo.si_pid)
    print_yellow(f"Process info for PID {siginfo.si_pid}:\n  → {process_info}\n")

    raise SignalCONT(f"CONT-signal received ({signum})")

signal.signal(signal.SIGUSR1, receive_usr_signal)
signal.signal(signal.SIGUSR2, receive_usr_signal)
signal.signal(signal.SIGTERM, receive_usr_signal_term)
signal.signal(signal.SIGCONT, receive_signal_cont)

print_debug(f"Current PID: {os.getpid()}")

def is_executable_in_path(executable_name: str) -> bool:
    for path in os.environ.get('PATH', '').split(':'):
        executable_path = os.path.join(path, executable_name)
        if os.path.exists(executable_path) and os.access(executable_path, os.X_OK):
            return True
    return False

SYSTEM_HAS_SBATCH: bool = False
IS_NVIDIA_SMI_SYSTEM: bool = False

if is_executable_in_path("sbatch"):
    SYSTEM_HAS_SBATCH = True
if is_executable_in_path("nvidia-smi"):
    IS_NVIDIA_SMI_SYSTEM = True

if not SYSTEM_HAS_SBATCH:
    num_parallel_jobs = 1

if SYSTEM_HAS_SBATCH and not args.force_local_execution and args.raw_samples < args.num_parallel_jobs:
    _fatal_error(f"Has --raw_samples={args.raw_samples}, but --num_parallel_jobs={args.num_parallel_jobs}. Cannot continue, since --raw_samples must be larger or equal to --num_parallel_jobs.", 48)

def save_global_vars() -> None:
    with open(get_state_file_name('global_vars.json'), mode="w", encoding="utf-8") as f:
        json.dump(global_vars, f)

def check_slurm_job_id() -> None:
    if SYSTEM_HAS_SBATCH:
        slurm_job_id = os.environ.get('SLURM_JOB_ID')
        if slurm_job_id is not None and not slurm_job_id.isdigit():
            print_red("Not a valid SLURM_JOB_ID.")
        elif slurm_job_id is None and len(args.calculate_pareto_front_of_job) == 0:
            print_red(
                "You are on a system that has SLURM available, but you are not running the main-script in a SLURM-Environment. "
                "This may cause the system to slow down for all other users. It is recommended you run the main script in a SLURM-job."
            )

def create_folder_and_file(folder: str) -> str:
    with console.status(f"[bold green]Creating folder {folder}..."):
        print_debug(f"create_folder_and_file({folder})")

        makedirs(folder)

        file_path = os.path.join(folder, RESULTS_CSV_FILENAME)

        return file_path

def get_program_code_from_out_file(f: str) -> str:
    if not os.path.exists(f):
        if f.endswith(".err"):
            alt = f[:-4] + ".out"
        elif f.endswith(".out"):
            alt = f[:-4] + ".err"
        else:
            alt = ""

        if alt and os.path.exists(alt):
            f = alt
        else:
            print_red(f"\nget_program_code_from_out_file: {f} not found")
            return ""

    fs = get_file_as_string(f)

    for line in fs.split("\n"):
        if "Program-Code:" in line:
            return line

    return ""

def get_min_or_max_column_value(pd_csv: str, column: str, _default: Union[None, int, float], _type: str = "min") -> Optional[Union[np.int64, float]]:
    if not os.path.exists(pd_csv):
        raise FileNotFoundError(f"CSV file {pd_csv} not found")

    try:
        _value = _default

        df = pd.read_csv(pd_csv, float_precision='round_trip')

        if column not in df.columns:
            print_red(f"Cannot load data from {pd_csv}: column {column} does not exist. Returning default {_default}")
            return _value

        if _type == "min":
            _value = df[column].min()
        elif _type == "max":
            _value = df[column].max()
        else:
            dier(f"get_min_or_max_column_value: Unknown type {_type}")

        return _value
    except Exception as e:
        print_red(f"Error while getting {_type} value from column {column}: {str(e)}")
        raise

    return None

def _get_column_value(pd_csv: str, column: str, default: Union[None, float, int], mode: str) -> Tuple[Optional[Union[int, float]], bool]:
    found_in_file = False
    column_value = get_min_or_max_column_value(pd_csv, column, default, mode)

    if column_value is not None:
        found_in_file = True
        if isinstance(column_value, (int, float)) and isinstance(default, (int, float)):
            if (mode == "min" and default > column_value) or (mode == "max" and default < column_value):
                return column_value, found_in_file
    return default, found_in_file

def get_ret_value_from_pd_csv(pd_csv: str, _type: str, _column: str, _default: Union[None, float, int]) -> Tuple[Optional[Union[int, float]], bool]:
    if not helpers.file_exists(pd_csv):
        print_red(f"'{pd_csv}' was not found")
        return _default, False

    mode: str = "min" if _type == "lower" else "max"
    return _get_column_value(pd_csv, _column, _default, mode)

def get_bound_if_prev_data(_type: str, name: str, _default: Union[None, float, int]) -> Union[Tuple[Union[float, int], bool], Any]:
    ret_val = _default

    found_in_file = False

    if args.continue_previous_job:
        pd_csv = f"{args.continue_previous_job}/{RESULTS_CSV_FILENAME}"

        ret_val, found_in_file = get_ret_value_from_pd_csv(pd_csv, _type, name, _default)

    if isinstance(ret_val, (int, float)):
        return ret_val, found_in_file

    return ret_val, False

def switch_lower_and_upper_if_needed(name: Union[list, str], lower_bound: Union[float, int], upper_bound: Union[float, int]) -> Tuple[Union[int, float], Union[int, float]]:
    if lower_bound > upper_bound:
        print_yellow(f"Lower bound ({lower_bound}) was larger than upper bound ({upper_bound}) for parameter '{name}'. Switched them.")
        upper_bound, lower_bound = lower_bound, upper_bound

    return lower_bound, upper_bound

def round_lower_and_upper_if_type_is_int(value_type: str, lower_bound: Union[int, float], upper_bound: Union[int, float]) -> Tuple[Union[int, float], Union[int, float]]:
    if value_type == "int":
        if not helpers.looks_like_int(lower_bound):
            if not args.tests:
                print_yellow(f"{value_type} can only contain integers. You chose {lower_bound}. Will be rounded down to {math.floor(lower_bound)}.")
            lower_bound = math.floor(lower_bound)

        if not helpers.looks_like_int(upper_bound):
            if not args.tests:
                print_yellow(f"{value_type} can only contain integers. You chose {upper_bound}. Will be rounded up to {math.ceil(upper_bound)}.")
            upper_bound = math.ceil(upper_bound)

    return lower_bound, upper_bound

def get_bounds(this_args: Union[str, list], j: int) -> Tuple[float, float]:
    try:
        lower_bound = float(this_args[j + 2])
    except Exception:
        _fatal_error(f"\n{this_args[j + 2]} is not a number", 181)

    try:
        upper_bound = float(this_args[j + 3])
    except Exception:
        _fatal_error(f"\n{this_args[j + 3]} is not a number", 181)

    return lower_bound, upper_bound

def adjust_bounds_for_value_type(value_type: str, lower_bound: Union[int, float], upper_bound: Union[int, float]) -> Union[Tuple[float, float], Tuple[int, int]]:
    lower_bound, upper_bound = round_lower_and_upper_if_type_is_int(value_type, lower_bound, upper_bound)

    if value_type == "int":
        lower_bound = math.floor(lower_bound)
        upper_bound = math.ceil(upper_bound)

    return lower_bound, upper_bound

def generate_values(name: str, value_type: str, lower_bound: Union[int, float], upper_bound: Union[int, float]) -> list:
    if value_type == "int":
        return [str(i) for i in range(int(lower_bound), int(upper_bound) + 1)] if int(upper_bound) - int(lower_bound) + 1 <= 999 else list(dict.fromkeys([str(round(int(lower_bound) + i * (int(upper_bound) - int(lower_bound)) / 998)) for i in range(999)]))

    if value_type == "float":
        num_steps = 999
        step = (upper_bound - lower_bound) / num_steps

        print_debug(f"{name}: step_size for converting to float: {step}, num_steps: {num_steps}")
        return [str(lower_bound + i * step) for i in range(num_steps + 1)]

    raise ValueError("Unsupported value_type")

def create_range_param(name: str, lower_bound: Union[float, int], upper_bound: Union[float, int], value_type: str, log_scale: bool, force_classic: bool = False) -> dict:
    if args.force_choice_for_ranges and not force_classic:
        return {
            'is_ordered': False,
            'name': name,
            'type': 'choice',
            'value_type': 'str',
            'values': generate_values(name, value_type, lower_bound, upper_bound)
        }
    return {
        "name": name,
        "type": "range",
        "bounds": [lower_bound, upper_bound],
        "value_type": value_type,
        "log_scale": log_scale
    }

def handle_grid_search(name: Union[list, str], lower_bound: Union[float, int], upper_bound: Union[float, int], value_type: str) -> dict:
    if lower_bound is None or upper_bound is None:
        _fatal_error("handle_grid_search: lower_bound or upper_bound is None", 91)

        return {}

    values: List[float] = cast(List[float], np.linspace(lower_bound, upper_bound, args.max_eval, endpoint=True).tolist())

    if value_type == "int":
        values = [int(value) for value in values]

    values = sorted(set(values))
    values_str: List[str] = [str(helpers.to_int_when_possible(value)) for value in values]

    return {
        "name": name,
        "type": "choice",
        "is_ordered": True,
        "value_type": "str",
        "values": values_str
    }

def get_bounds_from_previous_data(name: str, lower_bound: Union[float, int], upper_bound: Union[float, int]) -> Tuple[Union[float, int], Union[float, int]]:
    lower_bound, _ = get_bound_if_prev_data("lower", name, lower_bound)
    upper_bound, _ = get_bound_if_prev_data("upper", name, upper_bound)
    return lower_bound, upper_bound

def check_bounds_change_due_to_previous_job(name: Union[list, str], lower_bound: Union[float, int], upper_bound: Union[float, int], search_space_reduction_warning: bool) -> bool:
    old_lower_bound = lower_bound
    old_upper_bound = upper_bound

    if args.continue_previous_job:
        if old_lower_bound != lower_bound:
            print_yellow(f"previous jobs contained smaller values for {name}. Lower bound adjusted from {old_lower_bound} to {lower_bound}")
            search_space_reduction_warning = True

        if old_upper_bound != upper_bound:
            print_yellow(f"previous jobs contained larger values for {name}. Upper bound adjusted from {old_upper_bound} to {upper_bound}")
            search_space_reduction_warning = True

    return search_space_reduction_warning

def get_value_type_and_log_scale(this_args: Union[str, list], j: int) -> Tuple[int, str, bool]:
    skip = 5
    try:
        value_type = this_args[j + 4]
    except Exception:
        value_type = "float"
        skip = 4

    try:
        log_scale = this_args[j + 5].lower() == "true"
    except Exception:
        log_scale = False
        skip = 5

    return skip, value_type, log_scale

def check_for_too_high_differences(lower_bound: Union[int, float], upper_bound: Union[int, float]) -> None:
    bound_diff = abs(lower_bound - upper_bound)

    if bound_diff > args.range_max_difference:
        print_red(f"The difference between {lower_bound} and {upper_bound} was too high, these large numbers can cause memory leaks. Difference was: {bound_diff}, max difference is {args.range_max_difference}")

        sys.exit(235)

def parse_range_param(classic_params: list, params: list, j: int, this_args: Union[str, list], name: str, search_space_reduction_warning: bool) -> Tuple[int, list, list, bool]:
    check_factorial_range()
    check_range_params_length(this_args)

    lower_bound: Union[float, int]
    upper_bound: Union[float, int]

    lower_bound, upper_bound = get_bounds(this_args, j)

    die_if_lower_and_upper_bound_equal_zero(lower_bound, upper_bound)

    lower_bound, upper_bound = switch_lower_and_upper_if_needed(name, lower_bound, upper_bound)

    skip, value_type, log_scale = get_value_type_and_log_scale(this_args, j)

    validate_value_type(value_type)

    lower_bound, upper_bound = adjust_bounds_for_value_type(value_type, lower_bound, upper_bound)

    lower_bound, upper_bound = get_bounds_from_previous_data(name, lower_bound, upper_bound)

    check_for_too_high_differences(lower_bound, upper_bound)

    if lower_bound == upper_bound:
        print_red(f"Lower bound {lower_bound} was equal to upper bound {upper_bound}. Please fix this. Cannot continue.")
        my_exit(181)

    search_space_reduction_warning = check_bounds_change_due_to_previous_job(name, lower_bound, upper_bound, search_space_reduction_warning)

    param = create_range_param(name, lower_bound, upper_bound, value_type, log_scale)
    classic_param = create_range_param(name, lower_bound, upper_bound, value_type, log_scale, True)

    if args.gridsearch:
        param = handle_grid_search(name, lower_bound, upper_bound, value_type)

    global_vars["parameter_names"].append(name)
    params.append(param)
    classic_params.append(classic_param)

    j += skip
    return j, params, classic_params, search_space_reduction_warning

def validate_value_type(value_type: str) -> None:
    valid_value_types = ["int", "float"]
    check_if_range_types_are_invalid(value_type, valid_value_types)

def parse_fixed_param(classic_params: list, params: list, j: int, this_args: Union[str, list], name: Union[list, str], search_space_reduction_warning: bool) -> Tuple[int, list, list, bool]:
    if len(this_args) != 3:
        _fatal_error("⚠ --parameter for type fixed must have 3 parameters: <NAME> fixed <VALUE>", 181)

    value = this_args[j + 2]

    value = value.replace('\r', ' ').replace('\n', ' ')

    param = {
        "name": name,
        "type": "fixed",
        "value": value
    }

    global_vars["parameter_names"].append(name)

    params.append(param)

    j += 3

    return j, params, classic_params, search_space_reduction_warning

def parse_choice_param(classic_params: list, params: list, j: int, this_args: Union[str, list], name: Union[list, str], search_space_reduction_warning: bool) -> Tuple[int, list, list, bool]:
    if len(this_args) != 3:
        _fatal_error("⚠ --parameter for type choice must have 3 parameters: <NAME> choice <VALUE,VALUE,VALUE,...>", 181)

    values = re.split(r'\s*,\s*', str(this_args[j + 2]))

    values[:] = [x for x in values if x != ""]

    param = {
        "name": name,
        "type": "choice",
        "is_ordered": False,
        "value_type": "str",
        "values": values
    }

    global_vars["parameter_names"].append(name)

    params.append(param)

    j += 3

    return j, params, classic_params, search_space_reduction_warning

def _parse_experiment_parameters_validate_name(name: str, invalid_names: List[str], param_names: List[str]) -> None:
    if name in invalid_names:
        _fatal_error(f"\n⚠ Name for argument is invalid: {name}. Invalid names are: {', '.join(invalid_names)}", 181)
    if name in param_names:
        _fatal_error(f"\n⚠ Parameter name '{name}' is not unique. Names for parameters must be unique!", 181)

def _parse_experiment_parameters_get_param_type(this_args: List[Any], j: int) -> str:
    try:
        return this_args[j + 1]
    except Exception:
        _fatal_error("Not enough arguments for --parameter", 181)

    return ""

def _parse_experiment_parameters_parse_this_args(
    this_args: List[Any],
    invalid_names: List[str],
    param_names: List[str],
    classic_params: List[Dict[str, Any]],
    params: List[Dict[str, Any]],
    search_space_reduction_warning: bool
) -> Tuple[int, List[Dict[str, Any]], List[Dict[str, Any]], bool]:
    j = 0
    param_parsers = {
        "range": parse_range_param,
        "fixed": parse_fixed_param,
        "choice": parse_choice_param
    }
    valid_types = list(param_parsers.keys())

    while j < len(this_args) - 1:
        name = this_args[j]
        _parse_experiment_parameters_validate_name(name, invalid_names, param_names)

        param_names.append(name)
        global_param_names.append(name)

        param_type = _parse_experiment_parameters_get_param_type(this_args, j)

        if param_type not in param_parsers:
            _fatal_error(f"⚠ Parameter type '{param_type}' not yet implemented.", 181)

        if param_type not in valid_types:
            valid_types_string = ', '.join(valid_types)
            _fatal_error(f"\n⚠ Invalid type {param_type}, valid types are: {valid_types_string}", 181)

        j, params, classic_params, search_space_reduction_warning = param_parsers[param_type](classic_params, params, j, this_args, name, search_space_reduction_warning)

    return j, params, classic_params, search_space_reduction_warning

def parse_experiment_parameters() -> None:
    global experiment_parameters

    params: List[Dict[str, Any]] = []
    classic_params: List[Dict[str, Any]] = []
    param_names: List[str] = []

    search_space_reduction_warning = False

    invalid_names = ["start_time", "end_time", "run_time", "program_string", *arg_result_names, "exit_code", "signal", *special_col_names]

    i = 0
    while args.parameter and i < len(args.parameter):
        this_args = args.parameter[i]
        if this_args is not None and isinstance(this_args, dict) and "param" in this_args:
            this_args = this_args["param"]

        _, params, classic_params, search_space_reduction_warning = _parse_experiment_parameters_parse_this_args(this_args, invalid_names, param_names, classic_params, params, search_space_reduction_warning)

        i += 1

    if search_space_reduction_warning:
        print_red("⚠ Search space reduction is not currently supported on continued runs or runs that have previous data.")

    # Remove duplicates by 'name' key preserving order
    params = list({p['name']: p for p in params}.values())

    experiment_parameters = params

def check_factorial_range() -> None:
    if args.model and args.model == "FACTORIAL":
        _fatal_error("\n⚠ --model FACTORIAL cannot be used with range parameter", 181)

def check_if_range_types_are_invalid(value_type: str, valid_value_types: list) -> None:
    if value_type not in valid_value_types:
        valid_value_types_string = ", ".join(valid_value_types)
        _fatal_error(f"⚠ {value_type} is not a valid value type. Valid types for range are: {valid_value_types_string}", 181)

def check_range_params_length(this_args: Union[str, list]) -> None:
    if len(this_args) != 5 and len(this_args) != 4 and len(this_args) != 6:
        _fatal_error("\n⚠ --parameter for type range must have 4 (or 5, the last one being optional and float by default, or 6, while the last one is true or false) parameters: <NAME> range <START> <END> (<TYPE (int or float)>, <log_scale: bool>)", 181)

def die_if_lower_and_upper_bound_equal_zero(lower_bound: Union[int, float], upper_bound: Union[int, float]) -> None:
    if upper_bound is None or lower_bound is None:
        _fatal_error("die_if_lower_and_upper_bound_equal_zero: upper_bound or lower_bound is None. Cannot continue.", 91)
    if upper_bound == lower_bound:
        if lower_bound == 0:
            _fatal_error(f"⚠ Lower bound and upper bound are equal: {lower_bound}, cannot automatically fix this, because they -0 = +0 (usually a quickfix would be to set lower_bound = -upper_bound)", 181)
        print_red(f"⚠ Lower bound and upper bound are equal: {lower_bound}, setting lower_bound = -upper_bound")
        if upper_bound is not None:
            lower_bound = -upper_bound

def format_value(value: Any, float_format: str = '.80f') -> str:
    try:
        if isinstance(value, float):
            s = format(value, float_format)
            s = s.rstrip('0').rstrip('.') if '.' in s else s
            return s
        return str(value)
    except Exception as e:
        print_red(f"⚠ Error formatting the number {value}: {e}")
        return str(value)

def replace_parameters_in_string(
    parameters: dict,
    input_string: str,
    float_format: str = '.20f',
    additional_prefixes: list[str] = [],
    additional_patterns: list[str] = [],
) -> str:
    try:
        prefixes = ['$', '%'] + additional_prefixes
        patterns = ['{key}', '({key})'] + additional_patterns

        for key, value in parameters.items():
            replacement = format_value(value, float_format=float_format)
            for prefix in prefixes:
                for pattern in patterns:
                    token = prefix + pattern.format(key=key)
                    input_string = input_string.replace(token, replacement)

        input_string = input_string.replace('\r', ' ').replace('\n', ' ')
        return input_string

    except Exception as e:
        print_red(f"\n⚠ Error: {e}")
        return ""

def get_memory_usage() -> float:
    user_uid = os.getuid()

    memory_usage = float(sum(
        p.memory_info().rss for p in psutil.process_iter(attrs=['memory_info', 'uids'])
        if p.info['uids'].real == user_uid
    ) / (1024 * 1024))

    return memory_usage

class MonitorProcess:
    def __init__(self: Any, pid: int, interval: float = 1.0) -> None:
        self.pid = pid
        self.interval = interval
        self.running = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.daemon = True

        fool_linter(f"self.thread.daemon was set to {self.thread.daemon}")

    def _monitor(self: Any) -> None:
        try:
            _internal_process = psutil.Process(self.pid)
            while self.running and _internal_process.is_running():
                crf = get_current_run_folder()

                if crf and crf != "":
                    log_file_path = os.path.join(crf, "eval_nodes_cpu_ram_logs.txt")

                    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

                    with open(log_file_path, mode="a", encoding="utf-8") as log_file:
                        hostname = socket.gethostname()

                        slurm_job_id = os.getenv("SLURM_JOB_ID")

                        if slurm_job_id:
                            hostname += f"-SLURM-ID-{slurm_job_id}"

                        total_memory = psutil.virtual_memory().total / (1024 * 1024)
                        cpu_usage = psutil.cpu_percent(interval=5)

                        memory_usage = get_memory_usage()

                        unix_timestamp = int(time.time())

                        log_file.write(f"\nUnix-Timestamp: {unix_timestamp}, Hostname: {hostname}, CPU: {cpu_usage:.2f}%, RAM: {memory_usage:.2f} MB / {total_memory:.2f} MB\n")
                time.sleep(self.interval)
        except psutil.NoSuchProcess:
            pass

    def __enter__(self: Any) -> None:
        self.thread.start()
        return self

    def __exit__(self: Any, exc_type: Any, exc_value: Any, _traceback: Any) -> None:
        self.running = False
        self.thread.join()

def execute_bash_code_log_time(code: str) -> list:
    process_item = subprocess.Popen(code, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    with MonitorProcess(process_item.pid):
        try:
            stdout, stderr = process_item.communicate()
            result = subprocess.CompletedProcess(
                args=code, returncode=process_item.returncode, stdout=stdout, stderr=stderr
            )
            return [result.stdout, result.stderr, result.returncode, None]
        except subprocess.CalledProcessError as e:
            real_exit_code = e.returncode
            signal_code = None
            if real_exit_code < 0:
                signal_code = abs(e.returncode)
                real_exit_code = 1
            return [e.stdout, e.stderr, real_exit_code, signal_code]

def execute_bash_code(code: str) -> list:
    try:
        result = subprocess.run(
            code,
            shell=True,
            check=True,
            text=True,
            capture_output=True
        )

        if result.returncode != 0:
            print(f"Exit-Code: {result.returncode}")

        real_exit_code = result.returncode

        signal_code = None
        if real_exit_code < 0:
            signal_code = abs(result.returncode)
            real_exit_code = 1

        return [result.stdout, result.stderr, real_exit_code, signal_code]

    except subprocess.CalledProcessError as e:
        real_exit_code = e.returncode

        signal_code = None
        if real_exit_code < 0:
            signal_code = abs(e.returncode)
            real_exit_code = 1

        if not args.tests:
            print(f"Error at execution of your program: {code}. Exit-Code: {real_exit_code}, Signal-Code: {signal_code}")
            if len(e.stdout):
                print(f"stdout: {e.stdout}")
            else:
                print("No stdout")

            if len(e.stderr):
                print(f"stderr: {e.stderr}")
            else:
                print("No stderr")

        return [e.stdout, e.stderr, real_exit_code, signal_code]

def get_results(input_string: Optional[Union[int, str]]) -> Optional[Union[Dict[str, Optional[float]], List[float]]]:
    if input_string is None:
        if not args.tests:
            print_red("get_results: Input-String is None")
        return None

    if not isinstance(input_string, str):
        if not args.tests:
            print_red(f"get_results: Type of input_string is not string, but {type(input_string)}")
        return None

    try:
        results: Dict[str, Optional[float]] = {}

        for column_name in arg_result_names:
            _pattern = rf'\s*{re.escape(column_name)}\d*:\s*(-?\d+(?:\.\d+)?)'

            matches = re.findall(_pattern, input_string)

            if matches:
                results[column_name] = [float(match) for match in matches][0]
            else:
                results[column_name] = None
                insensitive_matches = re.findall(_pattern, input_string, re.IGNORECASE)

                if insensitive_matches:
                    add_to_global_error_list(f"'{column_name}: <FLOAT>' not found (found ignoring case)")
                else:
                    add_to_global_error_list(f"'{column_name}: <FLOAT>' not found")

        if len(results):
            return results
    except Exception as e:
        print_red(f"Error extracting the RESULT-string: {e}")

    return None

def add_to_csv(file_path: str, new_heading: list, new_data: list) -> None:
    new_data = [helpers.to_int_when_possible(x) for x in new_data]
    formatted_data = _add_to_csv_format_data(new_data)
    _add_to_csv_with_lock(file_path, new_heading, formatted_data)

def _add_to_csv_format_data(new_data: List[object]) -> List[object]:
    return [
        ("{:.20f}".format(x).rstrip('0').rstrip('.')) if isinstance(x, float) else x
        for x in new_data
    ]

def _add_to_csv_with_lock(file_path: str, new_heading: list, formatted_data: list) -> None:
    lockfile = file_path + ".lock"
    if not _add_to_csv_acquire_lock(lockfile, os.path.dirname(file_path)):
        return
    try:
        _add_to_csv_handle_file(file_path, new_heading, formatted_data)
    finally:
        _add_to_csv_release_lock(lockfile)

def _add_to_csv_acquire_lock(lockfile: str, dir_path: str) -> bool:
    wait_time = 0.01
    max_wait = 30.0  # seconds
    while max_wait > 0:
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, dir=dir_path)
            tmp.close()
            os.link(tmp.name, lockfile)
            os.unlink(tmp.name)
            return True
        except FileExistsError:
            time.sleep(wait_time)
            max_wait -= wait_time
        except Exception as e:
            print("Lock error:", e)
            return False
    return False

def _add_to_csv_release_lock(lockfile: str) -> None:
    try:
        os.unlink(lockfile)
    except FileNotFoundError:
        pass

def _add_to_csv_handle_file(file_path: str, new_heading: list, formatted_data: list) -> None:
    if not os.path.exists(file_path):
        _add_to_csv_create_new_file(file_path, new_heading, formatted_data)
        return

    with open(file_path, 'r', encoding="utf-8", newline='') as f:
        rows = list(csv.reader(f))

    existing_heading = rows[0] if rows else []
    all_headings = list(dict.fromkeys(existing_heading + new_heading))

    if all_headings != existing_heading:
        _add_to_csv_rewrite_file(file_path, rows, existing_heading, all_headings, new_heading, formatted_data)
    else:
        _add_to_csv_append_row(file_path, existing_heading, new_heading, formatted_data)

def _add_to_csv_create_new_file(file_path: str, heading: list, data: list) -> None:
    with open(file_path, 'w', encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(heading)
        writer.writerow(data)

def _add_to_csv_rewrite_file(file_path: str, rows: List[list], existing_heading: list, all_headings: list, new_heading: list, formatted_data: list) -> None:
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(file_path), suffix=".csv")
    with os.fdopen(tmp_fd, 'w', encoding="utf-8", newline='') as tmp_file:
        writer = csv.writer(tmp_file)
        writer.writerow(all_headings)
        for row in rows[1:]:
            tmp_file.writerow([
                row[existing_heading.index(h)] if h in existing_heading else ""
                for h in all_headings
            ])
        tmp_file.writerow([
            formatted_data[new_heading.index(h)] if h in new_heading else ""
            for h in all_headings
        ])
    shutil.move(tmp_path, file_path)

def _add_to_csv_append_row(file_path: str, existing_heading: list, new_heading: list, formatted_data: list) -> None:
    with open(file_path, 'a', encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            formatted_data[new_heading.index(h)] if h in new_heading else ""
            for h in existing_heading
        ])

def find_file_paths(_text: str) -> List[str]:
    file_paths = []

    if isinstance(_text, str):
        words = _text.split()

        for word in words:
            if os.path.exists(word):
                file_paths.append(word)

        return file_paths

    return []

def check_file_info(file_path: str) -> str:
    if not os.path.exists(file_path):
        if not args.tests:
            print(f"check_file_info: The file {file_path} does not exist.")
        return ""

    if not os.access(file_path, os.R_OK):
        if not args.tests:
            print(f"check_file_info: The file {file_path} is not readable.")
        return ""

    file_stat = os.stat(file_path)

    uid = file_stat.st_uid
    gid = file_stat.st_gid

    username = pwd.getpwuid(uid).pw_name

    size = file_stat.st_size
    permissions = stat.filemode(file_stat.st_mode)

    access_time = file_stat.st_atime
    modification_time = file_stat.st_mtime
    status_change_time = file_stat.st_ctime

    string = f"pwd: {os.getcwd()}\n"
    string += f"File: {file_path}\n"
    string += f"UID: {uid}\n"
    string += f"GID: {gid}\n"
    _SLURM_JOB_ID = os.getenv('SLURM_JOB_ID')
    if _SLURM_JOB_ID is not None and _SLURM_JOB_ID is not False and _SLURM_JOB_ID != "":
        string += f"SLURM_JOB_ID: {_SLURM_JOB_ID}\n"
    string += f"Status-Change-Time: {status_change_time}\n"
    string += f"Size: {size} Bytes\n"
    string += f"Permissions: {permissions}\n"
    string += f"Owner: {username}\n"
    string += f"Last access: {access_time}\n"
    string += f"Last modification: {modification_time}\n"
    string += f"Hostname: {socket.gethostname()}"

    return string

def find_file_paths_and_print_infos(_text: str, program_code: str) -> str:
    file_paths = find_file_paths(_text)

    if len(file_paths) == 0:
        return ""

    string = "\n========\nDEBUG INFOS START:\n"

    string += f"Program-Code: {program_code}"
    if file_paths:
        for file_path in file_paths:
            string += "\n"
            string += check_file_info(file_path)
    string += "\n========\nDEBUG INFOS END\n"

    return string

def write_failed_logs(data_dict: Optional[dict], error_description: str = "") -> None:
    headers = []
    data = []

    if data_dict is not None:
        headers = list(data_dict.keys())
        data = [list(data_dict.values())]
    else:
        print_debug("No data_dict provided, writing only error description.")
        data = [[]]

    if error_description:
        headers.append('error_description')
        for row in data:
            row.append(error_description)

    try:
        failed_logs_dir = os.path.join(get_current_run_folder(), 'failed_logs')
        makedirs(failed_logs_dir)

        header_file_path = os.path.join(failed_logs_dir, 'headers.csv')
        data_file_path = os.path.join(failed_logs_dir, 'parameters.csv')

        if not os.path.exists(header_file_path):
            try:
                with open(header_file_path, mode='w', encoding='utf-8', newline='') as header_file:
                    writer = csv.writer(header_file)
                    writer.writerow(headers)
                    print_debug(f"Header file created with headers: {headers}")
            except Exception as e:
                print_red(f"Failed to write header file: {e}")

        try:
            with open(data_file_path, mode='a', encoding='utf-8', newline='') as data_file:
                writer = csv.writer(data_file)
                writer.writerows(data)
                print_debug(f"Data appended to file: {data_file_path}")
        except Exception as e:
            print_red(f"Failed to append data to file: {e}")

    except Exception as e:
        print_red(f"Unexpected error: {e}")

def count_defective_nodes(file_path: Union[str, None] = None, entry: Any = None) -> list:
    if file_path is None:
        file_path = os.path.join(get_current_run_folder(), "state_files", "defective_nodes")

    makedirs(os.path.dirname(file_path))

    try:
        with open(file_path, mode='a+', encoding="utf-8") as file:
            file.seek(0)
            lines = file.readlines()

            entries = [line.strip() for line in lines]

            if entry is not None and entry not in entries:
                file.write(entry + '\n')
                entries.append(entry)

        return sorted(set(entries))

    except Exception as e:
        print(f"An error has occurred: {e}")
        return []

def test_gpu_before_evaluate(return_in_case_of_error: dict) -> Union[None, dict]:
    if SYSTEM_HAS_SBATCH and args.gpus >= 1 and args.auto_exclude_defective_hosts and not args.force_local_execution:
        try:
            for i in range(torch.cuda.device_count()):
                tmp = torch.cuda.get_device_properties(i).name

                fool_linter(tmp)
        except RuntimeError:
            print(f"Node {socket.gethostname()} was detected as faulty. It should have had a GPU, but there is an error initializing the CUDA driver. Adding this node to the --exclude list.")
            count_defective_nodes(None, socket.gethostname())
            return return_in_case_of_error
        except Exception:
            pass

    return None

def extract_info(data: Optional[str]) -> Tuple[List[str], List[str]]:
    if data is None:
        return [], []

    names: List[str] = []
    values: List[str] = []

    _pattern = re.compile(r'\s*OO-Info:\s*([a-zA-Z0-9_]+):\s*(.+)\s*$', re.IGNORECASE)

    for line in data.splitlines():
        match = _pattern.search(line)
        if match:
            names.append(f"OO_Info_{match.group(1)}")
            values.append(match.group(2))

    return names, values

def ignore_signals() -> None:
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    signal.signal(signal.SIGUSR2, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGQUIT, signal.SIG_IGN)

def calculate_signed_harmonic_distance(_args: Union[dict, List[Union[int, float]]]) -> Union[int, float]:
    if not _args or len(_args) == 0:
        return 0

    abs_inverse_sum: float = sum(1 / abs(a) for a in _args if a != 0)
    harmonic_mean: float = len(_args) / abs_inverse_sum if abs_inverse_sum != 0 else 0

    num_negatives: float = sum(1 for a in _args if a < 0)
    sign: int = -1 if num_negatives % 2 != 0 else 1

    return sign * harmonic_mean

def calculate_signed_euclidean_distance(_args: Union[dict, List[float]]) -> float:
    _sum = sum(a ** 2 for a in _args)
    sign = -1 if any(a < 0 for a in _args) else 1
    return sign * math.sqrt(_sum)

def calculate_signed_geometric_distance(_args: Union[dict, List[float]]) -> float:
    product: float = 1
    for a in _args:
        product *= abs(a)

    num_negatives: float = sum(1 for a in _args if a < 0)
    sign: int = -1 if num_negatives % 2 != 0 else 1

    geometric_mean: float = product ** (1 / len(_args)) if _args else 0
    return sign * geometric_mean

def calculate_signed_minkowski_distance(_args: Union[dict, List[float]], p: Union[int, float] = 2) -> float:
    if p <= 0:
        raise ValueError("p must be greater than 0.")

    sign: int = -1 if any(a < 0 for a in _args) else 1
    minkowski_sum: float = sum(abs(a) ** p for a in _args) ** (1 / p)
    return sign * minkowski_sum

def calculate_signed_weighted_euclidean_distance(_args: Union[dict, List[float]], weights_string: str) -> float:
    pattern = r'^\s*-?\d+(\.\d+)?\s*(,\s*-?\d+(\.\d+)?\s*)*$'

    if not re.fullmatch(pattern, weights_string):
        _fatal_error(f"String '{weights_string}' does not match pattern {pattern}", 32)

    weights = [float(w.strip()) for w in weights_string.split(",") if w.strip()]

    if len(weights) > len(_args):
        if not args.tests:
            print_yellow(f"calculate_signed_weighted_euclidean_distance: Warning: Trimming {len(weights) - len(_args)} extra weight(s): {weights[len(_args):]}")
        weights = weights[:len(_args)]

    if len(weights) < len(_args):
        if not args.tests:
            print_yellow("calculate_signed_weighted_euclidean_distance: Warning: Not enough weights, filling with 1s")
        weights.extend([1] * (len(_args) - len(weights)))

    if len(_args) != len(weights):
        raise ValueError("Length of _args and weights must match.")

    weighted_sum: float = sum(w * (a ** 2) for a, w in zip(_args, weights))
    sign: int = -1 if any(a < 0 for a in _args) else 1
    return sign * (weighted_sum ** 0.5)

class invalidOccType(Exception):
    pass

def calculate_occ(_args: Optional[Union[dict, List[Union[int, float]]]]) -> Union[int, float]:
    if _args is None or len(_args) == 0:
        return VAL_IF_NOTHING_FOUND

    if args.occ_type == "euclid":
        return calculate_signed_euclidean_distance(_args)
    if args.occ_type == "geometric":
        return calculate_signed_geometric_distance(_args)
    if args.occ_type == "signed_harmonic":
        return calculate_signed_harmonic_distance(_args)
    if args.occ_type == "minkowski":
        return calculate_signed_minkowski_distance(_args, args.minkowski_p)
    if args.occ_type == "weighted_euclidean":
        return calculate_signed_weighted_euclidean_distance(_args, args.signed_weighted_euclidean_weights)

    raise invalidOccType(f"Invalid OCC (optimization with combined criteria) type {args.occ_type}. Valid types are: {joined_valid_occ_types}")

def get_return_in_case_of_errors() -> dict:
    return_in_case_of_error = {}

    i = 0
    for _rn in arg_result_names:
        if arg_result_min_or_max[i] == "min":
            return_in_case_of_error[_rn] = VAL_IF_NOTHING_FOUND
        else:
            return_in_case_of_error[_rn] = -VAL_IF_NOTHING_FOUND

        i = i + 1

    return return_in_case_of_error

def write_job_infos_csv(parameters: dict, stdout: Optional[str], program_string_with_params: str, exit_code: Optional[int], _signal: Optional[int], result: Optional[Union[Dict[str, Optional[float]], List[float], int, float]], start_time: Union[int, float], end_time: Union[int, float], run_time: Union[float, int], trial_index: int, submit_time: Union[float, int], queue_time: Union[float, int]) -> None:
    str_parameters_values = _write_job_infos_csv_parameters_to_str(parameters)
    extra_vars_names, extra_vars_values = _write_job_infos_csv_extract_extra_vars(stdout)
    extra_vars_names, extra_vars_values = _write_job_infos_csv_add_slurm_job_id(extra_vars_names, extra_vars_values)

    parameters_keys = list(parameters.keys())

    headline = _write_job_infos_csv_build_headline(parameters_keys, extra_vars_names)
    result_values = _write_job_infos_csv_result_to_strlist(result)

    values = _write_job_infos_csv_build_values(start_time, end_time, run_time, program_string_with_params, str_parameters_values, result_values, exit_code, _signal, extra_vars_values)

    headline = _write_job_infos_csv_replace_none_with_str(headline)
    values = _write_job_infos_csv_replace_none_with_str(values)

    headline = ["trial_index", "submit_time", "queue_time", "worker_generator_uuid", *headline]
    values = [str(trial_index), str(submit_time), str(queue_time), worker_generator_uuid, *values]

    run_folder = get_current_run_folder()
    if run_folder is not None and os.path.exists(run_folder):
        try:
            add_to_csv(f"{run_folder}/job_infos.csv", headline, values)
        except Exception as e:
            print_red(f"Error writing job_infos.csv: {e}")
    else:
        print_debug(f"evaluate: get_current_run_folder() {run_folder} could not be found")

def _write_job_infos_csv_parameters_to_str(parameters: dict) -> List[str]:
    return [str(v) for v in list(parameters.values())]

def _write_job_infos_csv_extract_extra_vars(stdout: Optional[str]) -> Tuple[List[str], List[str]]:
    return extract_info(stdout)

def _write_job_infos_csv_add_slurm_job_id(extra_vars_names: List[str], extra_vars_values: List[str]) -> Tuple[List[str], List[str]]:
    _SLURM_JOB_ID = os.getenv('SLURM_JOB_ID')
    if _SLURM_JOB_ID:
        extra_vars_names.append("OO_Info_SLURM_JOB_ID")
        extra_vars_values.append(str(_SLURM_JOB_ID))
    return extra_vars_names, extra_vars_values

def _write_job_infos_csv_build_headline(parameters_keys: List[str], extra_vars_names: List[str]) -> List[str]:
    return [
        "start_time",
        "end_time",
        "run_time",
        "program_string",
        *parameters_keys,
        *arg_result_names,
        "exit_code",
        "signal",
        "hostname",
        *extra_vars_names
    ]

def _write_job_infos_csv_result_to_strlist(result: Optional[Union[Dict[str, Optional[float]], List[float], int, float]]) -> List[str]:
    result_values: List[str] = []

    if isinstance(result, list):
        for rkey in result:
            result_values.append(str(rkey))
    elif isinstance(result, dict):
        for _rkey, rval in result.items():  # type: str, Optional[float]
            result_values.append(str(rval))
    elif result is not None:  # int or float
        result_values.append(str(result))

    return result_values

def _write_job_infos_csv_build_values(start_time: Union[int, float], end_time: Union[int, float], run_time: Union[float, int], program_string_with_params: str, str_parameters_values: List[str], result_values: List[str], exit_code: Optional[int], _signal: Optional[int], extra_vars_values: List[str]) -> List[str]:
    return [
        str(int(start_time)),
        str(int(end_time)),
        str(int(run_time)),
        program_string_with_params,
        *str_parameters_values,
        *result_values,
        str(exit_code),
        str(_signal),
        socket.gethostname(),
        *extra_vars_values
    ]

def _write_job_infos_csv_replace_none_with_str(elements: Optional[List[str]]) -> List[str]:
    if elements is None:
        return []
    result = []
    for element in elements:
        if element is None:
            result.append('None')
        else:
            result.append(element)
    return result

def print_evaluate_times() -> None:
    file_path = get_current_run_folder("job_infos.csv")

    if not Path(file_path).exists():
        print_debug(f"The file '{file_path}' was not found.")
        return

    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)

        if csv_reader.fieldnames is None:
            print_debug("CSV-field-names are empty")
            return

        if 'run_time' not in csv_reader.fieldnames:
            print_debug("The 'run_time' column does not exist.")
            return

        time_values = []
        for row in csv_reader:
            try:
                time_values.append(float(row['run_time']))
            except ValueError:
                continue

        if not time_values:
            print_debug("No valid run times found.")
            return

        min_time = min(time_values)
        max_time = max(time_values)
        avg_time = statistics.mean(time_values)
        median_time = statistics.median(time_values)

        if min_time != max_time or max_time != 0:
            headers = ["Number of evaluations", "Min time", "Max time", "Average time", "Median time"]

            cols = [
                str(len(time_values)),
                f"{min_time:.2f} sec {human_time_when_larger_than_a_min(min_time)}",
                f"{max_time:.2f} sec {human_time_when_larger_than_a_min(max_time)}",
                f"{avg_time:.2f} sec {human_time_when_larger_than_a_min(avg_time)}",
                f"{median_time:.2f} sec {human_time_when_larger_than_a_min(median_time)}"
            ]

            table = Table(title="Runtime Infos")
            for h in headers:
                table.add_column(h, justify="center")

            table.add_row(*cols)

            console.print(table)

            overview_file = get_current_run_folder("time_overview.txt")
            with open(overview_file, mode='w', encoding='utf-8') as overview:
                overview.write(f"Number of evaluations: {len(time_values)} sec\n")
                overview.write(f"Min Time: {min_time:.2f} sec\n")
                overview.write(f"Max Time: {max_time:.2f} sec\n")
                overview.write(f"Average Time: {avg_time:.2f} sec\n")
                overview.write(f"Median Time: {median_time:.2f} sec\n")

def print_debug_infos(program_string_with_params: str) -> None:
    string = find_file_paths_and_print_infos(program_string_with_params, program_string_with_params)

    if not args.tests:
        original_print("Debug-Infos:", string)

def print_stdout_and_stderr(stdout: Optional[str], stderr: Optional[str]) -> None:
    if not args.tests:
        if stdout:
            original_print("stdout:\n", stdout)
        else:
            original_print("stdout was empty")

        if stderr:
            original_print("stderr:\n", stderr)
        else:
            original_print("stderr was empty")

def _evaluate_print_stuff(parameters: dict, program_string_with_params: str, stdout: Optional[str], stderr: Optional[str], exit_code: Optional[int], _signal: Optional[int], result: Optional[Union[Dict[str, Optional[float]], List[float], int, float]], start_time: Union[float, int], end_time: Union[float, int], run_time: Union[float, int], final_result: dict, trial_index: int, submit_time: Union[float, int], queue_time: Union[float, int]) -> None:
    if not args.tests:
        original_print(f"Parameters: {json.dumps(parameters)}")

    print_debug_infos(program_string_with_params)

    original_print(program_string_with_params)

    print_stdout_and_stderr(stdout, stderr)

    if not args.tests:
        original_print(f"Result: {result}")

        original_print(f"Final-results: {final_result}")

    write_job_infos_csv(parameters, stdout, program_string_with_params, exit_code, _signal, result, start_time, end_time, run_time, trial_index, submit_time, queue_time)

    if not args.tests:
        original_print(f"EXIT_CODE: {exit_code}")

def get_results_with_occ(stdout: str) -> Union[int, float, Optional[Union[Dict[str, Optional[float]], List[float]]]]:
    result = get_results(stdout)

    if result and args.occ:
        occed_result = calculate_occ(result)

        if occed_result is not None:
            result = [occed_result]

    return result

def get_signal_name(sig: BaseException, signal_messages: dict) -> str:
    try:
        signal_name_candidates = [k for k, v in signal_messages.items() if isinstance(sig, v)]
        if signal_name_candidates:
            signal_name = signal_name_candidates[0]
        else:
            signal_name = f"UNKNOWN_SIGNAL({type(sig).__name__})"
    except Exception as e:
        signal_name = f"ERROR_IDENTIFYING_SIGNAL({e})"
    return signal_name

def get_result_sem(stdout: Optional[str], name: str) -> Optional[float]:
    if stdout is None:
        print_red("get_result_sem: stdout is None")
        return None

    if not isinstance(stdout, str):
        print_red(f"get_result_sem: Type of stdout is not string, but {type(stdout)}")
        return None

    try:
        pattern = rf'^SEM-{re.escape(name)}:\s*(-?\d+(?:\.\d+)?)$'

        for line in stdout.splitlines():
            match = re.match(pattern, line.strip())
            if match:
                return float(match.group(1))

    except Exception as e:
        print_red(f"get_result_sem: Error while parsing: {e}")

    return None

def attach_sem_to_result(stdout: str, name: str, value: Union[int, float, None, list]) -> Optional[Union[tuple, int, float, list]]:
    sem = get_result_sem(stdout, name)
    if sem:
        return (value, sem)

    return value

def die_for_debug_reasons() -> None:
    max_done_str = os.getenv("DIE_AFTER_THIS_NR_OF_DONE_JOBS")
    if max_done_str is not None:
        try:
            max_done = int(max_done_str)
            if count_done_jobs() > max_done:
                my_exit(34)
        except ValueError:
            print_red(f"Invalid value for DIE_AFTER_THIS_NR_OF_DONE_JOBS: '{max_done_str}', cannot be converted to int")

def _evaluate_preprocess_parameters(parameters: dict) -> dict:
    return {
        k: (int(float(v)) if isinstance(v, (int, float, str)) and re.fullmatch(r'^\d+(\.0+)?$', str(v)) else v)
        for k, v in parameters.items()
    }

def _evaluate_create_signal_map() -> Dict[str, type[BaseException]]:
    return {
        "USR1-signal": SignalUSR,
        "CONT-signal": SignalCONT,
        "INT-signal": SignalINT
    }

def sanitize_for_evaluate_handle_result(val: Optional[Union[int, float, list, tuple]]) -> Optional[Union[float, Tuple]]:
    if val is None:
        return None
    if isinstance(val, int):
        return float(val)
    if isinstance(val, float):
        return val
    if isinstance(val, list):
        return tuple(val)
    if isinstance(val, tuple):
        return val
    raise TypeError(f"Unexpected result type: {type(val)}")

def _evaluate_handle_result(
    stdout: str,
    result: Optional[Union[int, float, dict, list]],
    parameters: Optional[dict]
) -> Dict[str, Optional[Union[float, Tuple]]]:
    final_result: Dict[str, Optional[Union[float, Tuple]]] = {}

    if isinstance(result, (int, float)):
        for name in arg_result_names:
            value = attach_sem_to_result(stdout, name, float(result))
            final_result[name] = sanitize_for_evaluate_handle_result(value)

    elif isinstance(result, list):
        float_values = [float(r) for r in result]
        for name in arg_result_names:
            value = attach_sem_to_result(stdout, name, float_values)
            final_result[name] = sanitize_for_evaluate_handle_result(value)

    elif isinstance(result, dict):
        for name in arg_result_names:
            value = attach_sem_to_result(stdout, name, result.get(name))
            final_result[name] = sanitize_for_evaluate_handle_result(value)

    else:
        write_failed_logs(parameters, "No Result")

    return final_result

def is_valid_result(result: Union[int, float, Optional[Union[Dict[str, Optional[float]], List[float]]]]) -> bool:
    return result is not None and (isinstance(result, (int, float)) or (isinstance(result, list) and all(isinstance(x, (int, float)) and x is not None for x in result)) or (isinstance(result, dict) and all(isinstance(v, (int, float)) and v is not None for v in result.values())))

def pretty_process_output(stdout_path: str, stderr_path: str, exit_code: Optional[int], result: Union[int, float, Optional[Union[Dict[str, Optional[float]], List[float]]]]) -> None:
    stdout_txt = get_file_as_string(stdout_path)
    stderr_txt = get_file_as_string(stderr_path)

    # -------- header -------- #
    is_valid = is_valid_result(result)

    outcome = "SUCCESS" if is_valid else "FAILURE"
    header_style = "bold white on green" if exit_code == 0 else "bold white on red"
    console.rule(Text(f" {outcome}  (exit {exit_code}) ", style=header_style))
    console.rule(Text(f" RESULT: {result} ", style=header_style))

    def is_nonempty(s: Optional[str]) -> bool:
        return bool(s and s.strip())

    if is_nonempty(stdout_txt) and stdout_txt is not None:
        console.rule(Text(f" {stdout_path} ", style=header_style))
        panel = Panel(stdout_txt, title="STDOUT", border_style="cyan", padding=(0, 1))
        console.print(panel)

    if is_nonempty(stderr_txt) and stderr_txt is not None:
        console.rule(Text(f" {stderr_path} ", style=header_style))
        panel = Panel(stderr_txt, title="STDERR", border_style="magenta", padding=(0, 1))
        console.print(panel)

    if not (is_nonempty(stdout_txt) or is_nonempty(stderr_txt)):
        print("\n")
        console.print("[dim]No output captured.[/dim]")

def evaluate(parameters_with_trial_index: dict) -> Optional[Union[int, float, Dict[str, Optional[Union[int, float, Tuple]]], List[float]]]:
    parameters = parameters_with_trial_index["params"]
    trial_index = parameters_with_trial_index["trial_idx"]
    submit_time = parameters_with_trial_index["submit_time"]

    queue_time = abs(int(time.time()) - int(submit_time))

    start_nvidia_smi_thread()
    return_in_case_of_error: dict = get_return_in_case_of_errors()

    _test_gpu = test_gpu_before_evaluate(return_in_case_of_error)
    final_result: Optional[Union[int, float, Dict[str, Optional[Union[int, float, Tuple]]], List[float]]] = return_in_case_of_error

    if _test_gpu is None:
        parameters = _evaluate_preprocess_parameters(parameters)
        ignore_signals()
        signal_messages = _evaluate_create_signal_map()

        try:
            if args.raise_in_eval:
                raise SignalUSR("Raised in eval")

            program_string_with_params: str = replace_parameters_in_string(
                parameters,
                global_vars["joined_run_program"]
            )

            start_time: int = int(time.time())

            stdout, stderr, exit_code, _signal = execute_bash_code_log_time(program_string_with_params)

            original_print(stderr)

            end_time: int = int(time.time())

            result = get_results_with_occ(stdout)

            final_result = _evaluate_handle_result(stdout, result, parameters)

            _evaluate_print_stuff(
                parameters,
                program_string_with_params,
                stdout,
                stderr,
                exit_code,
                _signal,
                result,
                start_time,
                end_time,
                end_time - start_time,
                final_result,
                trial_index,
                submit_time,
                queue_time
            )

        except tuple(signal_messages.values()) as sig:
            signal_name = get_signal_name(sig, signal_messages)
            print(f"\n⚠ {signal_name} was sent. Cancelling evaluation.")
            write_failed_logs(parameters, signal_name)

    return final_result

def custom_warning_handler(
    message: Union[Warning, str],
    category: Type[Warning],
    filename: str,
    lineno: int,
    file: Union[TextIO, None] = None,
    line: Union[str, None] = None
) -> None:
    warning_message = f"{category.__name__}: {message} (in {filename}, line {lineno})"
    print_debug(f"{file}:{line}: {warning_message}")

def disable_logging() -> None:
    if args.verbose:
        return

    with spinner("Disabling logging..."):
        logging.basicConfig(level=logging.CRITICAL)
        logging.getLogger().setLevel(logging.CRITICAL)
        logging.getLogger().disabled = True

        fool_linter(f"logging.getLogger().disabled set to {logging.getLogger().disabled}")

        categories = [
            Warning,
            UserWarning,
            DeprecationWarning,
            PendingDeprecationWarning,
            SyntaxWarning,
            RuntimeWarning,
            FutureWarning,
            ImportWarning,
            UnicodeWarning,
            BytesWarning
        ]

        modules = [
            "ax",

            "ax.core.data",
            "ax.core.parameter",
            "ax.core.experiment",

            "ax.service.ax_client",

            "ax.models.torch.botorch_modular.acquisition",

            "ax.adapter"
            "ax.adapter.base",
            "ax.adapter.standardize_y",
            "ax.adapter.transforms",
            "ax.adapter.transforms.standardize_y",
            "ax.adapter.transforms.int_to_float",
            "ax.adapter.cross_validation",
            "ax.adapter.dispatch_utils",
            "ax.adapter.torch",
            "ax.adapter.generation_node",
            "ax.adapter.best_model_selector",

            "ax.generation_strategy.generation_strategy",
            "ax.generation_strategy.generation_node",
            "ax.generation_strategy.external_generation_node",
            "ax.generation_strategy.transition_criterion",
            "ax.generation_strategy.model_spec",

            "ax.service",
            "ax.service.utils",
            "ax.service.utils.instantiation",
            "ax.service.utils.report_utils",
            "ax.service.utils.best_point",
            "ax.service.utils.with_db_settings_base",

            "ax.storage.sqa_store.save",

            "botorch.optim.fit",
            "botorch.models.utils.assorted",
            "botorch.optim.optimize",

            "linear_operator.utils.cholesky",

            "torch.autograd",
            "torch.autograd.__init__",
        ]

        for module in modules:
            logging.getLogger(module).setLevel(logging.CRITICAL)
            logging.getLogger(module).disabled = True
            fool_linter(f"logging.getLogger('{module}.disabled') set to {logging.getLogger(module).disabled}")

        for cat in categories:
            warnings.filterwarnings("ignore", category=cat)
            for module in modules:
                warnings.filterwarnings("ignore", category=cat, module=module)

        warnings.showwarning = custom_warning_handler

        fool_linter(f"warnings.showwarning set to {warnings.showwarning}")

def display_failed_jobs_table() -> None:
    failed_jobs_file = get_current_run_folder("failed_logs")
    header_file = os.path.join(failed_jobs_file, "headers.csv")
    parameters_file = os.path.join(failed_jobs_file, "parameters.csv")

    if not os.path.exists(failed_jobs_file):
        print_debug(f"Failed jobs {failed_jobs_file} file does not exist.")
        return

    if not os.path.isfile(header_file):
        print_debug(f"Failed jobs Header file ({header_file}) does not exist.")
        return

    if not os.path.isfile(parameters_file):
        print_debug(f"Failed jobs Parameters file ({parameters_file}) does not exist.")
        return

    try:
        with open(header_file, mode='r', encoding="utf-8") as file:
            reader = csv.reader(file)
            headers = next(reader)
            #print_debug(f"Headers: {headers}")

        with open(parameters_file, mode='r', encoding="utf-8") as file:
            reader = csv.reader(file)
            parameters = [row for row in reader]
            #print_debug(f"Parameters: {parameters}")

        table = Table(show_header=True, header_style="bold red", title="Failed Jobs parameters")

        for header in headers:
            table.add_column(header)

        added_rows = set()

        for parameter_set in parameters:
            row = [str(helpers.to_int_when_possible(value)) for value in parameter_set]
            row_tuple = tuple(row)
            if row_tuple not in added_rows:
                table.add_row(*row, style='red')
                added_rows.add(row_tuple)

        console.print(table)
    except Exception as e:
        print_red(f"Error: {str(e)}")

def plot_command(_command: str, tmp_file: str, _width: str = "1300") -> None:
    if not helpers.looks_like_int(_width):
        print_red(f"Error: {_width} does not look like an int")
        sys.exit(8)

    width = int(_width)

    _show_sixel_graphics = args.show_sixel_scatter or args.show_sixel_general or args.show_sixel_scatter
    if not _show_sixel_graphics:
        return

    print_debug(f"command: {_command}")

    my_env = os.environ.copy()
    my_env["DONT_INSTALL_MODULES"] = "1"
    my_env["DONT_SHOW_DONT_INSTALL_MESSAGE"] = "1"

    _process = subprocess.Popen(_command.split(), stdout=subprocess.PIPE, env=my_env)
    _, error = _process.communicate()

    if os.path.exists(tmp_file):
        print_image_to_cli(tmp_file, width)
    else:
        print_debug(f"{tmp_file} not found, error: {str(error)}")

def replace_string_with_params(input_string: str, params: list) -> str:
    try:
        replaced_string = input_string
        i = 0
        for param in params:
            #print(f"param: {param}, type: {type(param)}")
            replaced_string = replaced_string.replace(f"%{i}", str(param))
            i += 1
        return replaced_string
    except AssertionError as e:
        error_text = f"Error in replace_string_with_params: {e}"
        print(error_text)
        raise

    return ""

def get_best_line_and_best_result(nparray: np.ndarray, result_idx: int, maximize: bool) -> Tuple[Optional[Union[str, np.ndarray]], Optional[Union[str, np.ndarray, int, float]]]:
    best_line: Optional[str] = None
    best_result: Optional[str] = None

    for i in range(len(nparray)):
        this_line = nparray[i]
        this_line_result = this_line[result_idx]

        if isinstance(this_line_result, str) and re.match(r'^-?\d+(?:\.\d+)$', this_line_result) is not None:
            this_line_result = float(this_line_result)

        if type(this_line_result) in [float, int]:
            if best_result is None:
                if this_line is not None and len(this_line) > 0:
                    best_line = this_line
                    best_result = this_line_result

            if (maximize and this_line_result >= best_result) or (not maximize and this_line_result <= best_result):
                best_line = this_line
                best_result = this_line_result

    return best_line, best_result

def get_res_name_is_maximized(res_name: str) -> bool:
    idx = -1

    k = 0
    for rn in arg_result_names:
        if rn == res_name:
            idx = k

        k = k + 1

    if idx == -1:
        print_red(f"!!! get_res_name_is_maximized could not find '{res_name}' in the arg_result_names.")

    maximize = False

    try:
        if arg_result_min_or_max[idx] == "max":
            maximize = True
    except IndexError as e:
        print_debug(f"Error: Failed with {e}")

    return maximize

def get_best_params_from_csv(res_name: str = "RESULT") -> Optional[dict]:
    maximize = get_res_name_is_maximized(res_name)

    results: dict = {
        res_name: None,
        "parameters": {}
    }

    if not os.path.exists(RESULT_CSV_FILE):
        return results

    df = None

    try:
        df = pd.read_csv(RESULT_CSV_FILE, index_col=0, float_precision='round_trip')
        df.dropna(subset=arg_result_names, inplace=True)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError, KeyError):
        return results

    cols = df.columns.tolist()
    nparray = df.to_numpy()

    lower_cols = [c.lower() for c in cols]
    if res_name.lower() in lower_cols:
        result_idx = lower_cols.index(res_name.lower())
    else:
        return results

    best_line, _ = get_best_line_and_best_result(nparray, result_idx, maximize)

    if best_line is None:
        print_debug(f"Could not determine best {res_name}")
        return results

    for i in range(len(cols)):
        col = cols[i]
        if col not in IGNORABLE_COLUMNS:
            if col == res_name:
                results[res_name] = repr(best_line[i]) if type(best_line[i]) in [int, float] else best_line[i]
            else:
                results["parameters"][col] = repr(best_line[i]) if type(best_line[i]) in [int, float] else best_line[i]

    return results

def get_best_params(res_name: str = "RESULT") -> Optional[dict]:
    if os.path.exists(RESULT_CSV_FILE):
        return get_best_params_from_csv(res_name)

    return None

def _count_sobol_or_completed(this_csv_file_path: str, _type: str) -> int:
    if _type not in ["Sobol", "COMPLETED", "SOBOL"]:
        print_red(f"_type is not in Sobol, SOBOL or COMPLETED, but is '{_type}'")
        return 0

    count = 0

    if not os.path.exists(this_csv_file_path):
        print_debug(f"_count_sobol_or_completed: path '{this_csv_file_path}' not found")
        return count

    df = None

    _err = False

    try:
        df = pd.read_csv(this_csv_file_path, index_col=0, float_precision='round_trip')
        df.dropna(subset=arg_result_names, inplace=True)
    except KeyError:
        _err = True
    except pd.errors.EmptyDataError:
        _err = True
    except pd.errors.ParserError as e:
        print_red(f"Error reading CSV file 2: {str(e)}")
        _err = True
    except UnicodeDecodeError as e:
        print_red(f"Error reading CSV file 3: {str(e)}")
        _err = True
    except Exception as e:
        print_red(f"Error reading CSV file 4: {str(e)}")
        _err = True

    if _err:
        return 0

    assert df is not None, "DataFrame should not be None after reading CSV file"

    if _type.lower() == "sobol":
        rows = df[df["generation_node"] == _type]
    else:
        rows = df[df["trial_status"] == _type]

    count = len(rows)

    return count

def _count_sobol_steps(this_csv_file_path: str) -> int:
    return _count_sobol_or_completed(this_csv_file_path, "SOBOL")

def _count_done_jobs(this_csv_file_path: str) -> int:
    return _count_sobol_or_completed(this_csv_file_path, "COMPLETED")

def count_sobol_steps() -> int:
    if os.path.exists(RESULT_CSV_FILE):
        return _count_sobol_steps(RESULT_CSV_FILE)

    return 0

def get_random_steps_from_prev_job() -> int:
    if not args.continue_previous_job:
        return count_sobol_steps()

    prev_step_file: str = f"{args.continue_previous_job}/{RESULTS_CSV_FILENAME}"

    if not os.path.exists(prev_step_file):
        return _count_sobol_steps(prev_step_file)

    return add_to_phase_counter("random", count_sobol_steps() + _count_sobol_steps(f"{args.continue_previous_job}/{RESULTS_CSV_FILENAME}"), args.continue_previous_job)

def failed_jobs(nr: int = 0) -> int:
    return append_and_read(get_state_file_name('failed_jobs'), nr)

def count_done_jobs() -> int:
    if os.path.exists(RESULT_CSV_FILE):
        return _count_done_jobs(RESULT_CSV_FILE)

    return 0

def get_plot_types(x_y_combinations: list, _force: bool = False) -> list:
    plot_types: list = []

    if args.show_sixel_trial_index_result or _force:
        plot_types.append(
            {
                "type": "trial_index_result",
                "min_done_jobs": 2
            }
        )

    if args.show_sixel_scatter or _force:
        plot_types.append(
            {
                "type": "scatter",
                "params": "--bubblesize=50 --allow_axes %0 --allow_axes %1",
                "iterate_through": x_y_combinations,
                "dpi": 76,
                "filename": "plot_%0_%1_%2"
            }
        )

    if args.show_sixel_general or _force:
        plot_types.append(
            {
                "type": "general"
            }
        )

    return plot_types

def get_x_y_combinations_parameter_names() -> list:
    return list(combinations(global_vars["parameter_names"], 2))

def get_plot_filename(plot: dict, _tmp: str) -> str:
    j = 0
    _fn = plot.get("filename", plot["type"])
    tmp_file = f"{_tmp}/{_fn}.png"

    while os.path.exists(tmp_file):
        j += 1
        tmp_file = f"{_tmp}/{_fn}_{j}.png"

    return tmp_file

def build_command(plot_type: str, plot: dict, _force: bool) -> str:
    maindir = os.path.dirname(os.path.realpath(__file__))
    base_command = "bash omniopt_plot" if _force else f"bash {maindir}/omniopt_plot"
    command = f"{base_command} --run_dir {get_current_run_folder()} --plot_type={plot_type}"

    if "dpi" in plot:
        command += f" --dpi={plot['dpi']}"

    return command

def get_sixel_graphics_data(_pd_csv: str, _force: bool = False) -> list:
    _show_sixel_graphics = args.show_sixel_scatter or args.show_sixel_general or args.show_sixel_scatter or args.show_sixel_trial_index_result

    if _force:
        _show_sixel_graphics = True

    data: list = []

    conditions = [
        (not os.path.exists(_pd_csv), f"Cannot find path {_pd_csv}"),
        (not _show_sixel_graphics, "_show_sixel_graphics was false. Will not plot."),
        (len(global_vars["parameter_names"]) == 0, "Cannot handle empty data in global_vars -> parameter_names"),
    ]

    for condition, message in conditions:
        if condition:
            print_debug(message)
            return data

    x_y_combinations = get_x_y_combinations_parameter_names()
    plot_types = get_plot_types(x_y_combinations, _force)

    for plot in plot_types:
        plot_type = plot["type"]
        min_done_jobs = plot.get("min_done_jobs", 1)

        if not _force and count_done_jobs() < min_done_jobs:
            print_debug(f"Cannot plot {plot_type}, because it needs {min_done_jobs}, but you only have {count_done_jobs()} jobs done")
            continue

        try:
            _tmp = f"{get_current_run_folder()}/plots/"
            _width = plot.get("width", "1200")

            if not _force and not os.path.exists(_tmp):
                makedirs(_tmp)

            tmp_file = get_plot_filename(plot, _tmp)
            _command = build_command(plot_type, plot, _force)

            _params = [_command, plot, _tmp, plot_type, tmp_file, _width]
            data.append(_params)
        except Exception as e:
            tb = traceback.format_exc()
            print_red(f"Error trying to print {plot_type} to CLI: {e}, {tb}")
            print_debug(f"Error trying to print {plot_type} to CLI: {e}")

    return data

def get_plot_commands(_command: str, plot: dict, _tmp: str, plot_type: str, tmp_file: str, _width: str) -> List[List[str]]:
    plot_commands: List[List[str]] = []
    if "params" in plot.keys():
        if "iterate_through" in plot.keys():
            iterate_through = plot["iterate_through"]
            if len(iterate_through):
                for j in range(len(iterate_through)):
                    this_iteration = iterate_through[j]
                    replaced_str = replace_string_with_params(plot["params"], [this_iteration[0], this_iteration[1]])
                    _iterated_command: str = f"{_command} {replaced_str}"

                    j = 0
                    tmp_file = f"{_tmp}/{plot_type}.png"
                    _fn = ""
                    if "filename" in plot:
                        _fn = plot['filename']
                        if len(this_iteration):
                            _p = [plot_type, this_iteration[0], this_iteration[1]]
                            if len(_p):
                                tmp_file = f"{_tmp}/{replace_string_with_params(_fn, _p)}.png"

                            while os.path.exists(tmp_file):
                                j += 1
                                tmp_file = f"{_tmp}/{plot_type}_{j}.png"
                                if "filename" in plot and len(_p):
                                    tmp_file = f"{_tmp}/{replace_string_with_params(_fn, _p)}_{j}.png"

                    _iterated_command += f" --save_to_file={tmp_file} "
                    plot_commands.append([_iterated_command, tmp_file, str(_width)])
    else:
        _command += f" --save_to_file={tmp_file} "
        plot_commands.append([_command, tmp_file, str(_width)])

    return plot_commands

def plot_sixel_imgs() -> None:
    if ci_env:
        print("Not printing sixel graphics in CI")
        return

    if not os.path.exists(RESULT_CSV_FILE):
        print_debug(f"File '{RESULT_CSV_FILE}' not found")
        return

    sixel_graphic_commands = get_sixel_graphics_data(RESULT_CSV_FILE)

    for c in sixel_graphic_commands:
        commands = get_plot_commands(*c)

        for command in commands:
            plot_command(*command)

def get_crf() -> str:
    crf = get_current_run_folder()
    if crf in ["", None]:
        if not args.tests:
            console.print("[red]Could not find current run folder[/]")
        return ""
    return crf

def write_to_file(file_path: str, content: str) -> None:
    with open(file_path, mode="a+", encoding="utf-8") as text_file:
        text_file.write(content)

def create_result_table(res_name: str, best_params: Optional[Dict[str, Any]], total_str: str, failed_error_str: str) -> Optional[Table]:
    arg_result_min_or_max_index = arg_result_names.index(res_name)

    try:
        min_or_max = arg_result_min_or_max[arg_result_min_or_max_index]
        bracket_string = f"{total_str}{failed_error_str}"

        table = Table(
            show_header=True,
            header_style="bold",
            title=f"Best {res_name}, {min_or_max} ({bracket_string})"
        )

        if best_params and "parameters" in best_params:
            row_data = {**best_params['parameters']}
            result_name = arg_result_names[0]
            row_data[result_name] = best_params.get(result_name, '?')

            for col in row_data.keys():
                table.add_column(col, style="bold")

            table.add_row(*[str(v) for v in row_data.values()])

            return table
    except IndexError as e:
        print_red(f"create_result_table: Error {e}")
    return None

def print_and_write_table(table: Table, print_to_file: bool, file_path: str) -> None:
    with console.capture() as capture:
        console.print(table)
    if print_to_file:
        write_to_file(file_path, capture.get())

def process_best_result(res_name: str, print_to_file: bool) -> int:
    best_params = get_best_params_from_csv(res_name)
    best_result = best_params.get(res_name, NO_RESULT) if best_params else NO_RESULT

    if str(best_result) in [NO_RESULT, None, "None"]:
        print_red(f"Best {res_name} could not be determined")
        return 87 # exit-code: 87

    total_str = f"total: {_count_done_jobs(RESULT_CSV_FILE) - NR_INSERTED_JOBS}"
    if NR_INSERTED_JOBS:
        total_str += f" + inserted jobs: {NR_INSERTED_JOBS}"

    failed_error_str = f", failed: {failed_jobs()}" if print_to_file and failed_jobs() >= 1 else ""

    table = create_result_table(res_name, best_params, total_str, failed_error_str)
    if table is not None:
        if len(arg_result_names) == 1:
            console.print(table)

        print_and_write_table(table, print_to_file, f"{get_crf()}/best_result.txt")
        plot_sixel_imgs()

    return 0

def _print_best_result(print_to_file: bool = True) -> int:
    global SHOWN_END_TABLE

    crf = get_crf()
    if not crf:
        return -1

    try:
        for res_name in arg_result_names:
            result_code = process_best_result(res_name, print_to_file)
            if result_code != 0:
                return result_code
        SHOWN_END_TABLE = True
    except Exception as e:
        print_red(f"[_print_best_result] Error: {e}, tb: {traceback.format_exc()}")
        return -1

    return 0

def print_best_result() -> int:
    if os.path.exists(RESULT_CSV_FILE):
        return _print_best_result(True)

    return 0

def show_end_table_and_save_end_files() -> int:
    print_debug("show_end_table_and_save_end_files()")

    ignore_signals()

    global ALREADY_SHOWN_WORKER_USAGE_OVER_TIME

    if SHOWN_END_TABLE:
        print("End table already shown, not doing it again")
        return -1

    _exit: int = 0

    display_failed_jobs_table()

    best_result_exit: int = print_best_result()

    if not args.dryrun:
        print_evaluate_times()

    if best_result_exit > 0:
        _exit = best_result_exit

    if args.show_worker_percentage_table_at_end and len(WORKER_PERCENTAGE_USAGE) and not ALREADY_SHOWN_WORKER_USAGE_OVER_TIME:
        ALREADY_SHOWN_WORKER_USAGE_OVER_TIME = True

        table = Table(header_style="bold", title="Worker usage over time")
        columns = ["Time", "Nr. workers", "Max. nr. workers", "%"]
        for column in columns:
            table.add_column(column)
        for row in WORKER_PERCENTAGE_USAGE:
            table.add_row(str(row["time"]), str(row["nr_current_workers"]), str(row["num_parallel_jobs"]), f'{row["percentage"]}%')
        console.print(table)

    return _exit

def abandon_job(job: Job, trial_index: int, reason: str) -> bool:
    if job:
        try:
            if ax_client:
                _trial = ax_client.get_trial(trial_index)
                _trial.mark_abandoned(reason=reason)
                print_debug(f"abandon_job: removing job {job}, trial_index: {trial_index}")
                global_vars["jobs"].remove((job, trial_index))
            else:
                _fatal_error("ax_client could not be found", 9)
        except Exception as e:
            print(f"ERROR in line {get_line_info()}: {e}")
            print_debug(f"ERROR in line {get_line_info()}: {e}")
            return False
        job.cancel()
        return True

    return False

def abandon_all_jobs() -> None:
    for job, trial_index in global_vars["jobs"][:]:
        abandoned = abandon_job(job, trial_index, "abandon_all_jobs was called")
        if not abandoned:
            print_debug(f"Job {job} could not be abandoned.")

def show_pareto_or_error_msg(path_to_calculate: str, res_names: list = arg_result_names, disable_sixel_and_table: bool = False) -> None:
    if args.dryrun:
        print_debug("Not showing Pareto-frontier data with --dryrun")
        return None

    if len(res_names) > 1:
        try:
            show_pareto_frontier_data(path_to_calculate, res_names, disable_sixel_and_table)
        except Exception as e:
            print_red(f"show_pareto_frontier_data() failed with exception '{e}'")
    else:
        print_debug(f"show_pareto_frontier_data will NOT be executed because len(arg_result_names) is {len(arg_result_names)}")
    return None

def end_program(_force: Optional[bool] = False, exit_code: Optional[int] = None) -> None:
    global END_PROGRAM_RAN

    wait_for_jobs_to_complete()

    show_pareto_or_error_msg(get_current_run_folder(), arg_result_names)

    if os.getpid() != main_pid:
        print_debug("returning from end_program, because it can only run in the main thread, not any forks")
        return

    if END_PROGRAM_RAN and not _force:
        print_debug("[end_program] END_PROGRAM_RAN was true. Returning.")
        return

    END_PROGRAM_RAN = True

    _exit: int = 0

    try:
        check_conditions = {
            get_current_run_folder(): "[end_program] get_current_run_folder() was empty. Not running end-algorithm.",
            bool(ax_client): "[end_program] ax_client was empty. Not running end-algorithm.",
            bool(console): "[end_program] console was empty. Not running end-algorithm."
        }

        for condition, message in check_conditions.items():
            if condition is None:
                print_debug(message)
                return

        new_exit = show_end_table_and_save_end_files()
        if new_exit > 0:
            _exit = new_exit
    except (SignalUSR, SignalINT, SignalCONT, KeyboardInterrupt):
        print_red("\n⚠ You pressed CTRL+C or a signal was sent. Program execution halted while ending program.")
        print("\n⚠ KeyboardInterrupt signal was sent. Ending program will still run.")
        new_exit = show_end_table_and_save_end_files()
        if new_exit > 0:
            _exit = new_exit
    except TypeError as e:
        print_red(f"\n⚠ The program has been halted without attaining any results. Error: {e}")

    abandon_all_jobs()

    if exit_code:
        _exit = exit_code

    show_time_debugging_table()

    if succeeded_jobs() == 0 and failed_jobs() > 0:
        _exit = 89

    force_live_share()

    my_exit(_exit)

def save_checkpoint(trial_nr: int = 0, eee: Union[None, str, Exception] = None) -> None:
    if trial_nr > 3:
        if eee:
            print(f"Error during saving checkpoint: {eee}")
        else:
            print("Error during saving checkpoint")
        return

    try:
        checkpoint_filepath = get_state_file_name('checkpoint.json')

        if ax_client:
            ax_client.save_to_json_file(filepath=checkpoint_filepath)
        else:
            _fatal_error("Something went wrong using the ax_client", 9)
    except Exception as e:
        save_checkpoint(trial_nr + 1, e)

def get_tmp_file_from_json(experiment_args: dict) -> str:
    _tmp_dir = "/tmp"

    k = 0

    while os.path.exists(f"/{_tmp_dir}/{k}"):
        k = k + 1

    try:
        with open(f'/{_tmp_dir}/{k}', mode="w", encoding="utf-8") as f:
            json.dump(experiment_args, f)
    except PermissionError as e:
        print_red(f"Error writing '{k}' in get_tmp_file_from_json: {e}")

    return f"/{_tmp_dir}/{k}"

def extract_differences(old: Dict[str, Any], new: Dict[str, Any], prefix: str = "") -> List[str]:
    differences = []
    for key in old:
        if key in new and old[key] != new[key]:
            old_value, new_value = old[key], new[key]

            if isinstance(old_value, dict) and isinstance(new_value, dict):
                if "name" in old_value and "name" in new_value and set(old_value.keys()) == {"__type", "name"}:
                    differences.append(f"{prefix}{key} from {old_value['name']} to {new_value['name']}")
                else:
                    differences.extend(extract_differences(old_value, new_value, prefix=f"{prefix}{key}."))
            else:
                differences.append(f"{prefix}{key} from {old_value} to {new_value}")
    return differences

def compare_parameters(old_param_json: str, new_param_json: str) -> str:
    try:
        old_param = json.loads(old_param_json)
        new_param = json.loads(new_param_json)

        differences = extract_differences(old_param, new_param)

        if differences:
            param_name = old_param.get("name", "?")
            return f"Changed parameter '{param_name}': " + ", ".join(differences)

        return "No differences found between the old and new parameters."
    except json.JSONDecodeError:
        return "Error: Invalid JSON input."
    except Exception as e:
        return f"Error: {str(e)}"

def get_ax_param_representation(data: dict) -> dict:
    if data["type"] == "range":
        parameter_type = data["value_type"].upper()
        return {
            "__type": "RangeParameter",
            "name": data["name"],
            "parameter_type": {
                "__type": "ParameterType",
                "name": parameter_type
            },
            "lower": data["bounds"][0],
            "upper": data["bounds"][1],
            "log_scale": False,
            "logit_scale": False,
            "digits": 32,
            "is_fidelity": False,
            "target_value": None
        }
    if data["type"] == "choice":
        #parameter_type = "FLOAT" if all(isinstance(i, float) for i in data["values"]) else ("INT" if all(isinstance(i, int) for i in data["values"]) else "STRING")
        parameter_type = "STRING"

        return {
            '__type': 'ChoiceParameter',
            'dependents': None,
            'is_fidelity': False,
            'is_ordered': data["is_ordered"],
            "value_type": "str",
            'is_task': False,
            'name': data["name"],
            'parameter_type': {
                "__type": "ParameterType",
                "name": parameter_type
            },
            'target_value': None,
            'values': [str(v) for v in data["values"]]
        }

    print("data:")
    pprint(data)
    _fatal_error(f"Unknown data range {data['type']}", 19)

    return {}

def set_torch_device_to_experiment_args(experiment_args: Union[None, dict]) -> Tuple[dict, str, str]:
    gpu_string = ""
    gpu_color = "green"
    torch_device = None
    try:
        cuda_is_available = torch.cuda.is_available()

        if not cuda_is_available or cuda_is_available == 0:
            gpu_string = "No CUDA devices found."
            gpu_color = "yellow"
        else:
            if args.gpus >= 1:
                if torch.cuda.device_count() >= 1:
                    try:
                        torch_device = torch.cuda.current_device()
                        gpu_string = f"Using CUDA device {torch.cuda.get_device_name(0)}."
                        gpu_color = "green"
                    except torch.cuda.DeferredCudaCallError as e:
                        print_red(f"Could not load GPU: {e}")
                        gpu_string = "Error loading the CUDA device."
                        gpu_color = "red"
                else:
                    gpu_string = "No CUDA devices found."
                    gpu_color = "yellow"
            else:
                gpu_string = "No CUDA devices searched."
                gpu_color = "yellow"
    except ModuleNotFoundError:
        print_red("Cannot load torch and thus, cannot use gpus")

    if torch_device:
        if experiment_args:
            experiment_args["choose_generation_strategy_kwargs"]["torch_device"] = torch_device
        else:
            _fatal_error("experiment_args could not be created.", 90)

    if experiment_args:
        return experiment_args, gpu_string, gpu_color

    return {}, gpu_string, gpu_color

def die_with_47_if_file_doesnt_exists(_file: str) -> None:
    if not os.path.exists(_file):
        _fatal_error(f"Cannot find {_file}", 47)

def copy_state_files_from_previous_job(continue_previous_job: str) -> None:
    for state_file in ["submitted_jobs"]:
        old_state_file = f"{continue_previous_job}/state_files/{state_file}"
        new_state_file = get_state_file_name(state_file)
        die_with_47_if_file_doesnt_exists(old_state_file)

        if not os.path.exists(new_state_file):
            shutil.copy(old_state_file, new_state_file)

def parse_equation_item(comparer_found: bool, item: str, parsed: list, parsed_order: list, variables: list, equation: str) -> Tuple[bool, bool, list, list]:
    return_totally = False

    if item in ["+", "*", "-", "/"]:
        parsed_order.append("operator")
        parsed.append({
            "type": "operator",
            "value": item
        })
    elif item in [">=", "<="]:
        if comparer_found:
            print("There is already one comparison operator! Cannot have more than one in an equation!")
            return_totally = True
        comparer_found = True

        parsed_order.append("comparer")
        parsed.append({
            "type": "comparer",
            "value": item
        })
    elif re.match(r'^\d+$', item):
        parsed_order.append("number")
        parsed.append({
            "type": "number",
            "value": item
        })
    elif item in variables:
        parsed_order.append("variable")
        parsed.append({
            "type": "variable",
            "value": item
        })
    else:
        print_red(f"constraint error: Invalid variable {item} in constraint '{equation}' is not defined in the parameters. Possible variables: {', '.join(variables)}")
        return_totally = True

    return return_totally, comparer_found, parsed, parsed_order

def is_valid_equation(expr: str, allowed_vars: list) -> bool:
    try:
        node = ast.parse(expr, mode='eval')
    except SyntaxError:
        return False

    if not isinstance(node, ast.Expression):
        return False

    def is_valid_op(op: Any) -> bool:
        return isinstance(op, (
            ast.LtE, ast.GtE, ast.Eq, ast.NotEq,
            ast.Add, ast.Sub, ast.Mult, ast.Div
        ))

    def is_only_allowed_vars(node: Any) -> bool:
        if isinstance(node, ast.Name):
            return node.id in allowed_vars
        if isinstance(node, ast.BinOp):
            return is_valid_op(node.op) and \
                   is_only_allowed_vars(node.left) and \
                   is_only_allowed_vars(node.right)
        if isinstance(node, ast.UnaryOp):
            return isinstance(node.op, (ast.UAdd, ast.USub)) and \
                   is_only_allowed_vars(node.operand)
        if isinstance(node, ast.Constant):
            return isinstance(node.value, (int, float))
        if isinstance(node, ast.Num):
            return isinstance(node.n, (int, float))
        return False

    def is_constant_expr(node: Any) -> bool:
        if isinstance(node, ast.Constant):
            return isinstance(node.value, (int, float))
        if isinstance(node, ast.Num):
            return isinstance(node.n, (int, float))
        if isinstance(node, ast.BinOp):
            return is_valid_op(node.op) and \
                   is_constant_expr(node.left) and \
                   is_constant_expr(node.right)
        if isinstance(node, ast.UnaryOp):
            return isinstance(node.op, (ast.UAdd, ast.USub)) and \
                   is_constant_expr(node.operand)
        return False

    body = node.body
    if not isinstance(body, ast.Compare):
        return False
    if len(body.ops) != 1 or len(body.comparators) != 1:
        return False

    left = body.left
    right = body.comparators[0]
    op = body.ops[0]

    if not isinstance(op, (ast.LtE, ast.GtE, ast.Eq, ast.NotEq)):
        return False

    if not is_only_allowed_vars(left):
        return False
    if not (is_constant_expr(right) or is_only_allowed_vars(right)):
        return False

    return True

def is_ax_compatible_constraint(equation: str, variables: List[str]) -> Union[str, bool]:
    equation = equation.replace("\\*", "*")
    equation = equation.replace(" * ", "*")
    equation = equation.replace(" + ", "+")
    equation = equation.replace(" - ", "-")
    equation = re.sub(r"\s+", "", equation)

    if ">=" not in equation and "<=" not in equation:
        return False

    if "==" in equation or re.search(r"(?<![<>])=(?!=)", equation):
        return False

    comparisons = re.findall(r"(<=|>=)", equation)
    if len(comparisons) != 1:
        return False

    operator = comparisons[0]
    lhs, rhs = equation.split(operator)

    def analyze_expression(expr: str) -> bool:
        terms = re.findall(r"[+-]?[^+-]+", expr)
        for term in terms:
            term = term.strip()
            if not term:
                continue

            if "*" in term:
                parts = term.split("*")
                if len(parts) != 2:
                    return False

                factor, var = parts
                if not re.fullmatch(r"[+-]?[0-9.]+(?:[eE][-+]?[0-9]+)?", factor):
                    return False
                if var not in variables:
                    return False
            else:
                if term not in variables:
                    if not re.fullmatch(r"[+-]?[0-9.]+(?:[eE][-+]?[0-9]+)?", term):
                        return False
        return True

    if lhs in variables and rhs in variables:
        return True

    if not analyze_expression(lhs):
        return False

    if not re.fullmatch(r"[+-]?[0-9.]+(?:[eE][-+]?[0-9]+)?", rhs):
        return False

    return True

def check_equation(variables: list, equation: str) -> Union[str, bool]:
    print_debug(f"check_equation({variables}, {equation})")

    _errors = []

    if not (">=" in equation or "<=" in equation):
        _errors.append(f"check_equation({variables}, {equation}): if not ('>=' in equation or '<=' in equation)")

    comparer_at_beginning = re.search("^\\s*((<=|>=)|(<=|>=))", equation)
    if comparer_at_beginning:
        _errors.append(f"The restraints {equation} contained comparison operator like <=, >= at at the beginning. This is not a valid equation.")

    comparer_at_end = re.search("((<=|>=)|(<=|>=))\\s*$", equation)
    if comparer_at_end:
        _errors.append(f"The restraints {equation} contained comparison operator like <=, >= at at the end. This is not a valid equation.")

    if len(_errors):
        for er in _errors:
            print_red(er)

        return False

    equation = equation.replace("\\*", "*")
    equation = equation.replace(" * ", "*")

    equation = equation.replace(">=", " >= ")
    equation = equation.replace("<=", " <= ")

    equation = re.sub(r'\s+', ' ', equation)
    #equation = equation.replace("", "")

    regex_pattern: str = r'\s+|(?=[+\-*\/()-])|(?<=[+\-*\/()-])'
    result_array = re.split(regex_pattern, equation)
    result_array = [item for item in result_array if item.strip()]

    parsed: list = []
    parsed_order: list = []

    comparer_found = False

    for item in result_array:
        return_totally, comparer_found, parsed, parsed_order = parse_equation_item(comparer_found, item, parsed, parsed_order, variables, equation)

        if return_totally:
            return False

    parsed_order_string = ";".join(parsed_order)

    number_or_variable = "(?:(?:number|variable);*)"
    number_or_variable_and_operator = f"(?:{number_or_variable};operator;*)"
    comparer = "(?:comparer;)"
    equation_part = f"{number_or_variable_and_operator}*{number_or_variable}"

    regex_order = f"^{equation_part}{comparer}{equation_part}$"

    order_check = re.match(regex_order, parsed_order_string)

    if order_check:
        return equation

    return False

def set_objectives() -> dict:
    objectives = {}

    for rn in args.result_names:
        key, value = "", ""

        if "=" in rn:
            key, value = rn.split('=', 1)
        else:
            key = rn
            value = ""

        if value not in ["min", "max"]:
            if value:
                print_yellow(f"Value '{value}' for --result_names {rn} is not a valid value. Must be min or max. Will be set to min.")

            value = "min"

        _min = True

        if value == "max":
            _min = False

        objectives[key] = ObjectiveProperties(minimize=_min)

    return objectives

def set_experiment_constraints(experiment_constraints: Optional[list], experiment_args: dict, _experiment_parameters: Union[dict, list]) -> dict:
    if experiment_constraints and len(experiment_constraints):

        experiment_args["parameter_constraints"] = []

        if experiment_constraints:
            for _l in range(len(experiment_constraints)):
                variables = [item['name'] for item in _experiment_parameters]

                constraints_string = experiment_constraints[_l]

                equation = check_equation(variables, constraints_string)

                if equation:
                    experiment_args["parameter_constraints"].append(constraints_string)
                else:
                    _fatal_error(f"Experiment constraint '{constraints_string}' is invalid. Cannot continue.", 19)

                file_path = os.path.join(get_current_run_folder(), "state_files", "constraints")

                os.makedirs(os.path.dirname(file_path), exist_ok=True)

                with open(file_path, "a", encoding="utf-8") as f:
                    f.write(constraints_string + "\n")
        else:
            print_debug(f"set_experiment_constraints: not set, content: {experiment_constraints}")
    else:
        print_debug("set_experiment_constraints: no constraints set")

    return experiment_args

def replace_parameters_for_continued_jobs(parameter: Optional[list], cli_params_experiment_parameters: Optional[list]) -> None:
    if args.worker_generator_path:
        return None

    def get_name(obj: Any) -> Optional[str]:
        """Extract a parameter name from dict, list, or tuple safely."""
        if isinstance(obj, dict):
            return obj.get("name")
        if isinstance(obj, (list, tuple)) and len(obj) > 0 and isinstance(obj[0], str):
            return obj[0]
        return None

    if parameter and cli_params_experiment_parameters:
        for _item in cli_params_experiment_parameters:
            _replaced = False
            item_name = get_name(_item)

            for _item_id_to_overwrite, param_entry in enumerate(
                experiment_parameters["experiment"]["search_space"]["parameters"]
            ):
                param_name = get_name(param_entry)

                if item_name and param_name and item_name == param_name:
                    old_param_json = json.dumps(param_entry)

                    experiment_parameters["experiment"]["search_space"]["parameters"][_item_id_to_overwrite] = get_ax_param_representation(_item)

                    new_param_json = json.dumps(
                        experiment_parameters["experiment"]["search_space"]["parameters"][_item_id_to_overwrite]
                    )

                    _replaced = True

                    compared_params = compare_parameters(old_param_json, new_param_json)
                    if compared_params and not args.worker_generator_path:
                        print_yellow(compared_params)

            if not _replaced:
                print_yellow(
                    f"--parameter named {item_name} could not be replaced. "
                    "It will be ignored instead. You cannot change the number of parameters "
                    "or their names when continuing a job, only update their values."
                )

    return None

def load_experiment_parameters_from_checkpoint_file(checkpoint_file: str, _die: bool = True) -> None:
    global experiment_parameters

    try:
        f = open(checkpoint_file, encoding="utf-8")
        experiment_parameters = json.load(f)
        f.close()

        with open(checkpoint_file, encoding="utf-8") as f:
            experiment_parameters = json.load(f)
    except json.decoder.JSONDecodeError:
        print_red(f"Error parsing checkpoint_file {checkpoint_file}")
        if _die:
            my_exit(47)

def get_username() -> str:
    _user = os.getenv('USER')

    if args.username:
        _user = args.username

    if _user is None:
        return 'defaultuser'

    return _user

def copy_continue_uuid() -> None:
    source_file = os.path.join(args.continue_previous_job, "state_files", "run_uuid")
    destination_file = os.path.join(get_current_run_folder(), "state_files", "continue_from_run_uuid")

    if os.path.exists(source_file):
        try:
            shutil.copy(source_file, destination_file)
            print_debug(f"copy_continue_uuid: Copied '{source_file}' to '{destination_file}'")
        except Exception as e:
            print_debug(f"copy_continue_uuid: Error copying file: {e}")
    else:
        print_debug(f"copy_continue_uuid: Source file does not exist: {source_file}")

def load_ax_client_from_experiment_parameters() -> None:
    #pprint(experiment_parameters)
    global ax_client

    tmp_file_path = get_tmp_file_from_json(experiment_parameters)
    ax_client = AxClient.load_from_json_file(tmp_file_path)
    ax_client = cast(AxClient, ax_client)
    os.unlink(tmp_file_path)

def save_checkpoint_for_continued() -> None:
    checkpoint_filepath = get_state_file_name('checkpoint.json')

    with open(checkpoint_filepath, mode="w", encoding="utf-8") as outfile:
        json.dump(experiment_parameters, outfile)

    if not os.path.exists(checkpoint_filepath):
        _fatal_error(f"{checkpoint_filepath} not found. Cannot continue_previous_job without.", 47)

def load_original_generation_strategy(original_ax_client_file: str) -> None:
    with open(original_ax_client_file, encoding="utf-8") as f:
        loaded_original_ax_client_json = json.load(f)
        original_generation_strategy = loaded_original_ax_client_json["generation_strategy"]

        if original_generation_strategy:
            experiment_parameters["generation_strategy"] = original_generation_strategy

def wait_for_checkpoint_file(checkpoint_file: str) -> None:
    start_time = time.time()

    while not os.path.exists(checkpoint_file):
        elapsed = int(time.time() - start_time)
        console.print(f"[yellow]Waiting for file {checkpoint_file}... {elapsed} seconds[/yellow]", end="\r")
        time.sleep(1)

    elapsed = int(time.time() - start_time)
    console.print(f"[green]Checkpoint file found after {elapsed} seconds[/green]   ")

def __get_experiment_parameters__check_ax_client() -> None:
    if not ax_client:
        _fatal_error("Something went wrong with the ax_client", 9)

def validate_experiment_parameters() -> None:
    if experiment_parameters is None:
        print_red("Error: experiment_parameters is None.")
        my_exit(95)

    if not isinstance(experiment_parameters, dict):
        print_red(f"Error: experiment_parameters is not a dict: {type(experiment_parameters).__name__}")
        my_exit(95)

        sys.exit(95)

    path_checks = [
        ("experiment", experiment_parameters),
        ("search_space", experiment_parameters.get("experiment")),
        ("parameters", experiment_parameters.get("experiment", {}).get("search_space")),
    ]

    for key, current_level in path_checks:
        if not isinstance(current_level, dict) or key not in current_level:
            print_red(f"Error: Missing key '{key}' at level: {current_level}")
            my_exit(95)

def __get_experiment_parameters__load_from_checkpoint(continue_previous_job: str, cli_params_experiment_parameters: Optional[list]) -> Tuple[Any, str, str]:
    print_debug(f"Load from checkpoint: {continue_previous_job}")

    checkpoint_file = f"{continue_previous_job}/state_files/checkpoint.json"
    checkpoint_parameters_filepath = f"{continue_previous_job}/state_files/checkpoint.json.parameters.json"
    original_ax_client_file = get_state_file_name("original_ax_client_before_loading_tmp_one.json")

    if args.worker_generator_path:
        wait_for_checkpoint_file(checkpoint_parameters_filepath)

    die_with_47_if_file_doesnt_exists(checkpoint_parameters_filepath)

    if args.worker_generator_path:
        wait_for_checkpoint_file(checkpoint_file)

    die_with_47_if_file_doesnt_exists(checkpoint_file)

    load_experiment_parameters_from_checkpoint_file(checkpoint_file)
    experiment_args, gpu_string, gpu_color = set_torch_device_to_experiment_args(None)

    copy_state_files_from_previous_job(continue_previous_job)

    validate_experiment_parameters()

    replace_parameters_for_continued_jobs(args.parameter, cli_params_experiment_parameters)

    ax_client.save_to_json_file(filepath=original_ax_client_file)

    load_original_generation_strategy(original_ax_client_file)
    load_ax_client_from_experiment_parameters()
    save_checkpoint_for_continued()

    with open(get_current_run_folder("checkpoint_load_source"), mode='w', encoding="utf-8") as f:
        print(f"Continuation from checkpoint {continue_previous_job}", file=f)

    if not args.worker_generator_path:
        copy_continue_uuid()
    else:
        print_debug(f"Not copying continue uuid because this is not a new job, because --worker_generator_path {args.worker_generator_path} is not a new job")

    experiment_constraints = get_constraints()
    if experiment_constraints:
        experiment_args = set_experiment_constraints(
            experiment_constraints,
            experiment_args,
            experiment_parameters["experiment"]["search_space"]["parameters"]
        )

    return experiment_args, gpu_string, gpu_color

def __get_experiment_parameters__create_new_experiment() -> Tuple[dict, str, str]:
    objectives = set_objectives()

    experiment_args = {
        "name": global_vars["experiment_name"],
        "parameters": experiment_parameters,
        "objectives": objectives,
        "choose_generation_strategy_kwargs": {
            "num_trials": max_eval,
            "num_initialization_trials": num_parallel_jobs,
            "use_batch_trials": True,
            "max_parallelism_override": -1,
            "random_seed": args.seed
        },
    }

    if args.seed:
        experiment_args["choose_generation_strategy_kwargs"]["random_seed"] = args.seed

    experiment_args, gpu_string, gpu_color = set_torch_device_to_experiment_args(experiment_args)
    experiment_args = set_experiment_constraints(get_constraints(), experiment_args, experiment_parameters)

    try:
        ax_client.create_experiment(**experiment_args)
        new_metrics = [Metric(k) for k in arg_result_names if k not in ax_client.metric_names]
        ax_client.experiment.add_tracking_metrics(new_metrics)
    except AssertionError as error:
        _fatal_error(f"An error has occurred while creating the experiment (0): {error}. This can happen when you have invalid parameter constraints.", 102)
    except ValueError as error:
        _fatal_error(f"An error has occurred while creating the experiment (1): {error}", 49)
    except TypeError as error:
        _fatal_error(f"An error has occurred while creating the experiment (2): {error}. This is probably a bug in OmniOpt2.", 49)
    except ax.exceptions.core.UserInputError as error:
        _fatal_error(f"An error occurred while creating the experiment (3): {error}", 49)

    return experiment_args, gpu_string, gpu_color

def get_experiment_parameters(cli_params_experiment_parameters: Optional[list]) -> Optional[Tuple[AxClient, dict, str, str]]:
    continue_previous_job = args.worker_generator_path or args.continue_previous_job

    __get_experiment_parameters__check_ax_client()

    if continue_previous_job:
        experiment_args, gpu_string, gpu_color = __get_experiment_parameters__load_from_checkpoint(continue_previous_job, cli_params_experiment_parameters)
    else:
        experiment_args, gpu_string, gpu_color = __get_experiment_parameters__create_new_experiment()

    return ax_client, experiment_args, gpu_string, gpu_color

def get_type_short(typename: str) -> str:
    if typename == "RangeParameter":
        return "range"

    if typename == "ChoiceParameter":
        return "choice"

    return typename

def parse_single_experiment_parameter_table(classic_params: Optional[Union[list, dict]]) -> list:
    rows: list = []

    if classic_params is None:
        print_red("parse_single_experiment_parameter_table: classic_param is None")
        return rows

    k = 0

    for param in classic_params:
        _type = ""
        _name = str(param["name"])

        if "__type" in param:
            _type = param["__type"].lower()
        else:
            _type = param["type"].lower()

        _short_type = get_type_short(_type)

        if "range" in _type:
            _lower = ""
            _upper = ""
            _type = ""
            value_type = ""

            log_scale = "No"

            if param["log_scale"]:
                log_scale = "Yes"

            if "parameter_type" in param:
                _type = param["parameter_type"]["name"].lower()
                value_type = _type
            else:
                _type = param["type"]
                value_type = param["value_type"]

            if "lower" in param:
                _lower = param["lower"]
            else:
                _lower = param["bounds"][0]
            if "upper" in param:
                _upper = param["upper"]
            else:
                _upper = param["bounds"][1]

            _possible_int_lower = str(helpers.to_int_when_possible(_lower))
            #print(f"name: {_name}, _possible_int_lower: {_possible_int_lower}, lower: {_lower}")
            _possible_int_upper = str(helpers.to_int_when_possible(_upper))

            rows.append([_name, _short_type, _possible_int_lower, _possible_int_upper, "", value_type, log_scale])
        elif "fixed" in _type:
            rows.append([_name, _short_type, "", "", str(helpers.to_int_when_possible(param["value"])), "", ""])
        elif "choice" in _type:
            values = param["values"]
            values = [str(helpers.to_int_when_possible(item)) for item in values]

            rows.append([_name, _short_type, "", "", ", ".join(values), "", ""])
        else:
            _fatal_error(f"Type {_type} is not yet implemented in the overview table.", 15)

        k = k + 1

    return rows

def print_non_ax_parameter_constraints_table() -> None:
    if not post_generation_constraints:
        return None

    table = Table(header_style="bold")
    columns = ["Post-Generation-Constraints"]

    for column in columns:
        table.add_column(column)

    for constraint in post_generation_constraints:
        table.add_row(constraint)

    with console.capture() as capture:
        console.print(table)

    table_str = capture.get()

    console.print(table)

    fn = f"{get_current_run_folder()}/post_generation_constraints.txt"
    try:
        with open(fn, mode="w", encoding="utf-8") as text_file:
            text_file.write(table_str)
    except Exception as e:
        print_red(f"Error writing {fn}: {e}")

    return None

def print_ax_parameter_constraints_table(experiment_args: dict) -> None:
    if not (experiment_args is not None and "parameter_constraints" in experiment_args and len(experiment_args["parameter_constraints"])):
        return None

    constraints = experiment_args["parameter_constraints"]
    table = Table(header_style="bold")
    columns = ["Constraints"]

    for column in columns:
        table.add_column(column)

    for constraint in constraints:
        table.add_row(constraint)

    with console.capture() as capture:
        console.print(table)

    table_str = capture.get()

    console.print(table)

    fn = get_current_run_folder("constraints.txt")
    try:
        with open(fn, mode="w", encoding="utf-8") as text_file:
            text_file.write(table_str)
    except Exception as e:
        print_red(f"Error writing {fn}: {e}")

    return None

def print_result_names_overview_table() -> None:
    if not ax_client:
        _fatal_error("Tried to access ax_client in print_result_names_overview_table, but it failed, because the ax_client was not defined.", 101)

        return None

    if args.continue_previous_job is not None and args.result_names is not None and len(args.result_names) != 0 and original_result_names is not None and len(original_result_names) != 0:
        print_yellow("--result_names will be ignored in continued jobs. The result names from the previous job will be used.")

    if ax_client.experiment.optimization_config.is_moo_problem:
        config_objectives = ax_client.experiment.optimization_config.objective.objectives
    else:
        config_objectives = [ax_client.experiment.optimization_config.objective]

    res_names = []
    res_min_max = []

    for obj in config_objectives:
        min_or_max = "max"
        if obj.minimize:
            min_or_max = "min"

        res_names.append(obj.metric_names[0])
        res_min_max.append(min_or_max)

    __table = Table(title="Result-Names")

    __table.add_column("Result-Name", justify="left", style="bold")
    __table.add_column("Min or max?", justify="right", style="bold")

    for __name, __value in zip(res_names, res_min_max):
        __table.add_row(str(__name), str(__value))

    console.print(__table)

    with console.capture() as capture:
        console.print(__table)

    table_str = capture.get()

    with open(f"{get_current_run_folder()}/result_names_overview.txt", mode="w", encoding="utf-8") as text_file:
        text_file.write(table_str)

    return None

def print_experiment_param_table_to_file(filtered_columns: list, filtered_data: list) -> None:
    table = Table(header_style="bold", title="Experiment parameters")
    for column in filtered_columns:
        table.add_column(column)

    for row in filtered_data:
        table.add_row(*[str(cell) if cell is not None else "" for cell in row], style="bright_green")

    console.print(table)

    with console.capture() as capture:
        console.print(table)

    table_str = capture.get()

    fn = get_current_run_folder("parameters.txt")

    try:
        with open(fn, mode="w", encoding="utf-8") as text_file:
            text_file.write(table_str)
    except FileNotFoundError as e:
        print_red(f"Error trying to write file {fn}: {e}")

def print_experiment_parameters_table(classic_param: Optional[Union[list, dict]]) -> None:
    if not classic_param:
        print_red("Cannot determine classic_param. No parameter table will be shown.")
        return

    if not classic_param:
        print_red("Experiment parameters could not be determined for display")
        return

    if isinstance(classic_param, dict) and "_type" in classic_param:
        classic_param = classic_param["experiment"]["search_space"]["parameters"]

    rows = parse_single_experiment_parameter_table(classic_param)

    columns = ["Name", "Type", "Lower bound", "Upper bound", "Values", "Type", "Log Scale?"]

    data = []
    for row in rows:
        data.append(row)

    non_empty_columns = []
    for col_index, _ in enumerate(columns):
        if any(row[col_index] not in (None, "") for row in data):
            non_empty_columns.append(col_index)

    filtered_columns = [columns[i] for i in non_empty_columns]
    filtered_data = [[row[i] for i in non_empty_columns] for row in data]

    print_experiment_param_table_to_file(filtered_columns, filtered_data)

def print_overview_tables(classic_params: Optional[Union[list, dict]], experiment_args: dict) -> None:
    print_experiment_parameters_table(classic_params)

    print_ax_parameter_constraints_table(experiment_args)
    print_non_ax_parameter_constraints_table()

    print_result_names_overview_table()

def update_progress_bar(nr: int) -> None:
    try:
        progress_bar.update(nr)
    except Exception as e:
        print(f"Error updating progress bar: {e}")

def get_current_model_name() -> str:
    if overwritten_to_random:
        return "Random*"

    if ax_client:
        try:
            if args.generation_strategy:
                idx = getattr(ax_client.generation_strategy, "current_step_index", None)
                if isinstance(idx, int):
                    if 0 <= idx < len(generation_strategy_names):
                        gs_model = generation_strategy_names[int(idx)]
                    else:
                        gs_model = "unknown model"
                else:
                    gs_model = "unknown model"
            else:
                gs_model = getattr(ax_client.generation_strategy, "current_node_name", "unknown model")

            if gs_model:
                return str(gs_model)

        except Exception as e:
            print_red(f"[WARN] Could not get current model name: {e}")
            return "error reading model name"

    return "initializing model"

def get_best_params_str(res_name: str = "RESULT") -> str:
    if count_done_jobs() >= 0:
        best_params = get_best_params(res_name)
        if best_params and best_params is not None and res_name in best_params:
            best_result = best_params[res_name]
            if isinstance(best_result, (int, float)) or helpers.looks_like_float(best_result):
                best_result_int_if_possible = helpers.to_int_when_possible(float(best_result))

                if str(best_result) != NO_RESULT and best_result is not None:
                    return f"{res_name}: {best_result_int_if_possible}"
    return ""

def state_from_job(job: Union[str, Job]) -> str:
    job_string = f'{job}'
    match = re.search(r'state="([^"]+)"', job_string)

    state = None

    if match:
        state = match.group(1).lower()
    else:
        state = f"{state}"

    return state

def get_workers_string() -> str:
    stats = _get_workers_string_collect_stats()
    string_keys, string_values, total_sum = _get_workers_string_format_keys_values(stats)

    if not (string_keys and string_values):
        return ""

    nr_current_workers, nr_current_workers_errmsg = count_jobs_in_squeue()

    if args.generate_all_jobs_at_once:
        return _get_workers_string_all_at_once(string_keys, string_values, total_sum)

    return _get_workers_string_dynamic(
        string_keys,
        string_values,
        total_sum,
        nr_current_workers,
        nr_current_workers_errmsg
    )

def _get_workers_string_collect_stats() -> dict:
    stats: dict = {}
    for job, _ in global_vars["jobs"][:]:
        state = state_from_job(job)
        stats[state] = stats.get(state, 0) + 1
    return stats

def _get_workers_string_format_keys_values(stats: dict) -> tuple[list, list, int]:
    string_keys = []
    string_values = []
    total_sum = 0

    for key, value in stats.items():
        string_keys.append(key.lower()[0] if args.abbreviate_job_names else key.lower())
        string_values.append(str(value))
        total_sum += int(value)

    return string_keys, string_values, total_sum

def _get_workers_string_all_at_once(keys: list, values: list, total_sum: int) -> str:
    _keys = "/".join(keys)
    _values = "/".join(values)
    return f"{_keys} {_values} = ∑{total_sum}/{num_parallel_jobs}"

def _get_workers_string_dynamic(
    keys: list,
    values: list,
    total_sum: int,
    nr_current_workers: int,
    nr_current_workers_errmsg: str
) -> str:
    _keys = "/".join(keys)
    _values = "/".join(values)

    if nr_current_workers_errmsg == "":
        percentage = round((nr_current_workers / num_parallel_jobs) * 100)
        _sum_and_percentage = ""
        if num_parallel_jobs > 1:
            _sum_and_percentage = f"∑{total_sum} ({percentage}%/{num_parallel_jobs})"
        return f"{_keys} {_values}{_sum_and_percentage}"

    print_debug(f"get_workers_string: {nr_current_workers_errmsg}")
    return f"{_keys} {_values} = ∑{total_sum}/{num_parallel_jobs}"

def submitted_jobs(nr: int = 0) -> int:
    return append_and_read(get_state_file_name('submitted_jobs'), nr)

def count_jobs_in_squeue() -> tuple[int, str]:
    global _last_count_time, _last_count_result

    now = time.time()
    if _last_count_result != (0, "") and now - _last_count_time < 15:
        return _last_count_result

    _len = len(global_vars["jobs"])

    if shutil.which('squeue') is None:
        _last_count_result = (_len, "count_jobs_in_squeue: squeue not found")
        _last_count_time = now
        return _last_count_result

    experiment_name = global_vars["experiment_name"]
    job_pattern = re.compile(rf"{experiment_name}_{run_uuid}_[a-f0-9-]+")
    err_msg = ""

    try:
        result = subprocess.run(
            ['squeue', '-o', '%j'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
        if "slurm_load_jobs error" in result.stderr:
            _last_count_result = (_len, "Detected slurm_load_jobs error in stderr.")
            _last_count_time = now
            return _last_count_result

        jobs = result.stdout.splitlines()
        job_count = sum(1 for job in jobs if job_pattern.match(job))
        _last_count_result = (job_count, "")
        _last_count_time = now
        return _last_count_result

    except subprocess.CalledProcessError:
        err_msg = "count_jobs_in_squeue: Error while executing squeue."
    except FileNotFoundError:
        err_msg = "count_jobs_in_squeue: squeue is not available on this host."

    _last_count_result = (-1, err_msg)
    _last_count_time = now
    return _last_count_result

def log_worker_numbers() -> None:
    if is_slurm_job():
        nr_current_workers, nr_current_workers_errmsg = count_jobs_in_squeue()
        if nr_current_workers_errmsg:
            print_debug(f"Failed to get count_jobs_in_squeue: {nr_current_workers_errmsg}")

        percentage = round((nr_current_workers / num_parallel_jobs) * 100)

        this_time: float = time.time()

        this_values = {
            "nr_current_workers": nr_current_workers,
            "num_parallel_jobs": num_parallel_jobs,
            "percentage": percentage,
            "time": this_time
        }

        if len(WORKER_PERCENTAGE_USAGE) == 0 or WORKER_PERCENTAGE_USAGE[len(WORKER_PERCENTAGE_USAGE) - 1] != this_values:
            WORKER_PERCENTAGE_USAGE.append(this_values)

def get_slurm_in_brackets(in_brackets: list) -> list:
    if is_slurm_job():
        workers_strings = get_workers_string()
        if workers_strings:
            in_brackets.append(workers_strings)

    return in_brackets

def get_types_of_errors_string() -> str:
    types_of_errors_str = ""

    _types_of_errors: list = read_errors_from_file()

    if len(_types_of_errors) > 0:
        types_of_errors_str = f" ({', '.join(_types_of_errors)})"

    return types_of_errors_str

def capitalized_string(s: str) -> str:
    return s[0].upper() + s[1:] if s else ""

def get_desc_progress_text(new_msgs: List[str] = []) -> str:
    global progress_bar_length

    current_model_name = get_current_model_name()

    if current_model_name == "SobolGenerator":
        current_model_name = "Sobol"

    in_brackets = []
    in_brackets.append(current_model_name)
    in_brackets.extend(_get_desc_progress_text_failed_jobs())
    in_brackets.extend(_get_desc_progress_text_best_params())
    in_brackets = get_slurm_in_brackets(in_brackets)

    if args.verbose_tqdm:
        in_brackets.extend(_get_desc_progress_text_submitted_jobs())

    if new_msgs:
        in_brackets.extend(_get_desc_progress_text_new_msgs(new_msgs))

    in_brackets_clean = [item for item in in_brackets if item]
    desc = ", ".join(in_brackets_clean) if in_brackets_clean else ""

    capitalized = capitalized_string(desc)

    if len(capitalized) > progress_bar_length:
        progress_bar_length = len(capitalized)
    else:
        capitalized = capitalized.ljust(progress_bar_length) if isinstance(capitalized, str) and isinstance(progress_bar_length, int) else capitalized

    return capitalized

def _get_desc_progress_text_failed_jobs() -> List[str]:
    if failed_jobs():
        return [f"{helpers.bcolors.red}failed: {failed_jobs()}{get_types_of_errors_string()}{helpers.bcolors.endc}"]
    return []

def _get_desc_progress_text_best_params() -> List[str]:
    best_params_res = [
        get_best_params_str(res_name) for res_name in arg_result_names if get_best_params_str(res_name)
    ]

    if best_params_res:
        done_jobs = count_done_jobs()
        return ["best " + ", ".join(best_params_res)] if len(arg_result_names) == 1 else [f"{done_jobs} done"]

    return []

def _get_desc_progress_text_submitted_jobs() -> List[str]:
    result = []
    if submitted_jobs():
        result.append(f"total submitted: {submitted_jobs()}")
        if max_eval:
            result.append(f"max_eval: {max_eval}")
    return result

def _get_desc_progress_text_new_msgs(new_msgs: List[str]) -> List[str]:
    return [msg for msg in new_msgs if msg]

def progressbar_description(new_msgs: Union[str, List[str]] = []) -> None:
    global last_progress_bar_desc
    global last_progress_bar_refresh_time

    if isinstance(new_msgs, str):
        new_msgs = [new_msgs]

    desc = get_desc_progress_text(new_msgs)
    print_debug_progressbar(desc)

    if progress_bar is not None:
        now = time.time()
        if last_progress_bar_desc != desc and (now - last_progress_bar_refresh_time) >= MIN_REFRESH_INTERVAL:
            progress_bar.set_description_str(desc)
            progress_bar.refresh()
            last_progress_bar_desc = desc
            last_progress_bar_refresh_time = now
    else:
        print_red("Cannot update progress bar! It is None.")

def clean_completed_jobs() -> None:
    job_states_to_be_removed = ["early_stopped", "abandoned", "cancelled", "timeout", "interrupted", "failed", "preempted", "node_fail", "boot_fail"]
    job_states_to_be_ignored = ["ready", "completed", "unknown", "pending", "running", "completing", "out_of_memory", "requeued", "resv_del_hold"]

    for job, trial_index in global_vars["jobs"][:]:
        _state = state_from_job(job)
        #print_debug(f'clean_completed_jobs: Job {job} (trial_index: {trial_index}) has state {_state}')
        if _state in job_states_to_be_removed:
            print_debug(f"clean_completed_jobs: removing job {job}, trial_index: {trial_index}, state: {_state}")
            global_vars["jobs"].remove((job, trial_index))
        elif _state in job_states_to_be_ignored:
            pass
        else:
            job_states_to_be_removed_string = "', '".join(job_states_to_be_removed)
            job_states_to_be_ignored_string = "', '".join(job_states_to_be_ignored)

            print_red(f"Job {job}, state not in ['{job_states_to_be_removed_string}'], which would be removed from the job list, or ['{job_states_to_be_ignored_string}'], which would be ignored: {_state}")

def simulate_load_data_from_existing_run_folders(_paths: List[str]) -> int:
    _counter: int = 0

    for this_path in _paths:
        this_path_json = f"{this_path}/state_files/ax_client.experiment.json"

        if not os.path.exists(this_path_json):
            print_red(f"{this_path_json} does not exist, cannot load data from it")
            return 0

        try:
            old_experiments = load_experiment(this_path_json, CORE_DECODER_REGISTRY)

            old_trials = old_experiments.trials

            for old_trial_index in old_trials:
                old_trial = old_trials[old_trial_index]
                trial_status = old_trial.status
                trial_status_str = trial_status.__repr__

                if "COMPLETED".lower() not in str(trial_status_str).lower():
                    continue

                _counter += 1
        except ValueError as e:
            print_red(f"Error while simulating loading data: {e}")

    return _counter

def get_nr_of_imported_jobs() -> int:
    nr_jobs: int = 0

    if args.continue_previous_job:
        nr_jobs += simulate_load_data_from_existing_run_folders([args.continue_previous_job])

    return nr_jobs

def load_existing_job_data_into_ax_client() -> None:
    nr_of_imported_jobs = get_nr_of_imported_jobs()
    set_nr_inserted_jobs(NR_INSERTED_JOBS + nr_of_imported_jobs)

def parse_parameter_type_error(_error_message: Union[str, None]) -> Optional[dict]:
    if not _error_message:
        return None

    error_message: str = str(_error_message)
    try:
        _pattern: str = r"Value for parameter (?P<parameter_name>\w+): .*? is of type <class '(?P<current_type>\w+)'>, expected\s*<class '(?P<expected_type>\w+)'>."
        match = re.search(_pattern, error_message)

        assert match is not None, "Pattern did not match the error message."

        parameter_name = match.group("parameter_name")
        current_type = match.group("current_type")
        expected_type = match.group("expected_type")

        assert parameter_name is not None, "Parameter name not found in the error message."
        assert current_type is not None, "Current type not found in the error message."
        assert expected_type is not None, "Expected type not found in the error message."

        return {
            "parameter_name": parameter_name,
            "current_type": current_type,
            "expected_type": expected_type
        }
    except AssertionError as e:
        print_debug(f"Assertion Error in parse_parameter_type_error: {e}")
        return None

def try_convert(value: Any) -> Any:
    try:
        if '.' in value or 'e' in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        return value

def parse_csv(csv_path: str) -> Tuple[List, List]:
    arm_params_list = []
    results_list = []

    if os.path.exists(csv_path):
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                arm_params = {}
                results = {}

                for col, value in row.items():
                    if col in special_col_names or col.startswith("OO_Info_"):
                        continue

                    if col in arg_result_names:
                        results[col] = try_convert(value)
                    else:
                        arm_params[col] = try_convert(value)

                arm_params_list.append(arm_params)
                results_list.append(results)

    return arm_params_list, results_list

def get_generation_node_for_index(
    this_csv_file_path: str,
    arm_params_list: List[Dict[str, Any]],
    results_list: List[Dict[str, Any]],
    index: int,
    __status: Any,
    base_str: Optional[str]
) -> str:
    __status.update(f"{base_str}: Getting generation node")
    try:
        if not _get_generation_node_for_index_index_valid(index, arm_params_list, results_list):
            return "MANUAL"

        target_arm_params = arm_params_list[index]
        target_result = results_list[index]

        __status.update(f"{base_str}: Getting generation node and combining dictionaries")
        target_combined = _get_generation_node_for_index_combine_dicts(target_arm_params, target_result)

        __status.update(f"{base_str}: Getting generation node and finding index for generation node")
        generation_node = _get_generation_node_for_index_find_generation_node(this_csv_file_path, target_combined)

        __status.update(f"{base_str}: Got generation node")

        return generation_node
    except Exception as e:
        print(f"Error while get_generation_node_for_index: {e}")
        return "MANUAL"

def _get_generation_node_for_index_index_valid(
    index: int,
    arm_params_list: List[Dict[str, Any]],
    results_list: List[Dict[str, Any]]
) -> bool:
    return 0 <= index < len(arm_params_list) and index < len(results_list)

def _get_generation_node_for_index_combine_dicts(
    dict1: Dict[str, Any],
    dict2: Dict[str, Any]
) -> Dict[str, Any]:
    combined = {}
    combined.update(dict1)
    combined.update(dict2)
    return combined

def _get_generation_node_for_index_find_generation_node(
    csv_file_path: str,
    target_combined: Dict[str, Any]
) -> str:
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        if reader.fieldnames is None or "generation_node" not in reader.fieldnames:
            return "MANUAL"

        for row in reader:
            if _get_generation_node_for_index_row_matches(row, target_combined):
                return row["generation_node"]

    return "MANUAL"

def _get_generation_node_for_index_row_matches(
    row: Dict[str, str],
    target_combined: Dict[str, Any]
) -> bool:
    for key, val in target_combined.items():
        row_val = row.get(key)
        if row_val is None:
            return False

        if isinstance(val, (int, float)):
            if not _get_generation_node_for_index_floats_match(val, row_val):
                return False
        else:
            if str(val) != row_val:
                return False

    return True

def _get_generation_node_for_index_floats_match(
    val: Union[int, float],
    row_val_str: str,
    tolerance: float = 1e-8
) -> bool:
    try:
        row_val_num = float(row_val_str)
    except ValueError:
        return False
    return abs(row_val_num - val) <= tolerance

def validate_and_convert_params_for_jobs_from_csv(arm_params: Dict) -> Dict:
    corrected_params: Dict[Any, Any] = {}

    if experiment_parameters is not None:
        for param in experiment_parameters:
            name = param["name"]
            expected_type = param.get("value_type", "str")

            if name not in arm_params:
                continue

            value = arm_params[name]

            try:
                if param["type"] == "range":
                    if expected_type == "int":
                        corrected_params[name] = int(value)
                    elif expected_type == "float":
                        corrected_params[name] = float(value)
                elif param["type"] == "choice":
                    corrected_params[name] = str(value)
            except (ValueError, TypeError):
                corrected_params[name] = None

    return corrected_params

def insert_jobs_from_csv(this_csv_file_path: str) -> None:
    with spinner(f"Inserting job into CSV from {this_csv_file_path}") as __status:
        this_csv_file_path = normalize_path(this_csv_file_path)

        if not helpers.file_exists(this_csv_file_path):
            print_red(f"--load_data_from_existing_jobs: Cannot find {this_csv_file_path}")
            return

        arm_params_list, results_list = parse_csv(this_csv_file_path)
        insert_jobs_from_lists(this_csv_file_path, arm_params_list, results_list, __status)

def normalize_path(file_path: str) -> str:
    return file_path.replace("//", "/")

def insert_jobs_from_lists(csv_path: str, arm_params_list: Any, results_list: Any, __status: Any) -> None:
    cnt = 0
    err_msgs = []

    for i, (arm_params, result) in enumerate(zip(arm_params_list, results_list)):
        base_str = f"[bold green]Loading job {i}/{len(results_list)} from {csv_path} into ax_client, result: {result}"
        __status.update(base_str)

        if not args.worker_generator_path:
            arm_params = validate_and_convert_params_for_jobs_from_csv(arm_params)

        cnt = try_insert_job(csv_path, arm_params, result, i, arm_params_list, results_list, __status, base_str, cnt, err_msgs)

    summarize_insertions(csv_path, cnt)
    update_global_job_counters(cnt)

def try_insert_job(csv_path: str, arm_params: Dict, result: Any, i: int, arm_params_list: Any, results_list: Any, __status: Any, base_str: Optional[str], cnt: int, err_msgs: Optional[Union[str, list[str]]]) -> int:
    try:
        gen_node_name = get_generation_node_for_index(csv_path, arm_params_list, results_list, i, __status, base_str)

        if not result:
            print_yellow("Encountered job without a result")
            return cnt

        if insert_job_into_ax_client(arm_params, result, gen_node_name, __status, base_str):
            cnt += 1
            print_debug(f"Inserted one job from {csv_path}, arm_params: {arm_params}, results: {result}")
        else:
            print_red(f"Failed to insert one job from {csv_path}, arm_params: {arm_params}, results: {result}")

    except ValueError as e:
        err_msg = (
            f"Failed to insert job(s) from {csv_path} into ax_client. "
            f"This can happen when the csv file has different parameters or results as the main job one's "
            f"or other imported jobs. Error: {e}"
        )
        if err_msg not in err_msgs:
            print_red(err_msg)
            err_msgs.append(err_msg)

    return cnt

def summarize_insertions(csv_path: str, cnt: int) -> None:
    if cnt == 0:
        return
    if cnt == 1:
        print_yellow(f"Inserted one job from {csv_path}")
    else:
        print_yellow(f"Inserted {cnt} jobs from {csv_path}")

def update_global_job_counters(cnt: int) -> None:
    if not args.worker_generator_path:
        set_max_eval(max_eval + cnt)
        set_nr_inserted_jobs(NR_INSERTED_JOBS + cnt)

def __insert_job_into_ax_client__update_status(__status: Optional[Any], base_str: Optional[str], new_text: str) -> None:
    if __status and base_str:
        __status.update(f"{base_str}: {new_text}")

def __insert_job_into_ax_client__check_ax_client() -> None:
    if ax_client is None or not ax_client:
        _fatal_error("insert_job_into_ax_client: ax_client was not defined where it should have been", 101)

def __insert_job_into_ax_client__attach_trial(arm_params: dict) -> Tuple[Any, int]:
    new_trial = ax_client.attach_trial(arm_params)
    if not isinstance(new_trial, tuple) or len(new_trial) < 2:
        raise RuntimeError("attach_trial didn't return the expected tuple")
    return new_trial

def __insert_job_into_ax_client__get_trial(trial_idx: int) -> Any:
    trial = ax_client.experiment.trials.get(trial_idx)
    if trial is None:
        raise RuntimeError(f"Trial with index {trial_idx} not found")
    return trial

def __insert_job_into_ax_client__create_generator_run(arm_params: dict, trial_idx: int, new_job_type: str) -> GeneratorRun:
    arm = Arm(parameters=arm_params, name=f'{trial_idx}_0')
    return GeneratorRun(arms=[arm], generation_node_name=new_job_type)

def __insert_job_into_ax_client__complete_trial_if_result(trial_idx: int, result: dict, __status: Optional[Any], base_str: Optional[str]) -> None:
    if f"{result}" != "":
        __insert_job_into_ax_client__update_status(__status, base_str, "Completing trial")
        is_ok = True

        for keyname in result.keys():
            if result[keyname] == "":
                is_ok = False

        if is_ok:
            ax_client.complete_trial(trial_index=trial_idx, raw_data=result)
            __insert_job_into_ax_client__update_status(__status, base_str, "Completed trial")
        else:
            print_debug("Empty job encountered")
    else:
        __insert_job_into_ax_client__update_status(__status, base_str, "Found trial without result. Not adding it.")

def __insert_job_into_ax_client__save_results_if_needed(__status: Optional[Any], base_str: Optional[str]) -> None:
    if not args.worker_generator_path:
        __insert_job_into_ax_client__update_status(__status, base_str, f"Saving {RESULTS_CSV_FILENAME}")
        save_results_csv()
        __insert_job_into_ax_client__update_status(__status, base_str, f"Saved {RESULTS_CSV_FILENAME}")

def __insert_job_into_ax_client__handle_type_error(e: Exception, arm_params: dict) -> bool:
    parsed_error = parse_parameter_type_error(e)
    if parsed_error is not None:
        param = parsed_error["parameter_name"]
        expected_type = parsed_error["expected_type"]
        current_type = parsed_error["current_type"]

        if expected_type == "int" and type(arm_params[param]).__name__ != "int":
            print_yellow(f"converted parameter {param} type {current_type} to {expected_type}")
            arm_params[param] = int(arm_params[param])
        elif expected_type == "float" and type(arm_params[param]).__name__ != "float":
            print_yellow(f"converted parameter {param} type {current_type} to {expected_type}")
            arm_params[param] = float(arm_params[param])
        return True

    print_red("Could not parse error while trying to insert_job_into_ax_client")
    return False

def insert_job_into_ax_client(
    arm_params: dict,
    result: dict,
    new_job_type: str = "MANUAL",
    __status: Optional[Any] = None,
    base_str: Optional[str] = None
) -> bool:
    __insert_job_into_ax_client__check_ax_client()

    done_converting = False
    while not done_converting:
        try:
            __insert_job_into_ax_client__update_status(__status, base_str, "Checking ax client")
            if ax_client is None:
                return False

            __insert_job_into_ax_client__update_status(__status, base_str, "Attaching new trial")
            _, new_trial_idx = __insert_job_into_ax_client__attach_trial(arm_params)

            __insert_job_into_ax_client__update_status(__status, base_str, "Getting new trial")
            trial = __insert_job_into_ax_client__get_trial(new_trial_idx)
            __insert_job_into_ax_client__update_status(__status, base_str, "Got new trial")

            __insert_job_into_ax_client__update_status(__status, base_str, "Creating new arm")
            manual_generator_run = __insert_job_into_ax_client__create_generator_run(arm_params, new_trial_idx, new_job_type)
            trial._generator_run = manual_generator_run
            fool_linter(trial._generator_run)

            __insert_job_into_ax_client__complete_trial_if_result(new_trial_idx, result, __status, base_str)
            done_converting = True

            __insert_job_into_ax_client__save_results_if_needed(__status, base_str)
            return True

        except ax.exceptions.core.UnsupportedError as e:
            if not __insert_job_into_ax_client__handle_type_error(e, arm_params):
                break

    return False

def get_first_line_of_file(file_paths: List[str]) -> str:
    first_line: str = ""
    if len(file_paths):
        first_file_as_string: str = ""
        try:
            first_file_as_string = get_file_as_string(file_paths[0])
            if isinstance(first_file_as_string, str) and first_file_as_string.strip().isprintable():
                first_line = first_file_as_string.split('\n')[0]
        except UnicodeDecodeError:
            pass

        if first_file_as_string == "":
            first_line = "#!/bin/bash"

    return first_line

def find_exec_errors(errors: List[str], file_as_string: str, file_paths: List[str]) -> List[str]:
    if "Exec format error" in file_as_string:
        current_platform = platform.machine()
        file_output = ""

        if len(file_paths):
            file_result = execute_bash_code(f"file {file_paths[0]}")
            if len(file_result) and isinstance(file_result[0], str):
                stripped_file_result = file_result[0].strip()
                file_output = f", {stripped_file_result}"

        errors.append(f"Was the program compiled for the wrong platform? Current system is {current_platform}{file_output}")

    return errors

def check_for_basic_string_errors(file_as_string: str, first_line: str, file_paths: List[str], program_code: str) -> List[str]:
    errors: List[str] = []

    if first_line and isinstance(first_line, str) and first_line.isprintable() and not first_line.startswith("#!"):
        errors.append(f"First line does not seem to be a shebang line: {first_line}")

    if "Permission denied" in file_as_string and "/bin/sh" in file_as_string:
        errors.append("Log file contains 'Permission denied'. Did you try to run the script without chmod +x?")

    errors = find_exec_errors(errors, file_as_string, file_paths)

    if "/bin/sh" in file_as_string and "not found" in file_as_string:
        errors.append("Wrong path? File not found")

    if len(file_paths) and os.stat(file_paths[0]).st_size == 0:
        errors.append(f"File in {program_code} is empty")

    if len(file_paths) == 0:
        errors.append(f"No files could be found in your program string: {program_code}")

    if "command not found" in file_as_string:
        errors.append("Some command was not found")

    return errors

def get_base_errors() -> list:
    base_errors: list = [
        "Segmentation fault",
        "Illegal division by zero",
        "OOM",
        ["Killed", "Detected kill, maybe OOM or Signal?"]
    ]

    return base_errors

def check_for_base_errors(file_as_string: str) -> list:
    errors: list = []
    for err in get_base_errors():
        if isinstance(err, list):
            if err[0] in file_as_string:
                errors.append(f"{err[0]} {err[1]}")
        elif isinstance(err, str):
            if err in file_as_string:
                errors.append(f"{err} detected")
        else:
            print_red(f"Wrong type, should be list or string, is {type(err)}")
    return errors

def get_exit_codes() -> dict:
    return {
        "3": "Command Invoked Cannot Execute - Permission problem or command is not an executable",
        "126": "Command Invoked Cannot Execute - Permission problem or command is not an executable or it was compiled for a different platform",
        "127": "Command Not Found - Usually this is returned when the file you tried to call was not found",
        "128": "Invalid Exit Argument - Exit status out of range",
        "129": "Hangup - Termination by the SIGHUP signal",
        "130": "Script Terminated by Control-C - Termination by Ctrl+C",
        "131": "Quit - Termination by the SIGQUIT signal",
        "132": "Illegal Instruction - Termination by the SIGILL signal",
        "133": "Trace/Breakpoint Trap - Termination by the SIGTRAP signal",
        "134": "Aborted - Termination by the SIGABRT signal",
        "135": "Bus Error - Termination by the SIGBUS signal",
        "136": "Floating Point Exception - Termination by the SIGFPE signal",
        "137": "Out of Memory - Usually this is done by the SIGKILL signal. May mean that the job has run out of memory",
        "138": "Killed by SIGUSR1 - Termination by the SIGUSR1 signal",
        "139": "Segmentation Fault - Usually this is done by the SIGSEGV signal. May mean that the job had a segmentation fault",
        "140": "Killed by SIGUSR2 - Termination by the SIGUSR2 signal",
        "141": "Pipe Error - Termination by the SIGPIPE signal",
        "142": "Alarm - Termination by the SIGALRM signal",
        "143": "Terminated by SIGTERM - Termination by the SIGTERM signal",
        "144": "Terminated by SIGSTKFLT - Termination by the SIGSTKFLT signal",
        "145": "Terminated by SIGCHLD - Termination by the SIGCHLD signal",
        "146": "Terminated by SIGCONT - Termination by the SIGCONT signal",
        "147": "Terminated by SIGSTOP - Termination by the SIGSTOP signal",
        "148": "Terminated by SIGTSTP - Termination by the SIGTSTP signal",
        "149": "Terminated by SIGTTIN - Termination by the SIGTTIN signal",
        "150": "Terminated by SIGTTOU - Termination by the SIGTTOU signal",
        "151": "Terminated by SIGURG - Termination by the SIGURG signal",
        "152": "Terminated by SIGXCPU - Termination by the SIGXCPU signal",
        "153": "Terminated by SIGXFSZ - Termination by the SIGXFSZ signal",
        "154": "Terminated by SIGVTALRM - Termination by the SIGVTALRM signal",
        "155": "Terminated by SIGPROF - Termination by the SIGPROF signal",
        "156": "Terminated by SIGWINCH - Termination by the SIGWINCH signal",
        "157": "Terminated by SIGIO - Termination by the SIGIO signal",
        "158": "Terminated by SIGPWR - Termination by the SIGPWR signal",
        "159": "Terminated by SIGSYS - Termination by the SIGSYS signal"
    }

def check_for_non_zero_exit_codes(file_as_string: str) -> List[str]:
    errors: List[str] = []
    for r in range(1, 255):
        special_exit_codes = get_exit_codes()
        search_for_exit_code = f"Exit-Code: {r},"
        if search_for_exit_code in file_as_string:
            _error: str = f"Non-zero exit-code detected: {r}"
            if str(r) in special_exit_codes:
                _error += f" (May mean {special_exit_codes[str(r)]}, unless you used that exit code yourself or it was part of any of your used libraries or programs)"
            errors.append(_error)
    return errors

def get_python_errors() -> List[List[str]]:
    synerr: str = "Python syntax error detected. Check log file."

    return [
        ["ModuleNotFoundError", "Module not found"],
        ["ImportError", "Module not found"],
        ["SyntaxError", synerr],
        ["NameError", synerr],
        ["ValueError", synerr],
        ["TypeError", synerr],
        ["FileNotFoundError", "This can happen when you don't have absolute paths for your data, or you haven't used the SCRIPT_PATH variable. See the documentation for the run.sh file."],
        ["AssertionError", "Assertion failed"],
        ["AttributeError", "Attribute Error"],
        ["EOFError", "End of file Error"],
        ["IndexError", "Wrong index for array. Check logs"],
        ["KeyError", "Wrong key for dict"],
        ["KeyboardInterrupt", "Program was cancelled using CTRL C"],
        ["MemoryError", "Python memory error detected"],
        ["NotImplementedError", "Something was not implemented"],
        ["OSError", "Something fundamentally went wrong in your program. Maybe the disk is full or a file was not found."],
        ["OverflowError", "There was an error with float overflow"],
        ["RecursionError", "Your program had a recursion error"],
        ["ReferenceError", "There was an error with a weak reference"],
        ["RuntimeError", "Something went wrong with your program. Try checking the log."],
        ["IndentationError", "There is something wrong with the intendation of your python code. Check the logs and your code."],
        ["TabError", "You used tab instead of spaces in your code"],
        ["SystemError", "Some error SystemError was found. Check the log."],
        ["UnicodeError", "There was an error regarding unicode texts or variables in your code"],
        ["ZeroDivisionError", "Your program tried to divide by zero and crashed"],
        ["error: argument", "Wrong argparse argument"],
        ["error: unrecognized arguments", "Wrong argparse argument"],
        ["CUDNN_STATUS_INTERNAL_ERROR", "Cuda had a problem. Try to delete ~/.nv and try again."],
        ["CUDNN_STATUS_NOT_INITIALIZED", "Cuda had a problem. Try to delete ~/.nv and try again."],
        ["BrokenPipeError", "Broken pipe: This usually happens when piping output to a process that closes early (e.g., head)"],
        ["ConnectionError", "Network connection failed. Check your internet or the remote server status."],
        ["TimeoutError", "An operation took too long to complete. Could be a network or file issue."],
        ["PermissionError", "You don't have permission to access this file or resource."],
        ["IsADirectoryError", "A directory was used where a file was expected."],
        ["NotADirectoryError", "A file was used where a directory was expected."],
        ["StopIteration", "An iterator was exhausted. Happens e.g. in loops over generators."],
        ["UnboundLocalError", "A local variable was referenced before it was assigned."],
        ["FloatingPointError", "A floating-point error occurred. This is rare and depends on system config."],
        ["json.decoder.JSONDecodeError", "Invalid JSON format. Check your JSON syntax."],
        ["SystemExit", "The program requested a system exit."],
        ["FloatingPointError", "A floating-point calculation failed fatally."],
        ["BrokenProcessPool", "A subprocess died unexpectedly."],
        ["zlib.error", "Compression/decompression failed (zlib)."],
        ["binascii.Error", "Binary/ASCII conversion failed. Often due to malformed base64 or hex."],
        ["SSL.SSLError", "A fatal SSL error occurred. Possibly a certificate or protocol problem."],
        ["socket.gaierror", "Address-related error in socket connection (e.g., DNS failure)."],
        ["socket.timeout", "Socket operation timed out fatally."],
        ["http.client.RemoteDisconnected", "The remote host closed the connection unexpectedly."],
        ["multiprocessing.ProcessError", "A multiprocessing error occurred, possibly a crash."],
        ["cudaErrorMemoryAllocation", "CUDA ran out of memory. Consider reducing batch size or model size."],
        ["cudaErrorIllegalAddress", "Illegal memory access by CUDA kernel. Could be a bug in kernel or index out-of-bounds."],
        ["cudaErrorLaunchFailure", "CUDA kernel launch failed. Often due to memory or driver issues."],
        ["cudaErrorUnknown", "Unknown fatal CUDA error. Could be hardware or driver related."],
        ["CUBLAS_STATUS_ALLOC_FAILED", "cuBLAS could not allocate GPU memory."],
        ["CUBLAS_STATUS_INTERNAL_ERROR", "cuBLAS encountered a fatal internal error."],
        ["CUBLAS_STATUS_EXECUTION_FAILED", "cuBLAS failed during execution. Usually a hardware/driver issue."],
        ["OpenBLAS blas_thread_init: pthread_create failed", "OpenBLAS failed to initialize threads. Could be OS/thread limit."],
        ["libc++abi.dylib: terminating", "The C++ runtime aborted due to a fatal exception."],
        ["terminate called after throwing an instance of", "Uncaught C++ exception crashed the program."],
        ["Segmentation fault", "Your program tried to access invalid memory. Often due to C extensions or CUDA."],
        ["Bus error", "Hardware-level memory error. Possibly caused by alignment issues or failed RAM."],
        ["Illegal instruction", "The CPU tried to execute an invalid or unsupported instruction. Possible binary mismatch."],
        ["Killed", "Your process was forcefully killed (e.g., OOM killer or SIGKILL)."],
        ["OOMKilled", "Out-of-memory killer terminated your process. Try reducing memory usage."],
        ["core dumped", "The program crashed and dumped a core file for debugging."],
        ["fatal error", "A general fatal error occurred (non-specific). See logs."],
    ]

def get_first_line_of_file_that_contains_string(stdout_path: str, s: str) -> str:
    stdout_path = check_alternate_path(stdout_path)
    if not os.path.exists(stdout_path):
        print_debug(f"File {stdout_path} not found")
        return ""

    f: str = get_file_as_string(stdout_path)

    lines: str = ""
    get_lines_until_end: bool = False

    for line in f.split("\n"):
        if s in line:
            if get_lines_until_end:
                lines += line
            else:
                line = line.strip()
                if line.endswith("(") and "raise" in line:
                    get_lines_until_end = True
                    lines += line
                else:
                    return line
    if lines != "":
        return lines

    return ""

def check_for_python_errors(stdout_path: str, file_as_string: str) -> List[str]:
    stdout_path = check_alternate_path(stdout_path)
    errors: List[str] = []

    for search_array in get_python_errors():
        search_for_string = search_array[0]
        search_for_error = search_array[1]

        if search_for_string in file_as_string:
            error_line = get_first_line_of_file_that_contains_string(stdout_path, search_for_string)
            if error_line:
                errors.append(error_line)
            else:
                errors.append(search_for_error)

    return errors

def get_errors_from_outfile(stdout_path: str) -> List[str]:
    stdout_path = check_alternate_path(stdout_path)
    file_as_string = get_file_as_string(stdout_path)

    program_code = get_program_code_from_out_file(stdout_path)
    file_paths = find_file_paths(program_code)

    first_line: str = get_first_line_of_file(file_paths)

    errors: List[str] = []

    for resname in arg_result_names:
        if f"{resname}: None" in file_as_string:
            errors.append("Got no result.")

            new_errors = check_for_basic_string_errors(file_as_string, first_line, file_paths, program_code)
            for n in new_errors:
                errors.append(n)

            new_errors = check_for_base_errors(file_as_string)
            for n in new_errors:
                errors.append(n)

            new_errors = check_for_non_zero_exit_codes(file_as_string)
            for n in new_errors:
                errors.append(n)

            new_errors = check_for_python_errors(stdout_path, file_as_string)
            for n in new_errors:
                errors.append(n)

        if f"{resname}: nan" in file_as_string:
            errors.append(f"The string '{resname}: nan' appeared. This may indicate the vanishing-gradient-problem, or a learning rate that is too high (if you are training a neural network).")

    return errors

def print_outfile_analyzed(stdout_path: str) -> None:
    stdout_path = check_alternate_path(stdout_path)
    errors = get_errors_from_outfile(stdout_path)

    _strs: List[str] = []
    j: int = 0

    if len(errors):
        if j == 0:
            _strs.append("")
        _strs.append(f"Out file {stdout_path} contains potential errors:\n")
        program_code = get_program_code_from_out_file(stdout_path)
        if program_code:
            _strs.append(program_code)

        for e in errors:
            _strs.append(f"- {e}\n")

        j = j + 1

    out_files_string: str = "\n".join(_strs)

    if len(_strs):
        try:
            with open(get_current_run_folder('evaluation_errors.log'), mode="a+", encoding="utf-8") as error_file:
                error_file.write(out_files_string)
        except Exception as e:
            print_debug(f"Error occurred while writing to evaluation_errors.log: {e}")

        print_red(out_files_string)

def get_parameters_from_outfile(stdout_path: str) -> Union[None, dict, str]:
    stdout_path = check_alternate_path(stdout_path)
    try:
        with open(stdout_path, mode='r', encoding="utf-8") as file:
            for line in file:
                if line.lower().startswith("parameters: "):
                    params = line.split(":", 1)[1].strip()
                    params = json.loads(params)
                    return params
    except FileNotFoundError:
        if not args.tests:
            original_print(f"get_parameters_from_outfile: The file '{stdout_path}' was not found.")
    except Exception as e:
        print(f"get_parameters_from_outfile: There was an error: {e}")

    return None

def get_hostname_from_outfile(stdout_path: Optional[str]) -> Optional[str]:
    if stdout_path is None:
        return None
    try:
        with open(stdout_path, mode='r', encoding="utf-8") as file:
            for line in file:
                if line.lower().startswith("hostname: "):
                    hostname = line.split(":", 1)[1].strip()
                    return hostname
        return None
    except FileNotFoundError:
        original_print(f"The file '{stdout_path}' was not found.")
        return None
    except Exception as e:
        print(f"There was an error: {e}")
        return None

def add_to_global_error_list(msg: str) -> None:
    crf = get_current_run_folder()

    if crf is not None and crf != "":
        error_file_path = f'{crf}/result_errors.log'

        if os.path.exists(error_file_path):
            with open(error_file_path, mode='r', encoding="utf-8") as file:
                errors = file.readlines()
            errors = [error.strip() for error in errors]
            if msg not in errors:
                with open(error_file_path, mode='a', encoding="utf-8") as file:
                    file.write(f"{msg}\n")
        else:
            with open(error_file_path, mode='w', encoding="utf-8") as file:
                file.write(f"{msg}\n")

def read_errors_from_file() -> list:
    error_file_path = get_current_run_folder('result_errors.log')
    if os.path.exists(error_file_path):
        with open(error_file_path, mode='r', encoding="utf-8") as file:
            errors = file.readlines()
        return [error.strip() for error in errors]
    return []

def mark_trial_as_failed(trial_index: int, _trial: Any) -> None:
    print_debug(f"Marking trial {_trial} as failed")
    try:
        if not ax_client:
            _fatal_error("mark_trial_as_failed: ax_client is not defined", 101)

            return None

        ax_client.log_trial_failure(trial_index=trial_index)
        _trial.mark_failed(unsafe=True)
    except ValueError as e:
        print_debug(f"mark_trial_as_failed error: {e}")

    return None

def _finish_job_core_helper_check_valid_result(result: Union[None, list, int, float, tuple]) -> bool:
    possible_val_not_found_values = [
        VAL_IF_NOTHING_FOUND,
        -VAL_IF_NOTHING_FOUND,
        -99999999999999997168788049560464200849936328366177157906432,
        99999999999999997168788049560464200849936328366177157906432
    ]
    values_to_check = result if isinstance(result, list) else [result]
    return result is not None and all(r not in possible_val_not_found_values for r in values_to_check)

def _finish_job_core_helper_complete_trial(trial_index: int, raw_result: dict) -> None:
    if ax_client is None:
        print_red("ax_client is not defined in _finish_job_core_helper_complete_trial")
        return None

    try:
        print_debug(f"Completing trial: {trial_index} with result: {raw_result}...")
        ax_client.complete_trial(trial_index=trial_index, raw_data=raw_result)
        print_debug(f"Completing trial: {trial_index} with result: {raw_result}... Done!")
    except ax.exceptions.core.UnsupportedError as e:
        if f"{e}":
            print_debug(f"Completing trial: {trial_index} with result: {raw_result} after failure. Trying to update trial...")
            ax_client.update_trial_data(trial_index=trial_index, raw_data=raw_result)
            print_debug(f"Completing trial: {trial_index} with result: {raw_result} after failure... Done!")
        else:
            _fatal_error(f"Error completing trial: {e}", 234)

    return None

def _finish_job_core_helper_mark_success(_trial: ax.core.trial.Trial, result: Union[float, int, tuple]) -> None:
    print_debug(f"Marking trial {_trial} as completed")
    _trial.mark_completed(unsafe=True)

    succeeded_jobs(1)

    progressbar_description(f"new result: {result}")
    update_progress_bar(1)

    save_results_csv()

def _finish_job_core_helper_mark_failure(job: Any, trial_index: int, _trial: Any) -> None:
    if ax_client is None:
        print_red("ax_client is not defined in _finish_job_core_helper_mark_failure")
        return None

    print_debug(f"Counting job {job} as failed, because the result is {job.result() if job else 'None'}")
    if job:
        try:
            progressbar_description("job_failed")
            ax_client.log_trial_failure(trial_index=trial_index)
            _trial.mark_failed(unsafe=True)
        except Exception as e:
            print_red(f"\nERROR while trying to mark job as failure: {e}")
        job.cancel()
        orchestrate_job(job, trial_index)

    mark_trial_as_failed(trial_index, _trial)
    failed_jobs(1)

    return None

def finish_job_core(job: Any, trial_index: int, this_jobs_finished: int) -> int:
    die_for_debug_reasons()

    result = job.result()
    print_debug(f"finish_job_core: trial-index: {trial_index}, job.result(): {result}, state: {state_from_job(job)}")

    raw_result = result
    result_keys = list(result.keys())
    result = result[result_keys[0]]
    this_jobs_finished += 1

    if ax_client:
        _trial = ax_client.get_trial(trial_index)

        if _finish_job_core_helper_check_valid_result(result):
            _finish_job_core_helper_complete_trial(trial_index, raw_result)

            try:
                _finish_job_core_helper_mark_success(_trial, result)

                if len(arg_result_names) > 1 and count_done_jobs() > 1 and not job_calculate_pareto_front(get_current_run_folder(), True):
                    print_red("job_calculate_pareto_front post job failed")
            except Exception as e:
                print(f"ERROR in line {get_line_info()}: {e}")
        else:
            _finish_job_core_helper_mark_failure(job, trial_index, _trial)
    else:
        _fatal_error("ax_client could not be found or used", 9)

    print_debug(f"finish_job_core: removing job {job}, trial_index: {trial_index}")
    global_vars["jobs"].remove((job, trial_index))

    force_live_share()

    return this_jobs_finished

def _finish_previous_jobs_helper_handle_failed_job(job: Any, trial_index: int) -> None:
    if ax_client is None:
        print_red("ax_client is not defined in _finish_job_core_helper_mark_failure")
        return None

    if job:
        try:
            progressbar_description("job_failed")
            _trial = ax_client.get_trial(trial_index)
            ax_client.log_trial_failure(trial_index=trial_index)
            mark_trial_as_failed(trial_index, _trial)
        except Exception as e:
            print(f"ERROR in line {get_line_info()}: {e}")
        job.cancel()
        orchestrate_job(job, trial_index)

    failed_jobs(1)
    print_debug(f"finish_previous_jobs: removing job {job}, trial_index: {trial_index}")

    with global_vars_jobs_lock:
        print_debug(f"finish_previous_jobs: removing job {job}, trial_index: {trial_index}")
        global_vars["jobs"].remove((job, trial_index))

    return None

def _finish_previous_jobs_helper_handle_exception(job: Any, trial_index: int, error: Exception) -> int:
    if "None for metric" in str(error):
        print_red(
            f"\n⚠ It seems like the program that was about to be run didn't have 'RESULT: <FLOAT>' in it's output string."
            f"\nError: {error}\nJob-result: {job.result()}"
        )
    else:
        print_red(f"\n⚠ {error}")

    _finish_previous_jobs_helper_handle_failed_job(job, trial_index)
    return 1

def _finish_previous_jobs_helper_process_job(job: Any, trial_index: int, this_jobs_finished: int) -> int:
    try:
        this_jobs_finished = finish_job_core(job, trial_index, this_jobs_finished)

        if args.prettyprint:
            pretty_print_job_output(job)
    except (SignalINT, SignalUSR, SignalCONT) as e:
        print_red(f"Cancelled finish_job_core: {e}")
    except (FileNotFoundError, submitit.core.utils.UncompletedJobError, ax.exceptions.core.UserInputError) as error:
        this_jobs_finished += _finish_previous_jobs_helper_handle_exception(job, trial_index, error)
    return this_jobs_finished

def _finish_previous_jobs_helper_check_and_process(job: Any, trial_index: int, this_jobs_finished: int) -> int:
    if job is None:
        print_debug(f"finish_previous_jobs: job {job} is None")
        return this_jobs_finished

    if job.done() or type(job) in [LocalJob, DebugJob]:
        this_jobs_finished = _finish_previous_jobs_helper_process_job(job, trial_index, this_jobs_finished)
    else:
        if not isinstance(job, SlurmJob):
            print_debug(f"finish_previous_jobs: job was neither done, nor LocalJob nor DebugJob, but {job}")

    return this_jobs_finished

def _finish_previous_jobs_helper_wrapper(__args: Tuple[Any, int]) -> int:
    job, trial_index = __args
    return _finish_previous_jobs_helper_check_and_process(job, trial_index, 0)

def finish_previous_jobs(new_msgs: List[str] = []) -> None:
    global JOBS_FINISHED

    if not ax_client:
        _fatal_error("ax_client failed", 101)

        return None

    this_jobs_finished = 0

    jobs_copy = global_vars["jobs"][:]

    if len(jobs_copy) > 0:
        print_debug(f"jobs in finish_previous_jobs: {jobs_copy}")

    finishing_jobs_start_time = time.time()

    with ThreadPoolExecutor() as finish_job_executor:
        futures = [finish_job_executor.submit(_finish_previous_jobs_helper_wrapper, (job, trial_index)) for job, trial_index in jobs_copy]

        for future in as_completed(futures):
            try:
                this_jobs_finished += future.result()
            except Exception as e:
                print_red(f"⚠ Exception in parallel job handling: {e}")

    finishing_jobs_end_time = time.time()

    finishing_jobs_runtime = finishing_jobs_end_time - finishing_jobs_start_time

    print_debug(f"Finishing jobs took {finishing_jobs_runtime} second(s)")

    if this_jobs_finished > 0:
        save_results_csv()
        save_checkpoint()
        progressbar_description([*new_msgs, f"finished {this_jobs_finished} {'job' if this_jobs_finished == 1 else 'jobs'}"])

    JOBS_FINISHED += this_jobs_finished
    clean_completed_jobs()
    return None

def get_alt_path_for_orchestrator(stdout_path: str) -> Optional[str]:
    stdout_path = check_alternate_path(stdout_path)
    alt_path = None
    if stdout_path.endswith(".err"):
        alt_path = stdout_path[:-4] + ".out"
    elif stdout_path.endswith(".out"):
        alt_path = stdout_path[:-4] + ".err"

    return alt_path

def check_orchestrator(stdout_path: str, trial_index: int) -> Optional[List[str]]:
    stdout_path = check_alternate_path(stdout_path)
    if not orchestrator or "errors" not in orchestrator:
        return []

    stdout = _check_orchestrator_read_stdout_with_fallback(stdout_path, trial_index)
    if stdout is None:
        return None

    return _check_orchestrator_find_behaviors(stdout, orchestrator["errors"])

def _check_orchestrator_read_stdout_with_fallback(stdout_path: str, trial_index: int) -> Optional[str]:
    stdout_path = check_alternate_path(stdout_path)
    try:
        return Path(stdout_path).read_text("UTF-8")
    except FileNotFoundError:
        alt_path = get_alt_path_for_orchestrator(stdout_path)

        if alt_path and Path(alt_path).exists():
            try:
                return Path(alt_path).read_text("UTF-8")
            except FileNotFoundError:
                return None

        _check_orchestrator_register_missing_file(stdout_path, trial_index)
        return None

def _check_orchestrator_register_missing_file(stdout_path: str, trial_index: int) -> None:
    stdout_path = check_alternate_path(stdout_path)
    if stdout_path not in ORCHESTRATE_TODO:
        ORCHESTRATE_TODO[stdout_path] = trial_index
        print_red(f"File not found: {stdout_path}, will try again later")
    else:
        print_red(f"File not found: {stdout_path}, not trying again")

def _check_orchestrator_find_behaviors(stdout: str, errors: List[Dict[str, Any]]) -> List[str]:
    behaviors: List[str] = []
    stdout_lower = stdout.lower()

    for error in errors:
        name = error.get("name", "")
        match_strings = error.get("match_strings", [])
        behavior = error.get("behavior", "")

        for match_string in match_strings:
            if match_string.lower() in stdout_lower:
                if behavior not in behaviors:
                    print_debug(f"Appending behavior {behavior}, orchestrator-error-name: {name}")
                    behaviors.append(behavior)

    return behaviors

def get_exit_code_from_stderr_or_stdout_path(stderr_path: str, stdout_path: str) -> Optional[int]:
    def extract_last_exit_code(path: str) -> Optional[int]:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                matches = re.findall(r"EXIT_CODE:\s*(-?\d+)", f.read())
                if matches:
                    return int(matches[-1])
        except Exception:
            pass
        return None

    code = extract_last_exit_code(stdout_path)
    if code is not None:
        return code

    return extract_last_exit_code(stderr_path)

def pretty_print_job_output(job: Job) -> None:
    stdout_path = get_stderr_or_stdout_from_job(job, "stdout")
    stderr_path = get_stderr_or_stdout_from_job(job, "stderr")
    exit_code = get_exit_code_from_stderr_or_stdout_path(stderr_path, stdout_path)
    result = get_results_with_occ(get_file_as_string(stdout_path))

    pretty_process_output(stdout_path, stderr_path, exit_code, result)

def get_stderr_or_stdout_from_job(job: Job, path_type: str) -> str:
    if path_type == "stderr":
        _path = str(job.paths.stderr.resolve())
    elif path_type == "stdout":
        _path = str(job.paths.stdout.resolve())
    else:
        print_red(f"ERROR: path_type {path_type} was neither stdout nor stderr. Using stdout")
        _path = str(job.paths.stdout.resolve())

    _path = _path.replace('\n', ' ').replace('\r', '')
    _path = _path.rstrip('\r\n')
    _path = _path.rstrip('\n')
    _path = _path.rstrip('\r')
    _path = _path.rstrip(' ')

    return _path

def orchestrate_job(job: Job, trial_index: int) -> None:
    stdout_path = get_stderr_or_stdout_from_job(job, "stdout")
    stderr_path = get_stderr_or_stdout_from_job(job, "stderr")

    print_outfile_analyzed(stdout_path)
    print_outfile_analyzed(stderr_path)

    _orchestrate(stdout_path, trial_index)
    _orchestrate(stderr_path, trial_index)

    orchestrate_todo_copy = ORCHESTRATE_TODO
    for todo_stdout_file in orchestrate_todo_copy.keys():
        old_behavs = check_orchestrator(todo_stdout_file, ORCHESTRATE_TODO[todo_stdout_file])
        if old_behavs is not None:
            del ORCHESTRATE_TODO[todo_stdout_file]

def is_already_in_defective_nodes(hostname: str) -> bool:
    file_path = os.path.join(get_current_run_folder(), "state_files", "defective_nodes")

    makedirs(os.path.dirname(file_path))

    if not os.path.isfile(file_path):
        print_red(f"is_already_in_defective_nodes: Error: The file {file_path} does not exist.")
        return False

    try:
        with open(file_path, mode="r", encoding="utf-8") as file:
            for line in file:
                if line.strip() == hostname:
                    return True
    except Exception as e:
        print_red(f"is_already_in_defective_nodes: Error reading the file {file_path}: {e}")
        return False

    return False

def submit_new_job(parameters: Union[dict, str], trial_index: int) -> Any:
    print_debug(f"Submitting new job for trial_index {trial_index}, parameters {parameters}")

    start = time.time()

    new_job = submitit_executor.submit(evaluate, {"params": parameters, "trial_idx": trial_index, "submit_time": int(time.time())})

    elapsed = time.time() - start

    print_debug(f"Done submitting new job, took {elapsed} seconds")

    return new_job

def orchestrator_start_trial(parameters: Union[dict, str], trial_index: int) -> None:
    if submitit_executor and ax_client:
        new_job = submit_new_job(parameters, trial_index)
        submitted_jobs(1)

        _trial = ax_client.get_trial(trial_index)

        try:
            _trial.mark_staged(unsafe=True)
        except Exception as e:
            print_debug(f"orchestrator_start_trial: error {e}")
        _trial.mark_running(unsafe=True, no_runner_required=True)

        print_debug(f"orchestrator_start_trial: appending job {new_job} to global_vars['jobs'], trial_index: {trial_index}")
        global_vars["jobs"].append((new_job, trial_index))
    else:
        _fatal_error("submitit_executor or ax_client could not be found properly", 9)

def handle_exclude_node(stdout_path: str, hostname_from_out_file: Union[None, str]) -> None:
    stdout_path = check_alternate_path(stdout_path)
    if hostname_from_out_file:
        if not is_already_in_defective_nodes(hostname_from_out_file):
            print_yellow(f"\nExcludeNode was triggered for node {hostname_from_out_file}")
            count_defective_nodes(None, hostname_from_out_file)
        else:
            print_yellow(f"\nExcludeNode was triggered for node {hostname_from_out_file}, but it was already in defective nodes and won't be added again")
    else:
        print_red(f"Cannot do ExcludeNode because the host could not be determined from {stdout_path}")

def handle_restart(stdout_path: str, trial_index: int) -> None:
    stdout_path = check_alternate_path(stdout_path)
    parameters = get_parameters_from_outfile(stdout_path)
    if parameters:
        orchestrator_start_trial(parameters, trial_index)
    else:
        print(f"Could not determine parameters from outfile {stdout_path} for restarting job")

def check_alternate_path(path: str) -> str:
    if os.path.exists(path):
        return path
    if path.endswith('.out'):
        alt_path = path[:-4] + '.err'
    elif path.endswith('.err'):
        alt_path = path[:-4] + '.out'
    else:
        alt_path = None
    if alt_path and os.path.exists(alt_path):
        return alt_path
    return path

def handle_restart_on_different_node(stdout_path: str, hostname_from_out_file: Union[None, str], trial_index: int) -> None:
    stdout_path = check_alternate_path(stdout_path)
    if hostname_from_out_file:
        if not is_already_in_defective_nodes(hostname_from_out_file):
            print_yellow(f"\nRestartOnDifferentNode was triggered for node {hostname_from_out_file}. Adding node to defective hosts list and restarting on another host.")
            count_defective_nodes(None, hostname_from_out_file)
        else:
            print_yellow(f"\nRestartOnDifferentNode was triggered for node {hostname_from_out_file}, but it was already in defective nodes. Job will only be resubmitted.")
        handle_restart(stdout_path, trial_index)
    else:
        print_red(f"Cannot do RestartOnDifferentNode because the host could not be determined from {stdout_path}")

def _orchestrate(stdout_path: str, trial_index: int) -> None:
    stdout_path = check_alternate_path(stdout_path)

    behavs = check_orchestrator(stdout_path, trial_index)

    if not behavs or behavs is None:
        return

    hostname_from_out_file = get_hostname_from_outfile(stdout_path)

    behavior_handlers = {
        "ExcludeNode": lambda: handle_exclude_node(stdout_path, hostname_from_out_file),
        "Restart": lambda: handle_restart(stdout_path, trial_index),
        "RestartOnDifferentNode": lambda: handle_restart_on_different_node(stdout_path, hostname_from_out_file, trial_index),
    }

    for behav in behavs:
        handler = behavior_handlers.get(behav)
        if handler:
            handler()
        else:
            _fatal_error(f"Orchestrator: {behav} not yet implemented!", 210)

def write_continue_run_uuid_to_file() -> None:
    if args.continue_previous_job:
        continue_dir = args.continue_previous_job

        with open(f'{continue_dir}/state_files/run_uuid', mode='r', encoding='utf-8') as f:
            continue_from_uuid = f.readline()

            write_state_file("uuid_of_continued_run", str(continue_from_uuid))

def save_state_files() -> None:
    if len(args.calculate_pareto_front_of_job) == 0:
        with spinner("Saving state files..."):
            write_state_file("joined_run_program", global_vars["joined_run_program"])
            write_state_file("experiment_name", global_vars["experiment_name"])
            write_state_file("mem_gb", str(global_vars["mem_gb"]))
            write_state_file("max_eval", str(max_eval))
            write_state_file("gpus", str(args.gpus))
            write_state_file("model", str(args.model))
            write_state_file("time", str(global_vars["_time"]))
            write_state_file("run.sh", "omniopt '" + " ".join(sys.argv[1:]) + "'")
            write_state_file("run_uuid", str(run_uuid))
            if args.external_generator:
                write_state_file("external_generator", str(args.external_generator))

            if args.follow:
                write_state_file("follow", "True")

            if args.main_process_gb:
                write_state_file("main_process_gb", str(args.main_process_gb))

            if args.force_choice_for_ranges:
                write_state_file("force_choice_for_ranges", str(args.main_process_gb))

def execute_evaluation(_params: list) -> Optional[int]:
    print_debug(f"execute_evaluation({_params})")
    trial_index, parameters, trial_counter, phase = _params
    if not ax_client:
        _fatal_error("Failed to get ax_client", 9)

        return None

    if not submitit_executor:
        _fatal_error("submitit_executor could not be found", 9)

        return None

    _trial = ax_client.get_trial(trial_index)

    def mark_trial_stage(stage: str, error_msg: str) -> None:
        try:
            getattr(_trial, stage)()
        except Exception as e:
            print_debug(f"execute_evaluation({_params}): {error_msg} with error: {e}")

    mark_trial_stage("mark_staged", "Marking the trial as staged failed")

    new_job = None

    try:
        initialize_job_environment()
        new_job = submit_new_job(parameters, trial_index)
        submitted_jobs(1)

        print_debug(f"execute_evaluation: appending job {new_job} to global_vars['jobs'], trial_index: {trial_index}")
        global_vars["jobs"].append((new_job, trial_index))

        mark_trial_stage("mark_running", "Marking the trial as running failed")
        trial_counter += 1

        progressbar_description("started new job")
    except submitit.core.utils.FailedJobError as error:
        handle_failed_job(error, trial_index, new_job)
        trial_counter += 1
    except (SignalUSR, SignalINT, SignalCONT):
        handle_exit_signal()
    except Exception as e:
        handle_generic_error(e)

    add_to_phase_counter(phase, 1)
    return trial_counter

def initialize_job_environment() -> None:
    progressbar_description("starting new job")
    set_sbatch_environment()
    exclude_defective_nodes()

def set_sbatch_environment() -> None:
    if args.reservation:
        os.environ['SBATCH_RESERVATION'] = args.reservation
    if args.account:
        os.environ['SBATCH_ACCOUNT'] = args.account

def exclude_defective_nodes() -> None:
    excluded_string: str = ",".join(count_defective_nodes())
    if len(excluded_string) > 1:
        if submitit_executor:
            submitit_executor.update_parameters(exclude=excluded_string)
        else:
            _fatal_error("submitit_executor could not be found", 9)

def handle_failed_job(error: Union[None, Exception, str], trial_index: int, new_job: Optional[Job]) -> None:
    if "QOSMinGRES" in str(error) and args.gpus == 0:
        print_red("\n⚠ It seems like, on the chosen partition, you need at least one GPU. Use --gpus=1 (or more) as parameter.")
    else:
        print_red(f"\n⚠ FAILED: {error}")

        if "CPU count per node can not be satisfied" in f"{error}":
            print_red("Cannot continue. This can happen when you have too requested much memory.")
            my_exit(144)

    if new_job is None:
        print_red("handle_failed_job: job is None")

        return None

    try:
        cancel_failed_job(trial_index, new_job)
    except Exception as e:
        print_red(f"\n⚠ Cancelling failed job FAILED: {e}")

    return None

def cancel_failed_job(trial_index: int, new_job: Job) -> None:
    print_debug("Trying to cancel job that failed")
    if new_job:
        try:
            if ax_client:
                ax_client.log_trial_failure(trial_index=trial_index)
            else:
                _fatal_error("ax_client not defined", 101)
        except Exception as e:
            print(f"ERROR in line {get_line_info()}: {e}")
        new_job.cancel()

        print_debug(f"cancel_failed_job: removing job {new_job}, trial_index: {trial_index}")
        global_vars["jobs"].remove((new_job, trial_index))
        print_debug("Removed failed job")
        save_results_csv()
    else:
        print_debug("cancel_failed_job: new_job was undefined")

def handle_exit_signal() -> None:
    print_red("\n⚠ Detected signal. Will exit.")
    end_program(False, 1)

def handle_generic_error(e: Union[Exception, str]) -> None:
    tb = traceback.format_exc()
    print(tb)
    print_red(f"\n⚠ Starting job failed with error: {e}")

def succeeded_jobs(nr: int = 0) -> int:
    return append_and_read(get_state_file_name('succeeded_jobs'), nr)

def show_debug_table_for_break_run_search(_name: str, _max_eval: Optional[int]) -> None:
    table = Table(show_header=True, header_style="bold", title=f"break_run_search for {_name}")

    headers = ["Variable", "Value"]
    table.add_column(headers[0])
    table.add_column(headers[1])

    rows = [
        ("args.max_failed_jobs", args.max_failed_jobs),
        ("succeeded_jobs()", succeeded_jobs()),
        ("submitted_jobs()", submitted_jobs()),
        ("failed_jobs()", failed_jobs()),
        ("count_done_jobs()", count_done_jobs()),
        ("_max_eval", _max_eval),
        ("progress_bar.total", progress_bar.total),
        ("NR_INSERTED_JOBS", NR_INSERTED_JOBS)
    ]

    for row in rows:
        table.add_row(str(row[0]), str(row[1]))

    console.print(table)

def break_run_search(_name: str, _max_eval: Optional[int]) -> bool:
    _ret = False

    _counted_done_jobs = count_done_jobs()
    _submitted_jobs = submitted_jobs()
    _failed_jobs = failed_jobs()

    max_failed_jobs = max_eval

    if args.max_failed_jobs is not None and args.max_failed_jobs > 0:
        max_failed_jobs = args.max_failed_jobs

    conditions = [
        (lambda: (_counted_done_jobs - _failed_jobs) >= max_eval, f"1. _counted_done_jobs {_counted_done_jobs} - (_failed_jobs {_failed_jobs}) >= max_eval {max_eval}"),
        (lambda: max_failed_jobs < _failed_jobs, f"3. max_failed_jobs {max_failed_jobs} < failed_jobs {_failed_jobs}"),
    ]

    if progress_bar is not None:
        conditions.append(
            (lambda: (_submitted_jobs - _failed_jobs) >= progress_bar.total + 1, f"2. _submitted_jobs {_submitted_jobs} - _failed_jobs {_failed_jobs} >= progress_bar.total {progress_bar.total} + 1"),
        )

    if _max_eval:
        conditions.append((lambda: succeeded_jobs() >= _max_eval + 1, f"4. succeeded_jobs() {succeeded_jobs()} >= _max_eval {_max_eval} + 1"),)
        conditions.append((lambda: _counted_done_jobs >= _max_eval, f"5. _counted_done_jobs {_counted_done_jobs} >= _max_eval {_max_eval}"),)
        conditions.append((lambda: (_submitted_jobs - _failed_jobs) >= _max_eval + 1, f"6. _submitted_jobs {_submitted_jobs} - _failed_jobs {_failed_jobs} > _max_eval {_max_eval} + 1"),)
        conditions.append((lambda: 0 >= abs(_counted_done_jobs - _max_eval - NR_INSERTED_JOBS), f"7. 0 >= abs(_counted_done_jobs {_counted_done_jobs} - _max_eval {_max_eval} - NR_INSERTED_JOBS {NR_INSERTED_JOBS})"))

    for condition_func, debug_msg in conditions:
        if condition_func():
            print_debug(f"breaking {_name}: {debug_msg}")
            _ret = True

    if args.verbose_break_run_search_table:
        show_debug_table_for_break_run_search(_name, _max_eval)

    return _ret

def _calculate_nr_of_jobs_to_get(simulated_jobs: int, currently_running_jobs: int) -> int:
    """Calculates the number of jobs to retrieve."""
    return min(
        max_eval + simulated_jobs - count_done_jobs(),
        max_eval + simulated_jobs - (submitted_jobs() - failed_jobs()),
        num_parallel_jobs - currently_running_jobs
    )

def remove_extra_spaces(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    return re.sub(r'\s+', ' ', text).strip()

def _get_trials_message(nr_of_jobs_to_get: int, full_nr_of_jobs_to_get: int, trial_durations: List[float]) -> str:
    """Generates the appropriate message for the number of trials being retrieved."""
    ret = ""
    if full_nr_of_jobs_to_get > 1:
        base_msg = f"getting new HP set #{nr_of_jobs_to_get}/{full_nr_of_jobs_to_get}"
    else:
        base_msg = "getting new HP set"

    if SYSTEM_HAS_SBATCH and not args.force_local_execution:
        ret = base_msg
    else:
        ret = f"{base_msg} (no sbatch)"

    ret = remove_extra_spaces(ret)

    if trial_durations and len(trial_durations) > 0 and full_nr_of_jobs_to_get > 1:
        avg_time = sum(trial_durations) / len(trial_durations)
        remaining = full_nr_of_jobs_to_get - nr_of_jobs_to_get + 1

        eta = avg_time * remaining

        if int(eta) > 0:
            hours = int(eta // 3600)
            minutes = int((eta % 3600) // 60)
            seconds = int(eta % 60)

            if hours > 0:
                eta_str = f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                eta_str = f"{minutes}m {seconds}s"
            else:
                eta_str = f"{seconds}s"

            ret += f" | ETA: {eta_str}"

    return ret

def has_no_post_generation_constraints_or_matches_constraints(_post_generation_constraints: list, params: dict) -> bool:
    if not _post_generation_constraints or len(_post_generation_constraints) == 0:
        return True

    for constraint in _post_generation_constraints:
        try:
            expression = constraint

            for var in params:
                expression = expression.replace(var, f"({repr(params[var])})")

            result = eval(expression, {"__builtins__": {}}, {})

            if not result:
                return False

        except Exception as e:
            print_debug(f"Error checking constraint '{constraint}': {e}")
            return False

    return True

def die_101_if_no_ax_client_or_experiment_or_gs() -> None:
    if ax_client is None:
        _fatal_error("Error: ax_client is not defined", 101)
    elif ax_client.experiment is None:
        _fatal_error("Error: ax_client.experiment is not defined", 101)
    elif global_gs is None:
        _fatal_error("Error: global_gs is not defined", 101)

def deduplicated_arm(arm: Any) -> bool:
    if arm.name in arms_by_name_for_deduplication:
        return True

    return False

def get_batched_arms(nr_of_jobs_to_get: int) -> list:
    batched_arms: list = []
    attempts = 0

    if global_gs is None:
        _fatal_error("Global generation strategy is not set. This is a bug in OmniOpt2.", 107)
        return []

    if ax_client is None:
        print_red("get_batched_arms: ax_client was None")
        return []

    # Experiment-Status laden
    load_experiment_state()

    while len(batched_arms) != nr_of_jobs_to_get:
        if attempts > args.max_attempts_for_generation:
            print_debug(f"get_batched_arms: Stopped after {attempts} attempts: could not generate enough arms "
                        f"(got {len(batched_arms)} out of {nr_of_jobs_to_get}).")
            break

        remaining = nr_of_jobs_to_get - len(batched_arms)
        print_debug(f"get_batched_arms: Attempt {attempts + 1}: requesting {remaining} more arm(s).")

        print_debug("get pending observations")
        t0 = time.time()
        pending_observations = get_pending_observation_features(experiment=ax_client.experiment)
        dt = time.time() - t0
        print_debug(f"got pending observations: {pending_observations} (took {dt:.2f} seconds)")

        print_debug("getting global_gs.gen()")
        batched_generator_run = global_gs.gen(
            experiment=ax_client.experiment,
            n=remaining,
            pending_observations=pending_observations
        )
        print_debug(f"got global_gs.gen(): {batched_generator_run}")

        depth = 0
        path = "batched_generator_run"
        while isinstance(batched_generator_run, (list, tuple)) and len(batched_generator_run) > 0:
            print_debug(f"Depth {depth}, path {path}, type {type(batched_generator_run).__name__}, length {len(batched_generator_run)}: {batched_generator_run}")
            batched_generator_run = batched_generator_run[0]
            path += "[0]"
            depth += 1

        print_debug(f"Final flat object at depth {depth}, path {path}: {batched_generator_run} (type {type(batched_generator_run).__name__})")

        print_debug("got new arms")
        new_arms = batched_generator_run.arms
        print_debug(f"new_arms: {new_arms}")
        if not new_arms:
            print_debug("get_batched_arms: No new arms were generated in this attempt.")
        else:
            print_debug(f"get_batched_arms: Generated {len(new_arms)} new arm(s), wanted {nr_of_jobs_to_get}.")

        batched_arms.extend(new_arms)
        attempts += 1

    print_debug(f"get_batched_arms: Finished with {len(batched_arms)} arm(s) after {attempts} attempt(s).")

    return batched_arms

def fetch_next_trials(nr_of_jobs_to_get: int, recursion: bool = False) -> Tuple[Dict[int, Any], bool]:
    die_101_if_no_ax_client_or_experiment_or_gs()

    if not ax_client:
        _fatal_error("ax_client was not defined", 9)

    if global_gs is None:
        _fatal_error("Global generation strategy is not set. This is a bug in OmniOpt2.", 107)

    return generate_trials(nr_of_jobs_to_get, recursion)

def generate_trials(n: int, recursion: bool) -> Tuple[Dict[int, Any], bool]:
    trials_dict: Dict[int, Any] = {}
    trial_durations: List[float] = []

    start_time = time.time()
    cnt = 0
    retries = 0

    try:
        while cnt < n and retries < args.max_abandoned_retrial:
            for arm in get_batched_arms(n - cnt):
                if cnt >= n:
                    break

                try:
                    arm = Arm(parameters=arm.parameters)
                except Exception as arm_err:
                    print_red(f"Error while creating new Arm: {arm_err}")
                    retries += 1
                    continue

                print_debug(f"Fetching trial {cnt + 1}/{n}...")
                progressbar_description(_get_trials_message(cnt + 1, n, trial_durations))

                try:
                    result = create_and_handle_trial(arm)
                    if result is not None:
                        trial_index, trial_duration, trial_successful = result

                except TrialRejected as e:
                    print_debug(f"Trial rejected: {e}")
                    retries += 1
                    continue

                trial_durations.append(trial_duration)

                if trial_successful:
                    cnt += 1
                    trials_dict[trial_index] = arm.parameters

        return _finalize_generation(trials_dict, cnt, n, start_time)

    except Exception as e:
        return _handle_generation_failure(e, n, recursion)

class TrialRejected(Exception):
    pass

def create_and_handle_trial(arm: Any) -> Optional[Tuple[int, float, bool]]:
    if ax_client is None:
        print_red("ax_client is None in create_and_handle_trial")
        return None

    start = time.time()

    if global_gs is None:
        _fatal_error("global_gs is not set", 107)

        return None

    _current_node_name = global_gs.current_node_name

    trial_index = ax_client.experiment.num_trials
    generator_run = GeneratorRun(
        arms=[arm],
        generation_node_name=_current_node_name
    )

    trial = ax_client.experiment.new_trial(generator_run)

    arm = trial.arms[0]
    if deduplicated_arm(arm):
        print_debug(f"Duplicated arm: {arm}")
        trial.mark_abandoned(reason="Duplication detected")
        raise TrialRejected("Duplicate arm.")

    arms_by_name_for_deduplication[arm.name] = arm

    params = arm.parameters

    if not has_no_post_generation_constraints_or_matches_constraints(post_generation_constraints, params):
        print_debug(f"Trial {trial_index} does not meet post-generation constraints. Marking abandoned. Params: {params}, constraints: {post_generation_constraints}")
        trial.mark_abandoned(reason="Post-Generation-Constraint failed")
        abandoned_trial_indices.append(trial_index)
        raise TrialRejected("Post-generation constraints not met.")

    trial.mark_running(no_runner_required=True)
    end = time.time()
    return trial_index, float(end - start), True

def _finalize_generation(trials_dict: Dict[int, Any], cnt: int, requested: int, start_time: float) -> Tuple[Dict[int, Any], bool]:
    total_time = time.time() - start_time

    log_gen_times.append(total_time)
    log_nr_gen_jobs.append(cnt)

    avg_time_str = f"{total_time / cnt:.2f} s/job" if cnt else "n/a"
    progressbar_description(f"requested {requested} jobs, got {cnt}, {avg_time_str}")

    return trials_dict, False

def _handle_generation_failure(
    e: Exception,
    requested: int,
    recursion: bool
) -> Tuple[Dict[int, Any], bool]:
    if isinstance(e, np.linalg.LinAlgError):
        _handle_linalg_error(e)
        my_exit(242)

    elif isinstance(e, (
        ax.exceptions.core.SearchSpaceExhausted,
        ax.exceptions.generation_strategy.GenerationStrategyRepeatedPoints,
        ax.exceptions.generation_strategy.MaxParallelismReachedException
    )):
        msg = str(e)
        if msg not in error_8_saved:
            _print_exhaustion_warning(e, recursion)
            error_8_saved.append(msg)

        if not recursion and args.revert_to_random_when_seemingly_exhausted:
            print_debug("Switching to random search strategy.")
            set_global_gs_to_random()
            return fetch_next_trials(requested, True)

    print_red(f"_handle_generation_failure: General Exception: {e}")

    return {}, True

def _print_exhaustion_warning(e: Exception, recursion: bool) -> None:
    if not recursion and args.revert_to_random_when_seemingly_exhausted:
        print_yellow(f"\n⚠Error 8: {e} From now (done jobs: {count_done_jobs()}) on, random points will be generated.")
    else:
        print_red(f"\n⚠Error 8: {e}")

def get_model_kwargs() -> dict:
    if 'Cont_X_trans_Y_trans' in args.transforms:
        return {
            "transforms": Cont_X_trans + Y_trans
        }

    if 'Cont_X_trans' in args.transforms:
        return {
            "transforms": Cont_X_trans
        }

    return {}

def get_model_gen_kwargs() -> dict:
    return {
        "model_gen_options": {
            "optimizer_kwargs": {
                "num_restarts": args.num_restarts,
                "raw_samples": args.raw_samples,
                # "sequential": False, # TODO, when https://github.com/facebook/Ax/issues/3819 is solved
            },
        },
        "fallback_to_sample_polytope": True,
        "normalize_y": not args.no_normalize_y,
        "transform_inputs": not args.no_transform_inputs,
        "optimizer_kwargs": get_optimizer_kwargs(),
        "enforce_num_trials": True,
        "torch_device": get_torch_device_str(),
        "random_seed": args.seed,
        "check_duplicates": True,
        "deduplicate_strict": True,
        "enforce_num_arms": True,
        "warm_start_refitting": not args.dont_warm_start_refitting,
        "jit_compile": not args.dont_jit_compile,
        "refit_on_cv": args.refit_on_cv,
        "fit_abandoned": args.fit_abandoned,
        "fit_out_of_design": args.fit_out_of_design
    }

def set_global_gs_to_random() -> None:
    global global_gs
    global overwritten_to_random

    global_gs = GenerationStrategy(
        name="Random*",
        nodes=[
            GenerationNode(
                node_name="Sobol",
                generator_specs=[
                    GeneratorSpec(
                        Models.SOBOL,
                        model_gen_kwargs=get_model_gen_kwargs()
                    )
                ]
            )
        ]
    )

    overwritten_to_random = True

    print_debug(f"New global_gs: {global_gs}")

def save_table_as_text(table: Table, filepath: str) -> None:
    try:
        with open(filepath, "w", encoding="utf-8") as file:
            from io import StringIO
            sio = StringIO()
            console_for_save = Console(file=sio, force_terminal=True, width=120)
            console_for_save.print(table)
            text_output = sio.getvalue()
            file.write(text_output)
    except Exception as e:
        print_debug(f"save_table_as_text: error at writing the file '{filepath}': {e}")

def show_time_debugging_table() -> None:
    if not args.dryrun:
        generate_time_table_rich()
        generate_job_submit_table_rich()
        plot_times_for_creation_and_submission()

def generate_time_table_rich() -> None:
    if not isinstance(log_gen_times, list):
        print_debug("generate_time_table_rich: Error: log_gen_times is not a list.")
        return

    if not isinstance(log_nr_gen_jobs, list):
        print_debug("generate_time_table_rich: Error: log_nr_gen_jobs is not a list.")
        return

    if len(log_gen_times) != len(log_nr_gen_jobs):
        print_debug("generate_time_table_rich: Error: Mismatched lengths of times and job counts.")
        return

    if len(log_gen_times) == 0:
        print_debug("generate_time_table_rich: No times to display.")
        return

    for i, (val, jobs) in enumerate(zip(log_gen_times, log_nr_gen_jobs)):
        try:
            float(val)
            int(jobs)
        except (ValueError, TypeError):
            print_debug(f"generate_time_table_rich: Error: Invalid data at index {i}.")
            return

    table = Table(show_header=True, header_style="bold", title="Model generation times")
    table.add_column("Iteration", justify="right")
    table.add_column("Seconds", justify="right")
    table.add_column("Jobs", justify="right")
    table.add_column("Time per job", justify="right")

    times_float: List[float] = []
    times_per_job: List[float] = []

    for idx, (time_val, job_count) in enumerate(zip(log_gen_times, log_nr_gen_jobs), start=1):
        seconds = float(time_val)
        jobs = int(job_count)
        per_job = seconds / jobs if jobs != 0 else 0.0
        times_float.append(seconds)
        times_per_job.append(per_job)
        table.add_row(str(idx), f"{seconds:.3f}", str(jobs), f"{per_job:.3f}")

    table.add_section()

    table.add_row("Average", f"{statistics.mean(times_float):.3f}", "", "")
    table.add_row("Median", f"{statistics.median(times_float):.3f}", "", "")
    table.add_row("Total", f"{sum(times_float):.3f}", "", "")
    table.add_row("Max", f"{max(times_float):.3f}", "", "")
    table.add_row("Min", f"{min(times_float):.3f}", "", "")

    if args.show_generate_time_table:
        console.print(table)

    folder = get_current_run_folder()
    filename = "generation_times.txt"
    filepath = os.path.join(folder, filename)
    save_table_as_text(table, filepath)

def validate_job_submit_data(durations: List[float], job_counts: List[int]) -> bool:
    if not durations or not job_counts:
        print_debug("No durations or job counts to display.")
        return False

    if len(durations) != len(job_counts):
        print_debug("Length mismatch between durations and job counts.")
        return False

    return True

def convert_durations_to_float(raw_durations: List) -> List[float] | None:
    try:
        return [float(val) for val in raw_durations]
    except (ValueError, TypeError) as e:
        print_debug(f"Invalid float in durations: {e}")
        return None

def convert_job_counts_to_int(raw_counts: List) -> List[int] | None:
    try:
        return [int(val) for val in raw_counts]
    except (ValueError, TypeError) as e:
        print_debug(f"Invalid int in job counts: {e}")
        return None

def build_job_submission_table(durations: List[float], job_counts: List[int]) -> Table:
    table = Table(show_header=True, header_style="bold", title="Job submission durations")
    table.add_column("Batch", justify="right")
    table.add_column("Seconds", justify="right")
    table.add_column("Jobs", justify="right")
    table.add_column("Time per job", justify="right")

    for idx, (duration, jobs) in enumerate(zip(durations, job_counts), start=1):
        time_per_job = duration / jobs if jobs > 0 else 0
        table.add_row(str(idx), f"{duration:.3f}", str(jobs), f"{time_per_job:.3f}")

    table.add_section()
    table.add_row("Average", f"{statistics.mean(durations):.3f}", "", "")
    table.add_row("Median", f"{statistics.median(durations):.3f}", "", "")
    table.add_row("Total", f"{sum(durations):.3f}", "", "")
    table.add_row("Max", f"{max(durations):.3f}", "", "")
    table.add_row("Min", f"{min(durations):.3f}", "", "")

    return table

def export_table_to_file(table: Table, filename: str) -> None:
    folder = get_current_run_folder()
    filepath = os.path.join(folder, filename)
    save_table_as_text(table, filepath)

def generate_job_submit_table_rich() -> None:
    if not isinstance(job_submit_durations, list) or not isinstance(job_submit_nrs, list):
        print_debug("job_submit_durations or job_submit_nrs is not a list.")
        return

    durations = convert_durations_to_float(job_submit_durations)
    job_counts = convert_job_counts_to_int(job_submit_nrs)

    if durations is None or job_counts is None:
        return

    if not validate_job_submit_data(durations, job_counts):
        return

    table = build_job_submission_table(durations, job_counts)

    if args.show_generate_time_table:
        console.print(table)

    export_table_to_file(table, "job_submit_durations.txt")

def plot_times_for_creation_and_submission() -> None:
    if not args.show_generation_and_submission_sixel:
        return

    plot_times_vs_jobs_sixel(
        times=job_submit_durations,
        job_counts=job_submit_nrs,
        xlabel="Index",
        ylabel="Duration (seconds)",
        title="Job Submission Durations vs Number of Jobs"
    )

    plot_times_vs_jobs_sixel(
        times=log_gen_times,
        job_counts=log_nr_gen_jobs,
        xlabel="Index",
        ylabel="Generation Time (seconds)",
        title="Model Generation Times vs Number of Jobs"
    )

def plot_times_vs_jobs_sixel(
    times: List[float],
    job_counts: List[int],
    xlabel: str = "Iteration",
    ylabel: str = "Duration (seconds)",
    title: str = "Times vs Jobs"
) -> None:
    if not times or not job_counts or len(times) != len(job_counts):
        print("[italic yellow]plot_times_vs_jobs_sixel: No valid data or mismatched lengths to plot.[/]")
        return

    if not supports_sixel():
        print("[italic yellow]Your console does not support sixel-images. Cannot show plot.[/]")
        return

    import matplotlib.pyplot as plt

    fig, _ax = plt.subplots()

    iterations = list(range(1, len(times) + 1))
    sizes = [max(20, min(200, jc * 10)) for jc in job_counts]

    scatter = _ax.scatter(iterations, times, s=sizes, c=job_counts, cmap='viridis', alpha=0.7, edgecolors='black')

    _ax.set_xlabel(xlabel)
    _ax.set_ylabel(ylabel)
    _ax.set_title(title)
    _ax.grid(True)

    cbar = plt.colorbar(scatter, ax=_ax)
    cbar.set_label('Number of Jobs')

    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp_file:
        plt.savefig(tmp_file.name, dpi=300)
        print_image_to_cli(tmp_file.name, width=1000)

    plt.close(fig)

def _handle_linalg_error(error: Union[None, str, Exception]) -> None:
    """Handles the np.linalg.LinAlgError based on the model being used."""
    print_red(f"Error: {error}")

def get_next_trials(nr_of_jobs_to_get: int) -> Tuple[Union[None, dict], bool]:
    finish_previous_jobs(["finishing jobs (get_next_trials)"])

    if break_run_search("get_next_trials", max_eval) or nr_of_jobs_to_get == 0:
        return {}, True

    try:
        trial_index_to_param, optimization_complete = fetch_next_trials(nr_of_jobs_to_get)

        cf = currentframe()
        if cf:
            _frame_info = getframeinfo(cf)
            if _frame_info:
                lineno: int = _frame_info.lineno
                print_debug_get_next_trials(
                    len(trial_index_to_param.items()),
                    nr_of_jobs_to_get,
                    lineno
                )

        _log_trial_index_to_param(trial_index_to_param)

        return trial_index_to_param, optimization_complete
    except OverflowError as e:
        if len(arg_result_names) > 1:
            print_red(f"Error while trying to create next trials. The number of result-names are probably too large. You have {len(arg_result_names)} parameters. Error: {e}")
        else:
            print_red(f"Error while trying to create next trials. Error: {e}")

        return None, True

def get_next_nr_steps(_num_parallel_jobs: int, _max_eval: int) -> int:
    if not SYSTEM_HAS_SBATCH:
        return 1

    simulated_nr_inserted_jobs = get_nr_of_imported_jobs()

    max_eval_plus_inserted = _max_eval + simulated_nr_inserted_jobs

    num_parallel_jobs_minus_existing_jobs = _num_parallel_jobs - len(global_vars["jobs"])

    max_eval_plus_nr_inserted_jobs_minus_submitted_jobs = max_eval_plus_inserted - submitted_jobs()

    max_eval_plus_nr_inserted_jobs_minus_done_jobs = max_eval_plus_inserted - count_done_jobs()

    min_of_all_options = min(
        num_parallel_jobs_minus_existing_jobs,
        max_eval_plus_nr_inserted_jobs_minus_submitted_jobs,
        max_eval_plus_nr_inserted_jobs_minus_done_jobs
    )

    requested = max(
        1,
        min_of_all_options
    )

    set_requested_to_zero_because_already_enough_jobs = False

    if count_done_jobs() >= max_eval_plus_inserted or (submitted_jobs() - failed_jobs()) >= max_eval_plus_inserted:
        requested = 0

        set_requested_to_zero_because_already_enough_jobs = True

    table = Table(title="Debugging get_next_nr_steps")
    table.add_column("Variable", justify="right")
    table.add_column("Wert", justify="left")

    table.add_row("max_eval", str(max_eval))
    if max_eval != _max_eval:
        table.add_row("_max_eval", str(_max_eval))

    table.add_row("", "")

    table.add_row("submitted_jobs()", str(submitted_jobs()))
    table.add_row("failed_jobs()", str(failed_jobs()))
    table.add_row("count_done_jobs()", str(count_done_jobs()))

    table.add_row("", "")

    table.add_row("simulated_nr_inserted_jobs", str(simulated_nr_inserted_jobs))
    table.add_row("max_eval_plus_inserted", str(max_eval_plus_inserted))

    table.add_row("", "")

    table.add_row("num_parallel_jobs_minus_existing_jobs", str(num_parallel_jobs_minus_existing_jobs))
    table.add_row("max_eval_plus_nr_inserted_jobs_minus_submitted_jobs", str(max_eval_plus_nr_inserted_jobs_minus_submitted_jobs))
    table.add_row("max_eval_plus_nr_inserted_jobs_minus_done_jobs", str(max_eval_plus_nr_inserted_jobs_minus_done_jobs))

    table.add_row("", "")

    table.add_row("min_of_all_options", str(min_of_all_options))

    table.add_row("", "")

    table.add_row("set_requested_to_zero_because_already_enough_jobs", str(set_requested_to_zero_because_already_enough_jobs))
    table.add_row("requested", str(requested))

    with console.capture() as capture:
        console.print(table)

    table_str = capture.get()

    with open(get_current_run_folder("get_next_nr_steps_tables.txt"), mode="a", encoding="utf-8") as text_file:
        text_file.write(table_str)

    return requested

def select_model(model_arg: Any) -> ax.adapter.registry.Generators:
    """Selects the model based on user input or defaults to BOTORCH_MODULAR."""
    available_models = list(ax.adapter.registry.Generators.__members__.keys())
    chosen_model = ax.adapter.registry.Generators.BOTORCH_MODULAR

    if model_arg:
        model_upper = str(model_arg).upper()
        if model_upper in available_models:
            chosen_model = ax.adapter.registry.Generators.__members__[model_upper]
        else:
            print_red(f"⚠ Cannot use {model_arg}. Available models are: {', '.join(available_models)}. Using BOTORCH_MODULAR instead.")

        if model_arg.lower() != "factorial" and args.gridsearch:
            print_yellow("Gridsearch only really works when you chose the FACTORIAL model.")

    return chosen_model

def print_generation_strategy(generation_strategy_array: list[dict[str, int]]) -> None:
    table = Table(header_style="bold", title="Generation Strategy")
    table.add_column("Generation Strategy")
    table.add_column("Number of Generations")

    for elem in generation_strategy_array:
        model_name, num_generations = list(elem.items())[0]
        table.add_row(model_name, str(num_generations))

    console.print(table)

def get_model_from_name(name: str) -> Any:
    name = name.lower()
    for gen in ax.adapter.registry.Generators:
        if gen.name.lower() == name:
            return gen
    raise ValueError(f"Unknown or unsupported model: {name}")

def get_name_from_model(model) -> Optional[str]:
    if not isinstance(SUPPORTED_MODELS, (list, set, tuple)):
        return None

    model_str = model.value if hasattr(model, "value") else str(model)

    model_str_lower = model_str.lower()
    model_map = {m.lower(): m for m in SUPPORTED_MODELS}

    return model_map.get(model_str_lower, None)

def parse_generation_strategy_string(gen_strat_str: str) -> tuple[list[dict[str, int]], int]:
    gen_strat_list = []
    sum_nr = 0

    cleaned_string = re.sub(r"\s+", "", gen_strat_str)
    splitted_by_comma = cleaned_string.split(",")

    for s in splitted_by_comma:
        if "=" not in s:
            print(f"'{s}' does not contain '='")
            my_exit(123)
        if s.count("=") != 1:
            print(f"There can only be one '=' in the gen_strat_str's element '{s}'")
            my_exit(123)

        model_name, nr_str = s.split("=")
        matching_model = get_name_from_model(model_name)

        if matching_model in uncontinuable_models:
            _fatal_error(f"Model {matching_model} is not valid for custom generation strategy.", 56)

        if not matching_model:
            print(f"'{model_name}' not found in SUPPORTED_MODELS")
            my_exit(123)

        try:
            nr = int(nr_str)
        except ValueError:
            print(f"Invalid number of generations '{nr_str}' for model '{model_name}'")
            my_exit(123)

        gen_strat_list.append({matching_model: nr})
        sum_nr += nr

    return gen_strat_list, sum_nr

def write_state_file(name: str, var: str) -> None:
    file_path = get_state_file_name(name)

    if os.path.isdir(file_path):
        _fatal_error(f"{file_path} is a dir. Must be a file.", 246)

    makedirs(os.path.dirname(file_path))

    try:
        with open(file_path, mode="w", encoding="utf-8") as f:
            f.write(str(var))
    except Exception as e:
        print_red(f"Failed writing '{file_path}': {e}")

def get_state_file_content(name: str, run_folder: str = get_current_run_folder()) -> str:
    if args.continue_previous_job:
        run_folder = args.continue_previous_job

    file_path = f"{run_folder}/state_files/{name}"

    if os.path.isdir(file_path):
        _fatal_error(f"{file_path} is a dir. Must be a file.", 247)

    if not os.path.exists(file_path):
        print_red(f"State file '{file_path}' does not exist.")
        return ""

    try:
        with open(file_path, mode="r", encoding="utf-8") as f:
            return f.read().replace("\n", "").replace("\r", "")
    except Exception as e:
        print_red(f"Failed reading '{file_path}': {e}")
        return ""

def get_chosen_model() -> str:
    chosen_model = args.model

    if args.continue_previous_job and chosen_model is None:
        chosen_model = get_state_file_content("model")

        write_state_file("model", str(chosen_model))

        found_model = False

        if chosen_model not in SUPPORTED_MODELS:
            print_red(f"Wrong model >{chosen_model}<.")
        else:
            found_model = True

        if not found_model:
            if args.model is not None:
                chosen_model = args.model
            else:
                chosen_model = "BOTORCH_MODULAR"
            print_red(f"Could not find model in previous job. Will use the default model '{chosen_model}'")

    if chosen_model is None:
        chosen_model = "BOTORCH_MODULAR"

    return chosen_model

def continue_not_supported_on_custom_generation_strategy() -> None:
    if args.continue_previous_job:
        generation_strategy_file = f"{args.continue_previous_job}/state_files/custom_generation_strategy"

        if os.path.exists(generation_strategy_file):
            _fatal_error("Trying to continue a job which was started with --generation_strategy. This is currently not possible.", 247)

def get_step_name(model_name: str, nr: int) -> str:
    this_step_name = f"{model_name} for {nr} step"
    if nr != 1:
        this_step_name = f"{this_step_name}s"

    return this_step_name

def join_with_comma_and_then(items: list) -> str:
    length = len(items)

    if length == 0:
        return ""

    if length == 1:
        return items[0]

    if length == 2:
        return f"{items[0]} and then {items[1]}"

    return ", ".join(items[:-1]) + " and then " + items[-1]

def get_torch_device_str() -> str:
    try:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"

        return "cpu"
    except Exception as e:
        print_debug(f"Error detecting device: {e}")
        return "cpu"

def create_node(model_name: str, threshold: int, next_model_name: Optional[str]) -> Union[RandomForestGenerationNode, GenerationNode]:
    if model_name == "RANDOMFOREST":
        if len(arg_result_names) != 1:
            _fatal_error("Currently, RANDOMFOREST does not support Multi-Objective-Optimization", 251)
        return RandomForestGenerationNode(
            num_samples=threshold,
            regressor_options={"n_estimators": args.n_estimators_randomforest},
            seed=args.seed
        )

    target_model = next_model_name if next_model_name is not None else model_name

    if model_name == "TPE":
        if len(arg_result_names) != 1:
            _fatal_error(f"Has {len(arg_result_names)} results. TPE currently only supports single-objective-optimization.", 108)
        return ExternalProgramGenerationNode(external_generator=f"python3 {script_dir}/.tpe.py", node_name="EXTERNAL_GENERATOR")

    external_generators = {
        "PSEUDORANDOM": f"python3 {script_dir}/.random_generator.py",
        "EXTERNAL_GENERATOR": args.external_generator
    }

    if next_model_name in ["PSEUDORANDOM", "EXTERNAL_GENERATOR", "TPE"]:
        target_model = "EXTERNAL_GENERATOR"

    if model_name in external_generators:
        cmd = external_generators[model_name]
        if model_name == "EXTERNAL_GENERATOR" and not cmd:
            _fatal_error("--external_generator is missing. Cannot create points for EXTERNAL_GENERATOR without it.", 204)
        return ExternalProgramGenerationNode(external_generator=cmd, node_name="EXTERNAL_GENERATOR")

    trans_crit = [
        MaxTrials(
            threshold=threshold,
            block_transition_if_unmet=True,
            transition_to=target_model,
            count_only_trials_with_data=True
        )
    ]

    selected_model = select_model(model_name)

    kwargs = {
        "model_gen_kwargs": get_model_gen_kwargs()
    }
    if model_name.lower() != "sobol":
        kwargs["model_kwargs"] = get_model_kwargs()

    model_spec = [GeneratorSpec(selected_model, **kwargs)]

    res = GenerationNode(
        node_name=model_name,
        generator_specs=model_spec,
        transition_criteria=trans_crit
    )

    return res

def get_optimizer_kwargs() -> dict:
    return {
        "sequential": False
    }

def create_step(model_name: str, _num_trials: int = -1, index: Optional[int] = None) -> GenerationStep:
    model_enum = get_model_from_name(model_name)

    return GenerationStep(
        generator=model_enum,
        num_trials=_num_trials,
        max_parallelism=1000 * max_eval + 1000,
        model_kwargs=get_model_kwargs(),
        model_gen_kwargs=get_model_gen_kwargs(),
        should_deduplicate=True,
        enforce_num_trials=True,
        index=index
    )

def set_global_generation_strategy() -> None:
    with spinner("Setting global generation strategy"):
        continue_not_supported_on_custom_generation_strategy()

        try:
            if args.generation_strategy is None:
                setup_default_generation_strategy()
            else:
                setup_custom_generation_strategy()
        except Exception as e:
            print_red(f"Unexpected error in generation strategy setup: {e}")
            my_exit(111)

    if global_gs is None:
        print_red("global_gs is None after setup!")
        my_exit(111)

def setup_default_generation_strategy() -> None:
    global generation_strategy_human_readable

    generation_strategy_nodes: list = []

    num_imported_jobs = get_nr_of_imported_jobs()

    set_max_eval(max_eval + num_imported_jobs)
    set_random_steps(random_steps or 0)

    if max_eval is None:
        set_max_eval(max(1, random_steps))

    chosen_model = get_chosen_model()
    print_debug(f"Chosen model: {chosen_model}")

    if chosen_model == "SOBOL":
        set_random_steps(max_eval)

    if random_steps > num_imported_jobs:
        add_sobol_node_if_needed(generation_strategy_nodes, generation_strategy_names, chosen_model)

    remaining = max_eval - random_steps + num_imported_jobs

    add_main_node_if_needed(generation_strategy_nodes, generation_strategy_names, chosen_model, remaining)

    generation_strategy_human_readable = join_with_comma_and_then(generation_strategy_names)
    print_debug(f"Generation strategy human readable: {generation_strategy_human_readable}")

    try:
        global global_gs
        global_gs = GenerationStrategy(
            name="+".join(generation_strategy_names),
            nodes=generation_strategy_nodes
        )
    except ax.exceptions.generation_strategy.GenerationStrategyMisconfiguredException as e:
        print_red(f"Error creating GenerationStrategy: {e}\nnames: {generation_strategy_names}\nnodes: {generation_strategy_nodes}")
        my_exit(111)

def add_sobol_node_if_needed(nodes: list, names: list, chosen_model: str) -> None:
    if random_steps >= 1:
        next_node_name = None
        if max_eval - random_steps and chosen_model:
            next_node_name = chosen_model
        step_name = get_step_name("SOBOL", random_steps)
        nodes.append(create_node("SOBOL", random_steps, next_node_name))
        names.append(step_name)
        print_debug(f"Added SOBOL node: {step_name}")

def add_main_node_if_needed(nodes: list, names: list, chosen_model: str, remaining: int) -> None:
    if chosen_model != "SOBOL" and remaining > 0:
        node = create_node(chosen_model, remaining, None)
        nodes.append(node)
        step_name = get_step_name(chosen_model, remaining)
        names.append(step_name)
        print_debug(f"Added main node: {step_name}")

def setup_custom_generation_strategy() -> None:
    generation_strategy_array, new_max_eval = parse_generation_strategy_string(args.generation_strategy)
    new_max_eval_plus_jobs = new_max_eval + get_nr_of_imported_jobs()

    if max_eval < new_max_eval_plus_jobs:
        print_yellow(f"--generation_strategy {args.generation_strategy.upper()} has more tasks than --max_eval {max_eval}. Updating max_eval to {new_max_eval_plus_jobs}.")
        set_max_eval(new_max_eval_plus_jobs)

    print_generation_strategy(generation_strategy_array)
    start_index = int(len(generation_strategy_array) / 2)
    steps: list = []

    for gs_element in generation_strategy_array:
        try:
            model_name = list(gs_element.keys())[0]
            num_trials = int(gs_element[model_name])
            step_node = create_step(model_name, num_trials, start_index)
            step_name = get_step_name(model_name, num_trials)
            steps.append(step_node)
            generation_strategy_names.append(step_name)
            print_debug(f"Added custom step: {step_name}")
            start_index += 1
        except Exception as e:
            print_red(f"Error creating step for {gs_element}: {e}")
            my_exit(111)

    write_state_file("custom_generation_strategy", args.generation_strategy)

    global global_gs, generation_strategy_human_readable
    try:
        global_gs = GenerationStrategy(steps=steps)
        generation_strategy_human_readable = join_with_comma_and_then(generation_strategy_names)
    except Exception as e:
        print_red(f"Failed to create custom GenerationStrategy: {e}")
        my_exit(111)

def wait_for_jobs_or_break(_max_eval: Optional[int]) -> bool:
    while len(global_vars["jobs"]) > num_parallel_jobs:
        finish_previous_jobs([f"finishing previous jobs ({len(global_vars['jobs'])})"])

        if break_run_search("create_and_execute_next_runs", _max_eval):
            return True

        if is_slurm_job() and not args.force_local_execution:
            _sleep(1)

    if break_run_search("create_and_execute_next_runs", _max_eval):
        return True

    if _max_eval is not None and (JOBS_FINISHED - NR_INSERTED_JOBS) >= _max_eval:
        return True

    return False

def execute_trials(
    trial_index_to_param: dict,
    phase: Optional[str],
    _max_eval: Optional[int],
) -> None:
    index_param_list: List[List[Any]] = []
    i: int = 1

    for trial_index, parameters in trial_index_to_param.items():
        if wait_for_jobs_or_break(_max_eval):
            break
        if break_run_search("create_and_execute_next_runs", _max_eval):
            break

        progressbar_description(f"eval #{i}/{len(trial_index_to_param.items())} start")
        _args = [trial_index, parameters, i, phase]
        index_param_list.append(_args)
        i += 1

    start_time = time.time()

    cnt = 0

    nr_workers = max(1, min(len(index_param_list), args.max_num_of_parallel_sruns))

    with ThreadPoolExecutor(max_workers=nr_workers) as tp_executor:
        future_to_args = {tp_executor.submit(execute_evaluation, _args): _args for _args in index_param_list}

        for future in as_completed(future_to_args):
            cnt = cnt + 1
            try:
                result = future.result()
                print_debug(f"result in execute_trials: {result}")
            except Exception as exc:
                print_red(f"execute_trials: Error at executing a trial: {exc}")

    end_time = time.time()

    duration = float(end_time - start_time)
    job_submit_durations.append(duration)
    job_submit_nrs.append(cnt)

def handle_exceptions_create_and_execute_next_runs(e: Exception) -> int:
    if isinstance(e, TypeError):
        print_red(f"Error 1: {e}")
    elif isinstance(e, botorch.exceptions.errors.InputDataError):
        print_red(f"Error 2: {e}")
    elif isinstance(e, ax.exceptions.core.DataRequiredError):
        if "transform requires non-empty data" in str(e) and args.num_random_steps == 0:
            _fatal_error(f"Error 3: {e} Increase --num_random_steps to at least 1 to continue.", 233)
        else:
            print_debug(f"Error 4: {e}")
    elif isinstance(e, RuntimeError):
        print_red(f"\n⚠ Error 5: {e}")
    elif isinstance(e, botorch.exceptions.errors.ModelFittingError):
        print_red(f"\n⚠ Error 6: {e}")
        end_program(False, 1)
    elif isinstance(e, (ax.exceptions.core.SearchSpaceExhausted, ax.exceptions.generation_strategy.GenerationStrategyRepeatedPoints)):
        print_red(f"\n⚠ Error 7 {e}")
        end_program(False, 87)
    return 0

def create_and_execute_next_runs(next_nr_steps: int, phase: Optional[str], _max_eval: Optional[int]) -> int:
    if next_nr_steps == 0:
        print_debug(f"Warning: create_and_execute_next_runs(next_nr_steps: {next_nr_steps}, phase: {phase}, _max_eval: {_max_eval}, progress_bar)")
        return 0

    trial_index_to_param: Optional[Dict] = None
    done_optimizing: bool = False

    try:
        done_optimizing, trial_index_to_param = _create_and_execute_next_runs_run_loop(_max_eval, phase)
        _create_and_execute_next_runs_finish(done_optimizing)
    except Exception as e:
        stacktrace = traceback.format_exc()
        print_debug(f"Warning: create_and_execute_next_runs encountered an exception: {e}\n{stacktrace}")
        return handle_exceptions_create_and_execute_next_runs(e)

    return _create_and_execute_next_runs_return_value(trial_index_to_param)

def _create_and_execute_next_runs_run_loop(_max_eval: Optional[int], phase: Optional[str]) -> Tuple[bool, Optional[Dict]]:
    done_optimizing = False
    trial_index_to_param: Optional[Dict] = None

    nr_of_jobs_to_get = _calculate_nr_of_jobs_to_get(get_nr_of_imported_jobs(), len(global_vars["jobs"]))

    __max_eval = _max_eval if _max_eval is not None else 0
    new_nr_of_jobs_to_get = min(__max_eval - (submitted_jobs() - failed_jobs()), nr_of_jobs_to_get)

    range_nr = new_nr_of_jobs_to_get
    get_next_trials_nr = 1

    if getattr(args, "generate_all_jobs_at_once", False) or args.worker_generator_path:
        range_nr = 1
        get_next_trials_nr = new_nr_of_jobs_to_get

    for _ in range(range_nr):
        trial_index_to_param, done_optimizing = get_next_trials(get_next_trials_nr)
        if done_optimizing:
            continue

        if trial_index_to_param:
            nr_jobs_before_removing_abandoned = len(list(trial_index_to_param.keys()))

            filtered_trial_index_to_param = {k: v for k, v in trial_index_to_param.items() if k not in abandoned_trial_indices}

            if is_skip_search():
                print_yellow("Skipping search part")
                return True, {}

            if len(filtered_trial_index_to_param):
                execute_trials(filtered_trial_index_to_param, phase, _max_eval)
            else:
                if nr_jobs_before_removing_abandoned > 0:
                    print_debug(f"Could not get jobs. They've been deleted by abandoned_trial_indices: {abandoned_trial_indices}")
                else:
                    print_debug("Could not generate any jobs")

            trial_index_to_param = filtered_trial_index_to_param

    return done_optimizing, trial_index_to_param

def _create_and_execute_next_runs_finish(done_optimizing: bool) -> None:
    finish_previous_jobs(["finishing jobs"])

    if done_optimizing:
        end_program(False, 0)

def _create_and_execute_next_runs_return_value(trial_index_to_param: Optional[Dict]) -> int:
    try:
        if trial_index_to_param:
            res = len(trial_index_to_param.keys())
            print_debug(f"create_and_execute_next_runs: Returning len(trial_index_to_param.keys()): {res}")
            return res

        print_debug(f"Warning: trial_index_to_param is not true. It, stringified, looks like this: {trial_index_to_param}. Returning 0.")
        return 0
    except Exception as e:
        print_debug(f"Warning: create_and_execute_next_runs encountered an exception: {e}. Returning 0.")
        return 0

def get_number_of_steps(_max_eval: int) -> int:
    with spinner("Calculating number of steps..."):
        _random_steps = args.num_random_steps

        already_done_random_steps = get_random_steps_from_prev_job()

        _random_steps = _random_steps - already_done_random_steps

        if _random_steps > _max_eval:
            print_yellow(f"You have less --max_eval {_max_eval} than --num_random_steps {_random_steps}. Switched both.")
            _random_steps, _max_eval = _max_eval, _random_steps

        if _random_steps < num_parallel_jobs and SYSTEM_HAS_SBATCH:
            print_yellow(f"Warning: --num_random_steps {_random_steps} is smaller than --num_parallel_jobs {num_parallel_jobs}. It's recommended that --num_parallel_jobs is the same as or a multiple of --num_random_steps")

        if _random_steps > _max_eval:
            set_max_eval(_random_steps)

        return _random_steps

def _set_global_executor() -> None:
    global submitit_executor

    log_folder: str = f'{get_current_run_folder()}/single_runs/%j'
    subjob_uuid = str(uuid.uuid4())

    if args.force_local_execution:
        submitit_executor = LocalExecutor(folder=log_folder)
    else:
        submitit_executor = AutoExecutor(folder=log_folder)

    if submitit_executor:
        params = {
            "name": f'{global_vars["experiment_name"]}_{run_uuid}_{subjob_uuid}',
            "timeout_min": args.worker_timeout,
            "slurm_gres": f"gpu:{args.gpus}",
            "cpus_per_task": args.cpus_per_task,
            "nodes": args.nodes_per_job,
            "stderr_to_stdout": True,
            "mem_gb": args.mem_gb,
            "slurm_signal_delay_s": args.slurm_signal_delay_s,
            "slurm_use_srun": args.slurm_use_srun,
            "exclude": args.exclude,
        }

        submitit_executor.update_parameters(**params)

        print_debug("submitit_executor.update_parameters(\n" + json.dumps(params, indent=4) + "\n)")

        if args.exclude:
            print_yellow(f"Excluding the following nodes: {args.exclude}")
    else:
        _fatal_error("submitit_executor could not be found", 9)

def set_global_executor() -> None:
    try:
        _set_global_executor()
    except ModuleNotFoundError as e:
        print_red(f"_set_global_executor() failed with error {e}. It may help if you can delete and re-install the virtual Environment containing the OmniOpt2 modules.")
        sys.exit(244)
    except (IsADirectoryError, PermissionError, FileNotFoundError) as e:
        print_red(f"Error trying to set_global_executor: {e}")

def execute_nvidia_smi() -> None:
    if not IS_NVIDIA_SMI_SYSTEM:
        print_debug("Cannot find nvidia-smi. Cannot take GPU logs")
        return

    while True:
        try:
            host = socket.gethostname()

            if NVIDIA_SMI_LOGS_BASE and host:
                _file = f"{NVIDIA_SMI_LOGS_BASE}_{host}.csv"
                noheader = ",noheader"

                result = subprocess.run([
                    'nvidia-smi',
                    '--query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used',
                    f'--format=csv{noheader}'],
                    capture_output=True,
                    text=True,
                    check=True
                )
                assert result.returncode == 0, "nvidia-smi execution failed"

                output = result.stdout

                output = output.rstrip('\n')

                if host and output:
                    append_to_nvidia_smi_logs(_file, host, output)
            else:
                if not NVIDIA_SMI_LOGS_BASE:
                    print_debug("NVIDIA_SMI_LOGS_BASE not defined")
                if not host:
                    print_debug("host not defined")
        except Exception as e:
            print(f"execute_nvidia_smi: An error occurred: {e}")
        if is_slurm_job() and not args.force_local_execution:
            _sleep(30)

def start_nvidia_smi_thread() -> None:
    if IS_NVIDIA_SMI_SYSTEM:
        nvidia_smi_thread = threading.Thread(target=execute_nvidia_smi, daemon=True)
        nvidia_smi_thread.start()

def run_search() -> bool:
    global NR_OF_0_RESULTS
    NR_OF_0_RESULTS = 0

    write_process_info()

    while (submitted_jobs() - failed_jobs()) <= max_eval:
        wait_for_jobs_to_complete()
        finish_previous_jobs()

        if should_break_search():
            break

        next_nr_steps: int = get_next_nr_steps(num_parallel_jobs, max_eval)

        nr_of_items = execute_next_steps(next_nr_steps)

        finish_previous_jobs([f"finishing previous jobs ({len(global_vars['jobs'])})"])

        handle_slurm_execution()

        if check_search_space_exhaustion(nr_of_items):
            wait_for_jobs_to_complete()
            raise SearchSpaceExhausted("Search space exhausted")

    finalize_jobs()

    return False

async def start_logging_daemon() -> None:
    while True:
        log_data()
        time.sleep(30)

def should_break_search() -> bool:
    ret = False

    if not args.worker_generator_path:
        ret = (break_run_search("run_search", max_eval) or (JOBS_FINISHED - NR_INSERTED_JOBS) >= max_eval)
    else:
        print_debug("should_break_search: False because --worker_generator_path was set")

    print_debug(f"should_break_search: {ret}")

    return ret

def execute_next_steps(next_nr_steps: int) -> int:
    if next_nr_steps:
        print_debug(f"trying to get {next_nr_steps} next steps (current done: {count_done_jobs()}, max: {max_eval})")
        nr_of_items = create_and_execute_next_runs(next_nr_steps, "systematic", max_eval)

        log_worker_status(nr_of_items, next_nr_steps)

        return nr_of_items
    return 0

def log_worker_status(nr_of_items: int, next_nr_steps: int) -> None:
    nr_current_workers, nr_current_workers_errmsg = count_jobs_in_squeue()
    if nr_current_workers_errmsg:
        print_debug(f"log_worker_status: {nr_current_workers_errmsg}")
    _debug_worker_creation(f"{int(time.time())}, {nr_current_workers}, {nr_of_items}, {next_nr_steps}")

def handle_slurm_execution() -> None:
    if is_slurm_job() and not args.force_local_execution:
        _sleep(1)

def check_search_space_exhaustion(nr_of_items: int) -> bool:
    global NR_OF_0_RESULTS

    if nr_of_items == 0 and len(global_vars["jobs"]) == 0:
        NR_OF_0_RESULTS += 1
        _wrn = f"found {NR_OF_0_RESULTS} zero-jobs (max: {args.max_nr_of_zero_results})"
        progressbar_description(_wrn)
        print_debug(_wrn)
    else:
        NR_OF_0_RESULTS = 0

    if NR_OF_0_RESULTS >= args.max_nr_of_zero_results:
        _wrn = f"{NR_OF_0_RESULTS} empty jobs (>= {args.max_nr_of_zero_results})"
        print_debug(_wrn)
        progressbar_description(_wrn)

        live_share()
        return True

    live_share()
    return False

def finalize_jobs() -> None:
    while len(global_vars["jobs"]):
        wait_for_jobs_to_complete()

        jobs_left = len(global_vars['jobs'])

        finish_previous_jobs([f"waiting for {jobs_left} job{'' if jobs_left == 1 else 's'})"])

        handle_slurm_execution()

def go_through_jobs_that_are_not_completed_yet() -> None:
    print_debug(f"Waiting for jobs to finish (currently, len(global_vars['jobs']) = {len(global_vars['jobs'])}")

    nr_jobs_left = len(global_vars['jobs'])
    if nr_jobs_left == 1:
        progressbar_description(f"waiting for {nr_jobs_left} job")
    else:
        progressbar_description(f"waiting for {nr_jobs_left} jobs")

    if is_slurm_job() and not args.force_local_execution:
        _sleep(0.5)

    jobs_left = len(global_vars['jobs'])

    finish_previous_jobs([f"waiting for {jobs_left} job{'s' if jobs_left != 1 else ''}"])

    clean_completed_jobs()

def wait_for_jobs_to_complete() -> None:
    while len(global_vars["jobs"]):
        go_through_jobs_that_are_not_completed_yet()

def die_orchestrator_exit_code_206(_test: bool) -> None:
    if _test:
        print_yellow("Not exiting, because _test was True")
    else:
        my_exit(206)

def parse_orchestrator_file(_f: str, _test: bool = False) -> Union[dict, None]:
    if os.path.exists(_f):
        with open(_f, mode='r', encoding="utf-8") as file:
            try:
                data = yaml.safe_load(file)

                if "errors" not in data:
                    print_red(f"{_f} file does not contain key 'errors'")
                    die_orchestrator_exit_code_206(_test)

                valid_keys: list = ['name', 'match_strings', 'behavior']
                valid_behaviours: list = ["RestartOnDifferentNode", "ExcludeNode", "Restart"]

                for x in data["errors"]:
                    expected_types = {
                        "name": str,
                        "match_strings": list
                    }

                    if not isinstance(x, dict):
                        print_red(f"Entry is not of type dict but {type(x)}")
                        die_orchestrator_exit_code_206(_test)

                    if set(x.keys()) != set(valid_keys):
                        print_red(f"{x.keys()} does not match {valid_keys}")
                        die_orchestrator_exit_code_206(_test)

                    if x["behavior"] not in valid_behaviours:
                        print_red(f"behavior-entry {x['behavior']} is not in valid_behaviours: {', '.join(valid_behaviours)}")
                        die_orchestrator_exit_code_206(_test)

                    for key, expected_type in expected_types.items():
                        if not isinstance(x[key], expected_type):
                            print_red(f"{key}-entry is not {expected_type.__name__} but {type(x[key])}")
                            die_orchestrator_exit_code_206(_test)

                    for y in x["match_strings"]:
                        if not isinstance(y, str):
                            print_red("x['match_strings'] is not a string but {type(x['match_strings'])}")
                            die_orchestrator_exit_code_206(_test)

                return data
            except Exception as e:
                print(f"Error while parse_experiment_parameters({_f}): {e}")
    else:
        print_red(f"{_f} could not be found")

    return None

def set_orchestrator() -> None:
    with spinner("Setting orchestrator..."):
        global orchestrator

        if args.orchestrator_file:
            if SYSTEM_HAS_SBATCH:
                orchestrator = parse_orchestrator_file(args.orchestrator_file, False)
            else:
                print_yellow("--orchestrator_file will be ignored on non-sbatch-systems.")

def check_if_has_random_steps() -> None:
    if (not args.continue_previous_job and "--continue" not in sys.argv) and (args.num_random_steps == 0 or not args.num_random_steps) and args.model not in ["EXTERNAL_GENERATOR", "SOBOL", "PSEUDORANDOM"]:
        _fatal_error("You have no random steps set. This is only allowed in continued jobs. To start, you need either some random steps, or a continued run.", 233)

def add_exclude_to_defective_nodes() -> None:
    with spinner("Adding excluded nodes..."):
        if args.exclude:
            entries = [entry.strip() for entry in args.exclude.split(',')]

            for entry in entries:
                count_defective_nodes(None, entry)

def check_max_eval(_max_eval: int) -> None:
    with spinner("Checking max_eval..."):
        if not _max_eval:
            _fatal_error("--max_eval needs to be set!", 19)

def parse_parameters() -> Any:
    cli_params_experiment_parameters = None
    if args.parameter:
        parse_experiment_parameters()
        cli_params_experiment_parameters = experiment_parameters

    return cli_params_experiment_parameters

def create_pareto_front_table(idxs: List[int], metric_x: str, metric_y: str) -> Table:
    table = Table(title=f"Pareto-Front for {metric_y}/{metric_x}:", show_lines=True)

    rows = _pareto_front_table_read_csv()
    if not rows:
        table.add_column("No data found")
        return table

    filtered_rows = _pareto_front_table_filter_rows(rows, idxs)
    if not filtered_rows:
        table.add_column("No matching entries")
        return table

    param_cols, result_cols = _pareto_front_table_get_columns(filtered_rows[0])

    _pareto_front_table_add_headers(table, param_cols, result_cols)
    _pareto_front_table_add_rows(table, filtered_rows, param_cols, result_cols)

    return table

def _pareto_front_table_read_csv() -> List[Dict[str, str]]:
    with open(RESULT_CSV_FILE, mode="r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def _pareto_front_table_filter_rows(rows: List[Dict[str, str]], idxs: List[int]) -> List[Dict[str, str]]:
    result = []
    for row in rows:
        try:
            trial_index = int(row["trial_index"])
        except (KeyError, ValueError):
            continue

        if row.get("trial_status", "").strip().upper() == "COMPLETED" and trial_index in idxs:
            result.append(row)
    return result

def _pareto_front_table_get_columns(first_row: Dict[str, str]) -> Tuple[List[str], List[str]]:
    all_columns = list(first_row.keys())
    ignored_cols = set(special_col_names) - {"trial_index"}

    param_cols = [col for col in all_columns if col not in ignored_cols and col not in arg_result_names and not col.startswith("OO_Info_")]
    result_cols = [col for col in arg_result_names if col in all_columns]
    return param_cols, result_cols

def _pareto_front_table_add_headers(table: Table, param_cols: List[str], result_cols: List[str]) -> None:
    for col in param_cols:
        table.add_column(col, justify="center")
    for col in result_cols:
        table.add_column(Text(f"{col}", style="cyan"), justify="center")

def _pareto_front_table_add_rows(table: Table, rows: List[Dict[str, str]], param_cols: List[str], result_cols: List[str]) -> None:
    for row in rows:
        values = [str(helpers.to_int_when_possible(row[col])) for col in param_cols]
        result_values = [Text(str(helpers.to_int_when_possible(row[col])), style="cyan") for col in result_cols]
        table.add_row(*values, *result_values, style="bold green")

def pareto_front_as_rich_table(idxs: list, metric_x: str, metric_y: str) -> Optional[Table]:
    if not os.path.exists(RESULT_CSV_FILE):
        print_debug(f"pareto_front_as_rich_table: File '{RESULT_CSV_FILE}' not found")
        return None

    return create_pareto_front_table(idxs, metric_x, metric_y)

def supports_sixel() -> bool:
    term = os.environ.get("TERM", "").lower()
    if "xterm" in term or "mlterm" in term:
        return True

    try:
        output = subprocess.run(["tput", "setab", "256"], capture_output=True, text=True, check=True)
        if output.returncode == 0 and "sixel" in output.stdout.lower():
            return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return False

def plot_pareto_frontier_sixel(data: Any, x_metric: str, y_metric: str) -> None:
    if data is None:
        print("[italic yellow]The data seems to be empty. Cannot plot pareto frontier.[/]")
        return

    if not supports_sixel():
        print(f"[italic yellow]Your console does not support sixel-images. Will not print Pareto-frontier as a matplotlib-sixel-plot for {x_metric}/{y_metric}.[/]")
        return

    import matplotlib.pyplot as plt

    means = data[x_metric][y_metric]["means"]

    x_values = means[x_metric]
    y_values = means[y_metric]

    fig, _ax = plt.subplots()

    _ax.scatter(x_values, y_values, s=50, marker='x', c='blue', label='Data Points')

    _ax.set_xlabel(x_metric)
    _ax.set_ylabel(y_metric)

    _ax.set_title(f'Pareto-Front {x_metric}/{y_metric}')

    _ax.ticklabel_format(style='plain', axis='both', useOffset=False)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp_file:
        plt.savefig(tmp_file.name, dpi=300)

        print_image_to_cli(tmp_file.name, 1000)

    plt.close(fig)

def _pareto_front_general_validate_shapes(x: np.ndarray, y: np.ndarray) -> None:
    if x.shape != y.shape:
        raise ValueError("Input arrays x and y must have the same shape.")

def _pareto_front_general_compare(
    xi: float, yi: float, xj: float, yj: float,
    x_minimize: bool, y_minimize: bool
) -> bool:
    x_better_eq = xj <= xi if x_minimize else xj >= xi
    y_better_eq = yj <= yi if y_minimize else yj >= yi
    x_strictly_better = xj < xi if x_minimize else xj > xi
    y_strictly_better = yj < yi if y_minimize else yj > yi

    return bool(x_better_eq and y_better_eq and (x_strictly_better or y_strictly_better))

def _pareto_front_general_find_dominated(
    x: np.ndarray, y: np.ndarray, x_minimize: bool, y_minimize: bool
) -> np.ndarray:
    num_points = len(x)
    is_dominated = np.zeros(num_points, dtype=bool)

    for i in range(num_points):
        for j in range(num_points):
            if i == j:
                continue

            if _pareto_front_general_compare(x[i], y[i], x[j], y[j], x_minimize, y_minimize):
                is_dominated[i] = True
                break

    return is_dominated

def pareto_front_general(
    x: np.ndarray,
    y: np.ndarray,
    x_minimize: bool = True,
    y_minimize: bool = True
) -> np.ndarray:
    try:
        _pareto_front_general_validate_shapes(x, y)
        is_dominated = _pareto_front_general_find_dominated(x, y, x_minimize, y_minimize)
        return np.where(~is_dominated)[0]
    except Exception as e:
        print("Error in pareto_front_general:", str(e))
        return np.array([], dtype=int)

def _pareto_front_aggregate_data(path_to_calculate: str) -> Optional[Dict[Tuple[int, str], Dict[str, Dict[str, float]]]]:
    results_csv_file = f"{path_to_calculate}/{RESULTS_CSV_FILENAME}"
    result_names_file = f"{path_to_calculate}/result_names.txt"

    if not os.path.exists(results_csv_file) or not os.path.exists(result_names_file):
        return None

    with open(result_names_file, mode="r", encoding="utf-8") as f:
        result_names = [line.strip() for line in f if line.strip()]

    records: dict = defaultdict(lambda: {'means': {}})

    with open(results_csv_file, encoding="utf-8", mode="r", newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            trial_index = int(row['trial_index'])
            arm_name = row['arm_name']
            key = (trial_index, arm_name)

            for metric in result_names:
                if metric in row:
                    try:
                        records[key]['means'][metric] = float(row[metric])
                    except ValueError:
                        continue

    return records

def _pareto_front_filter_complete_points(
    path_to_calculate: str,
    records: Dict[Tuple[int, str], Dict[str, Dict[str, float]]],
    primary_name: str,
    secondary_name: str
) -> List[Tuple[Tuple[int, str], float, float]]:
    points = []
    for key, metrics in records.items():
        means = metrics['means']
        if primary_name in means and secondary_name in means:
            x_val = means[primary_name]
            y_val = means[secondary_name]
            points.append((key, x_val, y_val))
    if len(points) == 0:
        raise ValueError(f"No full data points with both objectives found in {path_to_calculate}.")
    return points

def _pareto_front_transform_objectives(
    points: List[Tuple[Any, float, float]],
    primary_name: str,
    secondary_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    primary_idx = arg_result_names.index(primary_name)
    secondary_idx = arg_result_names.index(secondary_name)

    x = np.array([p[1] for p in points])
    y = np.array([p[2] for p in points])

    if arg_result_min_or_max[primary_idx] == "max":
        x = -x
    elif arg_result_min_or_max[primary_idx] != "min":
        raise ValueError(f"Unknown mode for {primary_name}: {arg_result_min_or_max[primary_idx]}")

    if arg_result_min_or_max[secondary_idx] == "max":
        y = -y
    elif arg_result_min_or_max[secondary_idx] != "min":
        raise ValueError(f"Unknown mode for {secondary_name}: {arg_result_min_or_max[secondary_idx]}")

    return x, y

def _pareto_front_select_pareto_points(
    x: np.ndarray,
    y: np.ndarray,
    x_minimize: bool,
    y_minimize: bool,
    points: List[Tuple[Any, float, float]],
    num_points: int
) -> List[Tuple[Any, float, float]]:
    indices = pareto_front_general(x, y, x_minimize, y_minimize)
    sorted_indices = indices[np.argsort(x[indices])]
    sorted_indices = sorted_indices[:num_points]
    selected_points = [points[i] for i in sorted_indices]
    return selected_points

def _pareto_front_build_return_structure(
    path_to_calculate: str,
    selected_points: List[Tuple[Any, float, float]],
    records: Dict[Tuple[int, str], Dict[str, Dict[str, float]]],
    absolute_metrics: List[str],
    primary_name: str,
    secondary_name: str
) -> dict:
    results_csv_file = f"{path_to_calculate}/{RESULTS_CSV_FILENAME}"
    result_names_file = f"{path_to_calculate}/result_names.txt"

    with open(result_names_file, mode="r", encoding="utf-8") as f:
        result_names = [line.strip() for line in f if line.strip()]

    csv_rows = {}
    with open(results_csv_file, mode="r", encoding="utf-8", newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            trial_index = int(row['trial_index'])
            csv_rows[trial_index] = row

    ignored_columns = {'trial_index', 'arm_name', 'trial_status', 'generation_node'}
    ignored_columns.update(result_names)

    param_dicts = []
    idxs = []
    means_dict = defaultdict(list)

    for (trial_index, arm_name), _, _ in selected_points:
        row = csv_rows.get(trial_index, {})
        if row == {} or row is None or row['arm_name'] != arm_name:
            print_debug(f"_pareto_front_build_return_structure: trial_index '{trial_index}' could not be found and row returned as None")
            continue

        idxs.append(int(row["trial_index"]))

        param_dict: dict[str, int | float | str] = {}
        for key, value in row.items():
            if key not in ignored_columns:
                try:
                    param_dict[key] = int(value)
                except ValueError:
                    try:
                        param_dict[key] = float(value)
                    except ValueError:
                        param_dict[key] = value

        param_dicts.append(param_dict)

        for metric in absolute_metrics:
            means_dict[metric].append(records[(trial_index, arm_name)]['means'].get(metric, float("nan")))

    ret = {
        primary_name: {
            secondary_name: {
                "absolute_metrics": absolute_metrics,
                "param_dicts": param_dicts,
                "means": dict(means_dict),
                "idxs": idxs
            },
            "absolute_metrics": absolute_metrics
        }
    }

    return ret

def get_pareto_frontier_points(
    path_to_calculate: str,
    primary_objective: str,
    secondary_objective: str,
    x_minimize: bool,
    y_minimize: bool,
    absolute_metrics: List[str],
    num_points: int
) -> Optional[dict]:
    records = _pareto_front_aggregate_data(path_to_calculate)

    if records is None:
        return None

    points = _pareto_front_filter_complete_points(path_to_calculate, records, primary_objective, secondary_objective)
    x, y = _pareto_front_transform_objectives(points, primary_objective, secondary_objective)
    selected_points = _pareto_front_select_pareto_points(x, y, x_minimize, y_minimize, points, num_points)
    result = _pareto_front_build_return_structure(path_to_calculate, selected_points, records, absolute_metrics, primary_objective, secondary_objective)

    return result

def save_experiment_state() -> None:
    try:
        if ax_client is None or ax_client.experiment is None:
            print_red("save_experiment_state: ax_client or ax_client.experiment is None, cannot save.")
            return
        state_path = get_current_run_folder("experiment_state.json")
        ax_client.save_to_json_file(state_path)
    except Exception as e:
        print(f"Error saving experiment state: {e}")

def wait_for_state_file(state_path: str, min_size: int = 5, max_wait_seconds: int = 60) -> bool:
    try:
        if not os.path.exists(state_path):
            print(f"[ERROR] File '{state_path}' does not exist.")
            return False

        i = 0
        while True:
            try:
                file_size = os.path.getsize(state_path)
            except OSError as e:
                print_debug(f"[ERROR] File '{state_path}' cannot be read: {e}")
                return False

            if file_size >= min_size:
                print_debug(f"[INFO] File '{state_path}' is now large enough ({file_size} Bytes).")
                return True

            if i >= max_wait_seconds:
                print_debug(f"[ERROR] Timeout: File '{state_path}' was not larger than {min_size} bytes after waiting for {max_wait_seconds} seconds.")
                return False

            print_debug(f"\r[yellow] File '{state_path}' is too small ({file_size} Bytes), waiting ... {i}s")
            sys.stdout.flush()
            time.sleep(1)
            i += 1

    except Exception as e:
        print_red(f"[ERROR] Unexpected error: {e}")
        return False

def load_json_with_retry(state_path: str, timeout: int = 30, retry_interval: int = 1) -> Optional[dict]:
    start_time = time.time()

    while True:
        if not os.path.exists(state_path):
            print_debug(f"load_json_with_retry(state_path = {state_path}, timeout = {timeout}, retry_interval: {retry_interval}): File does not exist: {state_path}")

        try:
            with open(state_path, mode="r", encoding="utf-8") as f:
                data = json.load(f)
                return data
        except Exception as e:
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                print_debug(f"\nCould not load valid JSON after {elapsed} second of trying on path {state_path}: {e}")
                return None

            print_debug(f"Wait for valid JSON {state_path}... error: {e}")
            time.sleep(retry_interval)

    return None

def load_experiment_state() -> None:
    global ax_client
    state_path = get_current_run_folder("experiment_state.json")

    if not os.path.exists(state_path):
        return

    if args.worker_generator_path:
        if not wait_for_state_file(state_path):
            my_exit(188)

    data = load_json_with_retry(state_path)

    if data is None:
        print(f"Could not read valid JSON from {state_path}")
        return

    try:
        arms_seen: dict = {}
        for arm in data.get("arms", []):
            name = arm.get("name")
            sig = arm.get("parameters")
            if not name:
                continue
            if name in arms_seen and arms_seen[name] != sig:
                new_name = f"{name}_{uuid.uuid4().hex[:6]}"
                print(f"Renaming conflicting arm '{name}' -> '{new_name}'")
                arm["name"] = new_name
            arms_seen[name] = sig

        temp_path = state_path + ".no_conflicts.json"
        with open(temp_path, encoding="utf-8", mode="w") as f:
            json.dump(data, f)

        ax_client = AxClient.load_from_json_file(temp_path)
    except Exception as e:
        print(f"Error loading experiment state: {e}")

def sanitize_json(obj: Any) -> Any:
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_json(x) for x in obj]
    return obj

def set_arg_min_or_max_if_required(path_to_calculate: str) -> None:
    global arg_result_names
    global arg_result_min_or_max

    if path_to_calculate:
        _found_result_min_max = []
        _default_min_max = "min"

        _found_result_names = []

        _look_for_result_names_file = f"{path_to_calculate}/result_names.txt"

        if os.path.exists(_look_for_result_names_file):
            try:
                with open(_look_for_result_names_file, 'r', encoding='utf-8') as _file:
                    _content = _file.read()
                    _found_result_names = _content.split('\n')

                    if _found_result_names and _found_result_names[-1] == '':
                        _found_result_names.pop()
            except FileNotFoundError:
                print_red(f"Error: The file at '{_look_for_result_names_file}' was not found.")
            except IOError as e:
                print_red(f"Error reading file '{_look_for_result_names_file}': {e}")
        else:
            print_yellow(f"{_look_for_result_names_file} not found!")

        for _n in range(len(_found_result_names)):
            _min_max = get_min_max_from_file(path_to_calculate, _n, _default_min_max)

            _found_result_min_max.append(_min_max)

        arg_result_names = _found_result_names
        arg_result_min_or_max = _found_result_min_max

def get_calculated_frontier(path_to_calculate: str, metric_x: str, metric_y: str, x_minimize: bool, y_minimize: bool, res_names: list) -> Any:
    try:
        state_dir = os.path.join(get_current_run_folder(), "state_files")
        os.makedirs(state_dir, exist_ok=True)

        json_file = os.path.join(state_dir, "pareto_front_data.json")

        set_arg_min_or_max_if_required(path_to_calculate)

        frontier = get_pareto_frontier_points(
            path_to_calculate=path_to_calculate,
            primary_objective=metric_x,
            secondary_objective=metric_y,
            x_minimize=x_minimize,
            y_minimize=y_minimize,
            absolute_metrics=res_names,
            num_points=count_done_jobs()
        )

        if frontier is None:
            print_red(f"Could not get frontier for {path_to_calculate}")
            return None

        frontier = sanitize_json(frontier)

        pickled_data = pickle.dumps(frontier)
        b64_data = base64.b64encode(pickled_data).decode("utf-8")

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump({"pickle_data": b64_data}, f, indent=2)

        return frontier

    except Exception as e:
        print_red(f"Error in get_calculated_frontier: {str(e)}")
        return None

def live_share_after_pareto() -> None:
    if args.calculate_pareto_front_of_job is not None or args.live_share:
        if not args.live_share:
            live_share_file = f"{args.calculate_pareto_front_of_job}/state_files/live_share"
            if os.path.exists(live_share_file):
                try:
                    with open(live_share_file, "r", encoding="utf-8") as f:
                        first_line = f.readline().strip()
                        if first_line == "1":
                            args.live_share = True
                except Exception as e:
                    print_debug(f"Error reading {live_share_file}: {e}")
                    args.live_share = False

        force_live_share()

def get_result_minimize_flag(path_to_calculate: str, resname: str) -> bool:
    result_names_path = os.path.join(path_to_calculate, "result_names.txt")
    result_min_max_path = os.path.join(path_to_calculate, "result_min_max.txt")

    if not os.path.isdir(path_to_calculate):
        _fatal_error(f"Error: Directory '{path_to_calculate}' does not exist.", 24)

    if not os.path.isfile(result_names_path) or not os.path.isfile(result_min_max_path):
        _fatal_error(f"Error: Missing 'result_names.txt' or 'result_min_max.txt' in '{path_to_calculate}'.", 24)

    try:
        with open(result_names_path, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f]
    except Exception as e:
        _fatal_error(f"Error: Failed to read 'result_names.txt': {e}", 24)

    if resname not in names:
        _fatal_error(f"Error: Result name '{resname}' not found in 'result_names.txt'.", 24)

    index = names.index(resname)

    try:
        with open(result_min_max_path, "r", encoding="utf-8") as f:
            minmax = [line.strip().lower() for line in f]
    except Exception as e:
        _fatal_error(f"Error: Failed to read 'result_min_max.txt': {e}", 24)

    if index >= len(minmax):
        _fatal_error(f"Error: Not enough entries in 'result_min_max.txt' for index {index}.", 24)

    return minmax[index] == "min"

def get_pareto_front_data(path_to_calculate: str, res_names: list) -> dict:
    pareto_front_data: dict = {}

    all_combinations = list(combinations(range(len(arg_result_names)), 2))

    skip = False

    for i, j in all_combinations:
        if not skip:
            metric_x = arg_result_names[i]
            metric_y = arg_result_names[j]

            x_minimize = get_result_minimize_flag(path_to_calculate, metric_x)
            y_minimize = get_result_minimize_flag(path_to_calculate, metric_y)

            try:
                if metric_x not in pareto_front_data:
                    pareto_front_data[metric_x] = {}

                pareto_front_data[metric_x][metric_y] = get_calculated_frontier(path_to_calculate, metric_x, metric_y, x_minimize, y_minimize, res_names)
            except ax.exceptions.core.DataRequiredError as e:
                print_red(f"Error computing Pareto frontier for {metric_x} and {metric_y}: {e}")
            except SignalINT:
                print_red("Calculating Pareto-fronts was cancelled by pressing CTRL-c")
                skip = True

    return pareto_front_data

def show_pareto_frontier_data(path_to_calculate: str, res_names: list, disable_sixel_and_table: bool = False) -> None:
    if len(res_names) <= 1:
        print_debug(f"--result_names (has {len(res_names)} entries) must be at least 2.")
        return

    pareto_front_data: dict = get_pareto_front_data(path_to_calculate, res_names)

    pareto_points: dict = {}

    for metric_x in pareto_front_data.keys():
        if metric_x not in pareto_points:
            pareto_points[metric_x] = {}

        for metric_y in pareto_front_data[metric_x].keys():
            calculated_frontier = pareto_front_data[metric_x][metric_y]

            hide_pareto = os.environ.get('HIDE_PARETO_FRONT_TABLE_DATA')

            if not disable_sixel_and_table:
                if hide_pareto is None:
                    plot_pareto_frontier_sixel(calculated_frontier, metric_x, metric_y)
                else:
                    print(f"Not showing Pareto-front-sixel for {path_to_calculate}")

            if len(calculated_frontier[metric_x][metric_y]["idxs"]):
                pareto_points[metric_x][metric_y] = sorted(calculated_frontier[metric_x][metric_y]["idxs"])

            rich_table = pareto_front_as_rich_table(
                calculated_frontier[metric_x][metric_y]["idxs"],
                metric_y,
                metric_x
            )

            if rich_table is not None:
                if not disable_sixel_and_table:
                    if hide_pareto is None:
                        console.print(rich_table)
                    else:
                        print(f"Not showing Pareto-front-table for {path_to_calculate}")

                with open(get_current_run_folder("pareto_front_table.txt"), mode="a", encoding="utf-8") as text_file:
                    with console.capture() as capture:
                        console.print(rich_table)
                    text_file.write(capture.get())

    with open(get_current_run_folder("pareto_idxs.json"), mode="w", encoding="utf-8") as pareto_idxs_json_handle:
        json.dump(pareto_points, pareto_idxs_json_handle)

    live_share_after_pareto()

def show_available_hardware_and_generation_strategy_string(gpu_string: str, gpu_color: str) -> None:
    cpu_count = os.cpu_count()

    gs_string = get_generation_strategy_string()

    try:
        cpu_count = len(os.sched_getaffinity(0))
    except AttributeError:
        pass

    if gpu_string:
        console.print(f"[green]You have {cpu_count} CPUs available for the main process.[/green] [{gpu_color}]{gpu_string}[/{gpu_color}] [green]{gs_string}[/green]")
    else:
        print_green(f"You have {cpu_count} CPUs available for the main process. {gs_string}")

def write_args_overview_table() -> None:
    table = Table(title="Arguments Overview")
    table.add_column("Key", justify="left", style="bold")
    table.add_column("Value", justify="left", style="dim")

    for key, value in vars(args).items():
        table.add_row(key, str(value))

    table_str = ""

    with console.capture() as capture:
        console.print(table)

    table_str = capture.get()

    with open(get_current_run_folder("args_overview.txt"), mode="w", encoding="utf-8") as text_file:
        text_file.write(table_str)

def show_experiment_overview_table() -> None:
    table = Table(title="Experiment overview", show_header=True)

    #random_step = gs_data[0]
    #systematic_step = gs_data[1]

    table.add_column("Setting", style="green")
    table.add_column("Value", style="green")

    if args.model:
        table.add_row("Model for non-random steps", str(args.model))
    table.add_row("Max. nr. evaluations", str(max_eval))
    if args.max_eval and args.max_eval != max_eval:
        table.add_row("Max. nr. evaluations (from arguments)", str(args.max_eval))

    table.add_row("Number random steps", str(random_steps))
    if args.num_random_steps != random_steps:
        table.add_row("Number random steps (from arguments)", str(args.num_random_steps))

    table.add_row("Nr. of workers (parameter)", str(args.num_parallel_jobs))

    if SYSTEM_HAS_SBATCH:
        table.add_row("Main process memory (GB)", str(args.main_process_gb))
        table.add_row("Worker memory (GB)", str(args.mem_gb))

    if NR_INSERTED_JOBS:
        table.add_row("Nr. imported jobs", str(NR_INSERTED_JOBS))

    if args.seed is not None:
        table.add_row("Seed", str(args.seed))

    with console.capture() as capture:
        console.print(table)

    table_str = capture.get()

    with open(get_current_run_folder("experiment_overview.txt"), mode="w", encoding="utf-8") as text_file:
        text_file.write(table_str)

def write_files_and_show_overviews() -> None:
    with spinner("Write files and show overview"):
        write_state_file("num_random_steps", str(args.num_random_steps))
        set_global_executor()
        load_existing_job_data_into_ax_client()
        write_args_overview_table()
        show_experiment_overview_table()
        save_global_vars()
        write_process_info()
        write_continue_run_uuid_to_file()

def write_git_version() -> None:
    with spinner("Writing git information"):
        folder = get_current_run_folder()
        os.makedirs(folder, exist_ok=True)
        file_path = os.path.join(folder, "git_version")

        try:
            commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()

            git_tag = ""

            try:
                git_tag = subprocess.check_output(["git", "describe", "--tags"], text=True, stderr=subprocess.DEVNULL).strip()
                git_tag = f" ({git_tag})"
            except subprocess.CalledProcessError:
                pass

            if commit_hash:
                with open(file_path, mode="w", encoding="utf-8") as f:
                    f.write(f"Commit: {commit_hash}{git_tag}\n")

        except subprocess.CalledProcessError:
            pass

def write_job_start_file() -> None:
    with spinner("Writing job_start_time file..."):
        fn = get_current_run_folder("job_start_time.txt")
        try:
            with open(fn, mode='w', encoding="utf-8") as f:
                f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        except Exception as e:
            print_red(f"Error trying to write {fn}: {e}")

def write_live_share_file_if_needed() -> None:
    with spinner("Writing live_share file if it is present..."):
        if args.live_share:
            write_state_file("live_share", "1\n")

def write_username_statefile() -> None:
    with spinner("Writing username state file..."):
        _path = get_current_run_folder()
        if args.username:
            file_path = f"{_path}/state_files/username"

            try:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, mode="w", encoding="utf-8") as f:
                    f.write(args.username)
            except Exception as e:
                print_red(f"Error writing to file: {e}")

def write_revert_to_random_when_seemingly_exhausted_file() -> None:
    with spinner("Writing revert_to_random_when_seemingly_exhausted file ..."):
        _path = get_current_run_folder()
        file_path = f"{_path}/state_files/revert_to_random_when_seemingly_exhausted"

        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, mode="w", encoding="utf-8") as f:
                f.write("1\n")
        except Exception as e:
            print_red(f"Error writing to file: {e}")

def debug_vars_unused_by_python_for_linter() -> None:
    print_debug(
        f"partition: {args.partition}, "
        f"root_venv_dir: {args.root_venv_dir}, "
        f"checkout_to_latest_tested_version: {args.checkout_to_latest_tested_version}, "
        f"send_anonymized_usage_stats: {args.send_anonymized_usage_stats}, "
        f"show_ram_every_n_seconds: {args.show_ram_every_n_seconds}, "
        f"workdir: {args.workdir}, "
        f"run_mode: {args.run_mode}"
    )

def get_constraints() -> list:
    constraints_list: List[str] = []

    if _has_explicit_constraints():
        constraints_list = args.experiment_constraints
    elif _should_load_previous_constraints():
        constraints_list = _load_previous_constraints(args.continue_previous_job)

    if len(constraints_list):
        constraints_list = _normalize_constraints_list(constraints_list)
        constraints_list = _filter_valid_constraints(constraints_list)

    return constraints_list

def _has_explicit_constraints() -> bool:
    return bool(args.experiment_constraints)

def _should_load_previous_constraints() -> bool:
    return bool(args.continue_previous_job and not args.disable_previous_job_constraint and (args.experiment_constraints is None or not len(args.experiment_constraints)))

def _load_previous_constraints(job_path: str) -> list:
    constraint_file = os.path.join(job_path, "state_files", "constraints")

    if not os.path.exists(constraint_file):
        return []

    with open(constraint_file, "r", encoding="utf-8") as f:
        raw_constraints = [line.strip() for line in f if line.strip()]

    encoded_constraints = [
        base64.b64encode(c.encode("utf-8")).decode("utf-8") for c in raw_constraints
    ]

    return [encoded_constraints]

def _normalize_constraints_list(constraints_list: list) -> List[str]:
    if isinstance(constraints_list, list) and len(constraints_list) == 1 and isinstance(constraints_list[0], list):
        return constraints_list[0]
    return constraints_list

def _load_experiment_json(continue_previous_job_path: str) -> dict:
    experiment_json_data = {}

    json_file_path = os.path.join(continue_previous_job_path, "state_files", "ax_client.experiment.json")

    if os.path.isfile(json_file_path):
        try:
            with open(json_file_path, "r", encoding="utf-8") as f:
                experiment_json_data = json.load(f)
        except Exception as e:
            print_yellow(f"Error while reading the file: {json_file_path}\n{str(e)}")
    else:
        print_yellow(f"Warning: File not found: {json_file_path}")

    return experiment_json_data

def _filter_valid_constraints(constraints: List[str]) -> List[str]:
    global global_param_names

    final_constraints_list: List[str] = []

    if len(global_param_names) == 0:
        if args.continue_previous_job:
            experiment_json_data = _load_experiment_json(args.continue_previous_job)

            params = experiment_json_data["search_space"]["parameters"]

            global_param_names = [param["name"] for param in params]
        else:
            print_red("_filter_valid_constraints: No parameters found, and not a continued job. Constraints will stay empty.")

    for raw_constraint in constraints:
        decoded = decode_if_base64(" ".join(raw_constraint))
        decoded = decoded.rstrip("\n\r")

        is_ax = is_ax_compatible_constraint(decoded, global_param_names)
        is_equation = is_valid_equation(decoded, global_param_names)

        if is_ax:
            final_constraints_list.append(decoded)
        elif is_equation:
            print_debug(f"Added Post-Generation-constraint '{decoded}'")
            post_generation_constraints.append(decoded)
        else:
            print_red(f"Invalid constraint found: '{decoded}' (is valid ax? {is_ax}, is valid equation? {is_equation}).")

            my_exit(19)

    return final_constraints_list

def load_username_to_args(path_to_calculate: str) -> None:
    if not path_to_calculate:
        return

    username_file_path = os.path.join(path_to_calculate, "state_files", "username")
    if os.path.isfile(username_file_path) and not args.username:
        try:
            with open(username_file_path, mode="r", encoding="utf-8") as f:
                args.username = f.readline().strip()
        except Exception as e:
            print_red(f"Error reading from file: {e}")

def find_results_paths(base_path: str) -> list:
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Path not found: {base_path}")

    if not os.path.isdir(base_path):
        raise NotADirectoryError(f"No directory: {base_path}")

    direct_result_file = os.path.join(base_path, RESULTS_CSV_FILENAME)
    if os.path.isfile(direct_result_file):
        return [base_path]

    found_paths = []

    if "DO_NOT_SEARCH_FOLDERS_FOR_RESULTS_CSV" not in os.environ:
        with spinner(f"Searching for subfolders with {RESULTS_CSV_FILENAME}..."):
            for root, _, files in os.walk(base_path):
                if RESULTS_CSV_FILENAME in files:
                    found_paths.append(root)

    return list(set(found_paths))

def post_job_calculate_pareto_front() -> None:
    if not args.calculate_pareto_front_of_job:
        return

    failure = False

    _paths_to_calculate = []

    for _path_to_calculate in list(set(args.calculate_pareto_front_of_job)):
        try:
            found_paths = find_results_paths(_path_to_calculate)

            for _fp in found_paths:
                if _fp not in _paths_to_calculate:
                    _paths_to_calculate.append(_fp)
        except (FileNotFoundError, NotADirectoryError) as e:
            print_red(f"post_job_calculate_pareto_front: find_results_paths('{_path_to_calculate}') failed with {e}")

            failure = True

    for _path_to_calculate in _paths_to_calculate:
        for path_to_calculate in found_paths:
            if not job_calculate_pareto_front(path_to_calculate):
                failure = True

    if failure:
        my_exit(24)

    my_exit(0)

def job_calculate_pareto_front(path_to_calculate: str, disable_sixel_and_table: bool = False) -> bool:
    pf_start_time = time.time()

    if not path_to_calculate:
        return False

    global CURRENT_RUN_FOLDER
    global RESULT_CSV_FILE
    global arg_result_names

    if not path_to_calculate:
        print_red("Can only calculate pareto front of previous job when --calculate_pareto_front_of_job is set")
        return False

    if not os.path.exists(path_to_calculate):
        print_red(f"Path '{path_to_calculate}' does not exist")
        return False

    ax_client_json = f"{path_to_calculate}/state_files/ax_client.experiment.json"

    if not os.path.exists(ax_client_json):
        print_red(f"Path '{ax_client_json}' not found")
        return False

    checkpoint_file: str = f"{path_to_calculate}/state_files/checkpoint.json"
    if not os.path.exists(checkpoint_file):
        print_red(f"The checkpoint file '{checkpoint_file}' does not exist")
        return False

    RESULT_CSV_FILE = f"{path_to_calculate}/{RESULTS_CSV_FILENAME}"
    if not os.path.exists(RESULT_CSV_FILE):
        print_red(f"{RESULT_CSV_FILE} not found")
        return False

    res_names = []

    res_names_file = f"{path_to_calculate}/result_names.txt"
    if not os.path.exists(res_names_file):
        print_red(f"File '{res_names_file}' does not exist")
        return False

    try:
        with open(res_names_file, "r", encoding="utf-8") as file:
            lines = file.readlines()
    except Exception as e:
        print_red(f"Error reading file '{res_names_file}': {e}")
        return False

    for line in lines:
        entry = line.strip()
        if entry != "":
            res_names.append(entry)

    if len(res_names) < 2:
        print_red(f"Error: There are less than 2 result names (is: {len(res_names)}, {', '.join(res_names)}) in {path_to_calculate}. Cannot continue calculating the pareto front.")
        return False

    load_username_to_args(path_to_calculate)

    CURRENT_RUN_FOLDER = path_to_calculate

    arg_result_names = res_names

    load_experiment_parameters_from_checkpoint_file(checkpoint_file, False)

    if experiment_parameters is None:
        return False

    show_pareto_or_error_msg(path_to_calculate, res_names, disable_sixel_and_table)

    pf_end_time = time.time()

    print_debug(f"Calculating the Pareto-front took {pf_end_time - pf_start_time} seconds")

    return True

def set_arg_states_from_continue() -> None:
    if args.continue_previous_job and not args.num_random_steps:
        num_random_steps_file = f"{args.continue_previous_job}/state_files/num_random_steps"

        if os.path.exists(num_random_steps_file):
            args.num_random_steps = int(open(num_random_steps_file, mode="r", encoding="utf-8").readline().strip())
        else:
            print_red(f"Cannot find >{num_random_steps_file}<. Will use default, it being >{args.num_random_steps}<.")

    if args.continue_previous_job:
        if os.path.exists(f"{args.continue_previous_job}/state_files/revert_to_random_when_seemingly_exhausted"):
            args.revert_to_random_when_seemingly_exhausted = True

def write_result_min_max_file() -> None:
    with spinner("Writing result min/max file..."):
        try:
            fn = get_current_run_folder("result_min_max.txt")
            with open(fn, mode="a", encoding="utf-8") as myfile:
                for rarg in arg_result_min_or_max:
                    original_print(rarg, file=myfile)
        except Exception as e:
            print_red(f"Error trying to open file '{fn}': {e}")

def write_result_names_file() -> None:
    with spinner("Writing result names file..."):
        try:
            fn = get_current_run_folder("result_names.txt")
            with open(fn, mode="a", encoding="utf-8") as myfile:
                for rarg in arg_result_names:
                    original_print(rarg, file=myfile)
        except Exception as e:
            print_red(f"Error trying to open file '{fn}': {e}")

def run_program_once(params: Optional[dict] = None) -> None:
    if not args.run_program_once:
        print_debug("[yellow]No setup script specified (run_program_once). Skipping setup.[/yellow]")
        return

    if params is None:
        params = {}

    if isinstance(args.run_program_once, str):
        command_str = args.run_program_once
        for k, v in params.items():
            placeholder = f"%({k})"
            command_str = command_str.replace(placeholder, str(v))

        with spinner(f"Executing command: [cyan]{command_str}[/cyan]"):
            result = subprocess.run(command_str, shell=True, check=True)
            if result.returncode == 0:
                console.log("[bold green]Setup script completed successfully ✅[/bold green]")
            else:
                console.log(f"[bold red]Setup script failed with exit code {result.returncode} ❌[/bold red]")

                my_exit(57)

    elif isinstance(args.run_program_once, (list, tuple)):
        with spinner("run_program_once: Executing command list: [cyan]{args.run_program_once}[/cyan]"):
            result = subprocess.run(args.run_program_once, check=True)
            if result.returncode == 0:
                console.log("[bold green]Setup script completed successfully ✅[/bold green]")
            else:
                console.log(f"[bold red]Setup script failed with exit code {result.returncode} ❌[/bold red]")

                my_exit(57)

    else:
        console.print(f"[red]Invalid type for run_program_once: {type(args.run_program_once)}[/red]")

        my_exit(57)

def show_omniopt_call() -> None:
    def remove_ui_url(arg_str: str) -> str:
        return re.sub(r'(?:--ui_url(?:=\S+)?(?:\s+\S+)?)', '', arg_str).strip()

    original_argv = " ".join(sys.argv[1:])
    cleaned = remove_ui_url(original_argv)

    original_print(oo_call + " " + cleaned)

def main() -> None:
    global RESULT_CSV_FILE, ax_client, LOGFILE_DEBUG_GET_NEXT_TRIALS

    check_if_has_random_steps()

    log_worker_creation()

    show_omniopt_call()

    check_slurm_job_id()

    debug_vars_unused_by_python_for_linter()

    set_arg_states_from_continue()

    disable_logging()

    post_job_calculate_pareto_front()

    set_run_folder()

    RESULT_CSV_FILE = create_folder_and_file(get_current_run_folder())

    write_revert_to_random_when_seemingly_exhausted_file()

    write_username_statefile()

    write_result_names_file()

    write_result_min_max_file()

    if args.dryrun:
        set_max_eval(1)

    run_program_once()

    if os.getenv("CI"):
        data_dict: dict = {
            "param1": "value1",
            "param2": "value2",
            "param3": "value3"
        }

        error_description: str = "Some error occurred during execution (this is not a real error!)."

        write_failed_logs(data_dict, error_description)

    save_state_files()

    print_run_info()

    initialize_nvidia_logs()
    write_ui_url()

    LOGFILE_DEBUG_GET_NEXT_TRIALS = get_current_run_folder('get_next_trials.csv')
    cli_params_experiment_parameters = parse_parameters()

    write_live_share_file_if_needed()

    write_job_start_file()

    write_git_version()

    check_max_eval(max_eval)

    _random_steps = get_number_of_steps(max_eval)

    set_random_steps(_random_steps)

    add_exclude_to_defective_nodes()

    handle_random_steps()

    set_global_generation_strategy()

    if args.dryrun:
        set_global_gs_to_HUMAN_INTERVENTION_MINIMUM()

    initialize_ax_client()

    exp_params = get_experiment_parameters(cli_params_experiment_parameters)

    if exp_params is not None:
        ax_client, experiment_args, gpu_string, gpu_color = exp_params
        print_debug(f"experiment_parameters: {experiment_parameters}")

        set_orchestrator()

        init_live_share()

        start_periodic_live_share()

        show_available_hardware_and_generation_strategy_string(gpu_string, gpu_color)

        original_print(f"Run-Program: {global_vars['joined_run_program']}")

        if args.external_generator:
            original_print(f"External-Generator: {decode_if_base64(args.external_generator)}")

        checkpoint_parameters_filepath = get_state_file_name("checkpoint.json.parameters.json")
        save_experiment_parameters(checkpoint_parameters_filepath)

        print_overview_tables(experiment_parameters, experiment_args)

        write_files_and_show_overviews()

        #if args.continue_previous_job:
        #    insert_jobs_from_csv(f"{args.continue_previous_job}/{RESULTS_CSV_FILENAME}")

        for existing_run in args.load_data_from_existing_jobs:
            insert_jobs_from_csv(f"{existing_run}/{RESULTS_CSV_FILENAME}")

            set_global_generation_strategy()

        #start_worker_generators()

        try:
            run_search_with_progress_bar()

            time.sleep(2)
        except ax.exceptions.core.UnsupportedError:
            pass

        end_program()
    else:
        print_red("exp_params is None!")

def load_existing_data_for_worker_generation_path() -> None:
    if args.worker_generator_path:
        with spinner("Loading existing data for worker generators"):
            if not os.path.exists(args.worker_generator_path):
                print_red(f"Cannot continue. '--worker_generator_path {args.worker_generator_path}' does not exist.")
                my_exit(96)

            if not os.path.exists(f"{args.worker_generator_path}/{RESULTS_CSV_FILENAME}"):
                print_red(f"Cannot continue. '--worker_generator_path {args.worker_generator_path}' does not exist.")
                my_exit(96)

            insert_jobs_from_csv(f"{args.worker_generator_path}/{RESULTS_CSV_FILENAME}")

def log_worker_creation() -> None:
    with spinner("Writing worker creation log..."):
        _debug_worker_creation("time, nr_workers, got, requested, phase")

def set_run_folder() -> None:
    with spinner("Setting run folder..."):
        global CURRENT_RUN_FOLDER

        # Ensure run_dir is an absolute path
        run_dir = args.run_dir
        if not os.path.isabs(run_dir):
            run_dir = os.path.abspath(run_dir)

        if args.worker_generator_path:
            print_yellow(f"set_run_folder: Using {args.worker_generator_path} as worker-generation path, will append additional worker to it")

            CURRENT_RUN_FOLDER = args.worker_generator_path

            if not os.path.exists(CURRENT_RUN_FOLDER):
                print_red(f"Cannot join worker generator: --worker_generator_path {args.worker_generator_path} is not a valid directory")

                my_exit(96)

        else:
            RUN_FOLDER_NUMBER: int = 0
            CURRENT_RUN_FOLDER = f"{run_dir}/{global_vars['experiment_name']}/{RUN_FOLDER_NUMBER}"

            while os.path.exists(CURRENT_RUN_FOLDER):
                RUN_FOLDER_NUMBER += 1
                CURRENT_RUN_FOLDER = f"{run_dir}/{global_vars['experiment_name']}/{RUN_FOLDER_NUMBER}"

def print_run_info() -> None:
    console.print(f"[bold]Run-folder[/bold]: [underline]{get_current_run_folder()}[/underline]")
    if args.continue_previous_job:
        original_print(f"Continuation from {args.continue_previous_job}")

def initialize_nvidia_logs() -> None:
    global NVIDIA_SMI_LOGS_BASE
    NVIDIA_SMI_LOGS_BASE = get_current_run_folder('gpu_usage_')

def build_gui_url(config: argparse.Namespace) -> str:
    base_url = get_base_url()
    params = collect_params(config)
    ret = f"{base_url}?{urlencode(params, doseq=True)}"

    return ret

def get_result_names_for_url(value: List) -> str:
    d = dict(v.split("=", 1) if "=" in v else (v, "min") for v in value)
    s = " ".join(f"{k}={v}" for k, v in d.items())

    return s

def collect_params(config: argparse.Namespace) -> dict:
    params = {}
    user_home = os.path.expanduser("~")

    for attr, value in vars(config).items():
        if attr == "run_program":
            params[attr] = global_vars["joined_run_program"]
        elif attr == "result_names" and value:
            params[attr] = get_result_names_for_url(value)
        elif attr == "parameter" and value is not None:
            params.update(process_parameters(config.parameter))
        elif attr == "root_venv_dir":
            if value is not None and os.path.abspath(value) != os.path.abspath(user_home):
                params[attr] = value
        elif isinstance(value, bool):
            params[attr] = int(value)
        elif isinstance(value, list):
            params[attr] = value
        elif value is not None:
            params[attr] = value

    return params

def process_parameters(parameters: list) -> dict:
    params = {}
    for i, param in enumerate(parameters):
        if isinstance(param, dict):
            name = param.get("name", f"param_{i}")
            ptype = param.get("type", "unknown")
        else:
            name = param[0] if len(param) > 0 else f"param_{i}"
            ptype = param[1] if len(param) > 1 else "unknown"

        params[f"parameter_{i}_name"] = name
        params[f"parameter_{i}_type"] = ptype

        if ptype == "range":
            params.update(process_range_parameter(i, param))
        elif ptype == "choice":
            params.update(process_choice_parameter(i, param))
        elif ptype == "fixed":
            params.update(process_fixed_parameter(i, param))

    params["num_parameters"] = len(parameters)
    return params

def process_range_parameter(i: int, param: list) -> dict:
    return {
        f"parameter_{i}_min": param[2] if len(param) > 3 else 0,
        f"parameter_{i}_max": param[3] if len(param) > 3 else 1,
        f"parameter_{i}_number_type": param[4] if len(param) > 4 else "float",
        f"parameter_{i}_log_scale": "false",
    }

def process_choice_parameter(i: int, param: list) -> dict:
    choices = ""
    if len(param) > 2 and param[2]:
        choices = ",".join([c.strip() for c in str(param[2]).split(",")])
    return {f"parameter_{i}_values": choices}

def process_fixed_parameter(i: int, param: list) -> dict:
    return {f"parameter_{i}_value": param[2] if len(param) > 2 else ""}

def get_base_url() -> str:
    file_path = Path.home() / ".oo_base_url"
    if file_path.exists():
        return file_path.read_text().strip()

    return "https://imageseg.scads.de/omniax/"

def write_ui_url() -> None:
    url = build_gui_url(args)
    with open(get_current_run_folder("ui_url.txt"), mode="a", encoding="utf-8") as myfile:
        myfile.write(url)

def handle_random_steps() -> None:
    if args.parameter and args.continue_previous_job and random_steps <= 0:
        print(f"A parameter has been reset, but the earlier job already had its random phase. To look at the new search space, {args.num_random_steps} random steps will be executed.")
        set_random_steps(args.num_random_steps)

def initialize_ax_client() -> None:
    global ax_client

    with spinner("Initializing ax_client..."):
        ax_client = AxClient(
            verbose_logging=args.verbose,
            enforce_sequential_optimization=False,
            generation_strategy=global_gs
        )

        ax_client = cast(AxClient, ax_client)

def get_generation_strategy_string() -> str:
    if generation_strategy_human_readable:
        return f"Generation strategy: {generation_strategy_human_readable}."

    return ""

class NpEncoder(json.JSONEncoder):
    def default(self: Any, obj: Any) -> Union[int, float, list, str]:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def save_experiment_parameters(filepath: str) -> None:
    with open(filepath, mode="w", encoding="utf-8") as outfile:
        json.dump(experiment_parameters, outfile, cls=NpEncoder)

def run_search_with_progress_bar() -> None:
    global progress_bar

    disable_tqdm = args.disable_tqdm or ci_env

    total_jobs = max_eval

    with tqdm(total=total_jobs, disable=disable_tqdm, ascii="░▒█") as _progress_bar:
        progress_bar = _progress_bar
        write_process_info()

        progressbar_description("Started OmniOpt2 run...")

        update_progress_bar(count_done_jobs() + NR_INSERTED_JOBS)

        run_search()

    wait_for_jobs_to_complete()

def complex_tests(_program_name: str, wanted_stderr: str, wanted_exit_code: int, wanted_signal: Union[int, None], res_is_none: bool = False) -> int:
    #print_yellow(f"Test suite: {_program_name}")

    nr_errors: int = 0

    program_path: str = f"./.tests/test_wronggoing_stuff.bin/bin/{_program_name}"

    if not os.path.exists(program_path):
        _fatal_error(f"Program path {program_path} not found!", 18)

    program_path_with_program: str = f"{program_path}"

    program_string_with_params = replace_parameters_in_string(
        {
            "a": 1,
            "b": 2,
            "c": 3,
            "def": 45
        },
        f"{program_path_with_program} %a %(b) $c $(def)"
    )

    nr_errors += is_equal(
        f"replace_parameters_in_string {_program_name}",
        program_string_with_params,
        f"{program_path_with_program} 1 2 3 45"
    )

    try:
        stdout, stderr, exit_code, _signal = execute_bash_code(program_string_with_params)

        res = get_results(stdout)

        if res_is_none:
            nr_errors += is_equal(f"{_program_name} res is None", {"result": None}, res)
        else:
            nr_errors += is_equal(f"{_program_name} res type is dict", True, isinstance(res, dict))
        nr_errors += is_equal(f"{_program_name} stderr", True, wanted_stderr in stderr)
        nr_errors += is_equal(f"{_program_name} exit-code ", exit_code, wanted_exit_code)
        nr_errors += is_equal(f"{_program_name} signal", _signal, wanted_signal)

        return nr_errors
    except Exception as e:
        print_red(f"Error complex_tests: {e}")

        return 1

def get_files_in_dir(mypath: str) -> list:
    print_debug("get_files_in_dir")
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    return [f"{mypath}/{s}" for s in onlyfiles]

def test_find_paths(program_code: str) -> int:
    print_debug(f"test_find_paths({program_code})")
    nr_errors: int = 0

    files: list = [
        "omniopt",
        ".omniopt.py",
        "plot",
        ".plot.py",
        "/etc/passwd",
        "I/DO/NOT/EXIST",
        "I DO ALSO NOT EXIST",
        "NEITHER DO I!",
        *get_files_in_dir("./.tests/test_wronggoing_stuff.bin/bin/")
    ]

    text: str = " -- && !!  ".join(files)

    string = find_file_paths_and_print_infos(text, program_code)

    for i in files:
        if i not in string:
            if os.path.exists(i):
                print("Missing {i} in find_file_paths string!")
                nr_errors += 1

    return nr_errors

def run_tests() -> None:
    print_red("This should be red")
    print_yellow("This should be yellow")
    print_green("This should be green")

    print(f"Printing test from current line {get_line_info()}")

    nr_errors: int = 0

    try:
        ie = is_equal(f'get_min_or_max_column_value(".tests/_plot_example_runs/ten_params/0/IDONTEVENEXIST/{RESULTS_CSV_FILENAME}", "result", -123, "min")', str(get_min_or_max_column_value(f".tests/_plot_example_runs/ten_params/0/IDONTEVENEXIST/{RESULTS_CSV_FILENAME}", 'result', -123, "min")), '-123')

        if not ie:
            nr_errors += 1
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"get_min_or_max_column_value on a non-existing file path excepted with another exception than FileNotFoundError (only acceptable one!). Error: {e}")
        nr_errors += 1

    non_rounded_lower, non_rounded_upper = round_lower_and_upper_if_type_is_int("float", -123.4, 123.4)
    nr_errors += is_equal("non_rounded_lower", non_rounded_lower, -123.4)
    nr_errors += is_equal("non_rounded_upper", non_rounded_upper, 123.4)

    nr_errors += is_equal('is_ax_compatible_constraint("abc", ["abc"])', is_ax_compatible_constraint("abc", ["abc"]), False)
    nr_errors += is_equal('is_ax_compatible_constraint("abc >= 1", ["abc"])', is_ax_compatible_constraint("abc >= 1", ["abc"]), True)
    nr_errors += is_equal('is_ax_compatible_constraint("abc * xyz >= 1", ["abc", "xyz"])', is_ax_compatible_constraint("abc * xyz >= 1", ["abc", "xyz"]), False)
    nr_errors += is_equal('is_ax_compatible_constraint("abc <= xyz", ["abc", "xyz"])', is_ax_compatible_constraint("abc <= xyz", ["abc", "xyz"]), True)
    nr_errors += is_equal('is_ax_compatible_constraint("2*abc <= 3.5", ["abc", "xyz"])', is_ax_compatible_constraint("2*abc <= 3.5", ["abc", "xyz"]), True)
    nr_errors += is_equal('is_ax_compatible_constraint("2*abc <= 3.5*xyz", ["abc"])', is_ax_compatible_constraint("2*abc <= 3.5*xyz", ["abc"]), False)
    nr_errors += is_equal('is_ax_compatible_constraint("2*abc * xyz <= 3.5", ["abc"])', is_ax_compatible_constraint("2*abc * xyz <= 3.5", ["abc"]), False)
    nr_errors += is_equal('is_ax_compatible_constraint("2*abc * xyz <= 3.5", ["abc", "xyz"])', is_ax_compatible_constraint("2*abc * xyz <= 3.5", ["abc", "xyz"]), False)

    nr_errors += is_equal(
        "has_no_post_generation_constraints_or_matches_constraints([], {})",
        has_no_post_generation_constraints_or_matches_constraints([], {}),
        True
    )

    nr_errors += is_equal(
        "has_no_post_generation_constraints_or_matches_constraints(['a > 0'], {'a': 5})",
        has_no_post_generation_constraints_or_matches_constraints(['a > 0'], {'a': 5}),
        True
    )

    nr_errors += is_equal(
        "has_no_post_generation_constraints_or_matches_constraints(['a > 0'], {'a': -1})",
        has_no_post_generation_constraints_or_matches_constraints(['a > 0'], {'a': -1}),
        False
    )

    nr_errors += is_equal(
        "has_no_post_generation_constraints_or_matches_constraints(['a + b == 3'], {'a': 1, 'b': 2})",
        has_no_post_generation_constraints_or_matches_constraints(['a + b == 3'], {'a': 1, 'b': 2}),
        True
    )

    nr_errors += is_equal(
        "has_no_post_generation_constraints_or_matches_constraints(['a + b == 3'], {'a': 1, 'b': 1})",
        has_no_post_generation_constraints_or_matches_constraints(['a + b == 3'], {'a': 1, 'b': 1}),
        False
    )

    nr_errors += is_equal(
        "has_no_post_generation_constraints_or_matches_constraints(['unknown > 0'], {'a': 1})",
        has_no_post_generation_constraints_or_matches_constraints(['unknown > 0'], {'a': 1}),
        False
    )

    nr_errors += is_equal(
        "has_no_post_generation_constraints_or_matches_constraints(['a + '], {'a': 1})",
        has_no_post_generation_constraints_or_matches_constraints(['a + '], {'a': 1}),
        False
    )

    nr_errors += is_equal('is_valid_equation("abc", ["abc"])',
                          is_valid_equation("abc", ["abc"]), False)
    nr_errors += is_equal('is_valid_equation("abc >= 1", ["abc"])',
                          is_valid_equation("abc >= 1", ["abc"]), True)
    nr_errors += is_equal('is_valid_equation("abc * xyz >= 1", ["abc", "xyz"])',
                          is_valid_equation("abc * xyz >= 1", ["abc", "xyz"]), True)
    nr_errors += is_equal('is_valid_equation("abc <= xyz", ["abc", "xyz"])',
                          is_valid_equation("abc <= xyz", ["abc", "xyz"]), True)
    nr_errors += is_equal('is_valid_equation("2*abc <= 3.5", ["abc", "xyz"])',
                          is_valid_equation("2*abc <= 3.5", ["abc", "xyz"]), True)
    nr_errors += is_equal('is_valid_equation("2*abc <= 3.5*xyz", ["abc"])',
                          is_valid_equation("2*abc <= 3.5*xyz", ["abc"]), False)
    nr_errors += is_equal('is_valid_equation("2*abc * xyz <= 3.5", ["abc"])',
                          is_valid_equation("2*abc * xyz <= 3.5", ["abc"]), False)
    nr_errors += is_equal('is_valid_equation("2*abc * xyz <= 3.5", ["abc", "xyz"])',
                          is_valid_equation("2*abc * xyz <= 3.5", ["abc", "xyz"]), True)
    nr_errors += is_equal('is_valid_equation("a * b >= 10", ["a", "b"])',
                          is_valid_equation("a * b >= 10", ["a", "b"]), True)
    nr_errors += is_equal('is_valid_equation("1*sample_period >= 1/window_size", ["sample_period", "window_size"])',
                          is_valid_equation("1*sample_period >= 1/window_size", ["sample_period", "window_size"]), True)

    rounded_lower, rounded_upper = round_lower_and_upper_if_type_is_int("int", -123.4, 123.4)
    nr_errors += is_equal("rounded_lower", rounded_lower, -124)
    nr_errors += is_equal("rounded_upper", rounded_upper, 124)

    nr_errors += is_equal(f'get_min_or_max_column_value(".tests/_plot_example_runs/ten_params/0/{RESULTS_CSV_FILENAME}", "result", -123, "min")', str(get_min_or_max_column_value(f".tests/_plot_example_runs/ten_params/0/{RESULTS_CSV_FILENAME}", 'result', -123, "min")), '17143005390319.627')
    nr_errors += is_equal(f'get_min_or_max_column_value(".tests/_plot_example_runs/ten_params/0/{RESULTS_CSV_FILENAME}", "result", -123, "max")', str(get_min_or_max_column_value(f".tests/_plot_example_runs/ten_params/0/{RESULTS_CSV_FILENAME}", 'result', -123, "max")), '9.865416064838896e+29')

    nr_errors += is_equal('get_file_as_string("/i/do/not/exist/ANYWHERE/EVER")', get_file_as_string("/i/do/not/exist/ANYWHERE/EVER"), "")

    nr_errors += is_equal('makedirs("/proc/AOIKJSDAOLSD")', makedirs("/proc/AOIKJSDAOLSD"), False)

    nr_errors += is_equal('replace_string_with_params("hello %0 %1 world", [10, "hello"])', replace_string_with_params("hello %0 %1 world", [10, "hello"]), "hello 10 hello world")

    nr_errors += is_equal('_count_sobol_or_completed("", "")', _count_sobol_or_completed("", ""), 0)

    plot_params = get_plot_commands('_command', {"type": "trial_index_result", "min_done_jobs": 2}, '_tmp', 'plot_type', 'tmp_file', "1200")

    nr_errors += is_equal('get_plot_commands', json.dumps(plot_params), json.dumps([['_command --save_to_file=tmp_file ', 'tmp_file', "1200"]]))

    plot_params_complex = get_plot_commands('_command', {"type": "scatter", "params": "--bubblesize=50 --allow_axes %0 --allow_axes %1", "iterate_through": [["n_samples", "confidence"], ["n_samples", "feature_proportion"], ["n_samples", "n_clusters"], ["confidence", "feature_proportion"], ["confidence", "n_clusters"], ["feature_proportion", "n_clusters"]], "dpi": 76, "filename": "plot_%0_%1_%2"}, '_tmp', 'plot_type', 'tmp_file', "1200")

    expected_plot_params_complex = [['_command --bubblesize=50 --allow_axes n_samples --allow_axes confidence '
                                     '--save_to_file=_tmp/plot_plot_type_n_samples_confidence.png ',
                                     '_tmp/plot_plot_type_n_samples_confidence.png',
                                     "1200"],
                                    ['_command --bubblesize=50 --allow_axes n_samples --allow_axes '
                                     'feature_proportion '
                                     '--save_to_file=_tmp/plot_plot_type_n_samples_feature_proportion.png ',
                                     '_tmp/plot_plot_type_n_samples_feature_proportion.png',
                                     "1200"],
                                    ['_command --bubblesize=50 --allow_axes n_samples --allow_axes n_clusters '
                                     '--save_to_file=_tmp/plot_plot_type_n_samples_n_clusters.png ',
                                     '_tmp/plot_plot_type_n_samples_n_clusters.png',
                                     "1200"],
                                    ['_command --bubblesize=50 --allow_axes confidence --allow_axes '
                                     'feature_proportion '
                                     '--save_to_file=_tmp/plot_plot_type_confidence_feature_proportion.png ',
                                     '_tmp/plot_plot_type_confidence_feature_proportion.png',
                                     "1200"],
                                    ['_command --bubblesize=50 --allow_axes confidence --allow_axes n_clusters '
                                     '--save_to_file=_tmp/plot_plot_type_confidence_n_clusters.png ',
                                     '_tmp/plot_plot_type_confidence_n_clusters.png',
                                     "1200"],
                                    ['_command --bubblesize=50 --allow_axes feature_proportion --allow_axes '
                                     'n_clusters '
                                     '--save_to_file=_tmp/plot_plot_type_feature_proportion_n_clusters.png ',
                                     '_tmp/plot_plot_type_feature_proportion_n_clusters.png',
                                     "1200"]]

    nr_errors += is_equal("get_plot_commands complex", json.dumps(plot_params_complex), json.dumps(expected_plot_params_complex))

    nr_errors += is_equal('get_sixel_graphics_data("")', json.dumps(get_sixel_graphics_data('')), json.dumps([]))

    global_vars["parameter_names"] = [
        "n_samples",
        "confidence",
        "feature_proportion",
        "n_clusters"
    ]

    got: str = json.dumps(get_sixel_graphics_data(f'.gui/_share_test_case/test_user/ClusteredStatisticalTestDriftDetectionMethod_NOAAWeather/0/{RESULTS_CSV_FILENAME}', True))
    expected: str = '[["bash omniopt_plot --run_dir  --plot_type=trial_index_result", {"type": "trial_index_result", "min_done_jobs": 2}, "/plots/", "trial_index_result", "/plots//trial_index_result.png", "1200"], ["bash omniopt_plot --run_dir  --plot_type=scatter --dpi=76", {"type": "scatter", "params": "--bubblesize=50 --allow_axes %0 --allow_axes %1", "iterate_through": [["n_samples", "confidence"], ["n_samples", "feature_proportion"], ["n_samples", "n_clusters"], ["confidence", "feature_proportion"], ["confidence", "n_clusters"], ["feature_proportion", "n_clusters"]], "dpi": 76, "filename": "plot_%0_%1_%2"}, "/plots/", "scatter", "/plots//plot_%0_%1_%2.png", "1200"], ["bash omniopt_plot --run_dir  --plot_type=general", {"type": "general"}, "/plots/", "general", "/plots//general.png", "1200"]]'

    nr_errors += is_equal(f'get_sixel_graphics_data(".gui/_share_test_case/test_user/ClusteredStatisticalTestDriftDetectionMethod_NOAAWeather/0/{RESULTS_CSV_FILENAME}", True)', got, expected)

    nr_errors += is_equal('get_hostname_from_outfile("")', get_hostname_from_outfile(''), None)

    nr_errors += is_equal('get_parameters_from_outfile("")', get_parameters_from_outfile(''), None)

    nonzerodebug: str = """
Exit-Code: 159
    """

    nr_errors += is_equal(f'check_for_non_zero_exit_codes("{nonzerodebug}")', check_for_non_zero_exit_codes(nonzerodebug), [f"Non-zero exit-code detected: 159.  (May mean {get_exit_codes()[str(159)]}, unless you used that exit code yourself or it was part of any of your used libraries or programs)"])

    nr_errors += is_equal('state_from_job("")', state_from_job(''), "None")

    nr_errors += is_equal('print_image_to_cli("", "")', print_image_to_cli("", 1200), False)
    if supports_sixel():
        nr_errors += is_equal('print_image_to_cli(".tools/slimer.png", 200)', print_image_to_cli(".tools/slimer.png", 200), True)
    else:
        nr_errors += is_equal('print_image_to_cli(".tools/slimer.png", 200)', print_image_to_cli(".tools/slimer.png", 200), False)

    _check_for_basic_string_errors_example_str: str = """
    Exec format error
    """

    nr_errors += is_equal('check_for_basic_string_errors("_check_for_basic_string_errors_example_str", "", [], "")', check_for_basic_string_errors(_check_for_basic_string_errors_example_str, "", [], ""), [f"Was the program compiled for the wrong platform? Current system is {platform.machine()}", "No files could be found in your program string: "])

    nr_errors += is_equal('state_from_job("state=\"FINISHED\")', state_from_job('state="FINISHED"'), "finished")

    nr_errors += is_equal('state_from_job("state=\"FINISHED\")', state_from_job('state="FINISHED"'), "finished")

    nr_errors += is_equal('get_first_line_of_file_that_contains_string("IDONTEXIST", "HALLO")', get_first_line_of_file_that_contains_string("IDONTEXIST", "HALLO"), "")

    nr_errors += is_equal('extract_info("OO-Info: SLURM_JOB_ID: 123")', json.dumps(extract_info("OO-Info: SLURM_JOB_ID: 123")), '[["OO_Info_SLURM_JOB_ID"], ["123"]]')

    nr_errors += is_equal('get_min_max_from_file("/i/do/not/exist/hopefully/anytime/ever", 0, "-123")', get_min_max_from_file("/i/do/not/exist/hopefully/anytime/ever", 0, "-123"), '-123')

    if not SYSTEM_HAS_SBATCH or args.run_tests_that_fail_on_taurus:
        nr_errors += complex_tests("signal_but_has_output", "Killed", 137, None)
        nr_errors += complex_tests("signal", "Killed", 137, None, True)
    else:
        print_yellow("Ignoring tests complex_tests(signal_but_has_output) and complex_tests(signal) because SLURM is installed and --run_tests_that_fail_on_taurus was not set")

    _not_equal: list = [
        ["nr equal strings", 1, "1"],
        ["unequal strings", "hallo", "welt"]
    ]

    for _item in _not_equal:
        __name = _item[0]
        __should_be = _item[1]
        __is = _item[2]

        nr_errors += is_not_equal(__name, __should_be, __is)

    nr_errors += is_equal("nr equal nr", 1, 1)

    example_parse_parameter_type_error_result: dict = {
        "parameter_name": "xxx",
        "current_type": "int",
        "expected_type": "float"
    }

    global arg_result_names

    arg_result_names = ["RESULT"]

    equal: list = [
        ["helpers.convert_string_to_number('123.123')", 123.123],
        ["helpers.convert_string_to_number('1')", 1],
        ["helpers.convert_string_to_number('-1')", -1],
        ["helpers.convert_string_to_number(None)", None],
        ["get_results(None)", None],
        ["parse_parameter_type_error(None)", None],
        ["parse_parameter_type_error(\"Value for parameter xxx: bla is of type <class 'int'>, expected <class 'float'>.\")", example_parse_parameter_type_error_result],
        ["get_hostname_from_outfile(None)", None],
        ["get_results(123)", None],
        ["get_results('RESULT: 10')", {'RESULT': 10.0}],
        ["helpers.looks_like_float(10)", True],
        ["helpers.looks_like_float('hallo')", False],
        ["helpers.looks_like_int('hallo')", False],
        ["helpers.looks_like_int('1')", True],
        ["helpers.looks_like_int(False)", False],
        ["helpers.looks_like_int(True)", False],
        ["_count_sobol_steps('/etc/idontexist')", 0],
        ["_count_done_jobs('/etc/idontexist')", 0],
        ["get_program_code_from_out_file('/etc/doesntexist')", ""],
        ["get_type_short('RangeParameter')", "range"],
        ["get_type_short('ChoiceParameter')", "choice"],
        ["create_and_execute_next_runs(0, None, None)", 0]
    ]

    for _item in equal:
        _name = _item[0]
        _should_be = _item[1]

        nr_errors += is_equal(_name, eval(_name), _should_be)

    nr_errors += is_equal(
        "replace_parameters_in_string({\"x\": 123}, \"echo 'RESULT: %x'\")",
        replace_parameters_in_string({"x": 123}, "echo 'RESULT: %x'"),
        "echo 'RESULT: 123'"
    )

    global_vars["joined_run_program"] = "echo 'RESULT: %x'"

    nr_errors += is_equal(
            "evaluate({'x': 123})",
            json.dumps(evaluate({"params": {'x': 123.0}, "trial_idx": 0, "submit_time": int(time.time())})),
            json.dumps({'RESULT': 123.0})
    )

    nr_errors += is_equal(
            "evaluate({'x': -0.05})",
            json.dumps(evaluate({"params": {'x': -0.05}, "trial_idx": 0, "submit_time": int(time.time())})),
            json.dumps({'RESULT': -0.05})
    )

    #complex_tests (_program_name, wanted_stderr, wanted_exit_code, wanted_signal, res_is_none=False):
    _complex_tests: list = [
        ["simple_ok", "hallo", 0, None],
        ["divide_by_0", 'Illegal division by zero at ./.tests/test_wronggoing_stuff.bin/bin/divide_by_0 line 3.\n', 255, None, True],
        ["result_but_exit_code_stdout_stderr", "stderr", 5, None],
        ["exit_code_no_output", "", 5, None, True],
        ["exit_code_stdout", "STDERR", 5, None, False],
        ["exit_code_stdout_stderr", "This has stderr", 5, None, True],
        ["module_not_found", "ModuleNotFoundError", 1, None, True]
    ]

    if not SYSTEM_HAS_SBATCH:
        _complex_tests.append(["no_chmod_x", "Permission denied", 126, None, True])

    for _item in _complex_tests:
        nr_errors += complex_tests(*_item)

    nr_errors += is_equal("test_find_paths failed", bool(test_find_paths("ls")), False)

    orchestrator_yaml: str = ".tests/example_orchestrator_config.yaml"

    if os.path.exists(orchestrator_yaml):
        _is: str = json.dumps(parse_orchestrator_file(orchestrator_yaml, True))
        should_be: str = '{"errors": [{"name": "GPUDisconnected", "match_strings": ["AssertionError: ``AmpOptimizerWrapper`` is only available"], "behavior": "ExcludeNode"}, {"name": "Timeout", "match_strings": ["Timeout"], "behavior": "RestartOnDifferentNode"}]}'
        nr_errors += is_equal(f"parse_orchestrator_file({orchestrator_yaml})", should_be, _is)
    else:
        nr_errors += is_equal(".tests/example_orchestrator_config.yaml exists", True, False)

    _print_best_result(False)

    nr_errors += is_equal("get_workers_string()", get_workers_string(), "")

    nr_errors += is_equal("check_file_info('/dev/i/dont/exist')", check_file_info('/dev/i/dont/exist'), "")

    nr_errors += is_equal(
        "get_parameters_from_outfile()",
        get_parameters_from_outfile(""),
        None
    )

    nr_errors += is_equal("calculate_occ(None)", calculate_occ(None), VAL_IF_NOTHING_FOUND)
    nr_errors += is_equal("calculate_occ([])", calculate_occ([]), VAL_IF_NOTHING_FOUND)

    #nr_errors += is_equal("calculate_signed_harmonic_distance(None)", calculate_signed_harmonic_distance(None), 0)
    nr_errors += is_equal("calculate_signed_harmonic_distance([])", calculate_signed_harmonic_distance([]), 0)
    nr_errors += is_equal("calculate_signed_harmonic_distance([0.1])", calculate_signed_harmonic_distance([0.1]), 0.1)
    nr_errors += is_equal("calculate_signed_harmonic_distance([-0.1])", calculate_signed_harmonic_distance([-0.1]), -0.1)
    nr_errors += is_equal("calculate_signed_harmonic_distance([0.1, 0.1])", calculate_signed_harmonic_distance([0.1, 0.2]), 0.13333333333333333)

    nr_errors += is_equal("calculate_signed_euclidean_distance([0.1])", calculate_signed_euclidean_distance([0.1]), 0.1)
    nr_errors += is_equal("calculate_signed_euclidean_distance([-0.1])", calculate_signed_euclidean_distance([-0.1]), -0.1)
    nr_errors += is_equal("calculate_signed_euclidean_distance([0.1, 0.1])", calculate_signed_euclidean_distance([0.1, 0.2]), 0.223606797749979)

    nr_errors += is_equal("calculate_signed_geometric_distance([0.1])", calculate_signed_geometric_distance([0.1]), 0.1)
    nr_errors += is_equal("calculate_signed_geometric_distance([-0.1])", calculate_signed_geometric_distance([-0.1]), -0.1)
    nr_errors += is_equal("calculate_signed_geometric_distance([0.1, 0.1])", calculate_signed_geometric_distance([0.1, 0.2]), 0.14142135623730953)

    nr_errors += is_equal("calculate_signed_minkowski_distance([0.1], 3)", calculate_signed_minkowski_distance([0.1], 3), 0.10000000000000002)
    nr_errors += is_equal("calculate_signed_minkowski_distance([-0.1], 3)", calculate_signed_minkowski_distance([-0.1], 3), -0.10000000000000002)
    nr_errors += is_equal("calculate_signed_minkowski_distance([0.1, 0.2], 3)", calculate_signed_minkowski_distance([0.1, 0.2], 3), 0.20800838230519045)

    try:
        calculate_signed_minkowski_distance([0.1, 0.2], -1)
        nr_errors = nr_errors + 1
    except ValueError:
        pass

    nr_errors += is_equal(
        "calculate_signed_weighted_euclidean_distance([0.1], '1.0')",
        calculate_signed_weighted_euclidean_distance([0.1], "1.0"),
        0.1
    )
    nr_errors += is_equal(
        "calculate_signed_weighted_euclidean_distance([-0.1], '1.0')",
        calculate_signed_weighted_euclidean_distance([-0.1], "1.0"),
        -0.1
    )
    nr_errors += is_equal(
        "calculate_signed_weighted_euclidean_distance([0.1, 0.2], '0.5,2.0')",
        calculate_signed_weighted_euclidean_distance([0.1, 0.2], "0.5,2.0"),
        0.29154759474226505
    )
    nr_errors += is_equal(
        "calculate_signed_weighted_euclidean_distance([0.1], '1')",
        calculate_signed_weighted_euclidean_distance([0.1], "1"),
        0.1
    )
    nr_errors += is_equal(
        "calculate_signed_weighted_euclidean_distance([0.1, 0.1], '1')",
        calculate_signed_weighted_euclidean_distance([0.1, 0.1], "1"),
        0.14142135623730953
    )
    nr_errors += is_equal(
        "calculate_signed_weighted_euclidean_distance([0.1], '1,1,1,1')",
        calculate_signed_weighted_euclidean_distance([0.1], "1,1,1,1"),
        0.1
    )

    my_exit(nr_errors)

def main_outside() -> None:
    print(f"Run-UUID: {run_uuid}")

    auto_wrap_namespace(globals())

    print_logo()

    start_logging_daemon()

    fool_linter(args.num_cpus_main_job)
    fool_linter(args.flame_graph)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if args.tests:
            run_tests()

        try:
            main()
        except (SignalUSR, SignalINT, SignalCONT, KeyboardInterrupt):
            signal.signal(signal.SIGINT, signal.SIG_IGN)

            print_red("\n⚠ You pressed CTRL+C or got a signal. Optimization stopped.")

            end_program(False, 1)
        except SearchSpaceExhausted:
            _get_perc: int = abs(int(((count_done_jobs() - NR_INSERTED_JOBS) / max_eval) * 100))

            if _get_perc < 100:
                print_red(
                    f"\nIt seems like the search space was exhausted. "
                    f"You were able to get {_get_perc}% of the jobs you requested "
                    f"(got: {count_done_jobs() - NR_INSERTED_JOBS}, submitted: {submitted_jobs()}, failed: {failed_jobs()}, "
                    f"requested: {max_eval}) after main ran"
                )

            if _get_perc != 100:
                end_program(True, 87)
            else:
                end_program(True)

def stack_trace_wrapper(func: Any, regex: Any = None) -> Any:
    pattern = re.compile(regex) if regex else None

    def wrapped(*args: Any, **kwargs: Any) -> None:
        if pattern and not pattern.search(func.__name__):
            return func(*args, **kwargs)

        stack = inspect.stack()
        chain = []
        for frame in stack[1:]:
            fn = frame.function
            if fn in ("wrapped", "<module>"):
                continue
            chain.append(fn)

        if chain:
            sys.stderr.write(" ⇒ ".join(reversed(chain)) + "\n")

        return func(*args, **kwargs)

    return wrapped

def auto_wrap_namespace(namespace: Any) -> Any:
    enable_beartype = any(os.getenv(v) for v in ("ENABLE_BEARTYPE", "CI"))

    excluded_functions = {
        "log_time_and_memory_wrapper",
        "collect_runtime_stats",
        "_print_time_and_memory_functions_wrapper_stats",
        "print",
        "_record_stats",
        "_open",
        "_check_memory_leak",
        "start_periodic_live_share",
        "start_logging_daemon",
        "get_current_run_folder",
        "show_func_name_wrapper"
    }

    for name, obj in list(namespace.items()):
        if (isinstance(obj, types.FunctionType) and name not in excluded_functions):
            wrapped = obj
            if enable_beartype:
                wrapped = beartype(wrapped)

            if args.runtime_debug:
                wrapped = log_time_and_memory_wrapper(wrapped)

            if args.show_func_name:
                wrapped = show_func_name_wrapper(wrapped)

            if args.debug_stack_trace_regex:
                wrapped = stack_trace_wrapper(wrapped, args.debug_stack_trace_regex)

            namespace[name] = wrapped

    return namespace

if __name__ == "__main__":
    try:
        main_outside()
    except (SignalUSR, SignalINT, SignalCONT) as e:
        print_red(f"main_outside failed with exception {e}")
        end_program(True)
