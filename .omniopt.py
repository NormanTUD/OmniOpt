#!/bin/env python3

#from mayhemmonkey import MayhemMonkey
#mayhemmonkey = MayhemMonkey()
#mayhemmonkey.set_function_fail_after_count("open", 201)
#mayhemmonkey.set_function_error_rate("open", 0.1)
#mayhemmonkey.set_function_group_error_rate(["io", "math"], 0.8)
#mayhemmonkey.install_faulty()

import sys
import os
import re
import math
import time
import random
import statistics
import tempfile

last_progress_bar_desc: str = ""
generation_strategy_human_readable: str = ""
oo_call: str = "./omniopt"
progress_bar_length: int = 0

if os.environ.get("CUSTOM_VIRTUAL_ENV") == "1":
    oo_call = "omniopt"

gotten_jobs: int = 0

shown_run_live_share_command: bool = False
ci_env: bool = os.getenv("CI", "false").lower() == "true"
original_print = print
overwritten_to_random: bool = False

valid_occ_types: list = ["geometric", "euclid", "signed_harmonic", "signed_minkowski", "weighted_euclid", "composite"]
joined_valid_occ_types: str = ", ".join(valid_occ_types)

SUPPORTED_MODELS: list = ["SOBOL", "FACTORIAL", "SAASBO", "BOTORCH_MODULAR", "UNIFORM", "BO_MIXED", "RANDOMFOREST", "EXTERNAL_GENERATOR", "PSEUDORANDOM"]
joined_supported_models: str = ", ".join(SUPPORTED_MODELS)

uncontinueable_models: list = ["PSEUDORANDOM", "EXTERNAL_GENERATOR"]

special_col_names: list = ["arm_name", "generation_method", "trial_index", "trial_status", "generation_node"]
IGNORABLE_COLUMNS: list = ["start_time", "end_time", "hostname", "signal", "exit_code", "run_time", "program_string"] + special_col_names

figlet_loaded: bool = False

try:
    from rich.console import Console

    terminal_width = 150

    try:
        terminal_width = os.get_terminal_size().columns
    except OSError:
        pass

    console: Console = Console(
        force_interactive=True,
        soft_wrap=True,
        color_system="256",
        force_terminal=not ci_env,
        width=max(200, terminal_width)
    )

    with console.status("[bold green]Loading base modules...") as status:
        import logging
        logging.basicConfig(level=logging.CRITICAL)

        import warnings

        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            module="ax.modelbridge.best_model_selector"
        )

        import argparse
        import datetime

        from ax.utils.common.random import set_rng_seed

        from dataclasses import dataclass

        import hashlib

        import socket
        import stat
        import pwd
        import signal
        import base64

        import json
        import yaml
        import toml
        import csv

        from rich.progress import Progress, TimeRemainingColumn
        from rich.table import Table
        from rich import print
        from rich.pretty import pprint

        import fcntl

        from types import FunctionType
        from typing import Pattern, Optional, Tuple, Any, cast, Union, TextIO, List, Dict, Type, Sequence

        from submitit import LocalExecutor, AutoExecutor
        from submitit import Job

        import threading

        import importlib.util
        import inspect
        import platform

        from inspect import currentframe, getframeinfo
        from pathlib import Path

        import uuid

        import traceback

        import cowsay

        import psutil
        import shutil

        from itertools import combinations

        import pandas as pd

        from os import listdir
        from os.path import isfile, join

        from PIL import Image
        import sixel

        import subprocess

        from tqdm import tqdm

        from beartype import beartype
    try:
        from pyfiglet import Figlet
        figlet_loaded = True
    except ModuleNotFoundError:
        figlet_loaded = False
except ModuleNotFoundError as e:
    print(f"Some of the base modules could not be loaded. Most probably that means you have not loaded or installed the virtualenv properly. Error: {e}")
    print("Exit-Code: 2")
    sys.exit(2)
except ImportError as e:
    print(f"Error loading modules: {e}\nThis may be caused by forgetting to 'module load' the right python version or missing the python virtual environment.")
    sys.exit(2)
except KeyboardInterrupt:
    print("You pressed CTRL-C while modules were loading.")
    sys.exit(17)

@beartype
def fool_linter(*fool_linter_args: Any) -> Any:
    return fool_linter_args

with console.status("[bold green]Loading rich_argparse...") as status:
    try:
        from rich_argparse import RichHelpFormatter
    except ModuleNotFoundError:
        RichHelpFormatter: Any = argparse.HelpFormatter # type: ignore

@beartype
def makedirs(p: str) -> bool:
    if not os.path.exists(p):
        try:
            os.makedirs(p, exist_ok=True)
        except Exception as ee:
            print(f"Failed to create >{p}<. Error: {ee}")

    if os.path.exists(p):
        return True

    return False

print_debug_once_list: List = []

YELLOW: str = "\033[93m"
RESET: str = "\033[0m"

uuid_regex: Pattern = re.compile(r"^[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-4[a-fA-F0-9]{3}-[89aAbB][a-fA-F0-9]{3}-[a-fA-F0-9]{12}$")

new_uuid: str = str(uuid.uuid4())
run_uuid: str = os.getenv("RUN_UUID", new_uuid)

if not uuid_regex.match(run_uuid):
    print(f"{YELLOW}WARNING: The provided RUN_UUID is not a valid UUID. Using new UUID {new_uuid} instead.{RESET}")
    run_uuid = new_uuid

JOBS_FINISHED: int = 0
SHOWN_LIVE_SHARE_COUNTER: int = 0
PD_CSV_FILENAME: str = "results.csv"
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

@beartype
def get_current_run_folder() -> str:
    return CURRENT_RUN_FOLDER

script_dir = os.path.dirname(os.path.realpath(__file__))
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

ORCHESTRATE_TODO: dict = {}

class SignalUSR (Exception):
    pass

class SignalINT (Exception):
    pass

class SignalTERM (Exception):
    pass

class SignalCONT (Exception):
    pass

@beartype
def is_slurm_job() -> bool:
    if os.environ.get('SLURM_JOB_ID') is not None:
        return True
    return False

@beartype
def _sleep(t: int) -> int:
    if args is not None and not args.no_sleep:
        time.sleep(t)

    return t

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

@beartype
def print_red(text: str) -> None:
    helpers.print_color("red", text)

    print_debug(text)

    if get_current_run_folder():
        try:
            with open(f"{get_current_run_folder()}/oo_errors.txt", mode="a", encoding="utf-8") as myfile:
                myfile.write(text + "\n\n")
        except (OSError, FileNotFoundError) as e:
            helpers.print_color("red", f"Error: {e}. This may mean that the {get_current_run_folder()} was deleted during the run. Could not write '{text} to {get_current_run_folder()}/oo_errors.txt'")
            sys.exit(99)

@beartype
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

@beartype
def _get_debug_json(time_str: str, msg: str) -> str:
    function_stack = []

    try:
        stack = inspect.stack()

        for frame_info in stack[1:]:
            if str(frame_info.function) != "<module>" and str(frame_info.function) != "print_debug":
                if frame_info.function != "wrapper":
                    function_stack.append({
                        "function": frame_info.function,
                        "line_number": frame_info.lineno
                    })
    except (SignalUSR, SignalINT, SignalCONT):
        print_red("\n⚠ You pressed CTRL-C. This is ignored in _get_debug_json.")
    return json.dumps({"function_stack": function_stack, "time": time_str, "msg": msg}, indent=0).replace('\r', '').replace('\n', '')

@beartype
def print_debug(msg: str) -> None:
    original_msg = msg

    time_str: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    stack_trace_element = _get_debug_json(time_str, msg)

    msg = f"{stack_trace_element}"

    #if args is not None and args.debug:
    #    original_print(msg)

    _debug(msg)

    try:
        with open(logfile_bare, mode='a', encoding="utf-8") as f:
            original_print(original_msg, file=f)
    except FileNotFoundError:
        print_red("It seems like the run's folder was deleted during the run. Cannot continue.")
        sys.exit(99) # generalized code for run folder deleted during run
    except Exception as e:
        original_print(f"_debug: Error trying to write log file: {e}")

@beartype
def print_debug_once(msg: str) -> None:
    if msg not in print_debug_once_list:
        print_debug(msg)
        print_debug_once_list.append(msg)

@beartype
def my_exit(_code: int = 0) -> None:
    tb = traceback.format_exc()

    try:
        print_debug(f"Exiting with error code {_code}. Traceback: {tb}")
    except NameError:
        print(f"Exiting with error code {_code}. Traceback: {tb}")

    try:
        if (is_slurm_job() and not args.force_local_execution) and not (args.show_sixel_scatter or args.show_sixel_general or args.show_sixel_trial_index_result):
            _sleep(5)
        else:
            time.sleep(2)
    except KeyboardInterrupt:
        pass

    exit_code_string = f"\nExit-Code: {_code}"

    print(exit_code_string)
    print_debug(exit_code_string)
    sys.exit(_code)

@beartype
def print_green(text: str) -> None:
    helpers.print_color("green", text)

    print_debug(text)

@beartype
def print_yellow(text: str) -> None:
    helpers.print_color("yellow", f"{text}")

    print_debug(text)

@beartype
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

class ConfigLoader:
    disable_previous_job_constraint: bool
    run_tests_that_fail_on_taurus: bool
    enforce_sequential_optimization: bool
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
    no_sleep: bool
    username: Optional[str]
    max_nr_of_zero_results: int
    mem_gb: int
    continue_previous_job: Optional[str]
    revert_to_random_when_seemingly_exhausted: bool
    minkowski_p: float
    decimalrounding: int
    partition: str
    signed_weighted_euclidean_weights: str
    external_generator: Optional[str]
    generation_strategy: Optional[str]
    root_venv_dir: str
    pareto_front_confidence: float
    follow: bool
    n_estimators_randomforest: int
    checkout_to_latest_tested_version: bool
    load_data_from_existing_jobs: List[str]
    time: str
    generate_all_jobs_at_once: bool
    result_names: Optional[List[str]]
    verbose_break_run_search_table: bool
    send_anonymized_usage_stats: bool
    max_failed_jobs: Optional[int]
    show_ram_every_n_seconds: int
    config_toml: Optional[str]
    config_json: Optional[str]
    config_yaml: Optional[str]
    workdir: str
    occ: bool
    run_mode: str

    @beartype
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(
            prog="omniopt",
            description='A hyperparameter optimizer for slurm-based HPC-systems',
            epilog=f"Example:\n\n{oo_call} --partition=alpha --experiment_name=neural_network ...",
            formatter_class=RichHelpFormatter
        )

        # Add config arguments
        self.parser.add_argument('--config_yaml', help='YAML configuration file', type=str, default=None)
        self.parser.add_argument('--config_toml', help='TOML configuration file', type=str, default=None)
        self.parser.add_argument('--config_json', help='JSON configuration file', type=str, default=None)

        # Initialize the remaining arguments
        self.add_arguments()

    @beartype
    def add_arguments(self) -> None:
        required = self.parser.add_argument_group('Required arguments', 'These options have to be set')
        required_but_choice = self.parser.add_argument_group('Required arguments that allow a choice', 'Of these arguments, one has to be set to continue')
        optional = self.parser.add_argument_group('Optional', 'These options are optional')
        slurm = self.parser.add_argument_group('SLURM', 'Parameters related to SLURM')
        installing = self.parser.add_argument_group('Installing', 'Parameters related to installing')
        debug = self.parser.add_argument_group('Debug', 'These options are mainly useful for debugging')

        required.add_argument('--num_random_steps', help='Number of random steps to start with', type=int, default=20)
        required.add_argument('--max_eval', help='Maximum number of evaluations', type=int)
        required.add_argument('--run_program', action='append', nargs='+', help='A program that should be run. Use, for example, $x for the parameter named x', type=str)
        required.add_argument('--experiment_name', help='Name of the experiment', type=str)
        required.add_argument('--mem_gb', help='Amount of RAM for each worker in GB (default: 1GB)', type=int, default=1)

        required_but_choice.add_argument('--parameter', action='append', nargs='+', help='Experiment parameters in the formats (options in round brackets are optional): <NAME> range <LOWER BOUND> <UPPER BOUND> (<INT, FLOAT>, log_scale: True/False, default: false>) -- OR -- <NAME> fixed <VALUE> -- OR -- <NAME> choice <Comma-separated list of values>', default=None)
        required_but_choice.add_argument('--continue_previous_job', help='Continue from a previous checkpoint, use run-dir as argument', type=str, default=None)

        optional.add_argument('--experiment_constraints', action='append', nargs='+', help='Constraints for parameters. Example: x + y <= 2.0. Convert them to base64', type=str)
        optional.add_argument('--run_dir', help='Directory, in which runs should be saved. Default: runs', default='runs', type=str)
        optional.add_argument('--seed', help='Seed for random number generator', type=int)
        optional.add_argument('--decimalrounding', help='Number of decimal places for rounding', type=int, default=4)
        optional.add_argument('--enforce_sequential_optimization', help='Enforce sequential optimization (default: false)', action='store_true', default=False)
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
        optional.add_argument('--pareto_front_confidence', help='Confidence for pareto-front-plotting (between 0 and 1, default: 1)', type=float, default=1)
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
        optional.add_argument('--signed_weighted_euclidean_weights', help='A comma-seperated list of values for the signed weighted euclidean distance. Needs to be equal to the number of results. Else, default will be 1', default='', type=str)
        optional.add_argument('--generation_strategy', help='A string containing the generation_strategy. Example: SOBOL=10,BOTORCH_MODULAR=10,SOBOL=10. Cannot use --model EXTERNAL_GENERATOR or PSEUDORANDOM', type=str, default=None)
        optional.add_argument('--generate_all_jobs_at_once', help='Generate all jobs at once rather than to create them and start them as soon as possible', action='store_true', default=False)
        optional.add_argument('--revert_to_random_when_seemingly_exhausted', help='Generate random steps instead of systematic steps when the search space is (seemingly) exhausted', action='store_true', default=False)
        optional.add_argument('--load_data_from_existing_jobs', type=str, nargs='*', default=[], help='List of job data to load from existing jobs')
        optional.add_argument('--n_estimators_randomforest', help='The number of trees in the forest for RANDOMFOREST (default: 100)', type=int, default=100)
        optional.add_argument('--external_generator', help='Programm call for an external generator', type=str, default=None)
        optional.add_argument('--username', help='A username for live share', default=None, type=str)
        optional.add_argument('--max_failed_jobs', help='Maximum number of failed jobs before the search is cancelled. Is defaulted to the value of --max_eval', default=None, type=int)

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

        installing.add_argument('--run_mode', help='Either local or docker', default='local', type=str)

        debug.add_argument('--verbose', help='Verbose logging', action='store_true', default=False)
        debug.add_argument('--verbose_break_run_search_table', help='Verbose logging for break_run_search', action='store_true', default=False)
        debug.add_argument('--debug', help='Enable debugging', action='store_true', default=False)
        debug.add_argument('--no_sleep', help='Disables sleeping for fast job generation (not to be used on HPC)', action='store_true', default=False)
        debug.add_argument('--tests', help='Run simple internal tests', action='store_true', default=False)
        debug.add_argument('--show_worker_percentage_table_at_end', help='Show a table of percentage of usage of max worker over time', action='store_true', default=False)
        debug.add_argument('--auto_exclude_defective_hosts', help='Run a Test if you can allocate a GPU on each node and if not, exclude it since the GPU driver seems to be broken somehow', action='store_true', default=False)
        debug.add_argument('--run_tests_that_fail_on_taurus', help='Run tests on Taurus that usually fail', action='store_true', default=False)
        debug.add_argument('--raise_in_eval', help='Raise a signal in eval (only useful for debugging and testing)', action='store_true', default=False)
        debug.add_argument('--show_ram_every_n_seconds', help='Show RAM usage every n seconds (0 = disabled)', type=int, default=0)

    @beartype
    def load_config(self: Any, config_path: str, file_format: str) -> dict:
        if not os.path.isfile(config_path):
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
                print_red(f"Error parsing {file_format} file '{config_path}': {e}")
                print("Exit-Code: 5")
                sys.exit(5)

        return {}

    @beartype
    def validate_and_convert(self: Any, config: dict, arg_defaults: dict) -> dict:
        """
        Validates the config data and converts them to the right types based on argparse defaults.
        Warns about unknown or unused parameters.
        """
        converted_config = {}
        for key, value in config.items():
            if key in arg_defaults:
                # Get the expected type either from the default value or from the CLI argument itself
                default_value = arg_defaults[key]
                if default_value is not None:
                    expected_type = type(default_value)
                else:
                    # Fall back to using the current value's type, assuming it's not None
                    expected_type = type(value)

                try:
                    # Convert the value to the expected type
                    converted_config[key] = expected_type(value)
                except (ValueError, TypeError):
                    print(f"Warning: Cannot convert '{key}' to {expected_type.__name__}. Using default value.")
            else:
                print(f"Warning: Unknown config parameter '{key}' found in the config file and ignored.")

        return converted_config

    @beartype
    def merge_args_with_config(self: Any, config: Any, cli_args: Any) -> argparse.Namespace:
        """ Merge CLI args with config file args (CLI takes precedence) """
        arg_defaults = {arg.dest: arg.default for arg in self.parser._actions if arg.default is not argparse.SUPPRESS}

        # Validate and convert the config values
        validated_config = self.validate_and_convert(config, arg_defaults)

        for key, value in vars(cli_args).items():
            if key in validated_config:
                setattr(cli_args, key, validated_config[key])

        return cli_args

    @beartype
    def parse_arguments(self: Any) -> argparse.Namespace:
        # First, parse the CLI arguments to check if config files are provided
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

        # Merge CLI args with config file (CLI has priority)
        _args = self.merge_args_with_config(config, _args)

        return _args

loader = ConfigLoader()
args = loader.parse_arguments()

if args.seed is not None:
    set_rng_seed(args.seed)

if args.max_eval is None and args.generation_strategy is None and args.continue_previous_job is None:
    print_red("Either --max_eval or --generation_strategy must be set.")
    my_exit(104)

if not 0 <= args.pareto_front_confidence <= 1:
    print_yellow("--pareto_front_confidence must be between 0 and 1, will be set to 1")
    args.pareto_front_confidence = 1

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

disable_logs = None

try:
    with console.status("[bold green]Loading torch...") as status:
        import torch
    with console.status("[bold green]Loading numpy...") as status:
        import numpy as np
    with console.status("[bold green]Loading ax...") as status:
        import ax

        from ax.plot.pareto_utils import compute_posterior_pareto_frontier
        from ax.core.objective import MultiObjective
        from ax.core import Metric
        import ax.exceptions.core
        import ax.exceptions.generation_strategy
        import ax.modelbridge.generation_node
        from ax.modelbridge.model_spec import ModelSpec
        from ax.modelbridge.generation_strategy import (GenerationStep, GenerationStrategy)
        from ax.modelbridge.registry import Models
        from ax.modelbridge.modelbridge_utils import get_pending_observation_features
        from ax.storage.json_store.load import load_experiment
        from ax.storage.json_store.save import save_experiment

        from ax.core.base_trial import TrialStatus
        from ax.core.data import Data
        from ax.core.experiment import Experiment
        from ax.core.parameter import RangeParameter, FixedParameter, ChoiceParameter, ParameterType
        from ax.core.types import TParameterization
        from ax.modelbridge.external_generation_node import ExternalGenerationNode
        from ax.modelbridge.generation_node import GenerationNode
        from ax.modelbridge.transition_criterion import MaxTrials
        from ax.service.ax_client import AxClient, ObjectiveProperties
        from sklearn.ensemble import RandomForestRegressor
    with console.status("[bold green]Loading botorch...") as status:
        import botorch
    with console.status("[bold green]Loading submitit...") as status:
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

with console.status("[bold green]Loading ax logger...") as status:
    from ax.utils.common.logger import disable_loggers
disable_logs = disable_loggers(names=["ax.modelbridge.base"], level=logging.CRITICAL)

NVIDIA_SMI_LOGS_BASE = None
global_gs: Optional[GenerationStrategy] = None

@dataclass(init=False)
class RandomForestGenerationNode(ExternalGenerationNode):
    @beartype
    def __init__(self: Any, num_samples: int, regressor_options: Dict[str, Any], seed: Optional[int] = None) -> None:
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

    @beartype
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

    @beartype
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

    @beartype
    def _is_within_constraints(self: Any, params_list: list) -> bool:
        if self.experiment.search_space.parameter_constraints:
            param_names = list(self.parameters.keys())
            params = dict(zip(param_names, params_list))

            for constraint in self.experiment.search_space.parameter_constraints:
                if not constraint.check(params):
                    return False

                return True

        return True

    @beartype
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

    @beartype
    def _build_reverse_choice_map(self: Any, choice_parameters: dict) -> dict:
        choice_value_map = {}
        for name, param in choice_parameters.items():
            for value, idx in param.items():
                choice_value_map[value] = idx
        return {idx: value for value, idx in choice_value_map.items()}

    @beartype
    def _generate_ranged_samples(self: Any, ranged_parameters: list) -> np.ndarray:
        ranged_bounds = np.array([[low, high] for _, low, high in ranged_parameters])
        unit_samples = np.random.random_sample([self.num_samples, len(ranged_bounds)])
        return ranged_bounds[:, 0] + (ranged_bounds[:, 1] - ranged_bounds[:, 0]) * unit_samples

    @beartype
    def _build_all_samples(self: Any, ranged_parameters: list, ranged_samples: np.ndarray, fixed_values: dict, choice_parameters: dict) -> list:
        all_samples = []
        for sample_idx in range(self.num_samples):
            sample = self._build_single_sample(sample_idx, ranged_parameters, ranged_samples, fixed_values, choice_parameters)
            all_samples.append(sample)
        return all_samples

    @beartype
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

    @beartype
    def _cast_value(self: Any, param: Any, name: Any, value: Any) -> Union[int, float]:
        if isinstance(param, RangeParameter) and param.parameter_type == "INT":
            return int(round(value))
        if isinstance(param, RangeParameter) and param.parameter_type == "FLOAT":
            return float(value)

        return self._try_convert_to_float(value, name)

    @beartype
    def _try_convert_to_float(self: Any, value: Any, name: str) -> float:
        try:
            return float(value)
        except ValueError as e:
            raise ValueError(f"Parameter '{name}' has a non-numeric value: {value}") from e

    @beartype
    def _build_prediction_matrix(self: Any, all_samples: list) -> np.ndarray:
        x_pred = np.zeros([self.num_samples, len(self.parameters)])
        for sample_idx, sample in enumerate(all_samples):
            for dim, name in enumerate(self.parameters.keys()):
                x_pred[sample_idx, dim] = sample[name]
        return x_pred

    @beartype
    def _format_best_sample(self: Any, best_sample: TParameterization, reverse_choice_map: dict) -> None:
        for name in best_sample.keys():
            param = self.parameters.get(name)
            if isinstance(param, RangeParameter) and param.parameter_type == ParameterType.INT:
                best_sample[name] = int(round(best_sample[name]))
            elif isinstance(param, ChoiceParameter):
                best_sample[name] = str(reverse_choice_map.get(int(best_sample[name])))

@beartype
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
class ExternalProgramGenerationNode(ExternalGenerationNode):
    @beartype
    def __init__(self: Any, external_generator: str, node_name: str = "EXTERNAL_GENERATOR") -> None:
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

    @beartype
    def update_generator_state(self: Any, experiment: Any, data: Any) -> None:
        print_debug("Updating generator state...")

        self.data = data

        search_space = experiment.search_space
        self.parameters = search_space.parameters

        print_debug("Generator state updated successfully.")

    @beartype
    def _parameter_type_to_string(self: Any, param_type: Any) -> str:
        if param_type == ParameterType.INT:
            return "INT"

        if param_type == ParameterType.FLOAT:
            return "FLOAT"

        if param_type == ParameterType.STRING:
            return "STRING"

        print_red(f"Unknown data type {param_type}")
        my_exit(33)

        return ""

    @beartype
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
                print_red(f"Unknown parameter type: {param}")
                my_exit(15)

        return serialized

    @beartype
    def _serialize_constraints(self: Any, constraints: Optional[list]) -> Optional[list]:
        parsed_constraints = []
        if constraints and len(constraints):
            for constraint in constraints:
                representation = str(constraint)
                equation = representation[representation.find('(') + 1:representation.rfind(')')]

                parsed_constraints.append(equation)

        return parsed_constraints

    @beartype
    def get_next_candidate(self: Any, pending_parameters: List[Any]) -> Any:
        if self.parameters is None:
            raise RuntimeError("Parameters are not initialized. Call update_generator_state first.")

        print_debug("Getting next candidate...")

        try:
            temp_dir_counter = 0
            temp_dir = os.path.join(get_current_run_folder(), "external_generator_tmp", str(temp_dir_counter))
            while os.path.isdir(temp_dir):
                temp_dir_counter = temp_dir_counter + 1
                temp_dir = os.path.join(get_current_run_folder(), "external_generator_tmp", str(temp_dir_counter))

            os.makedirs(temp_dir, exist_ok=True)

            print_debug(f"Created temporary directory: {temp_dir}")

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

@beartype
def append_and_read(file: str, nr: int = 0, recursion: int = 0) -> int:
    try:
        with open(file, mode='a+', encoding="utf-8") as f:
            f.seek(0)  # Setze den Dateizeiger auf den Anfang der Datei
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

@beartype
def run_live_share_command() -> Tuple[str, str]:
    global shown_run_live_share_command

    if get_current_run_folder():
        try:
            # Environment variable USER
            _user = os.getenv('USER')
            if _user is None:
                _user = 'defaultuser'

            if args.username:
                _user = args.username

            _command = f"bash {script_dir}/omniopt_share {get_current_run_folder()} --update --username={_user} --no_color"

            if not shown_run_live_share_command:
                print_debug(f"run_live_share_command: {_command}")
                shown_run_live_share_command = True

            result = subprocess.run(_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # Return stdout and stderr
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

@beartype
def extract_and_print_qr(text: str) -> None:
    match = re.search(r"(https?://\S+|\b[\w.-]+@[\w.-]+\.\w+\b|\b\d{10,}\b)", text)
    if match:
        data = match.group(0)
        import qrcode

        qr = qrcode.QRCode(box_size=1, error_correction=qrcode.constants.ERROR_CORRECT_L, border=0)
        qr.add_data(data)
        qr.make()
        qr.print_ascii(out=sys.stdout)

@beartype
def live_share() -> bool:
    global SHOWN_LIVE_SHARE_COUNTER

    try:

        if not args.live_share:
            return False

        if not get_current_run_folder():
            return False

        stdout, stderr = run_live_share_command()

        if SHOWN_LIVE_SHARE_COUNTER == 0 and stderr:
            print_green(stderr)

            extract_and_print_qr(stderr)

            time.sleep(1)

        SHOWN_LIVE_SHARE_COUNTER = SHOWN_LIVE_SHARE_COUNTER + 1
    except KeyboardInterrupt:
        return False

    return True

@beartype
def compute_md5_hash(filepath: str) -> Optional[str]:
    if not os.path.exists(filepath):
        return None
    try:
        hasher = hashlib.new('md5', usedforsecurity=False)
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        print_red(f"Error computing MD5 for {filepath}: {e}")
        return None

@beartype
def save_results_csv() -> Optional[str]:
    pd_csv: str = f'{get_current_run_folder()}/{PD_CSV_FILENAME}'
    pd_json: str = f'{get_current_run_folder()}/state_files/pd.json'
    state_files_folder: str = f"{get_current_run_folder()}/state_files/"

    makedirs(state_files_folder)

    if ax_client is None:
        return None

    old_hash: Optional[str] = compute_md5_hash(pd_csv)

    try:
        ax_client.experiment.fetch_data()

        pd_frame = ax_client.get_trials_data_frame()
        pd_frame.to_csv(pd_csv, index=False, float_format="%.30f")

        json_snapshot = ax_client.to_json_snapshot()

        with open(pd_json, mode='w', encoding="utf-8") as json_file:
            json.dump(json_snapshot, json_file, indent=4)

        save_experiment(
            ax_client.experiment,
            f"{get_current_run_folder()}/state_files/ax_client.experiment.json"
        )
    except SignalUSR as e:
        raise SignalUSR(str(e)) from e
    except SignalCONT as e:
        raise SignalCONT(str(e)) from e
    except SignalINT as e:
        raise SignalINT(str(e)) from e
    except Exception as e:
        print_red(f"While saving all trials as a pandas-dataframe-csv, an error occurred: {e}")

    new_hash: Optional[str] = compute_md5_hash(pd_csv)

    if old_hash != new_hash:
        live_share()

    return pd_csv

@beartype
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
    print_red(f"--num_parallel_jobs must be 1 or larger, is {num_parallel_jobs}")
    my_exit(106)

class SearchSpaceExhausted (Exception):
    pass

NR_INSERTED_JOBS: int = 0
executor: Union[LocalExecutor, AutoExecutor, None] = None

NR_OF_0_RESULTS: int = 0

orchestrator = None

@beartype
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

VAL_IF_NOTHING_FOUND: int = 99999999999999999999999999999999999999999999999999999999999
NO_RESULT: str = "{:.0e}".format(VAL_IF_NOTHING_FOUND)

global_vars["jobs"] = []
global_vars["_time"] = None
global_vars["mem_gb"] = None
global_vars["num_parallel_jobs"] = None
global_vars["parameter_names"] = []

# max_eval usw. in unterordner
# grid ausblenden

main_pid = os.getpid()

@beartype
def set_nr_inserted_jobs(new_nr_inserted_jobs: int) -> None:
    global NR_INSERTED_JOBS

    print_debug(f"set_nr_inserted_jobs({new_nr_inserted_jobs})")

    NR_INSERTED_JOBS = new_nr_inserted_jobs

@beartype
def set_max_eval(new_max_eval: int) -> None:
    global max_eval

    print_debug(f"set_max_eval({new_max_eval})")

    max_eval = new_max_eval

@beartype
def write_worker_usage() -> None:
    if len(WORKER_PERCENTAGE_USAGE):
        csv_filename = f'{get_current_run_folder()}/worker_usage.csv'

        csv_columns = ['time', 'num_parallel_jobs', 'nr_current_workers', 'percentage']

        with open(csv_filename, mode='w', encoding="utf-8", newline='') as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            for row in WORKER_PERCENTAGE_USAGE:
                csv_writer.writerow(row)
    else:
        if is_slurm_job():
            print_debug("WORKER_PERCENTAGE_USAGE seems to be empty. Not writing worker_usage.csv")

@beartype
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

@beartype
def write_process_info() -> None:
    try:
        log_system_usage()
    except Exception as e:
        print_debug(f"Error retrieving process information: {str(e)}")

@beartype
def log_nr_of_workers() -> None:
    try:
        write_process_info()
    except Exception as e:
        print_debug(f"log_nr_of_workers: failed to write_process_info: {e}")
        return None

    if "jobs" not in global_vars:
        print_debug("log_nr_of_workers: Could not find jobs in global_vars")
        return None

    nr_of_workers: int = len(global_vars["jobs"])

    if not nr_of_workers:
        return None

    try:
        with open(logfile_nr_workers, mode='a+', encoding="utf-8") as f:
            f.write(str(nr_of_workers) + "\n")
    except FileNotFoundError:
        print_red(f"It seems like the folder for writing {logfile_nr_workers} was deleted during the run. Cannot continue.")
        my_exit(99)
    except OSError as e:
        print_red(f"Tried writing log_nr_of_workers to file {logfile_nr_workers}, but failed with error: {e}. This may mean that the file system you are running on is instable. OmniOpt2 probably cannot do anything about it.")
        my_exit(199)

    return None

@beartype
def log_what_needs_to_be_logged() -> None:
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

@beartype
def get_line_info() -> Tuple[str, str, int, str, str]:
    return (inspect.stack()[1][1], ":", inspect.stack()[1][2], ":", inspect.stack()[1][3])

@beartype
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
        _sleep(2)

        return True
    except Exception as e:
        print_debug(
            f"Error converting and resizing image: "
            f"{str(e)}, width: {width}, image_path: {image_path}"
        )

    return False

@beartype
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
        sys.exit(99) # generalized code for run folder deleted during run
    except Exception as e:
        original_print(f"Error trying to write log file: {e}")
        log_message_to_file(_logfile, message, _lvl + 1, e)

    return None

@beartype
def _log_trial_index_to_param(trial_index: dict, _lvl: int = 0, eee: Union[None, str, Exception] = None) -> None:
    log_message_to_file(logfile_trial_index_to_param_logs, str(trial_index), _lvl, str(eee))

@beartype
def _debug_worker_creation(msg: str, _lvl: int = 0, eee: Union[None, str, Exception] = None) -> None:
    log_message_to_file(logfile_worker_creation_logs, msg, _lvl, str(eee))

@beartype
def append_to_nvidia_smi_logs(_file: str, _host: str, result: str, _lvl: int = 0, eee: Union[None, str, Exception] = None) -> None:
    log_message_to_file(_file, result, _lvl, str(eee))

@beartype
def _debug_get_next_trials(msg: str, _lvl: int = 0, eee: Union[None, str, Exception] = None) -> None:
    log_message_to_file(LOGFILE_DEBUG_GET_NEXT_TRIALS, msg, _lvl, str(eee))

@beartype
def _debug_progressbar(msg: str, _lvl: int = 0, eee: Union[None, str, Exception] = None) -> None:
    log_message_to_file(logfile_progressbar, msg, _lvl, str(eee))

@beartype
def decode_if_base64(input_str: str) -> str:
    try:
        decoded_bytes = base64.b64decode(input_str)
        decoded_str = decoded_bytes.decode('utf-8')
        return decoded_str
    except Exception:
        return input_str

@beartype
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
        print_red(f"The previous job file {prev_job_file} could not be found. You may forgot to add the run number at the end.")
        my_exit(44)

if not args.tests and len(global_vars["joined_run_program"]) == 0:
    print_red("--run_program was empty")
    my_exit(19)

global_vars["experiment_name"] = args.experiment_name

@beartype
def load_global_vars(_file: str) -> None:
    global global_vars

    if not os.path.exists(_file):
        print_red(f"You've tried to continue a non-existing job: {_file}")
        my_exit(44)
    try:
        with open(_file, encoding="utf-8") as f:
            global_vars = json.load(f)
    except Exception as e:
        print_red(f"Error while loading old global_vars: {e}, trying to load {_file}")
        my_exit(44)

@beartype
def load_or_exit(filepath: str, error_msg: str, exit_code: int) -> None:
    if not os.path.exists(filepath):
        print_red(error_msg)
        my_exit(exit_code)

@beartype
def get_file_content_or_exit(filepath: str, error_msg: str, exit_code: int) -> str:
    load_or_exit(filepath, error_msg, exit_code)
    return get_file_as_string(filepath).strip()

@beartype
def check_param_or_exit(param: Optional[Union[list, str]], error_msg: str, exit_code: int) -> None:
    if param is None:
        print_red(error_msg)
        my_exit(exit_code)

@beartype
def check_continue_previous_job(continue_previous_job: Optional[str]) -> dict:
    if continue_previous_job:
        load_global_vars(f"{continue_previous_job}/state_files/global_vars.json")

        # Load experiment name from file if not already set
        if not global_vars.get("experiment_name"):
            exp_name_file = f"{continue_previous_job}/experiment_name"
            global_vars["experiment_name"] = get_file_content_or_exit(
                exp_name_file,
                f"{exp_name_file} not found, and no --experiment_name given. Cannot continue.",
                19
            )
    return global_vars

@beartype
def check_required_parameters(_args: Any) -> None:
    check_param_or_exit(
        _args.parameter or _args.continue_previous_job,
        "Either --parameter or --continue_previous_job is required. Both were not found.",
        19
    )
    check_param_or_exit(
        _args.run_program or _args.continue_previous_job,
        "--run_program needs to be defined when --continue_previous_job is not set",
        19
    )
    check_param_or_exit(
        global_vars.get("experiment_name") or _args.continue_previous_job,
        "--experiment_name needs to be defined when --continue_previous_job is not set",
        19
    )

@beartype
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
        print_red("Missing --time parameter. Cannot continue.")
        my_exit(19)

@beartype
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

    print_red("--mem_gb needs to be set")
    my_exit(19)

    return None

@beartype
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

@beartype
def load_max_eval_or_exit(_args: Any) -> None:
    if _args.max_eval:
        set_max_eval(_args.max_eval)
        if _args.max_eval <= 0:
            print_red("--max_eval must be larger than 0")
            my_exit(19)
    elif _args.continue_previous_job:
        max_eval_file = f"{_args.continue_previous_job}/state_files/max_eval"
        max_eval_content = get_file_content_or_exit(max_eval_file, f"neither --max_eval nor file {max_eval_file} found", 19)
        if max_eval_content.isdigit():
            set_max_eval(int(max_eval_content))
            print_yellow(f"Using old run's --max_eval: {max_eval_content}")
        else:
            print_yellow(f"max_eval-setting: The contents of {max_eval_file} do not contain a single number")
    else:
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

@beartype
def print_debug_get_next_trials(got: int, requested: int, _line: int) -> None:
    time_str: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg: str = f"{time_str}, {got}, {requested}"

    _debug_get_next_trials(msg)

@beartype
def print_debug_progressbar(msg: str) -> None:
    time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"{time_str}: {msg}"

    _debug_progressbar(msg)

@beartype
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

@beartype
def receive_usr_signal(signum: int, stack: Any) -> None:
    """Handle SIGUSR1 signal."""
    siginfo = signal.sigwaitinfo({signum})
    fool_linter(stack)
    print_yellow(f"\nReceived SIGUSR1 ({signum}) from PID {siginfo.si_pid}, sent by UID {siginfo.si_uid}\n")

    # Show process and hierarchy information
    process_info = get_process_info(siginfo.si_pid)
    print_yellow(f"Process info for PID {siginfo.si_pid}:\n  → {process_info}\n")

    raise SignalUSR(f"USR1-signal received ({signum})")

@beartype
def receive_usr_signal_term(signum: int, stack: Any) -> None:
    """Handle SIGTERM signal (termination)."""
    siginfo = signal.sigwaitinfo({signum})
    fool_linter(stack)
    print_yellow(f"\nReceived SIGTERM ({signum}) from PID {siginfo.si_pid}, sent by UID {siginfo.si_uid}\n")

    # Show process and hierarchy information
    process_info = get_process_info(siginfo.si_pid)
    print_yellow(f"Process info for PID {siginfo.si_pid}:\n  → {process_info}\n")

    # Additional specific handling for SIGTERM
    raise SignalTERM(f"TERM-signal received ({signum})")

@beartype
def receive_signal_cont(signum: int, stack: Any) -> None:
    """Handle SIGCONT signal (continue)."""
    siginfo = signal.sigwaitinfo({signum})
    fool_linter(stack)
    print_yellow(f"\nReceived SIGCONT ({signum}) from PID {siginfo.si_pid}, sent by UID {siginfo.si_uid}\n")

    # Show process and hierarchy information
    process_info = get_process_info(siginfo.si_pid)
    print_yellow(f"Process info for PID {siginfo.si_pid}:\n  → {process_info}\n")

    raise SignalCONT(f"CONT-signal received ({signum})")

# Signal handlers registration
signal.signal(signal.SIGUSR1, receive_usr_signal)
signal.signal(signal.SIGUSR2, receive_usr_signal)
signal.signal(signal.SIGTERM, receive_usr_signal_term)
signal.signal(signal.SIGCONT, receive_signal_cont)

print_debug(f"Current PID: {os.getpid()}")

@beartype
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

@beartype
def save_global_vars() -> None:
    state_files_folder = f"{get_current_run_folder()}/state_files"
    makedirs(state_files_folder)

    with open(f'{state_files_folder}/global_vars.json', mode="w", encoding="utf-8") as f:
        json.dump(global_vars, f)

@beartype
def check_slurm_job_id() -> None:
    if SYSTEM_HAS_SBATCH:
        slurm_job_id = os.environ.get('SLURM_JOB_ID')
        if slurm_job_id is not None and not slurm_job_id.isdigit():
            print_red("Not a valid SLURM_JOB_ID.")
        elif slurm_job_id is None:
            print_red(
                "You are on a system that has SLURM available, but you are not running the main-script in a SLURM-Environment. "
                "This may cause the system to slow down for all other users. It is recommended you run the main script in a SLURM-job."
            )

@beartype
def create_folder_and_file(folder: str) -> str:
    print_debug(f"create_folder_and_file({folder})")

    makedirs(folder)

    file_path = os.path.join(folder, "results.csv")

    return file_path

@beartype
def get_program_code_from_out_file(f: str) -> str:
    if not os.path.exists(f):
        print_debug(f"{f} not found")
        print_red(f"\n{f} not found")
        return ""

    fs = get_file_as_string(f)

    for line in fs.split("\n"):
        if "Program-Code:" in line:
            return line

    return ""

@beartype
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

@beartype
def _get_column_value(pd_csv: str, column: str, default: Union[None, float, int], mode: str) -> Tuple[Optional[Union[int, float]], bool]:
    found_in_file = False
    column_value = get_min_or_max_column_value(pd_csv, column, default, mode)

    if column_value is not None:
        found_in_file = True
        if isinstance(column_value, (int, float)) and isinstance(default, (int, float)):
            if (mode == "min" and default > column_value) or (mode == "max" and default < column_value):
                return column_value, found_in_file

    return default, found_in_file

@beartype
def get_ret_value_from_pd_csv(pd_csv: str, _type: str, _column: str, _default: Union[None, float, int]) -> Tuple[Optional[Union[int, float]], bool]:
    if not helpers.file_exists(pd_csv):
        print_red(f"'{pd_csv}' was not found")
        return _default, False

    mode: str = "min" if _type == "lower" else "max"
    return _get_column_value(pd_csv, _column, _default, mode)

@beartype
def get_bound_if_prev_data(_type: str, _column: str, _default: Union[None, float, int]) -> Union[Tuple[Union[float, int], bool], Any]:
    ret_val = _default

    found_in_file = False

    if args.continue_previous_job:
        pd_csv = f"{args.continue_previous_job}/{PD_CSV_FILENAME}"

        ret_val, found_in_file = get_ret_value_from_pd_csv(pd_csv, _type, _column, _default)

    if isinstance(ret_val, (int, float)):
        return round(ret_val, args.decimalrounding), found_in_file

    return ret_val, False

@beartype
def switch_lower_and_upper_if_needed(name: Union[list, str], lower_bound: Union[float, int], upper_bound: Union[float, int]) -> Tuple[Union[int, float], Union[int, float]]:
    if lower_bound > upper_bound:
        print_yellow(f"Lower bound ({lower_bound}) was larger than upper bound ({upper_bound}) for parameter '{name}'. Switched them.")
        upper_bound, lower_bound = lower_bound, upper_bound

    return lower_bound, upper_bound

@beartype
def round_lower_and_upper_if_type_is_int(value_type: str, lower_bound: Union[int, float], upper_bound: Union[int, float]) -> Tuple[Union[int, float], Union[int, float]]:
    if value_type == "int":
        if not helpers.looks_like_int(lower_bound):
            print_yellow(f"{value_type} can only contain integers. You chose {lower_bound}. Will be rounded down to {math.floor(lower_bound)}.")
            lower_bound = math.floor(lower_bound)

        if not helpers.looks_like_int(upper_bound):
            print_yellow(f"{value_type} can only contain integers. You chose {upper_bound}. Will be rounded up to {math.ceil(upper_bound)}.")
            upper_bound = math.ceil(upper_bound)

    return lower_bound, upper_bound

@beartype
def get_bounds(this_args: Union[str, list], j: int) -> Tuple[float, float]:
    try:
        lower_bound = float(this_args[j + 2])
    except Exception:
        print_red(f"\n{this_args[j + 2]} is not a number")
        my_exit(181)

    try:
        upper_bound = float(this_args[j + 3])
    except Exception:
        print_red(f"\n{this_args[j + 3]} is not a number")
        my_exit(181)

    return lower_bound, upper_bound

@beartype
def adjust_bounds_for_value_type(value_type: str, lower_bound: Union[int, float], upper_bound: Union[int, float]) -> Union[Tuple[float, float], Tuple[int, int]]:
    lower_bound, upper_bound = round_lower_and_upper_if_type_is_int(value_type, lower_bound, upper_bound)

    if value_type == "int":
        lower_bound = math.floor(lower_bound)
        upper_bound = math.ceil(upper_bound)

    return lower_bound, upper_bound

@beartype
def create_param(name: Union[list, str], lower_bound: Union[float, int], upper_bound: Union[float, int], value_type: str, log_scale: bool) -> dict:
    return {
        "name": name,
        "type": "range",
        "bounds": [lower_bound, upper_bound],
        "value_type": value_type,
        "log_scale": log_scale
    }

@beartype
def handle_grid_search(name: Union[list, str], lower_bound: Union[float, int], upper_bound: Union[float, int], value_type: str) -> dict:
    if lower_bound is None or upper_bound is None:
        print_red("handle_grid_search: lower_bound or upper_bound is None")
        my_exit(91)

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
        "values": values_str
    }

@beartype
def get_bounds_from_previous_data(name: str, lower_bound: Union[float, int], upper_bound: Union[float, int]) -> Tuple[Union[float, int], Union[float, int]]:
    lower_bound, _ = get_bound_if_prev_data("lower", name, lower_bound)
    upper_bound, _ = get_bound_if_prev_data("upper", name, upper_bound)
    return lower_bound, upper_bound

@beartype
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

@beartype
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

@beartype
def parse_range_param(params: list, j: int, this_args: Union[str, list], name: str, search_space_reduction_warning: bool) -> Tuple[int, list, bool]:
    check_factorial_range()
    check_range_params_length(this_args)

    lower_bound: Union[float, int]
    upper_bound: Union[float, int]

    lower_bound, upper_bound = get_bounds(this_args, j)

    die_181_or_91_if_lower_and_upper_bound_equal_zero(lower_bound, upper_bound)

    lower_bound, upper_bound = switch_lower_and_upper_if_needed(name, lower_bound, upper_bound)

    skip, value_type, log_scale = get_value_type_and_log_scale(this_args, j)

    validate_value_type(value_type)

    lower_bound, upper_bound = adjust_bounds_for_value_type(value_type, lower_bound, upper_bound)

    lower_bound, upper_bound = get_bounds_from_previous_data(name, lower_bound, upper_bound)

    search_space_reduction_warning = check_bounds_change_due_to_previous_job(name, lower_bound, upper_bound, search_space_reduction_warning)

    param = create_param(name, lower_bound, upper_bound, value_type, log_scale)

    if args.gridsearch:
        param = handle_grid_search(name, lower_bound, upper_bound, value_type)

    global_vars["parameter_names"].append(name)
    params.append(param)

    j += skip
    return j, params, search_space_reduction_warning

@beartype
def validate_value_type(value_type: str) -> None:
    valid_value_types = ["int", "float"]
    check_if_range_types_are_invalid(value_type, valid_value_types)

@beartype
def parse_fixed_param(params: list, j: int, this_args: Union[str, list], name: Union[list, str], search_space_reduction_warning: bool) -> Tuple[int, list, bool]:
    if len(this_args) != 3:
        print_red("⚠ --parameter for type fixed must have 3 parameters: <NAME> fixed <VALUE>")
        my_exit(181)

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

    return j, params, search_space_reduction_warning

@beartype
def parse_choice_param(params: list, j: int, this_args: Union[str, list], name: Union[list, str], search_space_reduction_warning: bool) -> Tuple[int, list, bool]:
    if len(this_args) != 3:
        print_red("⚠ --parameter for type choice must have 3 parameters: <NAME> choice <VALUE,VALUE,VALUE,...>")
        my_exit(181)

    values = re.split(r'\s*,\s*', str(this_args[j + 2]))

    values[:] = [x for x in values if x != ""]

    param = {
        "name": name,
        "type": "choice",
        "is_ordered": False,
        "values": values
    }

    global_vars["parameter_names"].append(name)

    params.append(param)

    j += 3

    return j, params, search_space_reduction_warning

@beartype
def parse_experiment_parameters() -> list:
    params: list = []
    param_names: List[str] = []

    i = 0

    search_space_reduction_warning = False

    valid_types = ["range", "fixed", "choice"]
    invalid_names = ["start_time", "end_time", "run_time", "program_string", *arg_result_names, "exit_code", "signal"]

    while args.parameter and i < len(args.parameter):
        this_args = args.parameter[i]
        j = 0

        if this_args is not None and isinstance(this_args, dict) and "param" in this_args:
            this_args = this_args["param"]

        while j < len(this_args) - 1:
            name = this_args[j]

            if name in invalid_names:
                print_red(f"\n⚠ Name for argument no. {j} is invalid: {name}. Invalid names are: {', '.join(invalid_names)}")
                my_exit(181)

            if name in param_names:
                print_red(f"\n⚠ Parameter name '{name}' is not unique. Names for parameters must be unique!")
                my_exit(181)

            param_names.append(name)

            try:
                param_type = this_args[j + 1]
            except Exception:
                print_red("Not enough arguments for --parameter")
                my_exit(181)

            if param_type not in valid_types:
                valid_types_string = ', '.join(valid_types)
                print_red(f"\n⚠ Invalid type {param_type}, valid types are: {valid_types_string}")
                my_exit(181)

            param_parsers = {
                "range": parse_range_param,
                "fixed": parse_fixed_param,
                "choice": parse_choice_param
            }

            if param_type in param_parsers:
                j, params, search_space_reduction_warning = param_parsers[param_type](params, j, this_args, name, search_space_reduction_warning)
            else:
                print_red(f"⚠ Parameter type '{param_type}' not yet implemented.")
                my_exit(181)

        i += 1

    if search_space_reduction_warning:
        print_red("⚠ Search space reduction is not currently supported on continued runs or runs that have previous data.")

    return params

@beartype
def check_factorial_range() -> None:
    if args.model and args.model == "FACTORIAL":
        print_red("\n⚠ --model FACTORIAL cannot be used with range parameter")
        my_exit(181)

@beartype
def check_if_range_types_are_invalid(value_type: str, valid_value_types: list) -> None:
    if value_type not in valid_value_types:
        valid_value_types_string = ", ".join(valid_value_types)
        print_red(f"⚠ {value_type} is not a valid value type. Valid types for range are: {valid_value_types_string}")
        my_exit(181)

@beartype
def check_range_params_length(this_args: Union[str, list]) -> None:
    if len(this_args) != 5 and len(this_args) != 4 and len(this_args) != 6:
        print_red("\n⚠ --parameter for type range must have 4 (or 5, the last one being optional and float by default, or 6, while the last one is true or false) parameters: <NAME> range <START> <END> (<TYPE (int or float)>, <log_scale: bool>)")
        my_exit(181)

@beartype
def die_181_or_91_if_lower_and_upper_bound_equal_zero(lower_bound: Union[int, float], upper_bound: Union[int, float]) -> None:
    if upper_bound is None or lower_bound is None:
        print_red("die_181_or_91_if_lower_and_upper_bound_equal_zero: upper_bound or lower_bound is None. Cannot continue.")
        my_exit(91)
    if upper_bound == lower_bound:
        if lower_bound == 0:
            print_red(f"⚠ Lower bound and upper bound are equal: {lower_bound}, cannot automatically fix this, because they -0 = +0 (usually a quickfix would be to set lower_bound = -upper_bound)")
            my_exit(181)
        print_red(f"⚠ Lower bound and upper bound are equal: {lower_bound}, setting lower_bound = -upper_bound")
        if upper_bound is not None:
            lower_bound = -upper_bound

@beartype
def replace_parameters_in_string(parameters: dict, input_string: str) -> str:
    try:
        for param_item in parameters:
            input_string = input_string.replace(f"${param_item}", str(parameters[param_item]))
            input_string = input_string.replace(f"$({param_item})", str(parameters[param_item]))

            input_string = input_string.replace(f"%{param_item}", str(parameters[param_item]))
            input_string = input_string.replace(f"%({param_item})", str(parameters[param_item]))

        input_string = input_string.replace('\r', ' ').replace('\n', ' ')

        return input_string
    except Exception as e:
        print_red(f"\n⚠ Error: {e}")
        return ""

@beartype
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

        print_debug_once(f"self.thread.daemon was set to {self.thread.daemon}") # only for deadcode to not complain

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

@beartype
def execute_bash_code_log_time(code: str) -> list:
    process_item = subprocess.Popen(code, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Monitor-Prozess starten
    with MonitorProcess(process_item.pid):  # Startet den Monitor im 'with'-Kontext
        try:
            stdout, stderr = process_item.communicate()  # Warten auf Beendigung des Prozesses
            result = subprocess.CompletedProcess(
                args=code, returncode=process_item.returncode, stdout=stdout, stderr=stderr
            )
            # Erfolgreiche Rückgabe der Ausgabe
            return [result.stdout, result.stderr, result.returncode, None]
        except subprocess.CalledProcessError as e:
            real_exit_code = e.returncode
            signal_code = None
            if real_exit_code < 0:  # falls Signalcode vorhanden ist
                signal_code = abs(e.returncode)
                real_exit_code = 1
            # Rückgabe im Fehlerfall
            return [e.stdout, e.stderr, real_exit_code, signal_code]

@beartype
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

@beartype
def get_results(input_string: Optional[Union[int, str]]) -> Optional[Union[Dict[str, Optional[float]], List[float]]]:
    if input_string is None:
        print_red("get_results: Input-String is None")
        return None

    if not isinstance(input_string, str):
        print_red(f"get_results: Type of input_string is not string, but {type(input_string)}")
        return None

    try:
        results: Dict[str, Optional[float]] = {}  # Typdefinition angepasst

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

@beartype
def add_to_csv(file_path: str, new_heading: list, new_data: list) -> None:
    new_data = [helpers.to_int_when_possible(x) for x in new_data]

    with open(file_path, 'a+', encoding="utf-8", newline='') as file:
        fcntl.flock(file, fcntl.LOCK_EX)  # Lock file

        file.seek(0)
        rows = list(csv.reader(file))  # Read entire file into memory
        existing_heading = rows[0] if rows else []

        # Merge old and new headers, preserving order
        all_headings = list(dict.fromkeys(existing_heading + new_heading))

        # Update all rows to match the new header order
        updated_rows = [all_headings] + [
            [row[existing_heading.index(h)] if h in existing_heading else "" for h in all_headings]
            for row in rows[1:]
        ]

        # Prepare new data line
        formatted_data = [
            ("{:.20f}".format(x).rstrip('0').rstrip('.')) if isinstance(x, float) else x
            for x in new_data
        ]
        new_row = [formatted_data[new_heading.index(h)] if h in new_heading else "" for h in all_headings]
        updated_rows.append(new_row)

        # Rewrite the entire file
        file.seek(0)
        file.truncate()
        csv.writer(file).writerows(updated_rows)

        fcntl.flock(file, fcntl.LOCK_UN)  # Unlock file

@beartype
def find_file_paths(_text: str) -> List[str]:
    file_paths = []

    if isinstance(_text, str):
        words = _text.split()

        for word in words:
            if os.path.exists(word):
                file_paths.append(word)

        return file_paths

    return []

@beartype
def check_file_info(file_path: str) -> str:
    if not os.path.exists(file_path):
        print(f"check_file_info: The file {file_path} does not exist.")
        return ""

    if not os.access(file_path, os.R_OK):
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

@beartype
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

@beartype
def write_failed_logs(data_dict: dict, error_description: str = "") -> None:
    headers = list(data_dict.keys())
    data = [list(data_dict.values())]

    if error_description:
        headers.append('error_description')
        for row in data:
            row.append(error_description)

    failed_logs_dir = os.path.join(get_current_run_folder(), 'failed_logs')

    data_file_path = os.path.join(failed_logs_dir, 'parameters.csv')
    header_file_path = os.path.join(failed_logs_dir, 'headers.csv')

    try:
        # Create directories if they do not exist
        makedirs(failed_logs_dir)

        # Write headers if the file does not exist
        if not os.path.exists(header_file_path):
            try:
                with open(header_file_path, mode='w', encoding='utf-8', newline='') as header_file:
                    writer = csv.writer(header_file)
                    writer.writerow(headers)
                    print_debug(f"Header file created with headers: {headers}")
            except Exception as e:
                print_red(f"Failed to write header file: {e}")

        # Append data to the data file
        try:
            with open(data_file_path, mode='a', encoding="utf-8", newline='') as data_file:
                writer = csv.writer(data_file)
                writer.writerows(data)
                print_debug(f"Data appended to file: {data_file_path}")

        except Exception as e:
            print_red(f"Failed to append data to file: {e}")

    except Exception as e:
        print_red(f"Unexpected error: {e}")

@beartype
def count_defective_nodes(file_path: Union[str, None] = None, entry: Any = None) -> list:
    if file_path is None:
        file_path = os.path.join(get_current_run_folder(), "state_files", "defective_nodes")

    # Sicherstellen, dass das Verzeichnis existiert
    makedirs(os.path.dirname(file_path))

    try:
        with open(file_path, mode='a+', encoding="utf-8") as file:
            file.seek(0)  # Zurück zum Anfang der Datei
            lines = file.readlines()

            entries = [line.strip() for line in lines]

            if entry is not None and entry not in entries:
                file.write(entry + '\n')
                entries.append(entry)

        return sorted(set(entries))

    except Exception as e:
        print(f"An error has occurred: {e}")
        return []

@beartype
def test_gpu_before_evaluate(return_in_case_of_error: dict) -> Union[None, dict]:
    if SYSTEM_HAS_SBATCH and args.gpus >= 1 and args.auto_exclude_defective_hosts and not args.force_local_execution:
        try:
            for i in range(torch.cuda.device_count()):
                tmp = torch.cuda.get_device_properties(i).name
                print_debug(f"Got CUDA device {tmp}")
        except RuntimeError:
            print(f"Node {socket.gethostname()} was detected as faulty. It should have had a GPU, but there is an error initializing the CUDA driver. Adding this node to the --exclude list.")
            count_defective_nodes(None, socket.gethostname())
            return return_in_case_of_error
        except Exception:
            pass

    return None

@beartype
def extract_info(data: Optional[str]) -> Tuple[List[str], List[str]]:
    if data is None:
        return [], []

    names: List[str] = []
    values: List[str] = []

    # Regex-Muster für OO-Info, das sowohl Groß- als auch Kleinschreibung berücksichtigt
    _pattern = re.compile(r'\s*OO-Info:\s*([a-zA-Z0-9_]+):\s*(.+)\s*$', re.IGNORECASE)

    # Gehe durch jede Zeile im String
    for line in data.splitlines():
        match = _pattern.search(line)
        if match:
            names.append(f"OO_Info_{match.group(1)}")
            values.append(match.group(2))

    return names, values

@beartype
def ignore_signals() -> None:
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    signal.signal(signal.SIGUSR2, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGQUIT, signal.SIG_IGN)

@beartype
def calculate_signed_harmonic_distance(_args: Union[dict, List[Union[int, float]]]) -> Union[int, float]:
    if not _args or len(_args) == 0: # Handle empty input gracefully
        return 0

    abs_inverse_sum: float = sum(1 / abs(a) for a in _args if a != 0)  # Avoid division by zero
    harmonic_mean: float = len(_args) / abs_inverse_sum if abs_inverse_sum != 0 else 0

    # Determine the sign based on the number of negatives
    num_negatives: float = sum(1 for a in _args if a < 0)
    sign: int = -1 if num_negatives % 2 != 0 else 1

    return sign * harmonic_mean

@beartype
def calculate_signed_euclidean_distance(_args: Union[dict, List[float]]) -> float:
    _sum = sum(a ** 2 for a in _args)
    sign = -1 if any(a < 0 for a in _args) else 1
    return sign * math.sqrt(_sum)

@beartype
def calculate_signed_geometric_distance(_args: Union[dict, List[float]]) -> float:
    product: float = 1
    for a in _args:
        product *= abs(a)

    num_negatives: float = sum(1 for a in _args if a < 0)
    sign: int = -1 if num_negatives % 2 != 0 else 1

    geometric_mean: float = product ** (1 / len(_args)) if _args else 0
    return sign * geometric_mean

@beartype
def calculate_signed_minkowski_distance(_args: Union[dict, List[float]], p: Union[int, float] = 2) -> float:
    if p <= 0:
        raise ValueError("p must be greater than 0.")

    sign: int = -1 if any(a < 0 for a in _args) else 1
    minkowski_sum: float = sum(abs(a) ** p for a in _args) ** (1 / p)
    return sign * minkowski_sum

@beartype
def calculate_signed_weighted_euclidean_distance(_args: Union[dict, List[float]], weights_string: str) -> float:
    pattern = r'^\s*-?\d+(\.\d+)?\s*(,\s*-?\d+(\.\d+)?\s*)*$'

    if not re.fullmatch(pattern, weights_string):
        print_red(f"String '{weights_string}' does not match pattern {pattern}")
        my_exit(32)

    weights = [float(w.strip()) for w in weights_string.split(",") if w.strip()]

    if len(weights) > len(_args):
        print_yellow(f"calculate_signed_weighted_euclidean_distance: Warning: Trimming {len(weights) - len(_args)} extra weight(s): {weights[len(_args):]}")
        weights = weights[:len(_args)]

    if len(weights) < len(_args):
        print_yellow("calculate_signed_weighted_euclidean_distance: Warning: Not enough weights, filling with 1s")
        weights.extend([1] * (len(_args) - len(weights)))

    if len(_args) != len(weights):
        raise ValueError("Length of _args and weights must match.")

    weighted_sum: float = sum(w * (a ** 2) for a, w in zip(_args, weights))
    sign: int = -1 if any(a < 0 for a in _args) else 1
    return sign * (weighted_sum ** 0.5)

class invalidOccType(Exception):
    pass

@beartype
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

@beartype
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

@beartype
def write_job_infos_csv(parameters: dict, stdout: Optional[str], program_string_with_params: str, exit_code: Optional[int], _signal: Optional[int], result: Optional[Union[Dict[str, Optional[float]], List[float], int, float]], start_time: Union[int, float], end_time: Union[int, float], run_time: Union[float, int]) -> None:
    str_parameters_values: List[str] = [str(v) for v in list(parameters.values())]

    extra_vars_names, extra_vars_values = extract_info(stdout)

    _SLURM_JOB_ID = os.getenv('SLURM_JOB_ID')
    if _SLURM_JOB_ID:
        extra_vars_names.append("OO_Info_SLURM_JOB_ID")
        extra_vars_values.append(str(_SLURM_JOB_ID))

    parameters_keys = list(parameters.keys())

    headline: List[str] = [
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

    result_values = []

    if isinstance(result, list):
        for rkey in result:
            result_values.append(str(rkey))
    elif isinstance(result, dict):
        result_keys: list = list(result.keys())
        for rkey in result_keys:
            rval = str(result[str(rkey)])

            result_values.append(rval)

    values: List[str] = [
        str(start_time),
        str(end_time),
        str(run_time),
        program_string_with_params,
        *str_parameters_values,
        *result_values,
        str(exit_code),
        str(_signal),
        socket.gethostname(),
        *extra_vars_values
    ]

    headline = ['None' if element is None else element for element in headline]
    values = ['None' if element is None else element for element in values]

    if get_current_run_folder() is not None and os.path.exists(get_current_run_folder()):
        add_to_csv(f"{get_current_run_folder()}/job_infos.csv", headline, values)
    else:
        print_debug(f"evaluate: get_current_run_folder() {get_current_run_folder()} could not be found")

@beartype
def print_evaluate_times() -> None:
    file_path = f"{get_current_run_folder()}/job_infos.csv"

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
            headers = ["Number of values", "Min time", "Max time", "Average time", "Median time"]
            cols = [str(len(time_values)), f"{min_time:.2f} sec", f"{max_time:.2f} sec", f"{avg_time:.2f} sec", f"{median_time:.2f} sec"]

            table = Table(title="Runtime Infos:")
            for h in headers:
                table.add_column(h, justify="center")

            table.add_row(*cols)

            console.print(table)

            overview_file = f"{get_current_run_folder()}/time_overview.txt"
            with open(overview_file, mode='w', encoding='utf-8') as overview:
                overview.write(f"Number of values: {len(time_values)} sec\n")
                overview.write(f"Min Time: {min_time:.2f} sec\n")
                overview.write(f"Max Time: {max_time:.2f} sec\n")
                overview.write(f"Average Time: {avg_time:.2f} sec\n")
                overview.write(f"Median Time: {median_time:.2f} sec\n")

@beartype
def print_debug_infos(program_string_with_params: str) -> None:
    string = find_file_paths_and_print_infos(program_string_with_params, program_string_with_params)

    original_print("Debug-Infos:", string)

@beartype
def print_stdout_and_stderr(stdout: Optional[str], stderr: Optional[str]) -> None:
    if stdout:
        original_print("stdout:\n", stdout)
    else:
        original_print("stdout was empty")

    if stderr:
        original_print("stderr:\n", stderr)
    else:
        original_print("stderr was empty")

@beartype
def evaluate_print_stuff(parameters: dict, program_string_with_params: str, stdout: Optional[str], stderr: Optional[str], exit_code: Optional[int], _signal: Optional[int], result: Optional[Union[Dict[str, Optional[float]], List[float], int, float]], start_time: Union[float, int], end_time: Union[float, int], run_time: Union[float, int], final_result: dict) -> None:
    original_print(f"Parameters: {json.dumps(parameters)}")

    print_debug_infos(program_string_with_params)

    original_print(program_string_with_params)

    print_stdout_and_stderr(stdout, stderr)

    original_print(f"Result: {result}")

    original_print(f"Final-results: {final_result}")

    write_job_infos_csv(parameters, stdout, program_string_with_params, exit_code, _signal, result, start_time, end_time, run_time)

    original_print(f"EXIT_CODE: {exit_code}")

@beartype
def get_results_with_occ(stdout: str) -> Union[int, float, Optional[Union[Dict[str, Optional[float]], List[float]]]]:
    result = get_results(stdout)

    if result and args.occ:
        occed_result = calculate_occ(result)

        if occed_result is not None:
            result = [occed_result]

    return result

@beartype
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

@beartype
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

@beartype
def attach_sem_to_result(stdout: str, name: str, value: Union[int, float, None, list]) -> Optional[Union[tuple, int, float, list]]:
    sem = get_result_sem(stdout, name)
    if sem:
        return (value, sem)

    return value

@beartype
def evaluate(parameters: dict) -> Optional[Union[int, float, Dict[str, Optional[Union[int, float, Tuple]]], List[float]]]:
    start_nvidia_smi_thread()

    return_in_case_of_error: dict = get_return_in_case_of_errors()

    _test_gpu = test_gpu_before_evaluate(return_in_case_of_error)
    final_result = return_in_case_of_error

    if _test_gpu is None:
        parameters = {
            k: (int(v) if isinstance(v, (int, float, str)) and re.fullmatch(r'^\d+(\.0+)?$', str(v)) else v)
            for k, v in parameters.items()
        }

        ignore_signals()

        signal_messages = {
            "USR1-signal": SignalUSR,
            "CONT-signal": SignalCONT,
            "INT-signal": SignalINT
        }

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

            final_result = {}

            if isinstance(result, (int, float)):
                for name in arg_result_names:
                    final_result[name] = attach_sem_to_result(stdout, name, float(result))

            elif isinstance(result, list):
                float_values = [float(r) for r in result]
                for name in arg_result_names:
                    final_result[name] = attach_sem_to_result(stdout, name, float_values)

            elif isinstance(result, dict):
                for name in arg_result_names:
                    final_result[name] = attach_sem_to_result(stdout, name, result.get(name))

            else:
                write_failed_logs(parameters, "No Result")

            evaluate_print_stuff(
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
                final_result
            )

        except tuple(signal_messages.values()) as sig:
            signal_name = get_signal_name(sig, signal_messages)
            print(f"\n⚠ {signal_name} was sent. Cancelling evaluation.")
            write_failed_logs(parameters, signal_name)

    return final_result

@beartype
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

@beartype
def disable_logging() -> None:
    if args.verbose:
        return

    with console.status("[bold green]Disabling logging..."):
        logging.basicConfig(level=logging.CRITICAL)
        logging.getLogger().setLevel(logging.CRITICAL)
        logging.getLogger().disabled = True

        fool_linter(f"logging.getLogger().disabled set to {logging.getLogger().disabled}")

        categories = [FutureWarning, RuntimeWarning, UserWarning, Warning]

        modules = [
            "ax",

            "ax.core.data",
            "ax.core.parameter",
            "ax.core.experiment",

            "ax.models.torch.botorch_modular.acquisition",

            "ax.modelbridge"
            "ax.modelbridge.base",
            "ax.modelbridge.standardize_y",
            "ax.modelbridge.transforms",
            "ax.modelbridge.transforms.standardize_y",
            "ax.modelbridge.transforms.int_to_float",
            "ax.modelbridge.cross_validation",
            "ax.modelbridge.dispatch_utils",
            "ax.modelbridge.torch",
            "ax.modelbridge.generation_node",
            "ax.modelbridge.best_model_selector",


            "ax.service",
            "ax.service.utils",
            "ax.service.utils.instantiation",
            "ax.service.utils.report_utils",
            "ax.service.utils.best_point",
            "ax.service.utils.with_db_settings_base",

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

@beartype
def display_failed_jobs_table() -> None:
    failed_jobs_file = f"{get_current_run_folder()}/failed_logs"
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

        # Create the table
        table = Table(show_header=True, header_style="bold red", title="Failed Jobs parameters:")

        for header in headers:
            table.add_column(header)

        added_rows = set()

        for parameter_set in parameters:
            row = [str(helpers.to_int_when_possible(value)) for value in parameter_set]
            row_tuple = tuple(row)  # Convert to tuple for set operations
            if row_tuple not in added_rows:
                table.add_row(*row, style='red')
                added_rows.add(row_tuple)

        # Print the table to the console
        console.print(table)
    except Exception as e:
        print_red(f"Error: {str(e)}")

@beartype
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

@beartype
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

@beartype
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

@beartype
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

    if arg_result_min_or_max[idx] == "max":
        maximize = True

    return maximize

@beartype
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

@beartype
def get_best_params(res_name: str = "RESULT") -> Optional[dict]:
    if os.path.exists(RESULT_CSV_FILE):
        return get_best_params_from_csv(res_name)

    return None

@beartype
def _count_sobol_or_completed(this_csv_file_path: str, _type: str) -> int:
    if _type not in ["Sobol", "COMPLETED"]:
        print_red(f"_type is not in Sobol or COMPLETED, but is '{_type}'")
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

    if _type == "Sobol":
        rows = df[df["generation_method"] == _type]
    else:
        rows = df[df["trial_status"] == _type]
    count = len(rows)

    return count

@beartype
def _count_sobol_steps(this_csv_file_path: str) -> int:
    return _count_sobol_or_completed(this_csv_file_path, "Sobol")

@beartype
def _count_done_jobs(this_csv_file_path: str) -> int:
    return _count_sobol_or_completed(this_csv_file_path, "COMPLETED")

@beartype
def count_sobol_steps() -> int:
    if os.path.exists(RESULT_CSV_FILE):
        return _count_sobol_steps(RESULT_CSV_FILE)

    return 0

@beartype
def get_random_steps_from_prev_job() -> int:
    if not args.continue_previous_job:
        return count_sobol_steps()

    prev_step_file: str = f"{args.continue_previous_job}/state_files/phase_random_steps"

    if not os.path.exists(prev_step_file):
        return count_sobol_steps()

    return add_to_phase_counter("random", count_sobol_steps() + _count_sobol_steps(f"{args.continue_previous_job}/results.csv"), args.continue_previous_job)

@beartype
def failed_jobs(nr: int = 0) -> int:
    state_files_folder = f"{get_current_run_folder()}/state_files/"

    makedirs(state_files_folder)

    return append_and_read(f'{get_current_run_folder()}/state_files/failed_jobs', nr)

@beartype
def count_done_jobs() -> int:
    if os.path.exists(RESULT_CSV_FILE):
        return _count_done_jobs(RESULT_CSV_FILE)

    return 0

@beartype
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
                "filename": "plot_%0_%1_%2" # omit file ending
            }
        )

    if args.show_sixel_general or _force:
        plot_types.append(
            {
                "type": "general"
            }
        )

    return plot_types

@beartype
def get_x_y_combinations_parameter_names() -> list:
    return list(combinations(global_vars["parameter_names"], 2))

@beartype
def get_plot_filename(plot: dict, _tmp: str) -> str:
    j = 0
    _fn = plot.get("filename", plot["type"])
    tmp_file = f"{_tmp}/{_fn}.png"

    while os.path.exists(tmp_file):
        j += 1
        tmp_file = f"{_tmp}/{_fn}_{j}.png"

    return tmp_file

@beartype
def build_command(plot_type: str, plot: dict, _force: bool) -> str:
    maindir = os.path.dirname(os.path.realpath(__file__))
    base_command = "bash omniopt_plot" if _force else f"bash {maindir}/omniopt_plot"
    command = f"{base_command} --run_dir {get_current_run_folder()} --plot_type={plot_type}"

    if "dpi" in plot:
        command += f" --dpi={plot['dpi']}"

    return command

@beartype
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

@beartype
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

@beartype
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

@beartype
def get_crf() -> str:
    crf = get_current_run_folder()
    if crf in ["", None]:
        console.print("[red]Could not find current run folder[/]")
        return ""
    return crf

@beartype
def write_to_file(file_path: str, content: str) -> None:
    with open(file_path, mode="w", encoding="utf-8") as text_file:
        text_file.write(content)

@beartype
def create_result_table(res_name: str, best_params: Optional[Dict[str, Any]], total_str: str, failed_error_str: str) -> Optional[Table]:
    min_or_max = arg_result_min_or_max[arg_result_names.index(res_name)]
    bracket_string = f"{total_str}{failed_error_str}"

    table = Table(
        show_header=True,
        header_style="bold",
        title=f"Best {res_name}, {min_or_max} ({bracket_string}):"
    )

    if best_params and "parameters" in best_params:
        row_data = {**best_params['parameters']}
        result_name = arg_result_names[0]
        row_data[result_name] = best_params.get(result_name, '?')

        for col in row_data.keys():
            table.add_column(col, style="bold")

        table.add_row(*[str(v) for v in row_data.values()])

        return table

    return None

@beartype
def print_and_write_table(table: Table, print_to_file: bool, file_path: str) -> None:
    with console.capture() as capture:
        console.print(table)
    if print_to_file:
        write_to_file(file_path, capture.get())

@beartype
def process_best_result(res_name: str, print_to_file: bool) -> int:
    best_params = get_best_params_from_csv(res_name)
    best_result = best_params.get(res_name, NO_RESULT) if best_params else NO_RESULT

    if str(best_result) in [NO_RESULT, None, "None"]:
        print_red(f"Best {res_name} could not be determined")
        return 87

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

@beartype
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

@beartype
def print_best_result() -> int:
    if os.path.exists(RESULT_CSV_FILE):
        return _print_best_result(True)

    return 0

@beartype
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

    print_evaluate_times()

    if best_result_exit > 0:
        _exit = best_result_exit

    if args.show_worker_percentage_table_at_end and len(WORKER_PERCENTAGE_USAGE) and not ALREADY_SHOWN_WORKER_USAGE_OVER_TIME:
        ALREADY_SHOWN_WORKER_USAGE_OVER_TIME = True

        table = Table(header_style="bold", title="Worker usage over time:")
        columns = ["Time", "Nr. workers", "Max. nr. workers", "%"]
        for column in columns:
            table.add_column(column)
        for row in WORKER_PERCENTAGE_USAGE:
            table.add_row(str(row["time"]), str(row["nr_current_workers"]), str(row["num_parallel_jobs"]), f'{row["percentage"]}%')
        console.print(table)

    return _exit

@beartype
def abandon_job(job: Job, trial_index: int) -> bool:
    if job:
        try:
            if ax_client:
                _trial = ax_client.get_trial(trial_index)
                _trial.mark_abandoned()
                print_debug(f"abandon_job: removing job {job}, trial_index: {trial_index}")
                global_vars["jobs"].remove((job, trial_index))
            else:
                print_red("ax_client could not be found")
                my_exit(9)
        except Exception as e:
            print(f"ERROR in line {get_line_info()}: {e}")
            print_debug(f"ERROR in line {get_line_info()}: {e}")
            return False
        job.cancel()
        return True

    return False

@beartype
def abandon_all_jobs() -> None:
    for job, trial_index in global_vars["jobs"][:]:
        abandoned = abandon_job(job, trial_index)
        if not abandoned:
            print_debug(f"Job {job} could not be abandoned.")

@beartype
def show_pareto_or_error_msg() -> None:
    if len(arg_result_names) > 1:
        try:
            show_pareto_frontier_data()
        except Exception as e:
            print_red(f"show_pareto_frontier_data() failed with exception {e}")
    else:
        print_debug(f"show_pareto_frontier_data will NOT be executed because len(arg_result_names) is {len(arg_result_names)}")

@beartype
def end_program(_force: Optional[bool] = False, exit_code: Optional[int] = None) -> None:
    global END_PROGRAM_RAN

    wait_for_jobs_to_complete()

    show_pareto_or_error_msg()

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

    save_results_csv()

    if exit_code:
        _exit = exit_code

    live_share()

    if succeeded_jobs() == 0 and failed_jobs() > 0:
        _exit = 89

    my_exit(_exit)

@beartype
def save_checkpoint(trial_nr: int = 0, eee: Union[None, str, Exception] = None) -> None:
    if trial_nr > 3:
        if eee:
            print(f"Error during saving checkpoint: {eee}")
        else:
            print("Error during saving checkpoint")
        return

    try:
        state_files_folder = f"{get_current_run_folder()}/state_files/"

        makedirs(state_files_folder)

        checkpoint_filepath = f'{state_files_folder}/checkpoint.json'
        if ax_client:
            ax_client.save_to_json_file(filepath=checkpoint_filepath)
        else:
            print_red("Something went wrong using the ax_client")
            my_exit(9)
    except Exception as e:
        save_checkpoint(trial_nr + 1, e)

@beartype
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

@beartype
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

@beartype
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


@beartype
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
            "digits": None,
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
    print_red(f"Unknown data range {data['type']}")
    my_exit(19)

    # only for linter, never reached because of die
    return {}

@beartype
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
            if torch.cuda.device_count() >= 1:
                torch_device = torch.cuda.current_device()
                gpu_string = f"Using CUDA device {torch.cuda.get_device_name(0)}."
                gpu_color = "green"
            else:
                gpu_string = "No CUDA devices found."
                gpu_color = "yellow"
    except ModuleNotFoundError:
        print_red("Cannot load torch and thus, cannot use gpus")

    if torch_device:
        if experiment_args:
            experiment_args["choose_generation_strategy_kwargs"]["torch_device"] = torch_device
        else:
            print_red("experiment_args could not be created.")
            my_exit(90)

    if experiment_args:
        return experiment_args, gpu_string, gpu_color

    return {}, gpu_string, gpu_color

@beartype
def die_with_47_if_file_doesnt_exists(_file: str) -> None:
    if not os.path.exists(_file):
        print_red(f"Cannot find {_file}")
        my_exit(47)

@beartype
def copy_state_files_from_previous_job(continue_previous_job: str) -> None:
    for state_file in ["submitted_jobs"]:
        old_state_file = f"{continue_previous_job}/state_files/{state_file}"
        new_state_file = f'{get_current_run_folder()}/state_files/{state_file}'
        die_with_47_if_file_doesnt_exists(old_state_file)

        if not os.path.exists(new_state_file):
            shutil.copy(old_state_file, new_state_file)

@beartype
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

@beartype
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

@beartype
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

@beartype
def set_parameter_constraints(experiment_constraints: Optional[list], experiment_args: dict, experiment_parameters: Union[dict, list]) -> dict:
    if experiment_constraints and len(experiment_constraints):
        experiment_constraints = experiment_constraints[0]

        experiment_args["parameter_constraints"] = []

        if experiment_constraints:
            for _l in range(len(experiment_constraints)):
                constraints_string = decode_if_base64(" ".join(experiment_constraints[_l]))

                constraints_string = constraints_string.rstrip("\n\r")

                variables = [item['name'] for item in experiment_parameters]

                equation = check_equation(variables, constraints_string)

                if equation:
                    experiment_args["parameter_constraints"].append(constraints_string)
                else:
                    print_red(f"Experiment constraint '{constraints_string}' is invalid. Cannot continue.")
                    my_exit(19)

                file_path = os.path.join(get_current_run_folder(), "state_files", "constraints")

                os.makedirs(os.path.dirname(file_path), exist_ok=True)

                with open(file_path, "a", encoding="utf-8") as f:
                    f.write(constraints_string + "\n")

    return experiment_args

@beartype
def replace_parameters_for_continued_jobs(parameter: Optional[list], cli_params_experiment_parameters: Optional[list], experiment_parameters: dict) -> dict:
    if parameter and cli_params_experiment_parameters:
        for _item in cli_params_experiment_parameters:
            _replaced = False
            for _item_id_to_overwrite in range(len(experiment_parameters["experiment"]["search_space"]["parameters"])):
                if _item["name"] == experiment_parameters["experiment"]["search_space"]["parameters"][_item_id_to_overwrite]["name"]:
                    old_param_json = json.dumps(experiment_parameters["experiment"]["search_space"]["parameters"][_item_id_to_overwrite])

                    experiment_parameters["experiment"]["search_space"]["parameters"][_item_id_to_overwrite] = get_ax_param_representation(_item)

                    new_param_json = json.dumps(experiment_parameters["experiment"]["search_space"]["parameters"][_item_id_to_overwrite])

                    _replaced = True

                    compared_params = compare_parameters(old_param_json, new_param_json)
                    if compared_params:
                        print_yellow(compared_params)

            if not _replaced:
                print_yellow(f"--parameter named {_item['name']} could not be replaced. It will be ignored, instead. You cannot change the number of parameters or their names when continuing a job, only update their values.")

    return experiment_parameters

@beartype
def load_experiment_parameters_from_checkpoint_file(checkpoint_file: str) -> dict:
    try:
        f = open(checkpoint_file, encoding="utf-8")
        experiment_parameters = json.load(f)
        f.close()

        with open(checkpoint_file, encoding="utf-8") as f:
            experiment_parameters = json.load(f)
    except json.decoder.JSONDecodeError:
        print_red(f"Error parsing checkpoint_file {checkpoint_file}")
        my_exit(47)

    return experiment_parameters

@beartype
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

@beartype
def get_experiment_parameters(_params: list) -> Tuple[AxClient, Union[list, dict], dict, str, str]:
    continue_previous_job, seed, experiment_constraints, parameter, cli_params_experiment_parameters, experiment_parameters = _params

    global ax_client

    gpu_string = ""
    gpu_color = "green"

    experiment_args = None

    if continue_previous_job:
        print_debug(f"Load from checkpoint: {continue_previous_job}")

        checkpoint_file: str = f"{continue_previous_job}/state_files/checkpoint.json"
        checkpoint_parameters_filepath: str = f"{continue_previous_job}/state_files/checkpoint.json.parameters.json"

        die_with_47_if_file_doesnt_exists(checkpoint_parameters_filepath)
        die_with_47_if_file_doesnt_exists(checkpoint_file)

        experiment_parameters = load_experiment_parameters_from_checkpoint_file(checkpoint_file)

        experiment_args, gpu_string, gpu_color = set_torch_device_to_experiment_args(experiment_args)

        copy_state_files_from_previous_job(continue_previous_job)

        replace_parameters_for_continued_jobs(parameter, cli_params_experiment_parameters, experiment_parameters)

        original_ax_client_file = f"{get_current_run_folder()}/state_files/original_ax_client_before_loading_tmp_one.json"

        if ax_client:
            ax_client.save_to_json_file(filepath=original_ax_client_file)

            with open(original_ax_client_file, encoding="utf-8") as f:
                loaded_original_ax_client_json = json.load(f)
                original_generation_strategy = loaded_original_ax_client_json["generation_strategy"]

                if original_generation_strategy:
                    experiment_parameters["generation_strategy"] = original_generation_strategy

            tmp_file_path = get_tmp_file_from_json(experiment_parameters)

            ax_client = AxClient.load_from_json_file(tmp_file_path)

            ax_client = cast(AxClient, ax_client)

            os.unlink(tmp_file_path)

            state_files_folder = f"{get_current_run_folder()}/state_files"

            checkpoint_filepath = f'{state_files_folder}/checkpoint.json'
            makedirs(state_files_folder)

            with open(checkpoint_filepath, mode="w", encoding="utf-8") as outfile:
                json.dump(experiment_parameters, outfile)

            if not os.path.exists(checkpoint_filepath):
                print_red(f"{checkpoint_filepath} not found. Cannot continue_previous_job without.")
                my_exit(47)

            with open(f'{get_current_run_folder()}/checkpoint_load_source', mode='w', encoding="utf-8") as f:
                print(f"Continuation from checkpoint {continue_previous_job}", file=f)

            copy_continue_uuid()
        else:
            print_red("Something went wrong with the ax_client")
            my_exit(9)

        if experiment_constraints:
            experiment_args = set_parameter_constraints(experiment_constraints, experiment_args, experiment_parameters["experiment"]["search_space"]["parameters"])
    else:
        objectives = set_objectives()

        experiment_args = {
            "name": global_vars["experiment_name"],
            "parameters": experiment_parameters,
            "objectives": objectives,
            "choose_generation_strategy_kwargs": {
                "num_trials": max_eval,
                "num_initialization_trials": num_parallel_jobs,
                #"use_batch_trials": True,
                "max_parallelism_override": -1,
                "random_seed": seed
            },
        }

        if seed:
            experiment_args["choose_generation_strategy_kwargs"]["random_seed"] = seed

        experiment_args, gpu_string, gpu_color = set_torch_device_to_experiment_args(experiment_args)

        experiment_args = set_parameter_constraints(experiment_constraints, experiment_args, experiment_parameters)

        try:
            if ax_client:
                ax_client.create_experiment(**experiment_args)

                new_metrics = [Metric(k) for k in arg_result_names if k not in ax_client.metric_names]
                ax_client.experiment.add_tracking_metrics(new_metrics)
            else:
                print_red("ax_client could not be found!")
                sys.exit(9)
        except AssertionError as error:
            print_red(f"An error has occurred while creating the experiment (0): {error}. This can happen when you have invalid parameter constraints.")
            my_exit(102)
        except ValueError as error:
            print_red(f"An error has occurred while creating the experiment (1): {error}")
            my_exit(49)
        except TypeError as error:
            print_red(f"An error has occurred while creating the experiment (2): {error}. This is probably a bug in OmniOpt2.")
            my_exit(49)
        except ax.exceptions.core.UserInputError as error:
            print_red(f"An error occurred while creating the experiment (3): {error}")
            my_exit(49)

    return ax_client, experiment_parameters, experiment_args, gpu_string, gpu_color

@beartype
def get_type_short(typename: str) -> str:
    if typename == "RangeParameter":
        return "range"

    if typename == "ChoiceParameter":
        return "choice"

    return typename

@beartype
def parse_single_experiment_parameter_table(experiment_parameters: Union[list, dict]) -> list:
    rows: list = []

    for param in experiment_parameters:
        _type = ""

        if "__type" in param:
            _type = param["__type"]
        else:
            _type = param["type"]

        if "range" in _type.lower():
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

            rows.append([str(param["name"]), get_type_short(_type), str(helpers.to_int_when_possible(_lower)), str(helpers.to_int_when_possible(_upper)), "", value_type, log_scale])
        elif "fixed" in _type.lower():
            rows.append([str(param["name"]), get_type_short(_type), "", "", str(helpers.to_int_when_possible(param["value"])), "", ""])
        elif "choice" in _type.lower():
            values = param["values"]
            values = [str(helpers.to_int_when_possible(item)) for item in values]

            rows.append([str(param["name"]), get_type_short(_type), "", "", ", ".join(values), "", ""])
        else:
            print_red(f"Type {_type} is not yet implemented in the overview table.")
            my_exit(15)

    return rows

@beartype
def print_parameter_constraints_table(experiment_args: dict) -> None:
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

    fn = f"{get_current_run_folder()}/constraints.txt"
    try:
        with open(fn, mode="w", encoding="utf-8") as text_file:
            text_file.write(table_str)
    except Exception as e:
        print_red(f"Error writing {fn}: {e}")

    return None

@beartype
def print_result_names_overview_table() -> None:
    if not ax_client:
        print_red("Tried to access ax_client in print_result_names_overview_table, but it failed, because the ax_client was not defined.")
        my_exit(101)

        return None

    if args.continue_previous_job is not None and args.result_names is not None:
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

    __table = Table(title="Result-Names:")

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

@beartype
def print_experiment_param_table_to_file(filtered_columns: list, filtered_data: list) -> None:
    table = Table(header_style="bold", title="Experiment parameters:")
    for column in filtered_columns:
        table.add_column(column)

    for row in filtered_data:
        table.add_row(*[str(cell) if cell is not None else "" for cell in row], style="bright_green")

    console.print(table)

    with console.capture() as capture:
        console.print(table)

    table_str = capture.get()

    fn = f"{get_current_run_folder()}/parameters.txt"

    try:
        with open(fn, mode="w", encoding="utf-8") as text_file:
            text_file.write(table_str)
    except FileNotFoundError as e:
        print_red(f"Error trying to write file {fn}: {e}")

@beartype
def print_experiment_parameters_table(experiment_parameters: Union[list, dict]) -> None:
    if not experiment_parameters:
        print_red("Cannot determine experiment_parameters. No parameter table will be shown.")
        return

    if not experiment_parameters:
        print_red("Experiment parameters could not be determined for display")
        return

    if isinstance(experiment_parameters, dict) and "_type" in experiment_parameters:
        experiment_parameters = experiment_parameters["experiment"]["search_space"]["parameters"]

    rows = parse_single_experiment_parameter_table(experiment_parameters)

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

@beartype
def print_overview_tables(experiment_parameters: Union[list, dict], experiment_args: dict) -> None:
    print_experiment_parameters_table(experiment_parameters)

    print_parameter_constraints_table(experiment_args)

    print_result_names_overview_table()

@beartype
def update_progress_bar(_progress_bar: Any, nr: int) -> None:
    #print(f"update_progress_bar(_progress_bar, {nr})")
    #traceback.print_stack()

    _progress_bar.update(nr)

@beartype
def get_current_model() -> str:
    if overwritten_to_random:
        return "Random*"

    if ax_client:
        gs_model = ax_client.generation_strategy.model

        if gs_model:
            return str(gs_model.model)

        gs_model = ax_client.generation_strategy.current_node_name

        if gs_model:
            return str(gs_model)

    return "initializing model"

@beartype
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

@beartype
def state_from_job(job: Union[str, Job]) -> str:
    job_string = f'{job}'
    match = re.search(r'state="([^"]+)"', job_string)

    state = None

    if match:
        state = match.group(1).lower()
    else:
        state = f"{state}"

    return state

@beartype
def get_workers_string() -> str:
    string = ""

    string_keys: list = []
    string_values: list = []

    stats: dict = {}

    for job, _ in global_vars["jobs"][:]:
        state = state_from_job(job)

        if state not in stats.keys():
            stats[state] = 0
        stats[state] += 1

    _sum = 0

    for key in stats.keys():
        if args.abbreviate_job_names:
            string_keys.append(key.lower()[0])
        else:
            string_keys.append(key.lower())
        string_values.append(str(stats[key]))

        _sum = _sum + int(stats[key])

    if len(string_keys) and len(string_values):
        _keys = "/".join(string_keys)
        _values = "/".join(string_values)

        if len(_keys):
            nr_current_workers = len(global_vars["jobs"])
            if args.generate_all_jobs_at_once:
                string = f"{_keys} {_values} = ∑{_sum}/{num_parallel_jobs}"
            else:
                percentage = round((nr_current_workers / num_parallel_jobs) * 100)
                _sum_and_percentage = ""
                if num_parallel_jobs > 1:
                    _sum_and_percentage = f"∑{_sum} ({percentage}%/{num_parallel_jobs})"
                string = f"{_keys} {_values}{_sum_and_percentage}"

    return string

@beartype
def submitted_jobs(nr: int = 0) -> int:
    state_files_folder = f"{get_current_run_folder()}/state_files/"
    makedirs(state_files_folder)
    return append_and_read(f'{get_current_run_folder()}/state_files/submitted_jobs', nr)

@beartype
def get_slurm_in_brackets(in_brackets: list) -> list:
    if is_slurm_job():
        nr_current_workers = len(global_vars["jobs"])
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

        workers_strings = get_workers_string()
        if workers_strings:
            in_brackets.append(workers_strings)

    return in_brackets

@beartype
def get_types_of_errors_string() -> str:
    types_of_errors_str = ""

    _types_of_errors: list = read_errors_from_file()

    if len(_types_of_errors) > 0:
        types_of_errors_str = f" ({', '.join(_types_of_errors)})"

    return types_of_errors_str

@beartype
def capitalized_string(s: str) -> str:
    return s[0].upper() + s[1:] if s else ""

@beartype
def get_desc_progress_text(new_msgs: List[str] = []) -> str:
    global progress_bar_length

    current_model = get_current_model()

    if current_model == "SobolGenerator":
        current_model = "Sobol"

    in_brackets = []
    in_brackets.append(current_model)
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

@beartype
def _get_desc_progress_text_failed_jobs() -> List[str]:
    if failed_jobs():
        return [f"{helpers.bcolors.red}failed: {failed_jobs()}{get_types_of_errors_string()}{helpers.bcolors.endc}"]
    return []

@beartype
def _get_desc_progress_text_best_params() -> List[str]:
    best_params_res = [
        get_best_params_str(res_name) for res_name in arg_result_names if get_best_params_str(res_name)
    ]

    if best_params_res:
        done_jobs = count_done_jobs()
        return ["best " + ", ".join(best_params_res)] if len(arg_result_names) == 1 else [f"{done_jobs} done"]

    return []

@beartype
def _get_desc_progress_text_submitted_jobs() -> List[str]:
    result = []
    if submitted_jobs():
        result.append(f"total submitted: {submitted_jobs()}")
        if max_eval:
            result.append(f"max_eval: {max_eval}")
    return result

@beartype
def _get_desc_progress_text_new_msgs(new_msgs: List[str]) -> List[str]:
    return [msg for msg in new_msgs if msg]

@beartype
def progressbar_description(new_msgs: List[str] = []) -> None:
    global last_progress_bar_desc

    desc = get_desc_progress_text(new_msgs)
    print_debug_progressbar(desc)

    if progress_bar is not None:
        if last_progress_bar_desc != desc:
            progress_bar.set_description_str(desc)
            progress_bar.refresh()

            last_progress_bar_desc = desc

    live_share()

@beartype
def clean_completed_jobs() -> None:
    job_states_to_be_removed = ["early_stopped", "abandoned", "cancelled", "timeout", "interrupted"]
    job_states_to_be_ignored = ["completed", "unknown", "pending", "running", "completing", "out_of_memory"]

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

@beartype
def simulate_load_data_from_existing_run_folders(_paths: List[str]) -> int:
    _counter: int = 0

    for this_path in _paths:
        this_path_json = f"{this_path}/state_files/ax_client.experiment.json"

        if not os.path.exists(this_path_json):
            print_red(f"{this_path_json} does not exist, cannot load data from it")
            return 0

        try:
            old_experiments = load_experiment(this_path_json)

            old_trials = old_experiments.trials

            for old_trial_index in old_trials:
                old_trial = old_trials[old_trial_index]
                trial_status = old_trial.status
                trial_status_str = trial_status.__repr__

                if "COMPLETED".lower() not in str(trial_status_str).lower():
                    # or "MANUAL".lower() in str(trial_status_str).lower()):
                    continue

                _counter += 1
        except ValueError as e:
            print_red(f"Error while simulating loading data: {e}")

    return _counter

@beartype
def get_nr_of_imported_jobs() -> int:
    nr_jobs: int = 0

    if args.continue_previous_job:
        nr_jobs += simulate_load_data_from_existing_run_folders([args.continue_previous_job])

    return nr_jobs

@beartype
def load_existing_job_data_into_ax_client() -> None:
    nr_of_imported_jobs = get_nr_of_imported_jobs()
    set_nr_inserted_jobs(NR_INSERTED_JOBS + nr_of_imported_jobs)

@beartype
def parse_parameter_type_error(_error_message: Union[str, None]) -> Optional[dict]:
    if not _error_message:
        return None

    error_message: str = str(_error_message)
    try:
        # Defining the regex pattern to match the required parts of the error message
        _pattern: str = r"Value for parameter (?P<parameter_name>\w+): .*? is of type <class '(?P<current_type>\w+)'>, expected\s*<class '(?P<expected_type>\w+)'>."
        match = re.search(_pattern, error_message)

        # Asserting the match is found
        assert match is not None, "Pattern did not match the error message."

        # Extracting values from the match object
        parameter_name = match.group("parameter_name")
        current_type = match.group("current_type")
        expected_type = match.group("expected_type")

        # Asserting the extracted values are correct
        assert parameter_name is not None, "Parameter name not found in the error message."
        assert current_type is not None, "Current type not found in the error message."
        assert expected_type is not None, "Expected type not found in the error message."

        # Returning the parsed values
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
                    if col in special_col_names:
                        continue

                    if col in arg_result_names:
                        results[col] = try_convert(value)
                    else:
                        arm_params[col] = try_convert(value)

                arm_params_list.append(arm_params)
                results_list.append(results)

    return arm_params_list, results_list

@beartype
def insert_jobs_from_csv(this_csv_file_path: str, experiment_parameters: Optional[Union[List[Any], dict]]) -> None:
    if not os.path.exists(this_csv_file_path):
        print_red(f"--load_data_from_existing_jobs: Cannot find {this_csv_file_path}")

        return

    def validate_and_convert_params(experiment_parameters: Optional[Union[List[Any], Dict[Any, Any]]], arm_params: Dict) -> Dict:
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

    arm_params_list, results_list = parse_csv(this_csv_file_path)

    cnt = 0

    err_msgs = []

    with console.status("[bold green]Loading existing jobs into ax_client...") as __status:
        for arm_params, result in zip(arm_params_list, results_list):
            __status.update(f"[bold green]Loading existing jobs from {this_csv_file_path} into ax_client")
            arm_params = validate_and_convert_params(experiment_parameters, arm_params)

            try:
                if insert_job_into_ax_client(arm_params, result):
                    cnt += 1

                    print_debug(f"Inserted one job from {this_csv_file_path}, arm_params: {arm_params}, results: {result}")
                else:
                    print_red(f"Failed to insert one job from {this_csv_file_path}, arm_params: {arm_params}, results: {result}")
            except ValueError as e:
                err_msg = f"Failed to insert job(s) from {this_csv_file_path} into ax_client. This can happen when the csv file has different parameters or results as the main job one's or other imported jobs. Error: {e}"
                if err_msg not in err_msgs:
                    print_red(err_msg)
                    err_msgs.append(err_msg)

    if cnt:
        if cnt == 1:
            print_yellow(f"Inserted one job from {this_csv_file_path}")
        else:
            print_yellow(f"Inserted {cnt} jobs from {this_csv_file_path}")

    set_max_eval(max_eval + cnt)
    set_nr_inserted_jobs(NR_INSERTED_JOBS + cnt)

@beartype
def insert_job_into_ax_client(arm_params: dict, result: dict) -> bool:
    done_converting = False

    if ax_client is None or not ax_client:
        print_red("insert_job_into_ax_client: ax_client was not defined where it should have been")
        my_exit(101)

    while not done_converting:
        try:
            if ax_client:
                new_trial = ax_client.attach_trial(arm_params)

                new_trial_idx = new_trial[1]

                ax_client.complete_trial(trial_index=new_trial_idx, raw_data=result)

                done_converting = True
                save_results_csv()

                return True

            print_red("Error getting ax_client")
            my_exit(9)

            return False
        except ax.exceptions.core.UnsupportedError as e:
            parsed_error = parse_parameter_type_error(e)

            if parsed_error is not None:
                error_expected_type = parsed_error["expected_type"]
                error_current_type = parsed_error["current_type"]
                error_param_name = parsed_error["parameter_name"]

                if error_expected_type == "int" and type(arm_params[error_param_name]).__name__ != "int":
                    print_yellow(f"converted parameter {error_param_name} type {error_current_type} to {error_expected_type}")
                    arm_params[error_param_name] = int(arm_params[error_param_name])
                elif error_expected_type == "float" and type(arm_params[error_param_name]).__name__ != "float":
                    print_yellow(f"converted parameter {error_param_name} type {error_current_type} to {error_expected_type}")
                    arm_params[error_param_name] = float(arm_params[error_param_name])
            else:
                print_red("Could not parse error while trying to insert_job_into_ax_client")

    return False

@beartype
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

@beartype
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

@beartype
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

@beartype
def get_base_errors() -> list:
    base_errors: list = [
        "Segmentation fault",
        "Illegal division by zero",
        "OOM",
        ["Killed", "Detected kill, maybe OOM or Signal?"]
    ]

    return base_errors

@beartype
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

@beartype
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

@beartype
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

@beartype
def get_python_errors() -> List[List[str]]:
    synerr: str = "Python syntax error detected. Check log file."

    return [
        ["ModuleNotFoundError", "Module not found"],
        ["ImportError", "Module not found"],
        ["SyntaxError", synerr],
        ["NameError", synerr],
        ["ValueError", synerr],
        ["TypeError", synerr],
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
        ["CUDNN_STATUS_NOT_INITIALIZED", "Cuda had a problem. Try to delete ~/.nv and try again."]
    ]

@beartype
def get_first_line_of_file_that_contains_string(stdout_path: str, s: str) -> str:
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

@beartype
def check_for_python_errors(stdout_path: str, file_as_string: str) -> List[str]:
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

@beartype
def get_errors_from_outfile(stdout_path: str) -> List[str]:
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

@beartype
def print_outfile_analyzed(stdout_path: str) -> None:
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
            with open(f'{get_current_run_folder()}/evaluation_errors.log', mode="a+", encoding="utf-8") as error_file:
                error_file.write(out_files_string)
        except Exception as e:
            print_debug(f"Error occurred while writing to evaluation_errors.log: {e}")

        print_red(out_files_string)

@beartype
def get_parameters_from_outfile(stdout_path: str) -> Union[None, dict, str]:
    try:
        with open(stdout_path, mode='r', encoding="utf-8") as file:
            for line in file:
                if line.lower().startswith("parameters: "):
                    params = line.split(":", 1)[1].strip()
                    params = json.loads(params)
                    return params
    except FileNotFoundError:
        original_print(f"get_parameters_from_outfile: The file '{stdout_path}' was not found.")
    except Exception as e:
        print(f"get_parameters_from_outfile: There was an error: {e}")

    return None

@beartype
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

@beartype
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

@beartype
def read_errors_from_file() -> list:
    error_file_path = f'{get_current_run_folder()}/result_errors.log'
    if os.path.exists(error_file_path):
        with open(error_file_path, mode='r', encoding="utf-8") as file:
            errors = file.readlines()
        # Entfernen des Zeilenumbruchs am Ende jeder Zeile
        return [error.strip() for error in errors]
    return []

@beartype
def mark_trial_as_failed(trial_index: int, _trial: Any) -> None:
    print_debug(f"Marking trial {_trial} as failed")
    try:
        if not ax_client:
            print_red("mark_trial_as_failed: ax_client is not defined")
            my_exit(101)

            return None

        ax_client.log_trial_failure(trial_index=trial_index)
        _trial.mark_failed(unsafe=True)
    except ValueError as e:
        print_debug(f"mark_trial_as_failed error: {e}")

    return None

@beartype
def finish_job_core(job: Any, trial_index: int, this_jobs_finished: int) -> int:
    result = job.result()

    print_debug(f"finish_job_core: trial-index: {trial_index}, job.result(): {result}, state: {state_from_job(job)}")

    raw_result = result
    result_keys = list(result.keys())
    result = result[result_keys[0]]
    this_jobs_finished += 1

    if ax_client:
        _trial = ax_client.get_trial(trial_index)

        possible_val_not_found_values = [VAL_IF_NOTHING_FOUND, -VAL_IF_NOTHING_FOUND, -99999999999999997168788049560464200849936328366177157906432, 99999999999999997168788049560464200849936328366177157906432]

        values_to_check = result if isinstance(result, list) else [result]

        if result is not None and all(r not in possible_val_not_found_values for r in values_to_check):
            print_debug(f"Completing trial: {trial_index} with result: {raw_result}...")
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
                    print_red(f"Error completing trial: {e}")
                    my_exit(234)

            #count_done_jobs(1)
            try:
                print_debug(f"Marking trial {_trial} as completed")
                _trial.mark_completed(unsafe=True)

                succeeded_jobs(1)

                progressbar_description([f"new result: {result}"])
                update_progress_bar(progress_bar, 1)
            except Exception as e:
                print(f"ERROR in line {get_line_info()}: {e}")
        else:
            print_debug(f"Counting job {job} as failed, because the result is {result}")
            if job:
                try:
                    progressbar_description(["job_failed"])
                    ax_client.log_trial_failure(trial_index=trial_index)
                    _trial.mark_failed(unsafe=True)
                except Exception as e:
                    print_red(f"\nERROR while trying to mark job as failure: {e}")
                job.cancel()
                orchestrate_job(job, trial_index)

            mark_trial_as_failed(trial_index, _trial)
            failed_jobs(1)
    else:
        print_red("ax_client could not be found or used")
        my_exit(9)

    print_debug(f"finish_job_core: removing job {job}, trial_index: {trial_index}")
    global_vars["jobs"].remove((job, trial_index))

    return this_jobs_finished

@beartype
def finish_previous_jobs(new_msgs: List[str]) -> None:
    global JOBS_FINISHED

    if not ax_client:
        print_red("ax_client failed")
        my_exit(101)

        return None

    this_jobs_finished = 0

    if len(global_vars["jobs"]) > 0:
        print_debug(f"jobs in finish_previous_jobs: {global_vars['jobs']}")

    for job, trial_index in global_vars["jobs"][:]:
        # Poll if any jobs completed
        # Local and debug jobs don't run until .result() is called.
        if job is None:
            print_debug(f"finish_previous_jobs: job {job} is None")
            continue

        #print_debug(f"finish_previous_jobs: single job {job}")

        if job.done() or type(job) in [LocalJob, DebugJob]:
            try:
                this_jobs_finished = finish_job_core(job, trial_index, this_jobs_finished)
            except (SignalINT, SignalUSR, SignalCONT) as e:
                print_red(f"Cancelled finish_job_core: {e}")
            except (FileNotFoundError, submitit.core.utils.UncompletedJobError, ax.exceptions.core.UserInputError) as error:
                if "None for metric" in str(error):
                    print_red(f"\n⚠ It seems like the program that was about to be run didn't have 'RESULT: <FLOAT>' in it's output string.\nError: {error}\nJob-result: {job.result()}")
                else:
                    print_red(f"\n⚠ {error}")

                if job:
                    try:
                        progressbar_description(["job_failed"])
                        _trial = ax_client.get_trial(trial_index)
                        ax_client.log_trial_failure(trial_index=trial_index)
                        mark_trial_as_failed(trial_index, _trial)
                    except Exception as e:
                        print(f"ERROR in line {get_line_info()}: {e}")
                    job.cancel()
                    orchestrate_job(job, trial_index)

                failed_jobs(1)
                this_jobs_finished += 1
                print_debug(f"finish_previous_jobs: removing job {job}, trial_index: {trial_index}")
                global_vars["jobs"].remove((job, trial_index))

            save_checkpoint()
        else:
            if not isinstance(job, SlurmJob):
                print_debug(f"finish_previous_jobs: job was neither done, nor LocalJob nor DebugJob, but {job}")

        save_results_csv()

    save_results_csv()

    if this_jobs_finished > 0:
        progressbar_description([*new_msgs, f"finished {this_jobs_finished} {'job' if this_jobs_finished == 1 else 'jobs'}"])

    JOBS_FINISHED += this_jobs_finished

    clean_completed_jobs()

    return None

@beartype
def get_alt_path_for_orchestrator(stdout_path: str) -> Optional[str]:
    alt_path = None
    if stdout_path.endswith(".err"):
        alt_path = stdout_path[:-4] + ".out"
    elif stdout_path.endswith(".out"):
        alt_path = stdout_path[:-4] + ".err"

    return alt_path

@beartype
def check_orchestrator(stdout_path: str, trial_index: int) -> Optional[list]:
    behavs: list = []

    if orchestrator and "errors" in orchestrator:
        try:
            stdout = Path(stdout_path).read_text("UTF-8")
        except FileNotFoundError:
            alt_path = get_alt_path_for_orchestrator(stdout_path)

            if alt_path and alt_path is not None and Path(alt_path).exists():
                stdout_path = alt_path
                try:
                    stdout = Path(stdout_path).read_text("UTF-8")
                except FileNotFoundError:
                    stdout = None
            else:
                stdout = None

            if stdout is None:
                orchestrate_todo_copy = ORCHESTRATE_TODO
                if stdout_path not in orchestrate_todo_copy.keys():
                    ORCHESTRATE_TODO[stdout_path] = trial_index
                    print_red(f"File not found: {stdout_path}, will try again later")
                else:
                    print_red(f"File not found: {stdout_path}, not trying again")
                return None

        for oc in orchestrator["errors"]:
            name = oc["name"]
            match_strings = oc["match_strings"]
            behavior = oc["behavior"]

            for match_string in match_strings:
                if match_string.lower() in stdout.lower():
                    if behavior not in behavs:
                        print_debug(f"Appending behavior {behavior}, orchestrator-error-name: {name}")
                        behavs.append(behavior)

    return behavs

@beartype
def orchestrate_job(job: Job, trial_index: int) -> None:
    stdout_path = str(job.paths.stdout.resolve())
    stderr_path = str(job.paths.stderr.resolve())

    stdout_path = stdout_path.replace('\n', ' ').replace('\r', '')
    stdout_path = stdout_path.rstrip('\r\n')
    stdout_path = stdout_path.rstrip('\n')
    stdout_path = stdout_path.rstrip('\r')
    stdout_path = stdout_path.rstrip(' ')

    stderr_path = stderr_path.replace('\n', ' ').replace('\r', '')
    stderr_path = stderr_path.rstrip('\r\n')
    stderr_path = stderr_path.rstrip('\n')
    stderr_path = stderr_path.rstrip('\r')
    stderr_path = stderr_path.rstrip(' ')

    print_outfile_analyzed(stdout_path)
    print_outfile_analyzed(stderr_path)

    _orchestrate(stdout_path, trial_index)
    _orchestrate(stderr_path, trial_index)

    orchestrate_todo_copy = ORCHESTRATE_TODO
    for todo_stdout_file in orchestrate_todo_copy.keys():
        old_behavs = check_orchestrator(todo_stdout_file, ORCHESTRATE_TODO[todo_stdout_file])
        if old_behavs is not None:
            del ORCHESTRATE_TODO[todo_stdout_file]

@beartype
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

@beartype
def orchestrator_start_trial(params_from_out_file: Union[dict, str], trial_index: int) -> None:
    if executor and ax_client:
        new_job = executor.submit(evaluate, params_from_out_file)
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
        print_red("executor or ax_client could not be found properly")
        my_exit(9)

@beartype
def handle_exclude_node(stdout_path: str, hostname_from_out_file: Union[None, str]) -> None:
    if hostname_from_out_file:
        if not is_already_in_defective_nodes(hostname_from_out_file):
            print_yellow(f"\nExcludeNode was triggered for node {hostname_from_out_file}")
            count_defective_nodes(None, hostname_from_out_file)
        else:
            print_yellow(f"\nExcludeNode was triggered for node {hostname_from_out_file}, but it was already in defective nodes and won't be added again")
    else:
        print_red(f"Cannot do ExcludeNode because the host could not be determined from {stdout_path}")

@beartype
def handle_restart(stdout_path: str, trial_index: int) -> None:
    params_from_out_file = get_parameters_from_outfile(stdout_path)
    if params_from_out_file:
        orchestrator_start_trial(params_from_out_file, trial_index)
    else:
        print(f"Could not determine parameters from outfile {stdout_path} for restarting job")

@beartype
def handle_restart_on_different_node(stdout_path: str, hostname_from_out_file: Union[None, str], trial_index: int) -> None:
    if hostname_from_out_file:
        if not is_already_in_defective_nodes(hostname_from_out_file):
            print_yellow(f"\nRestartOnDifferentNode was triggered for node {hostname_from_out_file}. Adding node to defective hosts list and restarting on another host.")
            count_defective_nodes(None, hostname_from_out_file)
        else:
            print_yellow(f"\nRestartOnDifferentNode was triggered for node {hostname_from_out_file}, but it was already in defective nodes. Job will only be resubmitted.")
        handle_restart(stdout_path, trial_index)
    else:
        print_red(f"Cannot do RestartOnDifferentNode because the host could not be determined from {stdout_path}")

@beartype
def handle_exclude_node_and_restart_all(stdout_path: str, hostname_from_out_file: Union[None, str]) -> None:
    if hostname_from_out_file:
        if not is_already_in_defective_nodes(hostname_from_out_file):
            # TODO: Implement ExcludeNodeAndRestartAll fully
            print_yellow(f"ExcludeNodeAndRestartAll not yet fully implemented. Adding {hostname_from_out_file} to unavailable hosts.")
            count_defective_nodes(None, hostname_from_out_file)
        else:
            print_yellow(f"ExcludeNodeAndRestartAll was triggered for node {hostname_from_out_file}, but it was already in defective nodes and won't be added again.")
    else:
        print_red(f"Cannot do ExcludeNodeAndRestartAll because the host could not be determined from {stdout_path}")

@beartype
def _orchestrate(stdout_path: str, trial_index: int) -> None:
    behavs = check_orchestrator(stdout_path, trial_index)

    if not behavs or behavs is None:
        return

    hostname_from_out_file = get_hostname_from_outfile(stdout_path)

    # Behavior handler mapping
    behavior_handlers = {
        "ExcludeNode": lambda: handle_exclude_node(stdout_path, hostname_from_out_file),
        "Restart": lambda: handle_restart(stdout_path, trial_index),
        "RestartOnDifferentNode": lambda: handle_restart_on_different_node(stdout_path, hostname_from_out_file, trial_index),
        "ExcludeNodeAndRestartAll": lambda: handle_exclude_node_and_restart_all(stdout_path, hostname_from_out_file)
    }

    for behav in behavs:
        handler = behavior_handlers.get(behav)
        if handler:
            handler()
        else:
            print_red(f"Orchestrator: {behav} not yet implemented!")
            my_exit(210)

@beartype
def write_continue_run_uuid_to_file() -> None:
    if args.continue_previous_job:
        continue_dir = args.continue_previous_job

        with open(f'{continue_dir}/state_files/run_uuid', mode='r', encoding='utf-8') as f:
            continue_from_uuid = f.readline()

            write_state_file("uuid_of_continued_run", str(continue_from_uuid))

@beartype
def save_state_files() -> None:
    with console.status("[bold green]Saving state files..."):
        write_state_file("joined_run_program", global_vars["joined_run_program"])
        write_state_file("experiment_name", global_vars["experiment_name"])
        write_state_file("mem_gb", str(global_vars["mem_gb"]))
        write_state_file("max_eval", str(max_eval))
        write_state_file("gpus", str(args.gpus))
        write_state_file("time", str(global_vars["_time"]))
        write_state_file("run.sh", "omniopt '" + " ".join(sys.argv[1:]) + "'")
        write_state_file("run_uuid", str(run_uuid))

        if args.follow:
            write_state_file("follow", "True")

        if args.main_process_gb:
            write_state_file("main_process_gb", str(args.main_process_gb))

@beartype
def submit_job(parameters: dict) -> Optional[Job]:
    try:
        if executor:
            new_job = executor.submit(evaluate, parameters)
            submitted_jobs(1)
            return new_job

        print_red("executor could not be found")
        my_exit(9)
    except Exception as e:
        print_debug(f"Error while trying to submit job: {e}")
        raise

    return None

@beartype
def execute_evaluation(_params: list) -> Optional[int]:
    print_debug(f"execute_evaluation({_params})")
    trial_index, parameters, trial_counter, next_nr_steps, phase = _params
    if ax_client:
        _trial = ax_client.get_trial(trial_index)

        # Helper function for trial stage marking with exception handling
        def mark_trial_stage(stage: str, error_msg: str) -> None:
            try:
                getattr(_trial, stage)()
            except Exception as e:
                print_debug(f"execute_evaluation({_params}): {error_msg} with error: {e}")

        mark_trial_stage("mark_staged", "Marking the trial as staged failed")

        new_job = None

        try:
            initialize_job_environment()
            new_job = submit_job(parameters)

            print_debug(f"execute_evaluation: appending job {new_job} to global_vars['jobs'], trial_index: {trial_index}")
            global_vars["jobs"].append((new_job, trial_index))

            if is_slurm_job() and not args.force_local_execution:
                _sleep(1)

            mark_trial_stage("mark_running", "Marking the trial as running failed")
            trial_counter += 1

            update_progress()
        except submitit.core.utils.FailedJobError as error:
            handle_failed_job(error, trial_index, new_job)
            trial_counter += 1
        except (SignalUSR, SignalINT, SignalCONT):
            handle_exit_signal()
        except Exception as e:
            handle_generic_error(e)

        add_to_phase_counter(phase, 1)
        return trial_counter

    print_red("Failed to get ax_client")
    my_exit(9)

    return None

@beartype
def initialize_job_environment() -> None:
    progressbar_description(["starting new job"])
    set_sbatch_environment()
    exclude_defective_nodes()

@beartype
def set_sbatch_environment() -> None:
    if args.reservation:
        os.environ['SBATCH_RESERVATION'] = args.reservation
    if args.account:
        os.environ['SBATCH_ACCOUNT'] = args.account

@beartype
def exclude_defective_nodes() -> None:
    excluded_string: str = ",".join(count_defective_nodes())
    if len(excluded_string) > 1:
        if executor:
            executor.update_parameters(exclude=excluded_string)
        else:
            print_red("executor could not be found")
            my_exit(9)

@beartype
def handle_failed_job(error: Union[None, Exception, str], trial_index: int, new_job: Job) -> None:
    if "QOSMinGRES" in str(error) and args.gpus == 0:
        print_red("\n⚠ It seems like, on the chosen partition, you need at least one GPU. Use --gpus=1 (or more) as parameter.")
    else:
        print_red(f"\n⚠ FAILED: {error}")

    try:
        cancel_failed_job(trial_index, new_job)
    except Exception as e:
        print_red(f"\n⚠ Cancelling failed job FAILED: {e}")

@beartype
def cancel_failed_job(trial_index: int, new_job: Job) -> None:
    print_debug("Trying to cancel job that failed")
    if new_job:
        try:
            if ax_client:
                ax_client.log_trial_failure(trial_index=trial_index)
            else:
                print_red("ax_client not defined")
                my_exit(101)
        except Exception as e:
            print(f"ERROR in line {get_line_info()}: {e}")
        new_job.cancel()

        print_debug(f"cancel_failed_job: removing job {new_job}, trial_index: {trial_index}")
        global_vars["jobs"].remove((new_job, trial_index))
        print_debug("Removed failed job")
        save_checkpoint()
        save_results_csv()
    else:
        print_debug("cancel_failed_job: new_job was undefined")

@beartype
def update_progress() -> None:
    progressbar_description(["started new job"])

@beartype
def handle_exit_signal() -> None:
    print_red("\n⚠ Detected signal. Will exit.")
    end_program(False, 1)

@beartype
def handle_generic_error(e: Union[Exception, str]) -> None:
    tb = traceback.format_exc()
    print(tb)
    print_red(f"\n⚠ Starting job failed with error: {e}")

@beartype
def succeeded_jobs(nr: int = 0) -> int:
    state_files_folder = f"{get_current_run_folder()}/state_files/"
    makedirs(state_files_folder)
    return append_and_read(f'{get_current_run_folder()}/state_files/succeeded_jobs', nr)

@beartype
def show_debug_table_for_break_run_search(_name: str, _max_eval: Optional[int], _progress_bar: Any, _ret: Any) -> None:
    table = Table(show_header=True, header_style="bold", title=f"break_run_search for {_name}")

    headers = ["Variable", "Value"]
    table.add_column(headers[0])
    table.add_column(headers[1])

    rows = [
        ("args.max_failed_jobs", args.max_failed_jobs),
        ("succeeded_jobs()", succeeded_jobs()),
        ("submitted_jobs()", submitted_jobs()),
        ("count_done_jobs()", count_done_jobs()),
        ("_max_eval", _max_eval),
        ("_progress_bar.total", _progress_bar.total),
        ("NR_INSERTED_JOBS", NR_INSERTED_JOBS),
        ("_ret", _ret)
    ]

    for row in rows:
        table.add_row(str(row[0]), str(row[1]))

    console.print(table)

@beartype
def break_run_search(_name: str, _max_eval: Optional[int], _progress_bar: Any) -> bool:
    _ret = False

    _counted_done_jobs = count_done_jobs()
    _submitted_jobs = submitted_jobs()
    _failed_jobs = failed_jobs()

    max_failed_jobs = max_eval

    if args.max_failed_jobs is not None and args.max_failed_jobs > 0:
        max_failed_jobs = args.max_failed_jobs

    conditions = [
        (lambda: (_counted_done_jobs - _failed_jobs) >= max_eval, f"1. _counted_done_jobs {_counted_done_jobs} - (_failed_jobs {_failed_jobs}) >= max_eval {max_eval}"),
        (lambda: (_submitted_jobs - _failed_jobs) >= _progress_bar.total + 1, f"2. _submitted_jobs {_submitted_jobs} - _failed_jobs {_failed_jobs} >= _progress_bar.total {_progress_bar.total} + 1"),
        (lambda: max_failed_jobs < _failed_jobs, f"3. max_failed_jobs {max_failed_jobs} < failed_jobs {_failed_jobs}"),
    ]

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
        show_debug_table_for_break_run_search(_name, _max_eval, _progress_bar, _ret)

    return _ret

@beartype
def _calculate_nr_of_jobs_to_get(simulated_jobs: int, currently_running_jobs: int) -> int:
    """Calculates the number of jobs to retrieve."""
    return min(
        max_eval + simulated_jobs - count_done_jobs(),
        max_eval + simulated_jobs - (submitted_jobs() - failed_jobs()),
        num_parallel_jobs - currently_running_jobs
    )

@beartype
def remove_extra_spaces(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    return re.sub(r'\s+', ' ', text).strip()

@beartype
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

        if eta > 0:
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

@disable_logs
@beartype
def _fetch_next_trials(nr_of_jobs_to_get: int, recursion: bool = False) -> Optional[Tuple[Dict[int, Any], bool]]:
    """Attempts to fetch the next trials using the ax_client."""

    global global_gs, overwritten_to_random, gotten_jobs
    if not ax_client:
        print_red("ax_client was not defined")
        my_exit(9)

    trials_dict: dict = {}

    trial_durations: List[float] = []

    try:
        for k in range(nr_of_jobs_to_get):
            progressbar_description([_get_trials_message(k + 1, nr_of_jobs_to_get, trial_durations)])

            start_time = time.time()

            print_debug(f"_fetch_next_trials: fetching trial {k + 1}/{nr_of_jobs_to_get}...")

            if ax_client is not None and ax_client.experiment is not None and global_gs is not None:
                trial_index = ax_client.experiment.num_trials

                generator_run = global_gs.gen(
                    experiment=ax_client.experiment,
                    n=1,
                    pending_observations=get_pending_observation_features(experiment=ax_client.experiment)
                )
                trial = ax_client.experiment.new_trial(generator_run)
                params = generator_run.arms[0].parameters

                trial.mark_running(no_runner_required=True)

                trials_dict[trial_index] = params
                print_debug(f"_fetch_next_trials: got trial {k + 1}/{nr_of_jobs_to_get} (trial_index: {trial_index} [gotten_jobs: {gotten_jobs}, k: {k}])")
                end_time = time.time()

                gotten_jobs = gotten_jobs + 1

                trial_durations.append(float(end_time - start_time))
            else:
                print_red("ax_client, ax_client.experiment or global_gs is not defined")
                my_exit(101)
        return trials_dict, False
    except np.linalg.LinAlgError as e:
        _handle_linalg_error(e)
        my_exit(242)
    except (ax.exceptions.core.SearchSpaceExhausted, ax.exceptions.generation_strategy.GenerationStrategyRepeatedPoints, ax.exceptions.generation_strategy.MaxParallelismReachedException) as e:
        if str(e) not in error_8_saved:
            if recursion is False and args.revert_to_random_when_seemingly_exhausted:
                print_yellow(f"\n⚠Error 8: {e} From now (done jobs: {count_done_jobs()}) on, random points will be generated.")
            else:
                print_red(f"\n⚠Error 8: {e}")

            error_8_saved.append(str(e))

        if recursion is False and args.revert_to_random_when_seemingly_exhausted:
            print_debug("The search space seems exhausted. Generating random points from here on.")

            global_gs = GenerationStrategy(
                    name="Random*",
                    nodes=[
                        GenerationNode(
                            node_name="Sobol",
                            model_specs=[ModelSpec(Models.SOBOL, model_gen_kwargs={"random_seed": args.seed})]
                        )
                    ]
            )

            overwritten_to_random = True

            print_debug(f"New global_gs: {global_gs}")

            return _fetch_next_trials(nr_of_jobs_to_get, True)

    return {}, True

@beartype
def _handle_linalg_error(error: Union[None, str, Exception]) -> None:
    """Handles the np.linalg.LinAlgError based on the model being used."""
    if args.model and args.model.upper() in ["THOMPSON", "EMPIRICAL_BAYES_THOMPSON"]:
        print_red(f"Error: {error}. This may happen because the THOMPSON model is used. Try another one.")
    else:
        print_red(f"Error: {error}")

@beartype
def _get_next_trials(nr_of_jobs_to_get: int) -> Tuple[Union[None, dict], bool]:
    finish_previous_jobs(["finishing jobs (_get_next_trials)"])

    if break_run_search("_get_next_trials", max_eval, progress_bar) or nr_of_jobs_to_get == 0:
        return {}, True

    try:
        trial_index_to_param, optimization_complete = _fetch_next_trials(nr_of_jobs_to_get)

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

@beartype
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

    with open(f"{get_current_run_folder()}/get_next_nr_steps_tables.txt", mode="a", encoding="utf-8") as text_file:
        text_file.write(table_str)

    return requested

@beartype
def select_model(model_arg: Any) -> ax.modelbridge.registry.Models:
    """Selects the model based on user input or defaults to BOTORCH_MODULAR."""
    available_models = list(Models.__members__.keys())
    chosen_model = Models.BOTORCH_MODULAR

    if model_arg:
        model_upper = str(model_arg).upper()
        if model_upper in available_models:
            chosen_model = Models.__members__[model_upper]
        else:
            print_red(f"⚠ Cannot use {model_arg}. Available models are: {', '.join(available_models)}. Using BOTORCH_MODULAR instead.")

        if model_arg.lower() != "factorial" and args.gridsearch:
            print_yellow("Gridsearch only really works when you chose the FACTORIAL model.")

    return chosen_model

@beartype
def get_matching_model_name(model_name: str) -> Optional[str]:
    if not isinstance(model_name, str):
        return None
    if not isinstance(SUPPORTED_MODELS, (list, set, tuple)):
        return None

    model_name_lower = model_name.lower()
    model_map = {m.lower(): m for m in SUPPORTED_MODELS}

    return model_map.get(model_name_lower, None)

@beartype
def parse_generation_strategy_string(gen_strat_str: str) -> Tuple[list, int]:
    gen_strat_list = []

    cleaned_string = re.sub(r"\s+", "", gen_strat_str)
    splitted_by_comma = cleaned_string.split(",")

    sum_nr = 0

    for s in splitted_by_comma:
        if "=" in s:
            if s.count("=") == 1:
                model_name, nr = s.split("=")
                matching_model = get_matching_model_name(model_name)

                if matching_model in ["RANDOMFOREST", "EXTERNAL_GENERATOR"]:
                    print_red(f"Model {matching_model} is not valid for custom generation strategy.")
                    my_exit(56)

                if matching_model:
                    gen_strat_list.append({matching_model: nr})
                    sum_nr += int(nr)
                else:
                    print(f"'{model_name}' not found in SUPPORTED_MODELS")
                    my_exit(123)
            else:
                print(f"There can only be one '=' in the gen_strat_str's element '{s}'")
                my_exit(123)
        else:
            print(f"'{s}' does not contain '='")
            my_exit(123)

    return gen_strat_list, sum_nr

@beartype
def print_generation_strategy(generation_strategy_array: list) -> None:
    table = Table(header_style="bold", title="Generation Strategy:")

    table.add_column("Generation Strategy")
    table.add_column("Number of Generations")

    for gs_element in generation_strategy_array:
        model_name, num_generations = next(iter(gs_element.items()))
        table.add_row(model_name, str(num_generations))

    console.print(table)

@beartype
def write_state_file(name: str, var: str) -> None:
    file_path = f"{get_current_run_folder()}/state_files/{name}"

    if os.path.isdir(file_path):
        print_red(f"{file_path} is a dir. Must be a file.")
        my_exit(246)

    makedirs(os.path.dirname(file_path))

    try:
        with open(file_path, mode="w", encoding="utf-8") as f:
            f.write(str(var))
    except Exception as e:
        print_red(f"Failed writing '{file_path}': {e}")

@beartype
def get_chosen_model() -> str:
    chosen_model = args.model

    if args.continue_previous_job and chosen_model is None:
        continue_model_file = f"{args.continue_previous_job}/state_files/model"

        found_model = False

        if os.path.exists(continue_model_file):
            chosen_model = open(continue_model_file, mode="r", encoding="utf-8").readline().strip()

            if chosen_model in uncontinueable_models:
                print_red(f"Models in {', '.join(uncontinueable_models)} cannot be continued.")
                my_exit(94)

            if chosen_model not in SUPPORTED_MODELS:
                print_red(f"Wrong model >{chosen_model}< in {continue_model_file}.")
            else:
                found_model = True
        else:
            print_red(f"Cannot find model under >{continue_model_file}<.")

        if not found_model:
            if args.model is not None:
                chosen_model = args.model
            else:
                chosen_model = "BOTORCH_MODULAR"
            print_red(f"Could not find model in previous job. Will use the default model '{chosen_model}'")

    if chosen_model is None:
        chosen_model = "BOTORCH_MODULAR"

    return chosen_model

@beartype
def continue_not_supported_on_custom_generation_strategy() -> None:
    if args.continue_previous_job:
        generation_strategy_file = f"{args.continue_previous_job}/state_files/custom_generation_strategy"

        if os.path.exists(generation_strategy_file):
            print_red("Trying to continue a job which was started with --generation_strategy. This is currently not possible.")
            my_exit(247)

@beartype
def get_step_name(model_name: str, nr: int) -> str:
    this_step_name = f"{model_name} for {nr} step"
    if nr != 1:
        this_step_name = f"{this_step_name}s"

    return this_step_name

@beartype
def join_with_comma_and_then(items: list) -> str:
    length = len(items)

    if length == 0:
        return ""

    if length == 1:
        return items[0]

    if length == 2:
        return f"{items[0]} and then {items[1]}"

    return ", ".join(items[:-1]) + " and then " + items[-1]

@beartype
def create_node(model_name: str, threshold: int, next_model_name: Optional[str]) -> Union[RandomForestGenerationNode, GenerationNode]:
    if model_name == "RANDOMFOREST":
        if len(arg_result_names) != 1:
            print_red("Currently, RANDOMFOREST does not support Multi-Objective-Optimization")
            my_exit(251)

        node = RandomForestGenerationNode(
            num_samples=threshold,
            regressor_options={
                "n_estimators": args.n_estimators_randomforest
            },
            seed=args.seed
        )

        return node

    if model_name == "PSEUDORANDOM":
        node = ExternalProgramGenerationNode(f"python3 {script_dir}/.random_generator.py", "PSEUDORANDOM")

        return node

    if model_name == "EXTERNAL_GENERATOR":
        if args.external_generator is None or args.external_generator == "":
            print_red("--external_generator is missing. Cannot create points for EXTERNAL_GENERATOR without it.")
            my_exit(204)

        node = ExternalProgramGenerationNode(args.external_generator)

        return node

    if next_model_name is not None:
        trans_crit = [
            MaxTrials(
                threshold=threshold,
                block_transition_if_unmet=True,
                transition_to=next_model_name,
                count_only_trials_with_data=True
            )
        ]
    else:
        # Handle the case where no transition is intended
        trans_crit = [
            MaxTrials(
                threshold=threshold,
                block_transition_if_unmet=True,
                transition_to=model_name,
                count_only_trials_with_data=True
            )
        ]

    selected_model = select_model(model_name)
    model_spec = [
        ModelSpec(selected_model, model_gen_kwargs={"random_seed": args.seed})
    ]

    res = GenerationNode(
        node_name=model_name,
        model_specs=model_spec,
        transition_criteria=trans_crit
    )

    return res

@beartype
def create_systematic_step(model: Any, _num_trials: int = -1, index: Optional[int] = None) -> GenerationStep:
    """Creates a generation step for Bayesian optimization."""
    step = GenerationStep(
        model=model,
        num_trials=_num_trials,
        max_parallelism=(1000 * max_eval + 1000),
        model_gen_kwargs={
            'enforce_num_arms': False
        },
        should_deduplicate=True,
        index=index
    )

    return step

@beartype
def set_global_generation_strategy() -> None:
    global global_gs, random_steps, generation_strategy_human_readable

    args_generation_strategy = args.generation_strategy

    continue_not_supported_on_custom_generation_strategy()

    gs_names: list = []
    gs_nodes: list = []

    if args_generation_strategy is None:
        num_imported_jobs: int = get_nr_of_imported_jobs()
        set_max_eval(max_eval + num_imported_jobs)
        random_steps = random_steps or 0

        if max_eval is None:
            set_max_eval(max(1, random_steps))

        chosen_model = get_chosen_model()

        if chosen_model == "SOBOL":
            random_steps = max_eval

        if random_steps >= 1:
            next_node_name = None
            if max_eval - random_steps and chosen_model:
                next_node_name = chosen_model

            gs_names.append(get_step_name("SOBOL", random_steps))
            gs_nodes.append(create_node("SOBOL", random_steps, next_node_name))

        write_state_file("model", str(chosen_model))

        if chosen_model != "SOBOL" and max_eval > random_steps:
            this_node = create_node(chosen_model, max_eval - random_steps, None)

            gs_names.append(get_step_name(chosen_model, max_eval - random_steps))
            gs_nodes.append(this_node)

        generation_strategy_human_readable = join_with_comma_and_then(gs_names)

        try:
            global_gs = GenerationStrategy(
                name="+".join(gs_names),
                nodes=gs_nodes
            )
        except ax.exceptions.generation_strategy.GenerationStrategyMisconfiguredException as e:
            print_red(f"Error: {e}\ngs_names: {gs_names}\ngs_nodes: {gs_nodes}")

            my_exit(55)
    else:
        generation_strategy_array, new_max_eval = parse_generation_strategy_string(args_generation_strategy)

        new_max_eval_plus_inserted_jobs = new_max_eval + get_nr_of_imported_jobs()

        if max_eval < new_max_eval_plus_inserted_jobs:
            print_yellow(f"--generation_strategy {args_generation_strategy.upper()} has, in sum, more tasks than --max_eval {max_eval}. max_eval will be set to {new_max_eval_plus_inserted_jobs}.")
            set_max_eval(new_max_eval_plus_inserted_jobs)

        print_generation_strategy(generation_strategy_array)

        start_index = int(len(generation_strategy_array) / 2)

        steps: list = []

        for gs_element in generation_strategy_array:
            model_name = list(gs_element.keys())[0]

            nr = int(gs_element[model_name])

            gs_elem = create_systematic_step(select_model(model_name), nr, start_index)
            steps.append(gs_elem)

            gs_names.append(get_step_name(model_name, nr))

            start_index = start_index + 1

        write_state_file("custom_generation_strategy", args_generation_strategy)

        global_gs = GenerationStrategy(steps=steps)

        generation_strategy_human_readable = join_with_comma_and_then(gs_names)

@beartype
def wait_for_jobs_or_break(_max_eval: Optional[int], _progress_bar: Any) -> bool:
    while len(global_vars["jobs"]) > num_parallel_jobs:
        finish_previous_jobs([f"finishing previous jobs ({len(global_vars['jobs'])})"])

        if break_run_search("create_and_execute_next_runs", _max_eval, _progress_bar):
            return True

        if is_slurm_job() and not args.force_local_execution:
            _sleep(5)

    if break_run_search("create_and_execute_next_runs", _max_eval, _progress_bar):
        return True

    if _max_eval is not None and (JOBS_FINISHED - NR_INSERTED_JOBS) >= _max_eval:
        return True

    return False

@beartype
def handle_optimization_completion(optimization_complete: bool) -> bool:
    if optimization_complete:
        return True
    return False

@beartype
def execute_trials(trial_index_to_param: dict, next_nr_steps: int, phase: Optional[str], _max_eval: Optional[int], _progress_bar: Any) -> list:
    results = []
    i = 1
    for trial_index, parameters in trial_index_to_param.items():
        if wait_for_jobs_or_break(_max_eval, _progress_bar):
            break
        if break_run_search("create_and_execute_next_runs", _max_eval, _progress_bar):
            break
        progressbar_description(["eval start"])
        _args = [trial_index, parameters, i, next_nr_steps, phase]
        results.append(execute_evaluation(_args))
        i += 1
    return results

@beartype
def handle_exceptions_create_and_execute_next_runs(e: Exception) -> int:
    if isinstance(e, TypeError):
        print_red(f"Error 1: {e}")
    elif isinstance(e, botorch.exceptions.errors.InputDataError):
        print_red(f"Error 2: {e}")
    elif isinstance(e, ax.exceptions.core.DataRequiredError):
        if "transform requires non-empty data" in str(e) and args.num_random_steps == 0:
            print_red(f"Error 3: {e} Increase --num_random_steps to at least 1 to continue.")
            my_exit(233)
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

@beartype
def create_and_execute_next_runs(next_nr_steps: int, phase: Optional[str], _max_eval: Optional[int], _progress_bar: Any) -> int:
    if next_nr_steps == 0:
        print_debug(f"Warning: create_and_execute_next_runs(next_nr_steps: {next_nr_steps}, phase: {phase}, _max_eval: {_max_eval}, progress_bar)")
        return 0

    trial_index_to_param = None
    done_optimizing = False

    try:
        nr_of_jobs_to_get = _calculate_nr_of_jobs_to_get(get_nr_of_imported_jobs(), len(global_vars["jobs"]))
        results = []

        new_nr_of_jobs_to_get = min(max_eval - (submitted_jobs() - failed_jobs()), nr_of_jobs_to_get)

        range_nr = new_nr_of_jobs_to_get
        get_next_trials_nr = 1

        if args.generate_all_jobs_at_once:
            range_nr = 1
            get_next_trials_nr = new_nr_of_jobs_to_get

        for _ in range(range_nr):
            trial_index_to_param, optimization_complete = _get_next_trials(get_next_trials_nr)
            done_optimizing = handle_optimization_completion(optimization_complete)
            if done_optimizing:
                continue
            if trial_index_to_param:
                results.extend(execute_trials(trial_index_to_param, next_nr_steps, phase, _max_eval, _progress_bar))

        finish_previous_jobs(["finishing jobs"])

        if done_optimizing:
            end_program(False, 0)
    except Exception as e:
        stacktrace = traceback.format_exc()
        print_debug(f"Warning: create_and_execute_next_runs encountered an exception: {e}\n{stacktrace}")
        return handle_exceptions_create_and_execute_next_runs(e)

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

@beartype
def get_number_of_steps(_max_eval: int) -> Tuple[int, int]:
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

    original_second_steps = _max_eval - _random_steps
    second_step_steps = max(0, original_second_steps)
    if second_step_steps != original_second_steps:
        original_print(f"? original_second_steps: {original_second_steps} = max_eval {_max_eval} - _random_steps {_random_steps}")
    if second_step_steps == 0:
        print_yellow("This is basically a random search. Increase --max_eval or reduce --num_random_steps")

    second_step_steps = second_step_steps - already_done_random_steps

    if args.continue_previous_job:
        second_step_steps = _max_eval

    return _random_steps, second_step_steps

@beartype
def _set_global_executor() -> None:
    global executor

    log_folder: str = f'{get_current_run_folder()}/single_runs/%j'

    if args.force_local_execution:
        executor = LocalExecutor(folder=log_folder)
    else:
        executor = AutoExecutor(folder=log_folder)

    if executor:
        executor.update_parameters(
            name=f'{global_vars["experiment_name"]}_{run_uuid}_{str(uuid.uuid4())}',
            timeout_min=args.worker_timeout,
            slurm_gres=f"gpu:{args.gpus}",
            cpus_per_task=args.cpus_per_task,
            nodes=args.nodes_per_job,
            stderr_to_stdout=True,
            mem_gb=args.mem_gb,
            slurm_signal_delay_s=args.slurm_signal_delay_s,
            slurm_use_srun=args.slurm_use_srun,
            exclude=args.exclude
        )

        print_debug(f"""
executor.update_parameters(
    "name"="{f'{global_vars["experiment_name"]}_{run_uuid}_{str(uuid.uuid4())}'}",
    "timeout_min"={args.worker_timeout},
    "slurm_gres"={f"gpu:{args.gpus}"},
    "cpus_per_task"={args.cpus_per_task},
    "nodes"={args.nodes_per_job},
    "stderr_to_stdout"=True,
    "mem_gb"={args.mem_gb},
    "slurm_signal_delay_s"={args.slurm_signal_delay_s},
    "slurm_use_srun"={args.slurm_use_srun},
    "exclude"={args.exclude}
)
"""
        )

        if args.exclude:
            print_yellow(f"Excluding the following nodes: {args.exclude}")
    else:
        print_red("executor could not be found")
        my_exit(9)

@beartype
def set_global_executor() -> None:
    try:
        _set_global_executor()
    except ModuleNotFoundError as e:
        print_red(f"_set_global_executor() failed with error {e}. It may help if you can delete and re-install the virtual Environment containing the OmniOpt2 modules.")
        sys.exit(244)
    except (IsADirectoryError, PermissionError, FileNotFoundError) as e:
        print_red(f"Error trying to set_global_executor: {e}")

@beartype
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

@beartype
def start_nvidia_smi_thread() -> None:
    if IS_NVIDIA_SMI_SYSTEM:
        nvidia_smi_thread = threading.Thread(target=execute_nvidia_smi, daemon=True)
        nvidia_smi_thread.start()

@beartype
def run_search(_progress_bar: Any) -> bool:
    global NR_OF_0_RESULTS
    NR_OF_0_RESULTS = 0

    log_what_needs_to_be_logged()
    write_process_info()

    while (submitted_jobs() - failed_jobs()) <= max_eval:
        log_what_needs_to_be_logged()
        wait_for_jobs_to_complete()
        finish_previous_jobs([])

        if should_break_search(_progress_bar):
            break

        next_nr_steps: int = get_next_nr_steps(num_parallel_jobs, max_eval)

        nr_of_items = execute_next_steps(next_nr_steps, _progress_bar)

        log_worker_status(nr_of_items, next_nr_steps)

        finish_previous_jobs([f"finishing previous jobs ({len(global_vars['jobs'])})"])

        handle_slurm_execution()

        if check_search_space_exhaustion(nr_of_items):
            wait_for_jobs_to_complete()
            raise SearchSpaceExhausted("Search space exhausted")

        log_what_needs_to_be_logged()

    finalize_jobs()
    log_what_needs_to_be_logged()

    return False

@beartype
def should_break_search(_progress_bar: Any) -> bool:
    return (break_run_search("run_search", max_eval, _progress_bar) or (JOBS_FINISHED - NR_INSERTED_JOBS) >= max_eval)

@beartype
def execute_next_steps(next_nr_steps: int, _progress_bar: Any) -> int:
    if next_nr_steps:
        print_debug(f"trying to get {next_nr_steps} next steps (current done: {count_done_jobs()}, max: {max_eval})")
        nr_of_items = create_and_execute_next_runs(next_nr_steps, "systematic", max_eval, _progress_bar)
        return nr_of_items
    return 0

@beartype
def log_worker_status(nr_of_items: int, next_nr_steps: int) -> None:
    _debug_worker_creation(f"{int(time.time())}, {len(global_vars['jobs'])}, {nr_of_items}, {next_nr_steps}")

@beartype
def handle_slurm_execution() -> None:
    if is_slurm_job() and not args.force_local_execution:
        _sleep(1)

@beartype
def check_search_space_exhaustion(nr_of_items: int) -> bool:
    global NR_OF_0_RESULTS

    if nr_of_items == 0 and len(global_vars["jobs"]) == 0:
        NR_OF_0_RESULTS += 1
        _wrn = f"found {NR_OF_0_RESULTS} zero-jobs (max: {args.max_nr_of_zero_results})"
        progressbar_description([_wrn])
        print_debug(_wrn)
    else:
        NR_OF_0_RESULTS = 0

    if NR_OF_0_RESULTS >= args.max_nr_of_zero_results:
        _wrn = f"{NR_OF_0_RESULTS} empty jobs (>= {args.max_nr_of_zero_results})"
        print_debug(_wrn)
        progressbar_description([_wrn])

        live_share()

        return True

    return False

@beartype
def finalize_jobs() -> None:
    while len(global_vars["jobs"]):
        wait_for_jobs_to_complete()

        jobs_left = len(global_vars['jobs'])

        if jobs_left == 1:
            finish_previous_jobs([f"waiting for {jobs_left} job)"])
        else:
            finish_previous_jobs([f"waiting for {jobs_left} jobs)"])

        handle_slurm_execution()

@beartype
def go_through_jobs_that_are_not_completed_yet() -> None:
    print_debug(f"Waiting for jobs to finish (currently, len(global_vars['jobs']) = {len(global_vars['jobs'])}")

    nr_jobs_left = len(global_vars['jobs'])
    if nr_jobs_left == 1:
        progressbar_description([f"waiting for {nr_jobs_left} job"])
    else:
        progressbar_description([f"waiting for {nr_jobs_left} jobs"])

    if is_slurm_job() and not args.force_local_execution:
        _sleep(5)

    jobs_left = len(global_vars['jobs'])

    if jobs_left == 1:
        finish_previous_jobs([f"waiting for {jobs_left} job)"])
    else:
        finish_previous_jobs([f"waiting for {jobs_left} jobs"])

    clean_completed_jobs()

@beartype
def wait_for_jobs_to_complete() -> None:
    while len(global_vars["jobs"]):
        go_through_jobs_that_are_not_completed_yet()

@beartype
def die_orchestrator_exit_code_206(_test: bool) -> None:
    if _test:
        print_yellow("Not exiting, because _test was True")
    else:
        my_exit(206)

@beartype
def parse_orchestrator_file(_f: str, _test: bool = False) -> Union[dict, None]:
    if os.path.exists(_f):
        with open(_f, mode='r', encoding="utf-8") as file:
            try:
                data = yaml.safe_load(file)

                if "errors" not in data:
                    print_red(f"{_f} file does not contain key 'errors'")
                    die_orchestrator_exit_code_206(_test)

                valid_keys: list = ['name', 'match_strings', 'behavior']
                valid_behaviours: list = ["ExcludeNodeAndRestartAll", "RestartOnDifferentNode", "ExcludeNode", "Restart"]

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

@beartype
def set_orchestrator() -> None:
    with console.status("[bold green]Setting orchestrator..."):
        global orchestrator

        if args.orchestrator_file:
            if SYSTEM_HAS_SBATCH:
                orchestrator = parse_orchestrator_file(args.orchestrator_file, False)
            else:
                print_yellow("--orchestrator_file will be ignored on non-sbatch-systems.")

@beartype
def check_if_has_random_steps() -> None:
    with console.status("[bold green]Checking if has random steps..."):
        if (not args.continue_previous_job and "--continue" not in sys.argv) and (args.num_random_steps == 0 or not args.num_random_steps) and args.model not in ["EXTERNAL_GENERATOR", "SOBOL", "PSEUDORANDOM"]:
            print_red("You have no random steps set. This is only allowed in continued jobs. To start, you need either some random steps, or a continued run.")
            my_exit(233)

@beartype
def add_exclude_to_defective_nodes() -> None:
    if args.exclude:
        entries = [entry.strip() for entry in args.exclude.split(',')]

        for entry in entries:
            count_defective_nodes(None, entry)

@beartype
def check_max_eval(_max_eval: int) -> None:
    if not _max_eval:
        print_red("--max_eval needs to be set!")
        my_exit(19)

@beartype
def parse_parameters() -> Union[Tuple[Union[Any, None], Union[Any, None]], Tuple[Union[Any, None], Union[Any, None]]]:
    with console.status("[bold green]Parsing parameters..."):
        experiment_parameters = None
        cli_params_experiment_parameters = None
        if args.parameter:
            experiment_parameters = parse_experiment_parameters()
            cli_params_experiment_parameters = experiment_parameters
        return experiment_parameters, cli_params_experiment_parameters

@beartype
def get_csv_data(csv_path: str) -> Tuple[Union[Sequence[str], None], List[Dict[Union[str, Any], Union[str, Any]]]]:
    with open(csv_path, encoding="utf-8", mode="r") as file:
        reader = csv.DictReader(file)
        all_columns = reader.fieldnames
        rows = list(reader)
    return all_columns, rows

@beartype
def extract_parameters_and_metrics(rows: List, all_columns: Optional[Sequence[str]], metrics: List) -> Tuple[List, dict, List]:
    if all_columns is None:
        return [], {}, []

    param_names = [col for col in all_columns if col not in metrics and col not in IGNORABLE_COLUMNS]
    metrics = [col for col in all_columns if col in arg_result_names]

    param_dicts = []
    means: dict = {metric: [] for metric in metrics}

    for row in rows:
        param_dict = {param: row[param] for param in param_names}
        for metric in metrics:
            if row[metric] != "":
                means[metric].append(float(row[metric]))
        param_dicts.append(param_dict)

    return param_dicts, means, metrics

@beartype
def create_table(param_dicts: List, means: dict, metrics: List, metric_i: str, metric_j: str) -> Table:
    table = Table(title=f"Pareto-Front for {metric_j}/{metric_i}:", show_lines=True)

    headers = list(param_dicts[0].keys()) + metrics
    for header in headers:
        table.add_column(header, justify="center")

    for i, params in enumerate(param_dicts):
        this_table_row = [str(params[k]) for k in params.keys()]
        for metric in metrics:
            try:
                mean = means[metric][i]
                this_table_row.append(f"{mean:.3f}")
            except IndexError:
                this_table_row.append("")

        table.add_row(*this_table_row, style="bold green")

    return table

@beartype
def pareto_front_as_rich_table(param_dicts: list, metrics: list, metric_i: str, metric_j: str) -> Optional[Table]:
    if not os.path.exists(RESULT_CSV_FILE):
        print_debug(f"pareto_front_as_rich_table: File '{RESULT_CSV_FILE}' not found")
        return None

    all_columns, rows = get_csv_data(RESULT_CSV_FILE)
    param_dicts, means, metrics = extract_parameters_and_metrics(rows, all_columns, metrics)
    return create_table(param_dicts, means, metrics, metric_i, metric_j)

@beartype
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

@beartype
def plot_pareto_frontier_sixel(data: Any, i: int, j: int) -> None:
    absolute_metrics = data.absolute_metrics

    x_metric = absolute_metrics[i]
    y_metric = absolute_metrics[j]

    if not supports_sixel():
        console.print(f"[italic yellow]Your console does not support sixel-images. Will not print pareto-frontier as a matplotlib-sixel-plot for {x_metric}/{y_metric}.[/]")
        return

    import matplotlib.pyplot as plt

    means = data.means

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

@beartype
def convert_to_serializable(obj: np.ndarray) -> Union[str, list]:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

@beartype
def show_pareto_frontier_data() -> None:
    if len(arg_result_names) <= 1:
        print_debug(f"--result_names (has {len(arg_result_names)} entries) must be at least 2.")
        return

    if ax_client is None:
        print_red("show_pareto_frontier_data: Cannot plot pareto-front. ax_client is undefined.")
        return

    objectives = ax_client.experiment.optimization_config.objective.objectives
    pareto_front_data = {}
    all_combinations = list(combinations(range(len(objectives)), 2))
    collected_data = []

    with Progress(
        "[progress.description]{task.description}",
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
        transient=True
    ) as progress:
        task = progress.add_task("Calculating Pareto-Front...", total=len(all_combinations))

        skip = False

        for i, j in all_combinations:
            if not skip:
                metric_i = objectives[i].metric
                metric_j = objectives[j].metric

                try:
                    calculated_frontier = compute_posterior_pareto_frontier(
                        experiment=ax_client.experiment,
                        data=ax_client.experiment.fetch_data(),
                        primary_objective=metric_i,
                        secondary_objective=metric_j,
                        absolute_metrics=arg_result_names,
                        num_points=count_done_jobs()
                    )

                    collected_data.append((i, j, metric_i, metric_j, calculated_frontier))
                except ax.exceptions.core.DataRequiredError as e:
                    print_red(f"Error computing Pareto frontier for {metric_i.name} and {metric_j.name}: {e}")
                except SignalINT:
                    print_red("Calculating pareto-fronts was cancelled by pressing CTRL-c")
                    skip = True

            progress.update(task, advance=1)

    for i, j, metric_i, metric_j, calculated_frontier in collected_data:
        plot_pareto_frontier_sixel(calculated_frontier, i, j)

        if metric_i.name not in pareto_front_data:
            pareto_front_data[metric_i.name] = {}

        pareto_front_data[metric_i.name][metric_j.name] = {
            "param_dicts": calculated_frontier.param_dicts,
            "means": calculated_frontier.means,
            "sems": calculated_frontier.sems,
            "absolute_metrics": calculated_frontier.absolute_metrics
        }

        rich_table = pareto_front_as_rich_table(
            calculated_frontier.param_dicts,
            calculated_frontier.absolute_metrics,
            metric_j.name,
            metric_i.name
        )

        if rich_table is not None:
            console.print(rich_table)

            with open(f"{get_current_run_folder()}/pareto_front_table.txt", mode="a", encoding="utf-8") as text_file:
                with console.capture() as capture:
                    console.print(rich_table)
                text_file.write(capture.get())

    with open(f"{get_current_run_folder()}/pareto_front_data.json", mode="w", encoding="utf-8") as pareto_front_json_handle:
        json.dump(pareto_front_data, pareto_front_json_handle, default=convert_to_serializable)

@beartype
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

@beartype
def write_args_overview_table() -> None:
    table = Table(title="Arguments Overview:")
    table.add_column("Key", justify="left", style="bold")
    table.add_column("Value", justify="left", style="dim")

    for key, value in vars(args).items():
        table.add_row(key, str(value))

    table_str = ""

    with console.capture() as capture:
        console.print(table)

    table_str = capture.get()

    with open(f"{get_current_run_folder()}/args_overview.txt", mode="w", encoding="utf-8") as text_file:
        text_file.write(table_str)

@beartype
def show_experiment_overview_table() -> None:
    table = Table(title="Experiment overview:", show_header=True)

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

    with open(f"{get_current_run_folder()}/experiment_overview.txt", mode="w", encoding="utf-8") as text_file:
        text_file.write(table_str)

@beartype
def write_files_and_show_overviews() -> None:
    write_state_file("num_random_steps", str(args.num_random_steps))
    set_global_executor()
    load_existing_job_data_into_ax_client()
    write_args_overview_table()
    show_experiment_overview_table()
    save_global_vars()
    write_process_info()
    write_continue_run_uuid_to_file()

@beartype
def write_git_version() -> None:
    folder = f"{get_current_run_folder()}/"
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

@beartype
def write_live_share_file_if_needed() -> None:
    if args.live_share:
        write_state_file("live_share", "1\n")

@beartype
def write_revert_to_random_when_seemingly_exhausted_file(_path: str) -> None:
    file_path = f"{_path}/state_files/revert_to_random_when_seemingly_exhausted"

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, mode="w", encoding="utf-8") as f:
            f.write("1\n")
    except Exception as e:
        print_red(f"Error writing to file: {e}")

@beartype
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

@beartype
def get_constraints() -> list:
    constraints_list = []

    if args.experiment_constraints:
        constraints_list = args.experiment_constraints
    elif args.continue_previous_job and not args.disable_previous_job_constraint and (args.experiment_constraints is None or not len(args.experiment_constraints)):
        prev_job_constraint_file = os.path.join(args.continue_previous_job, "state_files", "constraints")
        if os.path.exists(prev_job_constraint_file):
            if os.path.exists(prev_job_constraint_file):
                with open(prev_job_constraint_file, "r", encoding="utf-8") as f:
                    constraints_list = [line.strip() for line in f if line.strip()]

                    constraints_list = [
                        base64.b64encode(constraint.encode("utf-8")).decode("utf-8")
                        for constraint in constraints_list
                    ]

                    constraints_list = [constraints_list]

    return constraints_list

@beartype
def main() -> None:
    global RESULT_CSV_FILE, ax_client, LOGFILE_DEBUG_GET_NEXT_TRIALS, random_steps

    check_if_has_random_steps()

    log_worker_creation()

    original_print(oo_call + " " + " ".join(sys.argv[1:]))
    check_slurm_job_id()

    debug_vars_unused_by_python_for_linter()

    if args.continue_previous_job and not args.num_random_steps:
        num_random_steps_file = f"{args.continue_previous_job}/state_files/num_random_steps"

        if os.path.exists(num_random_steps_file):
            args.num_random_steps = int(open(num_random_steps_file, mode="r", encoding="utf-8").readline().strip())
        else:
            print_red(f"Cannot find >{num_random_steps_file}<. Will use default, it being >{args.num_random_steps}<.")

    if args.continue_previous_job:
        if os.path.exists(f"{args.continue_previous_job}/state_files/revert_to_random_when_seemingly_exhausted"):
            args.revert_to_random_when_seemingly_exhausted = True

    set_run_folder()

    RESULT_CSV_FILE = create_folder_and_file(get_current_run_folder())

    write_revert_to_random_when_seemingly_exhausted_file(get_current_run_folder())

    try:
        fn = f"{get_current_run_folder()}/result_names.txt"
        with open(fn, mode="a", encoding="utf-8") as myfile:
            for rarg in arg_result_names:
                original_print(rarg, file=myfile)
    except Exception as e:
        print_red(f"Error trying to open file '{fn}': {e}")

    try:
        fn = f"{get_current_run_folder()}/result_min_max.txt"
        with open(fn, mode="a", encoding="utf-8") as myfile:
            for rarg in arg_result_min_or_max:
                original_print(rarg, file=myfile)
    except Exception as e:
        print_red(f"Error trying to open file '{fn}': {e}")

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
    write_ui_url_if_present()

    LOGFILE_DEBUG_GET_NEXT_TRIALS = f'{get_current_run_folder()}/get_next_trials.csv'
    experiment_parameters, cli_params_experiment_parameters = parse_parameters()

    write_live_share_file_if_needed()

    fn = f'{get_current_run_folder()}/job_start_time.txt'
    try:
        with open(fn, mode='w', encoding="utf-8") as f:
            f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    except Exception as e:
        print_red(f"Error trying to write {fn}: {e}")

    write_git_version()

    disable_logging()
    check_max_eval(max_eval)

    random_steps, second_step_steps = get_number_of_steps(max_eval)
    add_exclude_to_defective_nodes()
    handle_random_steps()

    set_global_generation_strategy()

    initialize_ax_client()

    with console.status("[bold green]Getting experiment parameters..."):
        ax_client, experiment_parameters, experiment_args, gpu_string, gpu_color = get_experiment_parameters([
            args.continue_previous_job,
            args.seed,
            get_constraints(),
            args.parameter,
            cli_params_experiment_parameters,
            experiment_parameters,
        ])

    set_orchestrator()

    show_available_hardware_and_generation_strategy_string(gpu_string, gpu_color)

    original_print(f"Run-Program: {global_vars['joined_run_program']}")

    if args.external_generator:
        original_print(f"External-Generator: {decode_if_base64(args.external_generator)}")

    checkpoint_parameters_filepath = f"{get_current_run_folder()}/state_files/checkpoint.json.parameters.json"
    save_experiment_parameters(checkpoint_parameters_filepath, experiment_parameters)

    print_overview_tables(experiment_parameters, experiment_args)

    write_files_and_show_overviews()

    for existing_run in args.load_data_from_existing_jobs:
        insert_jobs_from_csv(f"{existing_run}/results.csv", experiment_parameters)

    try:
        run_search_with_progress_bar()

        live_share()

        time.sleep(2)
    except ax.exceptions.core.UnsupportedError:
        pass

    end_program()

@beartype
def log_worker_creation() -> None:
    with console.status("[bold green]Writing worker creation log..."):
        _debug_worker_creation("time, nr_workers, got, requested, phase")

@beartype
def set_run_folder() -> None:
    global CURRENT_RUN_FOLDER
    RUN_FOLDER_NUMBER: int = 0
    CURRENT_RUN_FOLDER = f"{args.run_dir}/{global_vars['experiment_name']}/{RUN_FOLDER_NUMBER}"

    while os.path.exists(f"{CURRENT_RUN_FOLDER}"):
        RUN_FOLDER_NUMBER += 1
        CURRENT_RUN_FOLDER = f"{args.run_dir}/{global_vars['experiment_name']}/{RUN_FOLDER_NUMBER}"

@beartype
def print_run_info() -> None:
    original_print(f"Run-folder: {get_current_run_folder()}")
    if args.continue_previous_job:
        original_print(f"Continuation from {args.continue_previous_job}")

@beartype
def initialize_nvidia_logs() -> None:
    global NVIDIA_SMI_LOGS_BASE
    NVIDIA_SMI_LOGS_BASE = f'{get_current_run_folder()}/gpu_usage_'

@beartype
def write_ui_url_if_present() -> None:
    if args.ui_url:
        with open(f"{get_current_run_folder()}/ui_url.txt", mode="a", encoding="utf-8") as myfile:
            myfile.write(decode_if_base64(args.ui_url))

@beartype
def handle_random_steps() -> None:
    global random_steps
    if args.parameter and args.continue_previous_job and random_steps <= 0:
        print(f"A parameter has been reset, but the earlier job already had its random phase. To look at the new search space, {args.num_random_steps} random steps will be executed.")
        random_steps = args.num_random_steps

@beartype
def initialize_ax_client() -> None:
    global ax_client

    with console.status("[bold green]Initializing ax_client..."):
        ax_client = AxClient(
            verbose_logging=args.verbose,
            enforce_sequential_optimization=args.enforce_sequential_optimization,
            generation_strategy=global_gs
        )

        ax_client = cast(AxClient, ax_client)

@beartype
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

@beartype
def save_experiment_parameters(filepath: str, experiment_parameters: Union[list, dict]) -> None:
    with open(filepath, mode="w", encoding="utf-8") as outfile:
        json.dump(experiment_parameters, outfile, cls=NpEncoder)

@beartype
def run_search_with_progress_bar() -> None:
    live_share()

    disable_tqdm = args.disable_tqdm or ci_env

    total_jobs = max_eval

    with tqdm(total=total_jobs, disable=disable_tqdm, ascii="░▒█") as _progress_bar:
        write_process_info()
        global progress_bar
        progress_bar = _progress_bar

        progressbar_description(["Started OmniOpt2 run..."])
        update_progress_bar(progress_bar, count_done_jobs() + NR_INSERTED_JOBS)

        run_search(progress_bar)

    wait_for_jobs_to_complete()

@beartype
def complex_tests(_program_name: str, wanted_stderr: str, wanted_exit_code: int, wanted_signal: Union[int, None], res_is_none: bool = False) -> int:
    #print_yellow(f"Test suite: {_program_name}")

    nr_errors: int = 0

    program_path: str = f"./.tests/test_wronggoing_stuff.bin/bin/{_program_name}"

    if not os.path.exists(program_path):
        print_red(f"Program path {program_path} not found!")
        my_exit(18)

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

@beartype
def get_files_in_dir(mypath: str) -> list:
    print_debug("get_files_in_dir")
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    return [f"{mypath}/{s}" for s in onlyfiles]

@beartype
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

@beartype
def run_tests() -> None:
    print_red("This should be red")
    print_yellow("This should be yellow")
    print_green("This should be green")

    print(f"Printing test from current line {get_line_info()}")

    nr_errors: int = 0

    try:
        ie = is_equal('get_min_or_max_column_value(".tests/_plot_example_runs/ten_params/0/IDONTEVENEXIST/results.csv", "result", -123, "min")', str(get_min_or_max_column_value(".tests/_plot_example_runs/ten_params/0/IDONTEVENEXIST/results.csv", 'result', -123, "min")), '-123')

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

    rounded_lower, rounded_upper = round_lower_and_upper_if_type_is_int("int", -123.4, 123.4)
    nr_errors += is_equal("rounded_lower", rounded_lower, -124)
    nr_errors += is_equal("rounded_upper", rounded_upper, 124)

    nr_errors += is_equal('get_min_or_max_column_value(".tests/_plot_example_runs/ten_params/0/results.csv", "result", -123, "min")', str(get_min_or_max_column_value(".tests/_plot_example_runs/ten_params/0/results.csv", 'result', -123, "min")), '17143005390319.627')
    nr_errors += is_equal('get_min_or_max_column_value(".tests/_plot_example_runs/ten_params/0/results.csv", "result", -123, "max")', str(get_min_or_max_column_value(".tests/_plot_example_runs/ten_params/0/results.csv", 'result', -123, "max")), '9.865416064838896e+29')

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

    got: str = json.dumps(get_sixel_graphics_data('.gui/_share_test_case/test_user/ClusteredStatisticalTestDriftDetectionMethod_NOAAWeather/0/results.csv', True))
    expected: str = '[["bash omniopt_plot --run_dir  --plot_type=trial_index_result", {"type": "trial_index_result", "min_done_jobs": 2}, "/plots/", "trial_index_result", "/plots//trial_index_result.png", "1200"], ["bash omniopt_plot --run_dir  --plot_type=scatter --dpi=76", {"type": "scatter", "params": "--bubblesize=50 --allow_axes %0 --allow_axes %1", "iterate_through": [["n_samples", "confidence"], ["n_samples", "feature_proportion"], ["n_samples", "n_clusters"], ["confidence", "feature_proportion"], ["confidence", "n_clusters"], ["feature_proportion", "n_clusters"]], "dpi": 76, "filename": "plot_%0_%1_%2"}, "/plots/", "scatter", "/plots//plot_%0_%1_%2.png", "1200"], ["bash omniopt_plot --run_dir  --plot_type=general", {"type": "general"}, "/plots/", "general", "/plots//general.png", "1200"]]'

    nr_errors += is_equal('get_sixel_graphics_data(".gui/_share_test_case/test_user/ClusteredStatisticalTestDriftDetectionMethod_NOAAWeather/0/results.csv", True)', got, expected)

    nr_errors += is_equal('get_hostname_from_outfile("")', get_hostname_from_outfile(''), None)

    res = get_hostname_from_outfile('.tests/_plot_example_runs/ten_params/0/single_runs/266908/266908_0_log.out')
    nr_errors += is_equal('get_hostname_from_outfile(".tests/_plot_example_runs/ten_params/0/single_runs/266908/266908_0_log.out")', res, 'arbeitsrechner')

    nr_errors += is_equal('get_parameters_from_outfile("")', get_parameters_from_outfile(''), None)
    #res = {"one": 678, "two": 531, "three": 569, "four": 111, "five": 127, "six": 854, "seven": 971, "eight": 332, "nine": 235, "ten": 867.6452040672302}
    #nr_errors += is_equal('get_parameters_from_outfile("".tests/_plot_example_runs/ten_params/0/single_runs/266908/266908_0_log.out")', get_parameters_from_outfile(".tests/_plot_example_runs/ten_params/0/single_runs/266908/266908_0_log.out"), res)

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
        nr_errors += complex_tests("signal_but_has_output", "Killed", 137, None) # Doesnt show Killed on taurus
        nr_errors += complex_tests("signal", "Killed", 137, None, True) # Doesnt show Killed on taurus
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
        ["create_and_execute_next_runs(0, None, None, None)", 0]
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
            json.dumps(evaluate({'x': 123.0})),
            json.dumps({'RESULT': 123.0})
    )

    nr_errors += is_equal(
            "evaluate({'x': -0.05})",
            json.dumps(evaluate({'x': -0.05})),
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
        should_be: str = '{"errors": [{"name": "GPUDisconnected", "match_strings": ["AssertionError: ``AmpOptimizerWrapper`` is only available"], "behavior": "ExcludeNode"}, {"name": "Timeout", "match_strings": ["Timeout"], "behavior": "RestartOnDifferentNode"}, {"name": "StorageError", "match_strings": ["Read/Write failure"], "behavior": "ExcludeNodeAndRestartAll"}]}'
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

    # Signed Weighted Euclidean Distance
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

@beartype
def main_outside() -> None:
    print(f"Run-UUID: {run_uuid}")

    print_logo()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if args.tests:
            run_tests()
        else:
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

if __name__ == "__main__":
    try:
        main_outside()
    except (SignalUSR, SignalINT, SignalCONT) as e:
        print_red(f"main_outside failed with exception {e}")
        end_program(True)
