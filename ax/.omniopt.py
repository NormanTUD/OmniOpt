#!/bin/env python3

import sys
import os

ci_env: bool = os.getenv("CI", "false").lower() == "true"
original_print = print

valid_moo_types: list = ["geometric", "euclid", "signed_harmonic"]

try:
    from rich.console import Console

    console: Console = Console(
        force_interactive=True,
        soft_wrap=True,
        color_system="256",
        force_terminal=not ci_env
    )

    with console.status("[bold green]Loading base modules...") as status:
        import json
        from types import FunctionType
        from typing import Pattern, Optional, Tuple, Any, cast, Union, TextIO
        from submitit import LocalExecutor, AutoExecutor
        from submitit import Job
        import threading
        from concurrent.futures import ThreadPoolExecutor
        import argparse
        import datetime
        import importlib.util
        import inspect
        import platform
        import random
        from inspect import currentframe, getframeinfo
        from pathlib import Path
        import uuid
        import yaml
        import re
        import toml
        import time
        import traceback
        import cowsay
        from rich_argparse import RichHelpFormatter
        import psutil
        import shutil
        import math
        from itertools import combinations
        import signal
        import warnings
        import pandas as pd
        from os import listdir
        from os.path import isfile, join
        import socket
        import stat
        import pwd
        import base64
        from pprint import pformat
        import sixel
        from PIL import Image
        from rich.table import Table
        from rich import print
        import csv
        from rich.pretty import pprint
        import subprocess
        import logging
        logging.basicConfig(level=logging.CRITICAL)
        from tqdm import tqdm
except ModuleNotFoundError as e: # pragma: no cover
    print(f"Some of the base modules could not be loaded. Most probably that means you have not loaded or installed the virtualenv properly. Error: {e}")
    print("Exit-Code: 2")
    sys.exit(2)

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

new_uuid: str = str(uuid.uuid4())
run_uuid: str = os.getenv("RUN_UUID", new_uuid)

if not uuid_regex.match(run_uuid): # pragma: no cover
    print(f"{YELLOW}WARNING: The provided RUN_UUID is not a valid UUID. Using new UUID {new_uuid} instead.{RESET}")
    run_uuid = new_uuid

JOBS_FINISHED: int = 0
SHOWN_LIVE_SHARE_COUNTER: int = 0
PD_CSV_FILENAME: str = "results.csv"
WORKER_PERCENTAGE_USAGE: list = []
END_PROGRAM_RAN: bool = False
ALREADY_SHOWN_WORKER_USAGE_OVER_TIME: bool = False
ax_client = None
TIME_NEXT_TRIALS_TOOK: list[float] = []
CURRENT_RUN_FOLDER: str = ""
RESULT_CSV_FILE: str = ""
SHOWN_END_TABLE: bool = False
max_eval: int = 1
random_steps: int = 1
progress_bar: Optional[tqdm] = None

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
else: # pragma: no cover
    raise ImportError(f"Could not load module from {helpers_file}")

dier: FunctionType = helpers.dier
is_equal: FunctionType = helpers.is_equal
is_not_equal: FunctionType = helpers.is_not_equal

SUPPORTED_MODELS: list = [
    "SOBOL",
    "GPEI",
    "FACTORIAL",
    "SAASBO",
    "LEGACY_BOTORCH",
    "BOTORCH_MODULAR",
    "UNIFORM",
    "BO_MIXED",
]

ORCHESTRATE_TODO: dict = {}

class SignalUSR (Exception):
    pass

class SignalINT (Exception):
    pass

class SignalCONT (Exception):
    pass

def is_slurm_job() -> bool:
    if os.environ.get('SLURM_JOB_ID') is not None: # pragma: no cover
        return True
    return False

def _sleep(t: int) -> int:
    if args is not None and not args.no_sleep:
        time.sleep(t)

    return t

LOG_DIR: str = "logs"
makedirs(LOG_DIR)

log_uuid_dir = f"{LOG_DIR}/{run_uuid}"
logfile: str = f'{log_uuid_dir}_log'
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
            with open(f"{get_current_run_folder()}/oo_errors.txt", mode="a", encoding="utf-8") as myfile:
                myfile.write(text + "\n\n")
        except FileNotFoundError as e: # pragma: no cover
            helpers.print_color("red", f"Error: {e}. This may mean that the {get_current_run_folder()} was deleted during the run. Could not write '{text} to {get_current_run_folder()}/oo_errors.txt'")
            sys.exit(99)

def _debug(msg: str, _lvl: int = 0, eee: Union[None, str, Exception] = None) -> None:
    if _lvl > 3: # pragma: no cover
        original_print(f"Cannot write _debug, error: {eee}")
        print("Exit-Code: 193")
        sys.exit(193)

    try:
        with open(logfile, mode='a', encoding="utf-8") as f:
            original_print(msg, file=f)
    except FileNotFoundError: # pragma: no cover
        print_red("It seems like the run's folder was deleted during the run. Cannot continue.")
        sys.exit(99) # generalized code for run folder deleted during run
    except Exception as e: # pragma: no cover
        original_print("_debug: Error trying to write log file: " + str(e))

        _debug(msg, _lvl + 1, e)

def _get_debug_json(time_str: str, msg: str) -> str:
    stack = inspect.stack()
    function_stack = []

    for frame_info in stack[1:]:
        if str(frame_info.function) != "<module>" and str(frame_info.function) != "print_debug":
            if frame_info.function != "wrapper":
                function_stack.append({
                    "function": frame_info.function,
                    "line_number": frame_info.lineno
                })

    return json.dumps({"function_stack": function_stack, "time": time_str, "msg": msg}, indent=0).replace('\r', '').replace('\n', '')

def print_debug(msg: str) -> None:
    time_str: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    stack_trace_element = _get_debug_json(time_str, msg)

    msg = f"{stack_trace_element}"

    #if args is not None and args.debug: # pragma: no cover
    #    original_print(msg)

    _debug(msg)

def my_exit(_code: int = 0) -> None:
    tb = traceback.format_exc()

    try:
        print_debug(f"Exiting with error code {_code}. Traceback: {tb}")
    except NameError: # pragma: no cover
        print(f"Exiting with error code {_code}. Traceback: {tb}")

    if (is_slurm_job() and not args.force_local_execution) and not (args.show_sixel_scatter or args.show_sixel_general or args.show_sixel_trial_index_result): # pragma: no cover
        _sleep(5)

    print("Exit-Code: " + str(_code))
    sys.exit(_code)

class ConfigLoader:
    run_tests_that_fail_on_taurus: bool
    enforce_sequential_optimization: bool
    num_random_steps: int
    verbose: bool
    disable_tqdm: bool
    slurm_use_srun: bool
    reservation: Optional[str]
    account: Optional[str]
    should_deduplicate: bool
    exclude: Optional[str]
    show_sixel_trial_index_result: bool
    num_parallel_jobs: int
    max_parallelism: int
    force_local_execution: bool
    moo_type: str
    raise_in_eval: bool
    maximize: bool
    show_sixel_general: bool
    show_sixel_scatter: bool
    gpus: int
    load_previous_job_data: Optional[str]
    model: str
    live_share: bool
    experiment_name: str
    show_worker_percentage_table_at_end: bool
    abbreviate_job_names: bool
    verbose_tqdm: bool
    tests: bool
    max_eval: int
    run_program: str
    orchestrator_file: Optional[str]
    run_dir: str
    ui_url: Optional[str]
    nodes_per_job: int
    seed: int
    cpus_per_task: int
    parameter: str
    experiment_constraints: Optional[list[str]]
    stderr_to_stdout: bool
    worker_timeout: int
    disable_search_space_exhaustion_detection: bool
    slurm_signal_delay_s: int
    gridsearch: bool
    auto_exclude_defective_hosts: bool
    debug: bool
    no_sleep: bool
    max_nr_of_zero_results: int
    mem_gb: int
    continue_previous_job: Optional[str]

    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(
            prog="omniopt",
            description='A hyperparameter optimizer for slurm-based HPC-systems',
            epilog="Example:\n\n./omniopt --partition=alpha --experiment_name=neural_network ...",
            formatter_class=RichHelpFormatter
        )

        # Add config arguments
        self.parser.add_argument('--config_yaml', help='YAML configuration file', type=str)
        self.parser.add_argument('--config_toml', help='TOML configuration file', type=str)
        self.parser.add_argument('--config_json', help='JSON configuration file', type=str)

        # Initialize the remaining arguments
        self.add_arguments()

    def add_arguments(self) -> None:
        required = self.parser.add_argument_group('Required arguments', "These options have to be set")
        required_but_choice = self.parser.add_argument_group('Required arguments that allow a choice', "Of these arguments, one has to be set to continue.")
        optional = self.parser.add_argument_group('Optional', "These options are optional")
        slurm = self.parser.add_argument_group('SLURM', "Parameters related to SLURM")
        installing = self.parser.add_argument_group('Installing', "Parameters related to installing")
        debug = self.parser.add_argument_group('Debug', "These options are mainly useful for debugging")

        required.add_argument('--num_random_steps', help='Number of random steps to start with', type=int, default=20)
        required.add_argument('--max_eval', help='Maximum number of evaluations', type=int)
        required.add_argument('--run_program', action='append', nargs='+', help='A program that should be run. Use, for example, $x for the parameter named x.', type=str)
        required.add_argument('--experiment_name', help='Name of the experiment.', type=str)
        required.add_argument('--mem_gb', help='Amount of RAM for each worker in GB (default: 1GB)', type=float, default=1)

        required_but_choice.add_argument('--parameter', action='append', nargs='+', help="Experiment parameters in the formats (options in round brackets are optional): <NAME> range <LOWER BOUND> <UPPER BOUND> (<INT, FLOAT>, log_scale: True/False, default: false>) -- OR -- <NAME> fixed <VALUE> -- OR -- <NAME> choice <Comma-separated list of values>", default=None)
        required_but_choice.add_argument('--continue_previous_job', help="Continue from a previous checkpoint, use run-dir as argument", type=str, default=None)

        optional.add_argument('--maximize', help='Maximize instead of minimize (which is default)', action='store_true', default=False)
        optional.add_argument('--experiment_constraints', action="append", nargs="+", help='Constraints for parameters. Example: x + y <= 2.0', type=str)
        optional.add_argument('--stderr_to_stdout', help='Redirect stderr to stdout for subjobs', action='store_true', default=False)
        optional.add_argument('--run_dir', help='Directory, in which runs should be saved. Default: runs', default="runs", type=str)
        optional.add_argument('--seed', help='Seed for random number generator', type=int)
        optional.add_argument('--enforce_sequential_optimization', help='Enforce sequential optimization (default: false)', action='store_true', default=False)
        optional.add_argument('--verbose_tqdm', help='Show verbose tqdm messages', action='store_true', default=False)
        optional.add_argument('--load_previous_job_data', action="append", nargs="+", help='Paths of previous jobs to load from', type=str)
        optional.add_argument('--hide_ascii_plots', help='Hide ASCII-plots.', action='store_true', default=False)
        optional.add_argument('--model', help=f'Use special models for nonrandom steps. Valid models are: {", ".join(SUPPORTED_MODELS)}', type=str, default=None)
        optional.add_argument('--gridsearch', help='Enable gridsearch.', action='store_true', default=False)
        optional.add_argument('--show_sixel_scatter', help='Show sixel graphics of scatter plots in the end', action='store_true', default=False)
        optional.add_argument('--show_sixel_general', help='Show sixel graphics of general plots in the end', action='store_true', default=False)
        optional.add_argument('--show_sixel_trial_index_result', help='Show sixel graphics of scatter plots in the end', action='store_true', default=False)
        optional.add_argument('--follow', help='Automatically follow log file of sbatch', action='store_true', default=False)
        optional.add_argument('--send_anonymized_usage_stats', help='Send anonymized usage stats', action='store_true', default=False)
        optional.add_argument('--ui_url', help='Site from which the OO-run was called', default=None, type=str)
        optional.add_argument('--root_venv_dir', help=f'Where to install your modules to ($root_venv_dir/.omniax_..., default: {os.getenv("HOME")})', default=os.getenv("HOME"), type=str)
        optional.add_argument('--exclude', help='A comma separated list of values of excluded nodes (taurusi8009,taurusi8010)', default=None, type=str)
        optional.add_argument('--main_process_gb', help='Amount of RAM for the main process in GB (default: 1GB)', type=float, default=4)
        optional.add_argument('--max_nr_of_zero_results', help='Max. nr of successive zero results by ax_client.get_next_trial() before the search space is seen as exhausted. Default is 20', type=int, default=20)
        optional.add_argument('--disable_search_space_exhaustion_detection', help='Disables automatic search space reduction detection', action='store_true', default=False)
        optional.add_argument('--abbreviate_job_names', help='Abbreviate pending job names (r = running, p = pending, u = unknown, c = cancelling)', action='store_true', default=False)
        optional.add_argument('--orchestrator_file', help='An orchestrator file', default=None, type=str)
        optional.add_argument('--checkout_to_latest_tested_version', help='Automatically checkout to latest version that was tested in the CI pipeline', action='store_true', default=False)
        optional.add_argument('--live_share', help='Automatically live-share the current optimization run automatically', action='store_true', default=False)
        optional.add_argument('--disable_tqdm', help='Disables the TQDM progress bar', action='store_true', default=False)
        optional.add_argument('--workdir', help='Work dir', action='store_true', default=False)
        optional.add_argument('--should_deduplicate', help='Try to de-duplicate ARMs', action='store_true', default=False)
        optional.add_argument('--max_parallelism', help='Set how the ax max parallelism flag should be set. Possible options: None, max_eval, num_parallel_jobs, twice_max_eval, max_eval_times_thousand_plus_thousand, twice_num_parallel_jobs and any integer.', type=str, default="max_eval_times_thousand_plus_thousand")
        optional.add_argument('--moo_type', help=f'MOO-type (valid types are {", ".join(valid_moo_types)})', type=str, default="euclid")

        slurm.add_argument('--num_parallel_jobs', help='Number of parallel slurm jobs (default: 20)', type=int, default=20)
        slurm.add_argument('--worker_timeout', help='Timeout for slurm jobs (i.e. for each single point to be optimized)', type=int, default=30)
        slurm.add_argument('--slurm_use_srun', help='Using srun instead of sbatch', action='store_true', default=False)
        slurm.add_argument('--time', help='Time for the main job', default="", type=str)
        slurm.add_argument('--partition', help='Partition to be run on', default="", type=str)
        slurm.add_argument('--reservation', help='Reservation', default=None, type=str)
        slurm.add_argument('--force_local_execution', help='Forces local execution even when SLURM is available', action='store_true', default=False)
        slurm.add_argument('--slurm_signal_delay_s', help='When the workers end, they get a signal so your program can react to it. Default is 0, but set it to any number of seconds you wish your program to be able to react to USR1.', type=int, default=0)
        slurm.add_argument('--nodes_per_job', help='Number of nodes per job due to the new alpha restriction', type=int, default=1)
        slurm.add_argument('--cpus_per_task', help='CPUs per task', type=int, default=1)
        slurm.add_argument('--account', help='Account to be used', type=str, default=None)
        slurm.add_argument('--gpus', help='Number of GPUs', type=int, default=0)
        #slurm.add_ argument('--tasks_per_node', help='ntasks', type=int, default=1)

        installing.add_argument('--run_mode', help='Either local or docker', default="local", type=str)

        debug.add_argument('--verbose', help='Verbose logging', action='store_true', default=False)
        debug.add_argument('--debug', help='Enable debugging', action='store_true', default=False)
        debug.add_argument('--no_sleep', help='Disables sleeping for fast job generation (not to be used on HPC)', action='store_true', default=False)
        debug.add_argument('--tests', help='Run simple internal tests', action='store_true', default=False)
        debug.add_argument('--show_worker_percentage_table_at_end', help='Show a table of percentage of usage of max worker over time', action='store_true', default=False)
        debug.add_argument('--auto_exclude_defective_hosts', help='Run a Test if you can allocate a GPU on each node and if not, exclude it since the GPU driver seems to be broken somehow.', action='store_true', default=False)
        debug.add_argument('--run_tests_that_fail_on_taurus', help='Run tests on Taurus that usually fail.', action='store_true', default=False)
        debug.add_argument('--raise_in_eval', help='Raise a signal in eval (only useful for debugging and testing).', action='store_true', default=False)

    def load_config(self, config_path: str, file_format: str) -> dict:
        if not os.path.isfile(config_path): # pragma: no cover
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

        return {} # pragma: no cover

    def validate_and_convert(self, config: dict, arg_defaults: dict) -> dict:
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
                except (ValueError, TypeError): # pragma: no cover
                    print(f"Warning: Cannot convert '{key}' to {expected_type.__name__}. Using default value.")
            else: # pragma: no cover
                print(f"Warning: Unknown config parameter '{key}' found in the config file and ignored.")

        return converted_config

    def merge_args_with_config(self: Any, config: Any, cli_args: Any) -> Any:
        """ Merge CLI args with config file args (CLI takes precedence) """
        arg_defaults = {arg.dest: arg.default for arg in self.parser._actions if arg.default is not argparse.SUPPRESS}

        # Validate and convert the config values
        validated_config = self.validate_and_convert(config, arg_defaults)

        for key, value in vars(cli_args).items():
            if key in validated_config:
                setattr(cli_args, key, validated_config[key])

        return cli_args

    def parse_arguments(self: Any) -> Any:
        # First, parse the CLI arguments to check if config files are provided
        _args = self.parser.parse_args()

        config = {}

        yaml_and_toml = _args.config_yaml and _args.config_toml
        yaml_and_json = _args.config_yaml and _args.config_json
        json_and_toml = _args.config_json and _args.config_toml

        if yaml_and_toml or yaml_and_json or json_and_toml: # pragma: no cover
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

def wrapper_print_debug(func: Any) -> Any:
    def wrapper(*__args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = func(*__args, **kwargs)
        end_time = time.time()

        runtime = end_time - start_time
        runtime_human_readable = f"{runtime:.4f} seconds"

        if runtime > 1:
            print_debug(f"@wrapper_print_debug: {func.__name__}(), runtime: {runtime_human_readable}")

        return result
    return wrapper

disable_logs = None

if not args.tests:
    try:
        with console.status("[bold green]Loading torch...") as status:
            import torch
        with console.status("[bold green]Loading numpy...") as status:
            import numpy as np
        with console.status("[bold green]Loading ax...") as status:
            import ax
            import ax.exceptions.core
            import ax.exceptions.generation_strategy
            import ax.modelbridge.generation_node
            from ax.modelbridge.generation_strategy import (GenerationStep, GenerationStrategy)
            from ax.modelbridge.registry import Models
            from ax.service.ax_client import AxClient, ObjectiveProperties
            from ax.storage.json_store.load import load_experiment
            from ax.storage.json_store.save import save_experiment
        with console.status("[bold green]Loading botorch...") as status:
            import botorch
        with console.status("[bold green]Loading submitit...") as status:
            import submitit
            from submitit import DebugJob, LocalJob
    except ModuleNotFoundError as ee: # pragma: no cover
        original_print(f"Base modules could not be loaded: {ee}")
        my_exit(31)
    except SignalINT: # pragma: no cover
        print("\n⚠ Signal INT was detected. Exiting with 128 + 2.")
        my_exit(130)
    except SignalUSR: # pragma: no cover
        print("\n⚠ Signal USR was detected. Exiting with 128 + 10.")
        my_exit(138)
    except SignalCONT: # pragma: no cover
        print("\n⚠ Signal CONT was detected. Exiting with 128 + 18.")
        my_exit(146)
    except KeyboardInterrupt: # pragma: no cover
        print("\n⚠ You pressed CTRL+C. Program execution halted.")
        my_exit(0)
    except AttributeError: # pragma: no cover
        print(f"\n⚠ This error means that your virtual environment is probably outdated. Try removing the virtual environment under '{os.getenv('VENV_DIR')}' and re-install your environment.")
        my_exit(7)

with console.status("[bold green]Loading ax logger...") as status:
    from ax.utils.common.logger import disable_loggers
disable_logs = disable_loggers(names=["ax.modelbridge.base"], level=logging.CRITICAL)

NVIDIA_SMI_LOGS_BASE = None

def append_and_read(file: str, nr: int = 0, recursion: int = 0) -> int:
    try:
        with open(file, mode='a+', encoding="utf-8") as f:
            f.seek(0)  # Setze den Dateizeiger auf den Anfang der Datei
            nr_lines = len(f.readlines())

            if nr == 1:
                f.write('1\n')

        return nr_lines

    except FileNotFoundError as e: # pragma: no cover
        original_print(f"File not found: {e}")
    except (SignalUSR, SignalINT, SignalCONT): # pragma: no cover
        if recursion:
            print_red("Recursion error in append_and_read.")
            sys.exit(199)
        append_and_read(file, nr, recursion + 1)
    except OSError as e: # pragma: no cover
        print_red(f"OSError: {e}. This may happen on unstable file systems.")
        sys.exit(199)
    except Exception as e: # pragma: no cover
        print(f"Error editing the file: {e}")

    return 0 # pragma: no cover

@wrapper_print_debug
def run_live_share_command() -> Tuple[str, str]:
    if get_current_run_folder():
        try:
            # Environment variable USER
            _user = os.getenv('USER')
            if _user is None: # pragma: no cover
                _user = 'defaultuser'

            _command = f"bash {script_dir}/omniopt_share {get_current_run_folder()} --update --username={_user} --no_color"

            print_debug(f"run_live_share_command: {_command}")

            result = subprocess.run(_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # Return stdout and stderr
            return str(result.stdout), str(result.stderr)
        except subprocess.CalledProcessError as e: # pragma: no cover
            if e.stderr:
                original_print(f"run_live_share_command: command failed with error: {e}, stderr: {e.stderr}")
            else:
                original_print(f"run_live_share_command: command failed with error: {e}")
            return "", str(e.stderr)
        except Exception as e: # pragma: no cover
            print(f"run_live_share_command: An error occurred: {e}")

    return "", "" # pragma: no cover

@wrapper_print_debug
def live_share() -> bool:
    global SHOWN_LIVE_SHARE_COUNTER

    if not args.live_share: # pragma: no cover
        return False

    if not get_current_run_folder(): # pragma: no cover
        return False

    stdout, stderr = run_live_share_command()

    if SHOWN_LIVE_SHARE_COUNTER == 0 and stderr:
        print_yellow(stderr)

    SHOWN_LIVE_SHARE_COUNTER = SHOWN_LIVE_SHARE_COUNTER + 1

    return True

@wrapper_print_debug
def save_pd_csv() -> str:
    #print_debug("save_pd_csv()")
    pd_csv: str = f'{get_current_run_folder()}/{PD_CSV_FILENAME}'
    pd_json: str = f'{get_current_run_folder()}/state_files/pd.json'

    state_files_folder: str = f"{get_current_run_folder()}/state_files/"

    makedirs(state_files_folder)

    if ax_client is None:
        return pd_csv

    try:
        pd_frame = ax_client.get_trials_data_frame()

        pd_frame.to_csv(pd_csv, index=False, float_format="%.30f")
        #pd_frame.to_json(pd_json)

        json_snapshot = ax_client.to_json_snapshot()

        with open(pd_json, mode='w', encoding="utf-8") as json_file:
            json.dump(json_snapshot, json_file, indent=4)

        save_experiment(ax_client.experiment, f"{get_current_run_folder()}/state_files/ax_client.experiment.json")
    except SignalUSR as e: # pragma: no cover
        raise SignalUSR(str(e)) from e
    except SignalCONT as e: # pragma: no cover
        raise SignalCONT(str(e)) from e
    except SignalINT as e: # pragma: no cover
        raise SignalINT(str(e)) from e
    except Exception as e: # pragma: no cover
        print_red(f"While saving all trials as a pandas-dataframe-csv, an error occurred: {e}")

    return pd_csv

def add_to_phase_counter(phase: str, nr: int = 0, run_folder: str = "") -> int:
    if run_folder == "":
        run_folder = get_current_run_folder()
    return append_and_read(f'{run_folder}/state_files/phase_{phase}_steps', nr)

if args.model and str(args.model).upper() not in SUPPORTED_MODELS:
    print(f"Unsupported model {args.model}. Cannot continue. Valid models are {', '.join(SUPPORTED_MODELS)}")
    my_exit(203)

if args.num_parallel_jobs:
    num_parallel_jobs = args.num_parallel_jobs

class SearchSpaceExhausted (Exception):
    pass

NR_INSERTED_JOBS: int = 0
changed_grid_search_params: dict = {}
executor: Union[LocalExecutor, AutoExecutor, None] = None

NR_OF_0_RESULTS: int = 0

orchestrator = None
double_hashes: dict = {}
missing_results: list = []
already_inserted_param_hashes: dict = {}
already_inserted_param_data: list = []

def print_logo() -> None:
    print_debug("print_logo()")
    if os.environ.get('NO_OO_LOGO') is not None:
        return

    sprueche = [
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

process = psutil.Process(os.getpid())

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

def set_max_eval(new_max_eval: int) -> None:
    global max_eval

    #print(f"set_max_eval(new_max_eval: {new_max_eval})")
    #traceback.print_stack()

    max_eval = new_max_eval

def write_worker_usage() -> None:
    if len(WORKER_PERCENTAGE_USAGE): # pragma: no cover
        csv_filename = f'{get_current_run_folder()}/worker_usage.csv'

        csv_columns = ['time', 'num_parallel_jobs', 'nr_current_workers', 'percentage']

        with open(csv_filename, mode='w', encoding="utf-8", newline='') as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            for row in WORKER_PERCENTAGE_USAGE:
                csv_writer.writerow(row)
    else:
        if is_slurm_job(): # pragma: no cover
            print_debug("WORKER_PERCENTAGE_USAGE seems to be empty. Not writing worker_usage.csv")

@wrapper_print_debug
def log_system_usage() -> None:
    if not get_current_run_folder(): # pragma: no cover
        return

    csv_file_path = os.path.join(get_current_run_folder(), "cpu_ram_usage.csv")

    makedirs(os.path.dirname(csv_file_path))

    file_exists = os.path.isfile(csv_file_path)

    with open(csv_file_path, mode='a', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(["timestamp", "ram_usage_mb", "cpu_usage_percent"])

        current_time = int(time.time())

        ram_usage_mb = process.memory_info().rss / (1024 * 1024)  # RSS in MB
        cpu_usage_percent = psutil.cpu_percent(percpu=False)  # Gesamt-CPU-Auslastung in Prozent

        writer.writerow([current_time, ram_usage_mb, cpu_usage_percent])

def write_process_info() -> None:
    try:
        log_system_usage()
    except Exception as e: # pragma: no cover
        print_debug(f"Error retrieving process information: {str(e)}")

def log_nr_of_workers() -> None:
    try:
        write_process_info()
    except Exception as e: # pragma: no cover
        print_debug(f"log_nr_of_workers: failed to write_process_info: {e}")
        return None

    if "jobs" not in global_vars: # pragma: no cover
        print_debug("log_nr_of_workers: Could not find jobs in global_vars")
        return None

    nr_of_workers: int = len(global_vars["jobs"])

    if not nr_of_workers:
        return None

    try: # pragma: no cover
        with open(logfile_nr_workers, mode='a+', encoding="utf-8") as f:
            f.write(str(nr_of_workers) + "\n")
    except FileNotFoundError: # pragma: no cover
        print_red(f"It seems like the folder for writing {logfile_nr_workers} was deleted during the run. Cannot continue.")
        my_exit(99)
    except OSError as e: # pragma: no cover
        print_red(f"Tried writing log_nr_of_workers to file {logfile_nr_workers}, but failed with error: {e}. This may mean that the file system you are running on is instable. OmniOpt probably cannot do anything about it.")
        my_exit(199)

    return None # pragma: no cover

@wrapper_print_debug
def log_what_needs_to_be_logged() -> None:
    if "write_worker_usage" in globals():
        try:
            write_worker_usage()
        except Exception: # pragma: no cover
            pass

    if "write_process_info" in globals():
        try:
            write_process_info()
        except Exception: # pragma: no cover
            pass

    if "log_nr_of_workers" in globals():
        try:
            log_nr_of_workers()
        except Exception: # pragma: no cover
            pass

def get_line_info() -> Tuple[str, str, int, str, str]:
    return (inspect.stack()[1][1], ":", inspect.stack()[1][2], ":", inspect.stack()[1][3])

#print(f"sys.path: {sys.path}")

def print_image_to_cli(image_path: str, width: int) -> bool:
    print("")
    try:
        image = Image.open(image_path)
        original_width, original_height = image.size

        height = int((original_height / original_width) * width)

        sixel_converter = sixel.converter.SixelConverter(image_path, w=width, h=height)

        sixel_converter.write(sys.stdout)
        _sleep(2)

        return True
    except Exception as e: # pragma: no cover
        print_debug(
            f"Error converting and resizing image: "
            f"{str(e)}, width: {width}, image_path: {image_path}"
        )

    return False

def log_message_to_file(_logfile: Union[str, None], message: str, _lvl: int = 0, eee: Union[None, str, Exception] = None) -> None:
    if not _logfile: # pragma: no cover
        return None

    if _lvl > 3: # pragma: no cover
        original_print(f"Cannot write _debug, error: {eee}")
        return None

    try:
        with open(_logfile, mode='a', encoding="utf-8") as f:
            #original_print(f"========= {time.time()} =========", file=f)
            original_print(message, file=f)
    except FileNotFoundError: # pragma: no cover
        print_red("It seems like the run's folder was deleted during the run. Cannot continue.")
        sys.exit(99) # generalized code for run folder deleted during run
    except Exception as e: # pragma: no cover
        original_print(f"Error trying to write log file: {e}")
        log_message_to_file(_logfile, message, _lvl + 1, e)

    return None

def _log_trial_index_to_param(trial_index: int, _lvl: int = 0, eee: Union[None, str, Exception] = None) -> None:
    log_message_to_file(logfile_trial_index_to_param_logs, str(trial_index), _lvl, str(eee))

def _debug_worker_creation(msg: str, _lvl: int = 0, eee: Union[None, str, Exception] = None) -> None:
    log_message_to_file(logfile_worker_creation_logs, msg, _lvl, str(eee))

def append_to_nvidia_smi_logs(_file: str, _host: str, result: str, _lvl: int = 0, eee: Union[None, str, Exception] = None) -> None: # pragma: no cover
    log_message_to_file(_file, result, _lvl, str(eee))

def _debug_get_next_trials(msg: str, _lvl: int = 0, eee: Union[None, str, Exception] = None) -> None:
    log_message_to_file(LOGFILE_DEBUG_GET_NEXT_TRIALS, msg, _lvl, str(eee))

def _debug_progressbar(msg: str, _lvl: int = 0, eee: Union[None, str, Exception] = None) -> None:
    log_message_to_file(logfile_progressbar, msg, _lvl, str(eee))

def print_green(text: str) -> None:
    helpers.print_color("green", text)

    print_debug(text)

def print_yellow(text: str) -> None:
    helpers.print_color("yellow", text)

    print_debug(text)

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
        else: # pragma: no cover
            datafile = "\n".join(_df)

    return "\n".join(datafile)

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
    prev_job_file = prev_job_folder + "/state_files/joined_run_program"
    if os.path.exists(prev_job_file):
        global_vars["joined_run_program"] = get_file_as_string(prev_job_file)
    else: # pragma: no cover
        print_red(f"The previous job file {prev_job_file} could not be found. You may forgot to add the run number at the end.")
        my_exit(44)

if not args.tests and len(global_vars["joined_run_program"]) == 0:
    print_red("--run_program was empty")
    my_exit(19)

global_vars["experiment_name"] = args.experiment_name

def load_global_vars(_file: str) -> None:
    if not os.path.exists(_file): # pragma: no cover
        print_red(f"You've tried to continue a non-existing job: {_file}")
        my_exit(44)
    try:
        global global_vars
        with open(_file, encoding="utf-8") as f:
            global_vars = json.load(f)
    except Exception as e: # pragma: no cover
        print_red("Error while loading old global_vars: " + str(e) + ", trying to load " + str(_file))
        my_exit(44)

def load_or_exit(filepath: str, error_msg: str, exit_code: int) -> None:
    if not os.path.exists(filepath): # pragma: no cover
        print_red(error_msg)
        my_exit(exit_code)

def get_file_content_or_exit(filepath: str, error_msg: str, exit_code: int) -> str:
    load_or_exit(filepath, error_msg, exit_code)
    return get_file_as_string(filepath).strip()

def check_param_or_exit(param: str, error_msg: str, exit_code: int) -> None:
    if param is None:
        print_red(error_msg)
        my_exit(exit_code)

def check_continue_previous_job(continue_previous_job: str) -> dict:
    global global_vars
    if continue_previous_job:
        load_global_vars(f"{continue_previous_job}/state_files/global_vars.json")

        # Load experiment name from file if not already set
        if not global_vars.get("experiment_name"): # pragma: no cover
            exp_name_file = f"{continue_previous_job}/experiment_name"
            global_vars["experiment_name"] = get_file_content_or_exit(
                exp_name_file,
                f"{exp_name_file} not found, and no --experiment_name given. Cannot continue.",
                19
            )
    return global_vars

def check_required_parameters(_args: Any) -> None:
    global global_vars
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

def load_time_or_exit(_args: Any) -> None:
    global global_vars
    if _args.time:
        global_vars["_time"] = _args.time
    elif _args.continue_previous_job:
        time_file = f"{_args.continue_previous_job}/state_files/time"
        time_content = get_file_content_or_exit(time_file, f"neither --time nor file {time_file} found", 19)
        if time_content.isdigit(): # pragma: no cover
            global_vars["_time"] = int(time_content)
            print_yellow(f"Using old run's --time: {global_vars['_time']}")
        else: # pragma: no cover
            print_yellow(f"Time-setting: The contents of {time_file} do not contain a single number")
    else:
        print_red("Missing --time parameter. Cannot continue.")
        my_exit(19)

def load_mem_gb_or_exit(_args: Any) -> Optional[int]: # pragma: no cover
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

def load_gpus_or_exit(_args: Any) -> Optional[int]:
    if _args.continue_previous_job and not _args.gpus:
        gpus_file = f"{_args.continue_previous_job}/state_files/gpus"
        gpus_content = get_file_content_or_exit(gpus_file, f"neither --gpus nor file {gpus_file} found", 19)
        if gpus_content.isdigit():
            gpus = int(gpus_content)
            print_yellow(f"Using old run's --gpus: {gpus}")
            return gpus

        print_yellow(f"--gpus: The contents of {gpus_file} do not contain a single number") # pragma: no cover
    return _args.gpus

def load_max_eval_or_exit(_args: Any) -> None:
    if _args.max_eval:
        set_max_eval(_args.max_eval)
        if _args.max_eval <= 0: # pragma: no cover
            print_red("--max_eval must be larger than 0")
            my_exit(19)
    elif _args.continue_previous_job: # pragma: no cover
        max_eval_file = f"{_args.continue_previous_job}/state_files/max_eval"
        max_eval_content = get_file_content_or_exit(max_eval_file, f"neither --max_eval nor file {max_eval_file} found", 19)
        if max_eval_content.isdigit():
            set_max_eval(int(max_eval_content))
            print_yellow(f"Using old run's --max_eval: {max_eval_content}")
        else:
            print_yellow(f"max_eval-setting: The contents of {max_eval_file} do not contain a single number")
    else:
        print_yellow("--max_eval needs to be set")

if not args.tests:
    global_vars = check_continue_previous_job(args.continue_previous_job)
    check_required_parameters(args)
    load_time_or_exit(args)

    loaded_mem_gb = load_mem_gb_or_exit(args)

    if loaded_mem_gb:
        args.mem_gb = loaded_mem_gb
        global_vars["mem_gb"] = args.mem_gb

    loaded_gpus = load_gpus_or_exit(args)

    if loaded_gpus: # pragma: no cover
        args.gpus = loaded_gpus
        global_vars["gpus"] = args.gpus

    load_max_eval_or_exit(args)

@wrapper_print_debug
def print_debug_get_next_trials(got: int, requested: int, _line: int) -> None:
    time_str: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg: str = f"{time_str}, {got}, {requested}"

    _debug_get_next_trials(msg)

@wrapper_print_debug
def print_debug_progressbar(msg: str) -> None:
    time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"{time_str}: {msg}"

    _debug_progressbar(msg)

def receive_usr_signal_one(signum: int, stack: Any) -> None: # pragma: no cover
    raise SignalUSR(f"USR1-signal received ({signum})")

def receive_usr_signal_int(signum: int, stack: Any) -> None: # pragma: no cover
    raise SignalINT(f"INT-signal received ({signum})")

def receive_signal_cont(signum: int, stack: Any) -> None: # pragma: no cover
    raise SignalCONT(f"CONT-signal received ({signum})")

signal.signal(signal.SIGUSR1, receive_usr_signal_one)
signal.signal(signal.SIGUSR2, receive_usr_signal_one)
signal.signal(signal.SIGINT, receive_usr_signal_int)
signal.signal(signal.SIGTERM, receive_usr_signal_int)
signal.signal(signal.SIGCONT, receive_signal_cont)

def is_executable_in_path(executable_name: str) -> bool:
    for path in os.environ.get('PATH', '').split(':'):
        executable_path = os.path.join(path, executable_name)
        if os.path.exists(executable_path) and os.access(executable_path, os.X_OK): # pragma: no cover
            return True
    return False

SYSTEM_HAS_SBATCH: bool = False
IS_NVIDIA_SMI_SYSTEM: bool = False

if is_executable_in_path("sbatch"): # pragma: no cover
    SYSTEM_HAS_SBATCH = True
if is_executable_in_path("nvidia-smi"): # pragma: no cover
    IS_NVIDIA_SMI_SYSTEM = True

if not SYSTEM_HAS_SBATCH:
    num_parallel_jobs = 1

def save_global_vars() -> None:
    state_files_folder = f"{get_current_run_folder()}/state_files"
    makedirs(state_files_folder)

    with open(f'{state_files_folder}/global_vars.json', mode="w", encoding="utf-8") as f:
        json.dump(global_vars, f)

def check_slurm_job_id() -> None:
    print_debug("check_slurm_job_id()")
    if SYSTEM_HAS_SBATCH: # pragma: no cover
        slurm_job_id = os.environ.get('SLURM_JOB_ID')
        if slurm_job_id is not None and not slurm_job_id.isdigit():
            print_red("Not a valid SLURM_JOB_ID.")
        elif slurm_job_id is None:
            print_red(
                "You are on a system that has SLURM available, but you are not running the main-script in a SLURM-Environment. "
                "This may cause the system to slow down for all other users. It is recommended you run the main script in a SLURM-job."
            )

def create_folder_and_file(folder: str) -> str:
    print_debug(f"create_folder_and_file({folder})")

    makedirs(folder)

    file_path = os.path.join(folder, "results.csv")

    return file_path

def sort_numerically_or_alphabetically(arr: list) -> list:
    try:
        new_arr = [float(item) for item in arr]
        arr = new_arr
    except ValueError:
        pass

    sorted_arr = sorted(arr)
    return sorted_arr

def get_program_code_from_out_file(f: str) -> str:
    if not os.path.exists(f):
        print_debug(f"{f} not found")
        original_print(f"{f} not found")
        return ""

    fs = get_file_as_string(f)

    for line in fs.split("\n"):
        if "Program-Code:" in line: # pragma: no cover
            return line

    return ""

def get_min_or_max_column_value(pd_csv: str, column: str, _default: Union[None, int, float], _type: str = "min") -> Optional[float]:
    if not os.path.exists(pd_csv):
        raise FileNotFoundError(f"CSV file {pd_csv} not found")

    try:
        _value = _default

        df = pd.read_csv(pd_csv, float_precision='round_trip')

        if column not in df.columns: # pragma: no cover
            print_red(f"Cannot load data from {pd_csv}: column {column} does not exist. Returning default {_default}")
            return _value

        if _type == "min":
            _value = df[column].min()
        elif _type == "max":
            _value = df[column].max()
        else: # pragma: no cover
            dier(f"get_min_or_max_column_value: Unknown type {_type}")

        return _value
    except Exception as e: # pragma: no cover
        print_red(f"Error while getting {_type} value from column {column}: {str(e)}")
        raise

    return None

@wrapper_print_debug
def get_max_column_value(pd_csv: str, column: str, _default: Union[None, float, int]) -> Optional[float]:
    return get_min_or_max_column_value(pd_csv, column, _default, "max")

@wrapper_print_debug
def get_min_column_value(pd_csv: str, column: str, _default: Union[None, float, int]) -> Optional[float]:
    return get_min_or_max_column_value(pd_csv, column, _default, "min")

@wrapper_print_debug
def get_ret_value_from_pd_csv(pd_csv: str, _type: str, _column: str, _default: Union[None, float, int]) -> Union[Tuple[int, bool], Tuple[float, bool]]:
    found_in_file = False
    if os.path.exists(pd_csv):
        if _type == "lower":
            _old_min_col = get_min_column_value(pd_csv, _column, _default)
            if _old_min_col:
                found_in_file = True

            if found_in_file and _default > _old_min_col: # pragma: no cover
                ret_val = _old_min_col
            else:
                ret_val = _default
        else:
            _old_max_col = get_max_column_value(pd_csv, _column, _default)
            if _old_max_col:
                found_in_file = True

            if found_in_file and _default < _old_max_col: # pragma: no cover
                ret_val = _old_max_col
            else:
                ret_val = _default
    else: # pragma: no cover
        print_red(f"{pd_csv} was not found")

    return ret_val, found_in_file

def get_bound_if_prev_data(_type: str, _column: Union[list, str], _default: Union[float, int]) -> Union[Tuple[float | int, bool], Any]:
    ret_val = _default

    found_in_file = False

    if args.load_previous_job_data and len(args.load_previous_job_data):
        prev_runs = helpers.flatten_extend(args.load_previous_job_data)
        for prev_run in prev_runs:
            pd_csv = f"{prev_run}/{PD_CSV_FILENAME}"

            ret_val, found_in_file = get_ret_value_from_pd_csv(pd_csv, _type, _column, _default)
    if args.continue_previous_job:
        pd_csv = f"{args.continue_previous_job}/{PD_CSV_FILENAME}"

        ret_val, found_in_file = get_ret_value_from_pd_csv(pd_csv, _type, _column, _default)

    if ret_val:
        return round(ret_val, 4), found_in_file

    return ret_val, False

def switch_lower_and_upper_if_needed(name: Union[list, str], lower_bound: Union[float, int], upper_bound: Union[float, int]) -> Tuple[int | float, int | float]:
    if lower_bound > upper_bound:
        print_yellow(f"⚠ Lower bound ({lower_bound}) was larger than upper bound ({upper_bound}) for parameter '{name}'. Switched them.")
        upper_bound, lower_bound = lower_bound, upper_bound

    return lower_bound, upper_bound

def round_lower_and_upper_if_type_is_int(value_type: str, lower_bound: Union[int, float], upper_bound: Union[int, float]) -> Tuple[int | float, int | float]:
    if value_type == "int":
        if not helpers.looks_like_int(lower_bound):
            print_yellow(f"⚠ {value_type} can only contain integers. You chose {lower_bound}. Will be rounded down to {math.floor(lower_bound)}.")
            lower_bound = math.floor(lower_bound)

        if not helpers.looks_like_int(upper_bound):
            print_yellow(f"⚠ {value_type} can only contain integers. You chose {upper_bound}. Will be rounded up to {math.ceil(upper_bound)}.")
            upper_bound = math.ceil(upper_bound)

    return lower_bound, upper_bound

def get_bounds(this_args: Union[str, list], j: int) -> Tuple[float, float]:
    try:
        lower_bound = float(this_args[j + 2])
    except Exception: # pragma: no cover
        print_red(f"\n⚠ {this_args[j + 2]} is not a number")
        my_exit(181)

    try:
        upper_bound = float(this_args[j + 3])
    except Exception:
        print_red(f"\n⚠ {this_args[j + 3]} is not a number")
        my_exit(181)

    return lower_bound, upper_bound

def adjust_bounds_for_value_type(value_type: str, lower_bound: Union[int, float], upper_bound: Union[int, float]) -> Union[Tuple[float, float], Tuple[int, int]]:
    lower_bound, upper_bound = round_lower_and_upper_if_type_is_int(value_type, lower_bound, upper_bound)

    if value_type == "int":
        lower_bound = math.floor(lower_bound)
        upper_bound = math.ceil(upper_bound)

    return lower_bound, upper_bound

def create_param(name: Union[list, str], lower_bound: Union[float, int], upper_bound: Union[float, int], value_type: str, log_scale: bool) -> dict:
    return {
        "name": name,
        "type": "range",
        "bounds": [lower_bound, upper_bound],
        "value_type": value_type,
        "log_scale": log_scale
    }

def handle_grid_search(name: Union[list, str], lower_bound: Union[float, int], upper_bound: Union[float, int], value_type: str) -> dict:
    if lower_bound is None or upper_bound is None: # pragma: no cover
        print_red("handle_grid_search: lower_bound or upper_bound is None")
        my_exit(91)

        return {}

    values = np.linspace(lower_bound, upper_bound, args.max_eval, endpoint=True).tolist()

    if value_type == "int":
        values = [int(value) for value in values]

    values = sorted(set(values))
    values = [str(helpers.to_int_when_possible(value)) for value in values]

    return {
        "name": name,
        "type": "choice",
        "is_ordered": True,
        "values": values
    }

def get_bounds_from_previous_data(name: Union[list, str], lower_bound: Union[float, int], upper_bound: Union[float, int]) -> Tuple[float | int, float | int]:
    lower_bound, _ = get_bound_if_prev_data("lower", name, lower_bound)
    upper_bound, _ = get_bound_if_prev_data("upper", name, upper_bound)
    return lower_bound, upper_bound

def check_bounds_change_due_to_previous_job(name: Union[list, str], lower_bound: Union[float, int], upper_bound: Union[float, int], search_space_reduction_warning: bool) -> bool:
    old_lower_bound = lower_bound
    old_upper_bound = upper_bound

    if args.continue_previous_job:
        if old_lower_bound != lower_bound: # pragma: no cover
            print_yellow(f"⚠ previous jobs contained smaller values for {name}. Lower bound adjusted from {old_lower_bound} to {lower_bound}")
            search_space_reduction_warning = True

        if old_upper_bound != upper_bound: # pragma: no cover
            print_yellow(f"⚠ previous jobs contained larger values for {name}. Upper bound adjusted from {old_upper_bound} to {upper_bound}")
            search_space_reduction_warning = True

    return search_space_reduction_warning

def get_value_type_and_log_scale(this_args: Union[str, list], j: int) -> Tuple[int, str, bool]:
    skip = 5
    try:
        value_type = this_args[j + 4]
    except Exception: # pragma: no cover
        value_type = "float"
        skip = 4

    try:
        log_scale = this_args[j + 5].lower() == "true"
    except Exception:
        log_scale = False
        skip = 5

    return skip, value_type, log_scale

def parse_range_param(params: list, j: int, this_args: Union[str, list], name: Union[list, str], search_space_reduction_warning: bool) -> Tuple[int, list, bool]:
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

def validate_value_type(value_type: str) -> None:
    valid_value_types = ["int", "float"]
    check_if_range_types_are_invalid(value_type, valid_value_types)

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

def parse_choice_param(params: list, j: int, this_args: Union[str, list], name: Union[list, str], search_space_reduction_warning: bool) -> Tuple[int, list, bool]:
    if len(this_args) != 3:
        print_red("⚠ --parameter for type choice must have 3 parameters: <NAME> choice <VALUE,VALUE,VALUE,...>")
        my_exit(181)

    values = re.split(r'\s*,\s*', str(this_args[j + 2]))

    values[:] = [x for x in values if x != ""]

    values = sort_numerically_or_alphabetically(values)

    param = {
        "name": name,
        "type": "choice",
        "is_ordered": True,
        "values": values
    }

    global_vars["parameter_names"].append(name)

    params.append(param)

    j += 3

    return j, params, search_space_reduction_warning

@wrapper_print_debug
def parse_experiment_parameters() -> list:
    global global_vars
    global changed_grid_search_params

    params: list = []
    param_names: list[str] = []

    i = 0

    search_space_reduction_warning = False

    valid_types = ["range", "fixed", "choice"]
    invalid_names = ["start_time", "end_time", "run_time", "program_string", "result", "exit_code", "signal"]

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
            except Exception: # pragma: no cover
                print_red("Not enough arguments for --parameter")
                my_exit(181)

            if param_type not in valid_types: # pragma: no cover
                valid_types_string = ', '.join(valid_types)
                print_red(f"\n⚠ Invalid type {param_type}, valid types are: {valid_types_string}")
                my_exit(181)

            if param_type == "range":
                j, params, search_space_reduction_warning = parse_range_param(params, j, this_args, name, search_space_reduction_warning)
            elif param_type == "fixed":
                j, params, search_space_reduction_warning = parse_fixed_param(params, j, this_args, name, search_space_reduction_warning)
            elif param_type == "choice":
                j, params, search_space_reduction_warning = parse_choice_param(params, j, this_args, name, search_space_reduction_warning)
            else: # pragma: no cover
                print_red(f"⚠ Parameter type '{param_type}' not yet implemented.")
                my_exit(181)
        i += 1

    if search_space_reduction_warning: # pragma: no cover
        print_red("⚠ Search space reduction is not currently supported on continued runs or runs that have previous data.")

    return params

def check_factorial_range() -> None: # pragma: no cover
    if args.model and args.model == "FACTORIAL":
        print_red("\n⚠ --model FACTORIAL cannot be used with range parameter")
        my_exit(181)

def check_if_range_types_are_invalid(value_type: str, valid_value_types: list) -> None:
    if value_type not in valid_value_types:
        valid_value_types_string = ", ".join(valid_value_types)
        print_red(f"⚠ {value_type} is not a valid value type. Valid types for range are: {valid_value_types_string}")
        my_exit(181)

def check_range_params_length(this_args: Union[str, list]) -> None:
    if len(this_args) != 5 and len(this_args) != 4 and len(this_args) != 6:
        print_red("\n⚠ --parameter for type range must have 4 (or 5, the last one being optional and float by default, or 6, while the last one is true or false) parameters: <NAME> range <START> <END> (<TYPE (int or float)>, <log_scale: bool>)")
        my_exit(181)

def die_181_or_91_if_lower_and_upper_bound_equal_zero(lower_bound: Union[int, float], upper_bound: Union[int, float]) -> None:
    if upper_bound is None or lower_bound is None: # pragma: no cover
        print_red("die_181_or_91_if_lower_and_upper_bound_equal_zero: upper_bound or lower_bound is None. Cannot continue.")
        my_exit(91)
    if upper_bound == lower_bound:
        if lower_bound == 0:
            print_red(f"⚠ Lower bound and upper bound are equal: {lower_bound}, cannot automatically fix this, because they -0 = +0 (usually a quickfix would be to set lower_bound = -upper_bound)")
            my_exit(181)
        print_red(f"⚠ Lower bound and upper bound are equal: {lower_bound}, setting lower_bound = -upper_bound") # pragma: no cover
        if upper_bound is not None: # pragma: no cover
            lower_bound = -upper_bound

def replace_parameters_in_string(parameters: dict, input_string: str) -> str:
    try:
        for param_item in parameters:
            input_string = input_string.replace(f"${param_item}", str(parameters[param_item]))
            input_string = input_string.replace(f"$({param_item})", str(parameters[param_item]))

            input_string = input_string.replace(f"%{param_item}", str(parameters[param_item]))
            input_string = input_string.replace(f"%({param_item})", str(parameters[param_item]))

        return input_string
    except Exception as e: # pragma: no cover
        print_red(f"\n⚠ Error: {e}")
        return ""

def execute_bash_code(code: str) -> list:
    try:
        result = subprocess.run(
            code,
            shell=True,
            check=True,
            text=True,
            capture_output=True
        )

        if result.returncode != 0: # pragma: no cover
            print(f"Exit-Code: {result.returncode}")

        real_exit_code = result.returncode

        signal_code = None
        if real_exit_code < 0: # pragma: no cover
            signal_code = abs(result.returncode)
            real_exit_code = 1

        return [result.stdout, result.stderr, real_exit_code, signal_code]

    except subprocess.CalledProcessError as e:
        real_exit_code = e.returncode

        signal_code = None
        if real_exit_code < 0: # pragma: no cover
            signal_code = abs(e.returncode)
            real_exit_code = 1

        if not args.tests: # pragma: no cover
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

def get_result(input_string: str) -> Optional[list[float]]:
    if input_string is None:
        print_red("get_result: Input-String is None")
        return None

    if not isinstance(input_string, str):
        print_red(f"get_result: Type of input_string is not string, but {type(input_string)}")
        return None

    try:
        _pattern: str = r'\s*RESULT\d*:\s*(-?\d+(?:\.\d+)?)'

        # Find all matches for the _pattern
        matches = re.findall(_pattern, input_string)

        if matches:
            # Convert matches to floats
            result_numbers = [float(match) for match in matches]
            return result_numbers  # Return list if multiple results are found
        return None
    except Exception as e: # pragma: no cover
        print_red(f"Error extracting the RESULT-string: {e}")
        return None

def add_to_csv(file_path: str, heading: list, data_line: list) -> None: # pragma: no cover
    print_debug(f"add_to_csv({file_path}, {heading}, {data_line})")
    is_empty = os.path.getsize(file_path) == 0 if os.path.exists(file_path) else True

    data_line = [helpers.to_int_when_possible(x) for x in data_line]

    with open(file_path, 'a+', encoding="utf-8", newline='') as file:
        csv_writer = csv.writer(file)

        if is_empty:
            csv_writer.writerow(heading)

        # desc += " (best loss: " + '{:f}'.format(best_result) + ")"
        data_line = ["{:.20f}".format(x) if isinstance(x, float) else x for x in data_line]
        csv_writer.writerow(data_line)

def find_file_paths(_text: str) -> list[str]:
    file_paths = []

    if isinstance(_text, str):
        words = _text.split()

        for word in words:
            if os.path.exists(word):
                file_paths.append(word)

        return file_paths

    return [] # pragma: no cover

def check_file_info(file_path: str) -> str:
    if not os.path.exists(file_path):
        print(f"check_file_info: The file {file_path} does not exist.")
        return ""

    if not os.access(file_path, os.R_OK): # pragma: no cover
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
    if _SLURM_JOB_ID is not None and _SLURM_JOB_ID is not False and _SLURM_JOB_ID != "": # pragma: no cover
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

    string += "Program-Code: " + program_code
    if file_paths:
        for file_path in file_paths:
            string += "\n"
            string += check_file_info(file_path)
    string += "\n========\nDEBUG INFOS END\n"

    return string

def write_failed_logs(data_dict: dict, error_description: str = "") -> None:
    assert isinstance(data_dict, dict), "The parameter must be a dictionary."
    assert isinstance(error_description, str), "The error_description must be a string."

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
            except Exception as e: # pragma: no cover
                print_red(f"Failed to write header file: {e}")

        # Append data to the data file
        try:
            with open(data_file_path, mode='a', encoding="utf-8", newline='') as data_file:
                writer = csv.writer(data_file)
                writer.writerows(data)
                print_debug(f"Data appended to file: {data_file_path}")

        except Exception as e: # pragma: no cover
            print_red(f"Failed to append data to file: {e}")

    except Exception as e: # pragma: no cover
        print_red(f"Unexpected error: {e}")

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

            if entry is not None and entry not in entries: # pragma: no cover
                file.write(entry + '\n')
                entries.append(entry)

        return sorted(set(entries))

    except Exception as e: # pragma: no cover
        print(f"An error has occurred: {e}")
        return []

def test_gpu_before_evaluate(return_in_case_of_error: dict) -> Union[None, dict]: # pragma: no cover
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

def extract_info(data: str) -> Tuple[list[str], list[str]]:
    names: list[str] = []
    values: list[str] = []

    # Regex-Muster für OO-Info, das sowohl Groß- als auch Kleinschreibung berücksichtigt
    _pattern = re.compile(r'\s*OO-Info:\s*([a-zA-Z0-9_]+):\s*(.+)\s*$', re.IGNORECASE)

    # Gehe durch jede Zeile im String
    for line in data.splitlines():
        match = _pattern.search(line)
        if match:
            names.append("OO_Info_" + match.group(1))
            values.append(match.group(2))

    return names, values

def ignore_signals() -> None:
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    signal.signal(signal.SIGUSR2, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGQUIT, signal.SIG_IGN)

def calculate_signed_harmonic_distance(_args: list[float]) -> float:
    if not _args or len(_args) == 0: # Handle empty input gracefully
        return 0

    abs_inverse_sum: float = sum(1 / abs(a) for a in _args if a != 0)  # Avoid division by zero
    harmonic_mean: float = len(_args) / abs_inverse_sum if abs_inverse_sum != 0 else 0

    # Determine the sign based on the number of negatives
    num_negatives: float = sum(1 for a in _args if a < 0)
    sign: int = -1 if num_negatives % 2 != 0 else 1

    return sign * harmonic_mean

def calculate_signed_euclidean_distance(_args: list[float]) -> float:
    _sum: float = 0
    for a in _args:
        _sum += a ** 2

    # Behalte das Vorzeichen des ersten Werts (oder ein beliebiges anderes Kriterium)
    sign: int = -1 if any(a < 0 for a in _args) else 1
    return sign * math.sqrt(_sum)

def calculate_signed_geometric_distance(_args: list[float]) -> float:
    product: float = 1  # Startwert für Multiplikation
    for a in _args:
        product *= abs(a)  # Absolutwerte für das Produkt verwenden

    # Behalte das Vorzeichen basierend auf der Anzahl negativer Werte
    num_negatives: float = sum(1 for a in _args if a < 0)
    sign: int = -1 if num_negatives % 2 != 0 else 1

    # Geometrisches Mittel: n-te Wurzel des Produkts
    geometric_mean: float = product ** (1 / len(_args)) if _args else 0
    return sign * geometric_mean

class invalidMooType(Exception):
    pass

def calculate_moo(_args: Optional[list[float]]) -> float:
    if _args is None or len(_args) == 0:
        return VAL_IF_NOTHING_FOUND

    if args.moo_type == "euclid":
        return calculate_signed_euclidean_distance(_args)
    if args.moo_type == "geometric": # pragma: no cover
        return calculate_signed_geometric_distance(_args)
    if args.moo_type == "signed_harmonic": # pragma: no cover
        return calculate_signed_harmonic_distance(_args)

    raise invalidMooType(f"Invalid moo type {args.moo_type}. Valid types are: {', '.join(valid_moo_types)}") # pragma: no cover

def evaluate(parameters: dict) -> dict:
    start_nvidia_smi_thread()

    return_in_case_of_error: dict = {"result": VAL_IF_NOTHING_FOUND}

    if args.maximize:
        return_in_case_of_error = {"result": -VAL_IF_NOTHING_FOUND}

    _test_gpu = test_gpu_before_evaluate(return_in_case_of_error)

    if _test_gpu is not None:
        return _test_gpu

    parameters = {k: (int(v) if isinstance(v, (int, float, str)) and re.fullmatch(r'^\d+(\.0+)?$', str(v)) else v) for k, v in parameters.items()}

    ignore_signals()

    try:
        if args.raise_in_eval: # pragma: no cover
            raise SignalUSR("Raised in eval")
        original_print(f"Parameters: {json.dumps(parameters)}")

        parameters_keys = list(parameters.keys())
        parameters_values = list(parameters.values())

        program_string_with_params: str = replace_parameters_in_string(parameters, global_vars["joined_run_program"])

        program_string_with_params = program_string_with_params.replace('\r', ' ').replace('\n', ' ')

        string = find_file_paths_and_print_infos(program_string_with_params, program_string_with_params)

        original_print("Debug-Infos:", string)

        original_print(program_string_with_params)

        start_time: float = int(time.time())

        stdout_stderr_exit_code_signal = execute_bash_code(program_string_with_params)

        end_time: float = int(time.time())

        stdout = stdout_stderr_exit_code_signal[0]
        exit_code = stdout_stderr_exit_code_signal[2]
        _signal = stdout_stderr_exit_code_signal[3]

        run_time: float = end_time - start_time

        original_print("stdout:")
        original_print(stdout)

        result = get_result(stdout)

        unmooed_result = result

        all_result_column_names = []

        _k = 1
        if unmooed_result:
            for a in unmooed_result:
                all_result_column_names.append(f"RESULT{_k}")
                _k = _k + 1

            mooed_result = calculate_moo(result)

            if mooed_result is not None:
                result = [mooed_result]

        extra_vars_names, extra_vars_values = extract_info(stdout)

        _SLURM_JOB_ID = os.getenv('SLURM_JOB_ID')
        if _SLURM_JOB_ID: # pragma: no cover
            extra_vars_names.append("OO_Info_SLURM_JOB_ID")
            extra_vars_values.append(str(_SLURM_JOB_ID))

        original_print(f"Result: {result}")

        str_parameters_values: list[str] = [str(v) for v in parameters_values]

        headline: list[str] = [
            "start_time",
            "end_time",
            "run_time",
            "program_string",
            *parameters_keys,
            "result",
            "exit_code",
            "signal",
            "hostname",
            *extra_vars_names,
            *all_result_column_names
        ]

        unmooed_result_str = []
        if unmooed_result:
            for v in unmooed_result:
                if v:
                    unmooed_result_str.append(str(v))

        values: list[str] = [
            str(start_time),
            str(end_time),
            str(run_time),
            program_string_with_params,
            *str_parameters_values,
            str(result),
            str(exit_code),
            str(_signal),
            socket.gethostname(),
            *extra_vars_values,
            *unmooed_result_str
        ]

        original_print(f"EXIT_CODE: {exit_code}")

        headline = ['None' if element is None else element for element in headline]
        values = ['None' if element is None else element for element in values]

        if get_current_run_folder() is not None and os.path.exists(get_current_run_folder()): # pragma: no cover
            add_to_csv(f"{get_current_run_folder()}/job_infos.csv", headline, values)
        else:
            print_debug(f"evaluate: get_current_run_folder() {get_current_run_folder()} could not be found")

        if isinstance(result, (int, float)): # pragma: no cover
            return {"result": float(result)}
        if isinstance(result, (list)) and len(result) == 1:
            return {"result": float(result[0])}
        if isinstance(result, (list)): # pragma: no cover
            return {"result": [float(r) for r in result]}

        write_failed_logs(parameters, "No Result") # pragma: no cover
    except SignalUSR: # pragma: no cover
        print("\n⚠ USR1-Signal was sent. Cancelling evaluation.")
        write_failed_logs(parameters, "USR1-signal")
    except SignalCONT: # pragma: no cover
        print("\n⚠ CONT-Signal was sent. Cancelling evaluation.")
        write_failed_logs(parameters, "CONT-signal")
    except SignalINT: # pragma: no cover
        print("\n⚠ INT-Signal was sent. Cancelling evaluation.")
        write_failed_logs(parameters, "INT-signal")

    return return_in_case_of_error # pragma: no cover

class NpEncoder(json.JSONEncoder):
    def default(self: Any, obj: Any) -> Union[int, float, list]: # pragma: no cover
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def custom_warning_handler(
    message: Warning | str,
    category: type[Warning],
    filename: str,
    lineno: int,
    file: TextIO | None = None,
    line: str | None = None
) -> None: # pragma: no cover
    warning_message = f"{category.__name__}: {message} (in {filename}, line {lineno})"
    print_debug(f"{file}:{line}: {warning_message}")

def disable_logging() -> None:
    if args.verbose: # pragma: no cover
        return

    logging.basicConfig(level=logging.CRITICAL)
    logging.getLogger().setLevel(logging.CRITICAL)
    logging.getLogger().disabled = True

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

        "ax.service",
        "ax.service.utils",
        "ax.service.utils.instantiation",
        "ax.service.utils.report_utils",
        "ax.service.utils.best_point",

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

    for cat in categories:
        warnings.filterwarnings("ignore", category=cat)
        for module in modules:
            warnings.filterwarnings("ignore", category=cat, module=module)

    warnings.showwarning = custom_warning_handler

@wrapper_print_debug
def display_failed_jobs_table() -> None:
    _console = Console()

    failed_jobs_file = f"{get_current_run_folder()}/failed_logs"
    header_file = os.path.join(failed_jobs_file, "headers.csv")
    parameters_file = os.path.join(failed_jobs_file, "parameters.csv")

    # Assert the existence of the folder and files
    if not os.path.exists(failed_jobs_file): # pragma: no cover
        print_debug(f"Failed jobs {failed_jobs_file} file does not exist.")
        return

    if not os.path.isfile(header_file): # pragma: no cover
        print_debug(f"Failed jobs Header file ({header_file}) does not exist.")
        return

    if not os.path.isfile(parameters_file): # pragma: no cover
        print_debug(f"Failed jobs Parameters file ({parameters_file}) does not exist.")
        return

    try:
        with open(header_file, mode='r', encoding="utf-8") as file:
            reader = csv.reader(file)
            headers = next(reader)
            print_debug(f"Headers: {headers}")

        with open(parameters_file, mode='r', encoding="utf-8") as file:
            reader = csv.reader(file)
            parameters = [row for row in reader]
            print_debug(f"Parameters: {parameters}")

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
        _console.print(table)
    except Exception as e: # pragma: no cover
        print_red(f"Error: {str(e)}")

@wrapper_print_debug
def plot_command(_command: str, tmp_file: str, _width: str = "1300") -> None: # pragma: no cover
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

@wrapper_print_debug
def replace_string_with_params(input_string: str, params: list) -> str:
    try:
        assert isinstance(input_string, str), "Input string must be a string"
        replaced_string = input_string
        i = 0
        for param in params:
            #print(f"param: {param}, type: {type(param)}")
            replaced_string = replaced_string.replace(f"%{i}", str(param))
            i += 1
        return replaced_string
    except AssertionError as e: # pragma: no cover
        error_text = f"Error in replace_string_with_params: {e}"
        print(error_text)
        raise

    return ""

@wrapper_print_debug
def get_best_params_from_csv(csv_file_path: str, maximize: bool) -> dict:
    results: dict = {
        "result": None,
        "parameters": {}
    }

    if not os.path.exists(csv_file_path): # pragma: no cover
        return results

    df = None

    try:
        df = pd.read_csv(csv_file_path, index_col=0, float_precision='round_trip')
        df.dropna(subset=["result"], inplace=True)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError, KeyError):
        return results

    cols = df.columns.tolist()
    nparray = df.to_numpy()

    best_line = None

    result_idx = cols.index("result")

    best_result = None

    for i in range(0, len(nparray)):
        this_line = nparray[i]
        this_line_result = this_line[result_idx]

        if isinstance(this_line_result, str) and re.match(r'^-?\d+(?:\.\d+)$', this_line_result) is not None: # pragma: no cover
            this_line_result = float(this_line_result)

        if type(this_line_result) in [float, int]:
            if best_result is None:
                if this_line is not None and len(this_line) > 0:
                    best_line = this_line
                    best_result = this_line_result

            if maximize:
                if this_line_result >= best_result:
                    best_line = this_line
                    best_result = this_line_result
            else:
                if this_line_result <= best_result:
                    best_line = this_line
                    best_result = this_line_result

    if best_line is None: # pragma: no cover
        print_debug("Could not determine best result")
        return results

    for i in range(0, len(cols)):
        col = cols[i]
        if col not in [
            "start_time",
            "end_time",
            "hostname",
            "signal",
            "exit_code",
            "run_time",
            "program_string"
        ]:
            if col == "result":
                results["result"] = repr(best_line[i]) if type(best_line[i]) in [int, float] else best_line[i]
            else:
                results["parameters"][col] = repr(best_line[i]) if type(best_line[i]) in [int, float] else best_line[i]

    return results

def get_best_params() -> dict:
    csv_file_path: str = save_pd_csv()

    return get_best_params_from_csv(csv_file_path, args.maximize)

def _count_sobol_or_completed(csv_file_path: str, _type: str) -> int:
    if _type not in ["Sobol", "COMPLETED"]:
        print_red(f"_type is not in Sobol or COMPLETED, but is {_type}")
        return 0

    count = 0

    if not os.path.exists(csv_file_path):
        print_debug(f"_count_sobol_or_completed: path '{csv_file_path}' not found")
        return count

    df = None

    _err = False

    try:
        df = pd.read_csv(csv_file_path, index_col=0, float_precision='round_trip')
        df.dropna(subset=["result"], inplace=True)
    except KeyError: # pragma: no cover
        _err = True
    except pd.errors.EmptyDataError: # pragma: no cover
        _err = True
    except pd.errors.ParserError as e: # pragma: no cover
        print_red(f"Error reading CSV file 2: {str(e)}")
        _err = True
    except UnicodeDecodeError as e: # pragma: no cover
        print_red(f"Error reading CSV file 3: {str(e)}")
        _err = True
    except Exception as e: # pragma: no cover
        print_red(f"Error reading CSV file 4: {str(e)}")
        _err = True

    if _err:
        return 0

    assert df is not None, "DataFrame should not be None after reading CSV file"
    assert "generation_method" in df.columns, "'generation_method' column must be present in the DataFrame"

    if _type == "Sobol":
        rows = df[df["generation_method"] == _type] # pragma: no cover
    else:
        rows = df[df["trial_status"] == _type]
    count = len(rows)

    return count

def _count_sobol_steps(csv_file_path: str) -> int:
    return _count_sobol_or_completed(csv_file_path, "Sobol")

def _count_done_jobs(csv_file_path: str) -> int:
    return _count_sobol_or_completed(csv_file_path, "COMPLETED")

def count_sobol_steps() -> int:
    csv_file_path: str = save_pd_csv()

    return _count_sobol_steps(csv_file_path)

def get_random_steps_from_prev_job() -> int:
    if not args.continue_previous_job:
        return count_sobol_steps()

    prev_step_file: str = args.continue_previous_job + "/state_files/phase_random_steps"

    if not os.path.exists(prev_step_file):
        return count_sobol_steps()

    return add_to_phase_counter("random", count_sobol_steps() + _count_sobol_steps(f"{args.continue_previous_job}/results.csv"), args.continue_previous_job) # pragma: no cover

def failed_jobs(nr: int = 0) -> int:
    state_files_folder = f"{get_current_run_folder()}/state_files/"

    makedirs(state_files_folder)

    return append_and_read(f'{get_current_run_folder()}/state_files/failed_jobs', nr)

def count_done_jobs() -> int:
    csv_file_path: str = save_pd_csv()

    return _count_done_jobs(csv_file_path)

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

@wrapper_print_debug
def get_sixel_graphics_data(_pd_csv: str, _force: bool = False) -> list:
    _show_sixel_graphics = args.show_sixel_scatter or args.show_sixel_general or args.show_sixel_scatter or args.show_sixel_trial_index_result

    if _force:
        _show_sixel_graphics = True

    data: list = []

    if not os.path.exists(_pd_csv):
        print_debug(f"Cannot find path {_pd_csv}")
        return data

    if not _show_sixel_graphics: # pragma: no cover
        print_debug("_show_sixel_graphics was false. Will not plot.")
        return data

    if len(global_vars["parameter_names"]) == 0: # pragma: no cover
        print_debug("Cannot handle empty data in global_vars -> parameter_names")
        return data

    x_y_combinations = list(combinations(global_vars["parameter_names"], 2))

    plot_types = get_plot_types(x_y_combinations, _force)

    for plot in plot_types:
        plot_type = plot["type"]
        min_done_jobs = 1

        if "min_done_jobs" in plot:
            min_done_jobs = plot["min_done_jobs"]

        if not _force and count_done_jobs() < min_done_jobs:
            print_debug(f"Cannot plot {plot_type}, because it needs {min_done_jobs}, but you only have {count_done_jobs()} jobs done")
            continue

        try:
            _tmp = f"{get_current_run_folder()}/plots/"
            _width = "1200"

            if "width" in plot: # pragma: no cover
                _width = plot["width"]

            if not _force and not os.path.exists(_tmp): # pragma: no cover
                makedirs(_tmp)

            j = 0
            _fn = plot_type

            if "filename" in plot:
                _fn = plot['filename']

            tmp_file = f"{_tmp}/{_fn}.png"

            while os.path.exists(tmp_file): # pragma: no cover
                j += 1
                tmp_file = f"{_tmp}/{_fn}_{j}.png"

            maindir = os.path.dirname(os.path.realpath(__file__))

            if _force:
                _command = f"bash omniopt_plot --run_dir {get_current_run_folder()} --plot_type={plot_type}"
            else: # pragma: no cover
                _command = f"bash {maindir}/omniopt_plot --run_dir {get_current_run_folder()} --plot_type={plot_type}"

            if "dpi" in plot:
                _command += " --dpi=" + str(plot["dpi"])

            _params = [_command, plot, _tmp, plot_type, tmp_file, _width]
            data.append(_params)
        except Exception as e: # pragma: no cover
            tb = traceback.format_exc()
            print_red(f"Error trying to print {plot_type} to to CLI: {e}, {tb}")
            print_debug(f"Error trying to print {plot_type} to to CLI: {e}")

    return data

def get_plot_commands(_command: str, plot: dict, _tmp: str, plot_type: str, tmp_file: str, _width: str) -> list[list[str]]:
    plot_commands: list[list[str]] = []
    if "params" in plot.keys():
        if "iterate_through" in plot.keys():
            iterate_through = plot["iterate_through"]
            if len(iterate_through):
                for j in range(0, len(iterate_through)):
                    this_iteration = iterate_through[j]
                    _iterated_command: str = _command + " " + replace_string_with_params(plot["params"], [this_iteration[0], this_iteration[1]])

                    j = 0
                    tmp_file = f"{_tmp}/{plot_type}.png"
                    _fn = ""
                    if "filename" in plot:
                        _fn = plot['filename']
                        if len(this_iteration):
                            _p = [plot_type, this_iteration[0], this_iteration[1]]
                            if len(_p):
                                tmp_file = f"{_tmp}/{replace_string_with_params(_fn, _p)}.png"

                            while os.path.exists(tmp_file): # pragma: no cover
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

def plot_sixel_imgs(csv_file_path: str) -> None:
    if ci_env:
        print("Not printing sixel graphics in CI")
        return

    sixel_graphic_commands = get_sixel_graphics_data(csv_file_path) # pragma: no cover

    for c in sixel_graphic_commands: # pragma: no cover
        commands = get_plot_commands(*c)

        for command in commands:
            plot_command(*command)

def _print_best_result(csv_file_path: str, maximize: bool, print_to_file: bool = True) -> int:
    global global_vars
    global SHOWN_END_TABLE

    try:
        best_params = get_best_params_from_csv(csv_file_path, maximize)

        best_result = None

        if best_params and "result" in best_params:
            best_result = best_params["result"]
        else: # pragma: no cover
            best_result = NO_RESULT

        if str(best_result) == NO_RESULT or best_result is None or best_result == "None":
            print_red("Best result could not be determined")
            return 87

        total_str = f"total: {_count_done_jobs(csv_file_path) - NR_INSERTED_JOBS}"

        if NR_INSERTED_JOBS:
            total_str += f" + inserted jobs: {NR_INSERTED_JOBS}"

        failed_error_str = ""
        if print_to_file:
            if failed_jobs() >= 1:
                failed_error_str = f", failed: {failed_jobs()}"

        table = Table(show_header=True, header_style="bold", title=f"Best result ({total_str}{failed_error_str}):")

        k = 0
        for key in best_params["parameters"].keys():
            if k > 2:
                table.add_column(key)
            k += 1

        table.add_column("result")

        row_without_result = [str(helpers.to_int_when_possible(best_params["parameters"][key])) for key in best_params["parameters"].keys()]
        row = [*row_without_result, str(helpers.to_int_when_possible(best_result))][3:]

        table.add_row(*row)

        console.print(table)

        with console.capture() as capture:
            console.print(table)

        if print_to_file:
            table_str = capture.get()
            with open(f'{get_current_run_folder()}/best_result.txt', mode="w", encoding="utf-8") as text_file:
                text_file.write(table_str)

        plot_sixel_imgs(csv_file_path)

        SHOWN_END_TABLE = True
    except Exception as e: # pragma: no cover
        tb = traceback.format_exc()
        print_red(f"[_print_best_result] Error during _print_best_result: {e}, tb: {tb}")

    return -1

def print_best_result() -> int:
    csv_file_path = save_pd_csv()

    return _print_best_result(csv_file_path, args.maximize, True)

@wrapper_print_debug
def show_end_table_and_save_end_files(csv_file_path: str) -> int:
    print_debug(f"show_end_table_and_save_end_files({csv_file_path})")

    ignore_signals()

    global ALREADY_SHOWN_WORKER_USAGE_OVER_TIME
    global global_vars

    if SHOWN_END_TABLE: # pragma: no cover
        print("End table already shown, not doing it again")
        return -1

    _exit: int = 0

    display_failed_jobs_table()

    best_result_exit: int = print_best_result()

    if best_result_exit > 0:
        _exit = best_result_exit

    if args.show_worker_percentage_table_at_end and len(WORKER_PERCENTAGE_USAGE) and not ALREADY_SHOWN_WORKER_USAGE_OVER_TIME: # pragma: no cover
        ALREADY_SHOWN_WORKER_USAGE_OVER_TIME = True

        table = Table(header_style="bold", title="Worker usage over time:")
        columns = ["Time", "Nr. workers", "Max. nr. workers", "%"]
        for column in columns:
            table.add_column(column)
        for row in WORKER_PERCENTAGE_USAGE:
            table.add_row(str(row["time"]), str(row["nr_current_workers"]), str(row["num_parallel_jobs"]), f'{row["percentage"]}%')
        console.print(table)

    return _exit

def abandon_job(job: Job, trial_index: int) -> bool: # pragma: no cover
    global global_vars

    if job:
        try:
            if ax_client:
                _trial = ax_client.get_trial(trial_index)
                _trial.mark_abandoned()
                global_vars["jobs"].remove((job, trial_index))
            else:
                print_red("ax_client could not be found")
                my_exit(9)
        except Exception as e: # pragma: no cover
            print(f"ERROR in line {get_line_info()}: {e}")
            print_debug(f"ERROR in line {get_line_info()}: {e}")
            return False
        job.cancel()
        return True

    return False

def abandon_all_jobs() -> None: # pragma: no cover
    for job, trial_index in global_vars["jobs"][:]:
        abandoned = abandon_job(job, trial_index)
        if not abandoned:
            print_debug(f"Job {job} could not be abandoned.")

def end_program(csv_file_path: str, _force: Optional[bool] = False, exit_code: Optional[int] = None) -> None:
    global global_vars, END_PROGRAM_RAN

    if os.getpid() != main_pid: # pragma: no cover
        print_debug("returning from end_program, because it can only run in the main thread, not any forks")
        return

    if END_PROGRAM_RAN and not _force: # pragma: no cover
        print_debug("[end_program] END_PROGRAM_RAN was true. Returning.")
        return

    END_PROGRAM_RAN = True

    _exit: int = 0

    try:
        if get_current_run_folder() is None: # pragma: no cover
            print_debug("[end_program] get_current_run_folder() was empty. Not running end-algorithm.")
            return

        if ax_client is None: # pragma: no cover
            print_debug("[end_program] ax_client was empty. Not running end-algorithm.")
            return

        if console is None: # pragma: no cover
            print_debug("[end_program] console was empty. Not running end-algorithm.")
            return

        new_exit = show_end_table_and_save_end_files(csv_file_path)
        if new_exit > 0:
            _exit = new_exit
    except (SignalUSR, SignalINT, SignalCONT, KeyboardInterrupt): # pragma: no cover
        print_red("\n⚠ You pressed CTRL+C or a signal was sent. Program execution halted.")
        print("\n⚠ KeyboardInterrupt signal was sent. Ending program will still run.")
        new_exit = show_end_table_and_save_end_files(csv_file_path)
        if new_exit > 0:
            _exit = new_exit
    except TypeError as e: # pragma: no cover
        print_red(f"\n⚠ The program has been halted without attaining any results. Error: {e}")

    abandon_all_jobs()

    save_pd_csv()

    if exit_code:
        _exit = exit_code

    live_share()

    my_exit(_exit)

def save_checkpoint(trial_nr: int = 0, eee: Union[None, str, Exception] = None) -> None:
    if trial_nr > 3: # pragma: no cover
        if eee:
            print("Error during saving checkpoint: " + str(eee))
        else:
            print("Error during saving checkpoint")
        return

    try:
        state_files_folder = f"{get_current_run_folder()}/state_files/"

        makedirs(state_files_folder)

        checkpoint_filepath = f'{state_files_folder}/checkpoint.json'
        if ax_client:
            ax_client.save_to_json_file(filepath=checkpoint_filepath)
        else: # pragma: no cover
            print_red("Something went wrong using the ax_client")
            my_exit(9)
    except Exception as e: # pragma: no cover
        save_checkpoint(trial_nr + 1, e)

@wrapper_print_debug
def get_tmp_file_from_json(experiment_args: dict) -> str:
    _tmp_dir = "/tmp"
    k = 0
    while os.path.exists(f"/{_tmp_dir}/{k}"): # pragma: no cover
        k = k + 1

    try:
        with open(f'/{_tmp_dir}/{k}', mode="w", encoding="utf-8") as f:
            json.dump(experiment_args, f)
    except PermissionError as e: # pragma: no cover
        print_red(f"Error writing '{k}' in get_tmp_file_from_json: {e}")

    return f"/{_tmp_dir}/{k}"

@wrapper_print_debug
def compare_parameters(old_param_json: str, new_param_json: str) -> str:
    try:
        old_param = json.loads(old_param_json)
        new_param = json.loads(new_param_json)

        differences = []
        for key in old_param:
            if old_param[key] != new_param[key]:
                differences.append(f"{key} from {old_param[key]} to {new_param[key]}")

        if differences:
            differences_message = f"Changed parameter {old_param['name']} " + ", ".join(differences)
            return differences_message

        return "No differences found between the old and new parameters." # pragma: no cover

    except AssertionError as e: # pragma: no cover
        print(f"Assertion error: {e}")
    except Exception as e: # pragma: no cover
        print(f"Unexpected error: {e}")

    return ""

def get_ax_param_representation(data: dict) -> dict:
    if data["type"] == "range":
        return {
            "__type": "RangeParameter",
            "name": data["name"],
            "parameter_type": {
                "__type": "ParameterType", "name": data["value_type"].upper()
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
        return {
            '__type': 'ChoiceParameter',
            'dependents': None,
            'is_fidelity': False,
            'is_ordered': data["is_ordered"],
            'is_task': False,
            'name': data["name"],
            'parameter_type': {'__type': 'ParameterType', 'name': 'STRING'},
            'target_value': None,
            'values': data["values"]
        }

    print("data:") # pragma: no cover
    pprint(data) # pragma: no cover
    print_red(f"Unknown data range {data['type']}") # pragma: no cover
    my_exit(19) # pragma: no cover

    # only for linter, never reached because of die
    return {} # pragma: no cover

def set_torch_device_to_experiment_args(experiment_args: Union[None, dict]) -> dict:
    torch_device = None
    try:
        cuda_is_available = torch.cuda.is_available()

        if not cuda_is_available or cuda_is_available == 0:
            print_yellow("No CUDA devices found. This means, the generation of new evaluation points will not be accelerated by a GPU.")
        else:
            if torch.cuda.device_count() >= 1: # pragma: no cover
                torch_device = torch.cuda.current_device()
                print_yellow(f"Using CUDA device {torch.cuda.get_device_name(0)}")
            else: # pragma: no cover
                print_yellow("No CUDA devices found. This means, the generation of new evaluation points will not be accelerated by a GPU.")
    except ModuleNotFoundError:
        print_red("Cannot load torch and thus, cannot use gpus")

    if torch_device: # pragma: no cover
        if experiment_args:
            experiment_args["choose_generation_strategy_kwargs"]["torch_device"] = torch_device
        else:
            print_red("experiment_args could not be created.")
            my_exit(90)

    if experiment_args:
        return experiment_args

    return {}

def die_with_47_if_file_doesnt_exists(_file: str) -> None:
    if not os.path.exists(_file): # pragma: no cover
        print_red(f"Cannot find {_file}")
        my_exit(47)

@wrapper_print_debug
def copy_state_files_from_previous_job(continue_previous_job: str) -> None:
    for state_file in ["submitted_jobs"]:
        old_state_file = f"{continue_previous_job}/state_files/{state_file}"
        new_state_file = f'{get_current_run_folder()}/state_files/{state_file}'
        die_with_47_if_file_doesnt_exists(old_state_file)

        if not os.path.exists(new_state_file):
            shutil.copy(old_state_file, new_state_file)

def die_something_went_wrong_with_parameters() -> None: # pragma: no cover
    my_exit(49)

@wrapper_print_debug
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

    parsed = []
    parsed_order = []

    comparer_found = False

    for item in result_array:
        if item in ["+", "*", "-", "/"]:
            parsed_order.append("operator")
            parsed.append({
                "type": "operator",
                "value": item
            })
        elif item in [">=", "<="]:
            if comparer_found:
                print("There is already one comparison operator! Cannot have more than one in an equation!")
                return False
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

@wrapper_print_debug
def get_experiment_parameters(_params: list) -> Any:
    continue_previous_job, seed, experiment_constraints, parameter, cli_params_experiment_parameters, experiment_parameters, minimize_or_maximize = _params

    global ax_client

    experiment_args = None

    if continue_previous_job:
        print_debug(f"Load from checkpoint: {continue_previous_job}")

        checkpoint_file: str = continue_previous_job + "/state_files/checkpoint.json"
        checkpoint_parameters_filepath: str = continue_previous_job + "/state_files/checkpoint.json.parameters.json"

        die_with_47_if_file_doesnt_exists(checkpoint_parameters_filepath)
        die_with_47_if_file_doesnt_exists(checkpoint_file)

        try:
            f = open(checkpoint_file, encoding="utf-8")
            experiment_parameters = json.load(f)
            f.close()

            with open(checkpoint_file, encoding="utf-8") as f:
                experiment_parameters = json.load(f)
        except json.decoder.JSONDecodeError:
            print_red(f"Error parsing checkpoint_file {checkpoint_file}")
            my_exit(47)

        experiment_args = set_torch_device_to_experiment_args(experiment_args)

        copy_state_files_from_previous_job(continue_previous_job)

        if parameter:
            for _item in cli_params_experiment_parameters:
                _replaced = False
                for _item_id_to_overwrite in range(0, len(experiment_parameters["experiment"]["search_space"]["parameters"])):
                    if _item["name"] == experiment_parameters["experiment"]["search_space"]["parameters"][_item_id_to_overwrite]["name"]:
                        old_param_json = json.dumps(
                            experiment_parameters["experiment"]["search_space"]["parameters"][_item_id_to_overwrite]
                        )
                        experiment_parameters["experiment"]["search_space"]["parameters"][_item_id_to_overwrite] = get_ax_param_representation(_item)
                        new_param_json = json.dumps(
                            experiment_parameters["experiment"]["search_space"]["parameters"][_item_id_to_overwrite]
                        )
                        _replaced = True

                        compared_params = compare_parameters(old_param_json, new_param_json)
                        if compared_params:
                            print_yellow(compared_params)

                if not _replaced:
                    print_yellow(f"--parameter named {_item['name']} could not be replaced. It will be ignored, instead. You cannot change the number of parameters or their names when continuing a job, only update their values.")

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
        else:
            print_red("Something went wrong with the ax_client")
            my_exit(9)
    else:
        experiment_args = {
            "name": global_vars["experiment_name"],
            "parameters": experiment_parameters,
            "objectives": {"result": ObjectiveProperties(minimize=minimize_or_maximize)},
            "choose_generation_strategy_kwargs": {
                "num_trials": max_eval,
                "num_initialization_trials": num_parallel_jobs,
                #"use_batch_trials": True,
                "max_parallelism_override": -1
            },
        }

        if seed:
            experiment_args["choose_generation_strategy_kwargs"]["random_seed"] = seed

        experiment_args = set_torch_device_to_experiment_args(experiment_args)

        if experiment_constraints and len(experiment_constraints):
            experiment_args["parameter_constraints"] = []
            for _l in range(0, len(experiment_constraints)):
                constraints_string = " ".join(experiment_constraints[_l])

                variables = [item['name'] for item in experiment_parameters]

                equation = check_equation(variables, constraints_string)

                if equation:
                    experiment_args["parameter_constraints"].append(constraints_string)
                else:
                    print_red(f"Experiment constraint '{constraints_string}' is invalid. Cannot continue.")
                    my_exit(19)

        try:
            if ax_client:
                ax_client.create_experiment(**experiment_args)
            else:
                print_red("ax_client could not be found!")
                sys.exit(9)
        except ValueError as error: # pragma: no cover
            print_red(f"An error has occurred while creating the experiment: {error}")
            die_something_went_wrong_with_parameters()
        except TypeError as error:
            print_red(f"An error has occurred while creating the experiment: {error}. This is probably a bug in OmniOpt.")
            die_something_went_wrong_with_parameters()

    return ax_client, experiment_parameters, experiment_args

def get_type_short(typename: str) -> str:
    if typename == "RangeParameter":
        return "range"

    if typename == "ChoiceParameter":
        return "choice"

    return typename

@wrapper_print_debug
def parse_single_experiment_parameter_table(experiment_parameters: dict) -> list:
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

@wrapper_print_debug
def print_overview_tables(experiment_parameters: dict, experiment_args: dict) -> None:
    if not experiment_parameters:
        print_red("Cannot determine experiment_parameters. No parameter table will be shown.")
        return

    if not experiment_parameters:
        print_red("Experiment parameters could not be determined for display")
        return

    min_or_max = "minimize"
    if args.maximize:
        min_or_max = "maximize"

    with open(f"{get_current_run_folder()}/state_files/{min_or_max}", mode='w', encoding="utf-8") as f:
        print('The contents of this file do not matter. It is only relevant that it exists.', file=f)

    if "_type" in experiment_parameters:
        experiment_parameters = experiment_parameters["experiment"]["search_space"]["parameters"]

    rows = parse_single_experiment_parameter_table(experiment_parameters)

    table = Table(header_style="bold", title="Experiment parameters:")
    columns = ["Name", "Type", "Lower bound", "Upper bound", "Values", "Type", "Log Scale?"]

    _param_name = ""

    for column in columns:
        table.add_column(column)

    k = 0

    for row in rows:
        _param_name = row[0]

        if _param_name in changed_grid_search_params:
            changed_text = changed_grid_search_params[_param_name]
            row[1] = "gridsearch"
            row[4] = changed_text

        table.add_row(*row, style='bright_green')

        k += 1

    console.print(table)

    with console.capture() as capture:
        console.print(table)
    table_str = capture.get()

    with open(f"{get_current_run_folder()}/parameters.txt", mode="w", encoding="utf-8") as text_file:
        text_file.write(table_str)

    if experiment_args is not None and "parameter_constraints" in experiment_args and len(experiment_args["parameter_constraints"]):
        constraints = experiment_args["parameter_constraints"]
        table = Table(header_style="bold", title="Constraints:")
        columns = ["Constraints"]
        for column in columns:
            table.add_column(column)
        for column in constraints:
            table.add_row(column)

        with console.capture() as capture:
            console.print(table)

        table_str = capture.get()

        console.print(table)

        with open(f"{get_current_run_folder()}/constraints.txt", mode="w", encoding="utf-8") as text_file:
            text_file.write(table_str)

@wrapper_print_debug
def update_progress_bar(_progress_bar: Any, nr: int) -> None:
    #print(f"update_progress_bar(_progress_bar, {nr})")
    #traceback.print_stack()

    _progress_bar.update(nr)

def get_current_model() -> str:
    global ax_client

    if ax_client:
        gs_model = ax_client.generation_strategy.model

        if gs_model:
            return str(gs_model.model)

    return "initializing model"

def get_best_params_str() -> str:
    if count_done_jobs() >= 0:
        best_params = get_best_params()
        if best_params and "result" in best_params:
            best_result = best_params["result"]
            if isinstance(best_result, (int, float)) or helpers.looks_like_float(best_result):
                best_result_int_if_possible = helpers.to_int_when_possible(float(best_result))

                if str(best_result) != NO_RESULT and best_result is not None:
                    return f"best result: {best_result_int_if_possible}"
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
    string = ""

    string_keys: list = []
    string_values: list = []

    stats: dict = {}

    for job, _ in global_vars["jobs"][:]: # pragma: no cover
        state = state_from_job(job)

        if state not in stats.keys():
            stats[state] = 0
        stats[state] += 1

    for key in stats.keys(): # pragma: no cover
        if args.abbreviate_job_names:
            string_keys.append(key.lower()[0])
        else:
            string_keys.append(key.lower())
        string_values.append(str(stats[key]))

    if len(string_keys) and len(string_values): # pragma: no cover
        _keys = "/".join(string_keys)
        _values = "/".join(string_values)

        if len(_keys):
            nr_current_workers = len(global_vars["jobs"])
            percentage = round((nr_current_workers / num_parallel_jobs) * 100)
            string = f"{_keys} {_values} ({percentage}%/{num_parallel_jobs})"

    return string

@wrapper_print_debug
def submitted_jobs(nr: int = 0) -> int:
    state_files_folder = f"{get_current_run_folder()}/state_files/"

    makedirs(state_files_folder)

    return append_and_read(f'{get_current_run_folder()}/state_files/submitted_jobs', nr)

@wrapper_print_debug
def get_desc_progress_text(new_msgs: list[str] = []) -> str:
    global global_vars
    global random_steps
    global max_eval

    desc: str = ""

    in_brackets: list[str] = []

    if failed_jobs():
        in_brackets.append(f"{helpers.bcolors.red}Failed jobs: {failed_jobs()}{helpers.bcolors.endc}")

    current_model = get_current_model()

    in_brackets.append(f"{current_model}")

    this_time: float = time.time()

    best_params_str: str = get_best_params_str()
    if best_params_str:
        in_brackets.append(best_params_str)

    if is_slurm_job():
        nr_current_workers = len(global_vars["jobs"])
        percentage = round((nr_current_workers / num_parallel_jobs) * 100)

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

    #in_brackets.append(f"jobs {count_done_jobs()}/{max_eval}")

    if args.verbose_tqdm:
        if submitted_jobs():
            in_brackets.append(f"total submitted: {submitted_jobs()}")

            if max_eval:
                in_brackets.append(f"max_eval: {max_eval}")

    if len(new_msgs):
        for new_msg in new_msgs:
            if new_msg:
                in_brackets.append(new_msg)

    if len(in_brackets):
        in_brackets_clean = []

        for item in in_brackets:
            if item:
                in_brackets_clean.append(item)

        if in_brackets_clean:
            desc += f"{', '.join(in_brackets_clean)}"

    def capitalized_string(s: str) -> str:
        return s[0].upper() + s[1:] if s else ""

    desc = capitalized_string(desc)

    return desc

@wrapper_print_debug
def progressbar_description(new_msgs: list[str] = []) -> None:
    desc = get_desc_progress_text(new_msgs)
    print_debug_progressbar(desc)
    if progress_bar is not None:
        progress_bar.set_description(desc)
        progress_bar.refresh()

@wrapper_print_debug
def clean_completed_jobs() -> None:
    #for job, trial_index in global_vars["jobs"][:]:
    #    _state = state_from_job(job)
    #    if _state != "running":
    #        print_red(f"clean_completed_jobs: {job} has state {_state}")
    #    while _state == "running":
    #        _state = state_from_job(job)
    #        print_yellow(f"clean_completed_jobs: Waiting for job {job}")
    #        _sleep(5)

    for job, trial_index in global_vars["jobs"][:]: # pragma: no cover
        _state = state_from_job(job)
        #print_debug(f'clean_completed_jobs: Job {job} (trial_index: {trial_index}) has state {_state}')
        if _state in ["completed", "early_stopped", "abandoned", "cancelled"]:
            global_vars["jobs"].remove((job, trial_index))
        elif _state in ["unknown", "pending", "running"]:
            pass
        else:
            print_red(f"Job {job}, state not in completed, early_stopped, abandoned, unknown, running or pending: {_state}")

def get_old_result_by_params(file_path: str, params: dict, float_tolerance: float = 1e-6) -> Any:
    """
    Open the CSV file and find the row where the subset of columns matching the keys in params have the same values.
    Return the value of the 'result' column from that row.

    :param file_path: The path to the CSV file.
    :param params: A dictionary of parameters with column names as keys and values to match.
    :param float_tolerance: The tolerance for comparing float values.
    :return: The value of the 'result' column from the matched row.
    """

    if not os.path.exists(file_path):
        print_red(f"{file_path} for getting old CSV results cannot be found")
        return None

    try:
        df = pd.read_csv(file_path, float_precision='round_trip')
    except Exception as e: # pragma: no cover
        print_red(f"Failed to read the CSV file: {str(e)}")
        return None

    if 'result' not in df.columns:
        print_red(f"Error: Could not get old result for {params} in {file_path}")
        return None

    try:
        matching_rows = df

        for param, value in params.items():
            if param in df.columns:
                if isinstance(value, float):
                    # Log current state before filtering

                    is_close_array = np.isclose(matching_rows[param], value, atol=float_tolerance)

                    matching_rows = matching_rows[is_close_array]

                    assert not matching_rows.empty, f"No matching rows found for float parameter '{param}' with value '{value}'"
                else:
                    # Ensure consistent types for comparison
                    if matching_rows[param].dtype == np.int64 and isinstance(value, str):
                        value = int(value)
                    elif matching_rows[param].dtype == np.float64 and isinstance(value, str):
                        value = float(value)

                    matching_rows = matching_rows[matching_rows[param] == value]

                    assert not matching_rows.empty, f"No matching rows found for parameter '{param}' with value '{value}'"

        if matching_rows.empty:
            return None

        return matching_rows
    except AssertionError as ae:
        print_red(f"Assertion error: {str(ae)}")
        raise
    except Exception as e: # pragma: no cover
        print_red(f"Error during filtering or extracting result: {str(e)}")
        raise

@wrapper_print_debug
def get_old_result_simple(this_path: str, old_arm_parameter: dict) -> Union[float, None, int]:
    tmp_old_res = get_old_result_by_params(f"{this_path}/{PD_CSV_FILENAME}", old_arm_parameter)
    if "result" in tmp_old_res:
        tmp_old_res = tmp_old_res["result"]
        tmp_old_res_list = list(set(list(tmp_old_res)))

        if len(tmp_old_res_list) == 1:
            print_debug(f"Got a list of length {len(tmp_old_res_list)}. This means the result was found properly and will be added.")
            old_result_simple = float(tmp_old_res_list[0])
        else:
            print_debug(f"Got a list of length {len(tmp_old_res_list)}. Cannot add this to previous jobs.")
            old_result_simple = None

        return old_result_simple

    return None

@wrapper_print_debug
def simulate_load_data_from_existing_run_folders(_paths: list[str]) -> int:
    _counter: int = 0

    for this_path in _paths:
        this_path_json = str(this_path) + "/state_files/ax_client.experiment.json"

        if not os.path.exists(this_path_json):
            print_red(f"{this_path_json} does not exist, cannot load data from it")
            return 0

        old_experiments = load_experiment(this_path_json)

        old_trials = old_experiments.trials

        trial_idx = 0
        for old_trial_index in old_trials:
            trial_idx += 1

            old_trial = old_trials[old_trial_index]
            trial_status = old_trial.status
            trial_status_str = trial_status.__repr__

            if "COMPLETED".lower() not in str(trial_status_str).lower(): # or "MANUAL".lower() in str(trial_status_str).lower()):
                continue

            old_arm_parameter = old_trial.arm.parameters

            old_result_simple = None
            try:
                old_result_simple = get_old_result_simple(this_path, old_arm_parameter)
            except Exception:
                pass

            if old_result_simple and helpers.looks_like_number(old_result_simple) and str(old_result_simple) != "nan":
                _counter += 1

    return _counter

@wrapper_print_debug
def get_nr_of_imported_jobs() -> int:
    nr_jobs: int = 0

    if args.load_previous_job_data:
        for this_path in args.load_previous_job_data:
            nr_jobs += simulate_load_data_from_existing_run_folders(this_path)

    if args.continue_previous_job:
        nr_jobs += simulate_load_data_from_existing_run_folders([args.continue_previous_job])

    return nr_jobs

@wrapper_print_debug
def load_existing_job_data_into_ax_client() -> None:
    global NR_INSERTED_JOBS

    if args.load_previous_job_data:
        for this_path in args.load_previous_job_data:
            load_data_from_existing_run_folders(this_path)

    if len(already_inserted_param_hashes.keys()):
        if len(missing_results):
            print(f"Missing results: {len(missing_results)}")
            #NR_INSERTED_JOBS += len(double_hashes)

        if len(double_hashes):
            print(f"Double parameters not inserted: {len(double_hashes)}")
            #NR_INSERTED_JOBS += len(double_hashes)

        if len(double_hashes) - len(already_inserted_param_hashes.keys()):
            print(f"Restored trials: {len(already_inserted_param_hashes.keys())}")
            NR_INSERTED_JOBS += len(already_inserted_param_hashes.keys())
    else:
        nr_of_imported_jobs = get_nr_of_imported_jobs()
        NR_INSERTED_JOBS += nr_of_imported_jobs

@wrapper_print_debug
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
    except AssertionError:
        # Logging the error
        return None

@wrapper_print_debug
def extract_headers_and_rows(data_list: list) -> Union[Tuple[None, None], Tuple[list, list]]:
    try:
        if not data_list:
            return None, None

        # Extract headers from the first dictionary
        first_entry = data_list[0]
        headers = list(first_entry.keys())

        # Initialize rows list
        rows: list = []

        # Extract rows based on headers order
        for entry in data_list:
            row: list = [str(entry.get(header, None)) for header in headers]
            rows.append(row)

        return headers, rows
    except Exception as e: # pragma: no cover
        print(f"extract_headers_and_rows: An error occurred: {e}")
        return None, None

def get_list_import_as_string(_brackets: bool = True, _comma: bool = False) -> str:
    _str: list = []

    if len(double_hashes):
        _str.append(f"double hashes: {len(double_hashes)}")

    if len(missing_results):
        _str.append(f"missing_results: {len(missing_results)}")

    if len(_str):
        if _brackets:
            if _comma:
                return ", (" + (", ".join(_str)) + ")"
            return " (" + (", ".join(_str)) + ")"

        if _comma:
            return ", " + (", ".join(_str))
        return ", ".join(_str)

    return ""

@wrapper_print_debug
def insert_job_into_ax_client(old_arm_parameter: dict, old_result: dict, hashed_params_result: Union[None, int, float]) -> None:
    done_converting = False

    if ax_client is None or not ax_client:
        print_red("insert_job_into_ax_client: ax_client was not defined where it should have been")
        my_exit(101)

    while not done_converting:
        try:
            if ax_client:
                new_old_trial = ax_client.attach_trial(old_arm_parameter)

                ax_client.complete_trial(trial_index=new_old_trial[1], raw_data=old_result)

                already_inserted_param_hashes[hashed_params_result] = 1

                done_converting = True
                save_pd_csv()
            else:
                print_red("Error getting ax_client")
                my_exit(9)
        except ax.exceptions.core.UnsupportedError as e:
            parsed_error = parse_parameter_type_error(e)

            if parsed_error["expected_type"] == "int" and type(old_arm_parameter[parsed_error["parameter_name"]]).__name__ != "int":
                print_yellow(f"⚠ converted parameter {parsed_error['parameter_name']} type {parsed_error['current_type']} to {parsed_error['expected_type']}")
                old_arm_parameter[parsed_error["parameter_name"]] = int(old_arm_parameter[parsed_error["parameter_name"]])
            elif parsed_error["expected_type"] == "float" and type(old_arm_parameter[parsed_error["parameter_name"]]).__name__ != "float":
                print_yellow(f"⚠ converted parameter {parsed_error['parameter_name']} type {parsed_error['current_type']} to {parsed_error['expected_type']}")
                old_arm_parameter[parsed_error["parameter_name"]] = float(old_arm_parameter[parsed_error["parameter_name"]])

@wrapper_print_debug
def load_data_from_existing_run_folders(_paths: list[str]) -> None:
    global already_inserted_param_hashes
    global already_inserted_param_data
    global double_hashes
    global missing_results

    def update_status(message: str, path_idx: Union[int, None] = None, trial_idx: Union[int, None] = None, total_trials: Union[int, None] = None) -> str:
        if len(_paths) > 1:
            folder_msg = f"(folder {path_idx + 1}/{len(_paths)})" if path_idx is not None else ""
            trial_msg = f", trial {trial_idx + 1}/{total_trials}" if trial_idx is not None else ""
            return f"{message} {folder_msg}{trial_msg}{get_list_import_as_string(False, True)}..."
        return f"{message}{get_list_import_as_string()}..."

    def generate_hashed_params(parameters: dict, path: str) -> Union[Tuple[str, str], Tuple[str, float, Tuple[str, int], Tuple[str, None]]]:
        try:
            result = get_old_result_simple(path, parameters)
        except Exception:
            result = None
        return pformat(parameters) + "====" + pformat(result), result

    def should_insert(hashed_params_result: Union[tuple[str, str] | tuple[str, float, tuple[str, int], tuple[str, None]]]) -> bool:
        result = hashed_params_result[1]
        res = result and helpers.looks_like_number(result) and str(result) != "nan" and hashed_params_result[0] not in already_inserted_param_hashes

        if res:
            return True
        return False

    def insert_or_log_result(parameters: Union[Tuple[str, str] | Tuple[str, float, Tuple[str, int], Tuple[str, None]]], hashed_params_result: Union[Tuple[str, str] | Tuple[str, float, Tuple[str, int], Tuple[str, None]]]) -> None:
        try:
            insert_job_into_ax_client(parameters, {'result': hashed_params_result[1]}, hashed_params_result[0])
            print_debug(f"ADDED: old_result_simple: {hashed_params_result[1]}, type: {type(hashed_params_result[1])}")
        except ValueError as e: # pragma: no cover
            print_red(f"Error while trying to insert parameter: {e}. Do you have parameters in your old run that are not in the new one?")
        else:
            already_inserted_param_hashes[hashed_params_result[0]] += 1
            double_hashes[hashed_params_result[0]] = 1

    def log_missing_result(parameters: dict, hashed_params_result: Union[Tuple[str, str] | tuple[str, float, tuple[str, int], Tuple[str, None]]]) -> None:
        print_debug("Prevent inserting a parameter set without result")
        missing_results.append(hashed_params_result[0])
        parameters["result"] = hashed_params_result[1]
        already_inserted_param_data.append(parameters)

    def load_and_insert_trials(_status: Any, old_trials: Any, this_path: str, path_idx: int) -> None:
        trial_idx = 0
        for old_trial_index, old_trial in old_trials.items():
            _status.update(update_status(f"[bold green]Loading existing jobs from {this_path} into ax_client", path_idx, trial_idx, len(old_trials)))

            trial_idx += 1
            if "COMPLETED".lower() not in str(old_trial.status).lower():
                continue

            old_arm_parameter = old_trial.arm.parameters
            hashed_params_result = generate_hashed_params(old_arm_parameter, this_path)

            if should_insert(hashed_params_result):
                insert_or_log_result(old_arm_parameter, hashed_params_result)
            else:
                log_missing_result(old_arm_parameter, hashed_params_result)

    def display_table() -> None:
        headers, rows = extract_headers_and_rows(already_inserted_param_data)
        if headers and rows:
            table = Table(show_header=True, header_style="bold", title="Duplicate parameters only inserted once or without result:")
            for header in headers:
                table.add_column(header)
            for row in rows:
                table.add_row(*row)
            console.print(table)

    with console.status("[bold green]Loading existing jobs into ax_client...") as __status:
        for path_idx, this_path in enumerate(_paths):
            __status.update(update_status(f"[bold green]Loading existing jobs from {this_path} into ax_client", path_idx))
            this_path_json = f"{this_path}/state_files/ax_client.experiment.json"

            if not os.path.exists(this_path_json):
                print_red(f"{this_path_json} does not exist, cannot load data from it")
                return

            old_experiments = load_experiment(this_path_json)
            load_and_insert_trials(__status, old_experiments.trials, this_path, path_idx)

    display_table()

@wrapper_print_debug
def get_first_line_of_file(file_paths: list[str]) -> str:
    first_line: str = ""
    if len(file_paths): # pragma: no cover
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

def check_for_basic_string_errors(file_as_string: str, first_line: str, file_paths: list[str], program_code: str) -> list[str]:
    errors: list[str] = []

    if first_line and isinstance(first_line, str) and first_line.isprintable() and not first_line.startswith("#!"): # pragma: no cover
        errors.append("First line does not seem to be a shebang line: " + first_line)

    if "Permission denied" in file_as_string and "/bin/sh" in file_as_string: # pragma: no cover
        errors.append("Log file contains 'Permission denied'. Did you try to run the script without chmod +x?")

    if "Exec format error" in file_as_string:
        current_platform = platform.machine()
        file_output = ""

        if len(file_paths): # pragma: no cover
            file_result = execute_bash_code("file " + file_paths[0])
            if len(file_result) and isinstance(file_result[0], str):
                file_output = ", " + file_result[0].strip()

        errors.append(f"Was the program compiled for the wrong platform? Current system is {current_platform}{file_output}")

    if "/bin/sh" in file_as_string and "not found" in file_as_string: # pragma: no cover
        errors.append("Wrong path? File not found")

    if len(file_paths) and os.stat(file_paths[0]).st_size == 0: # pragma: no cover
        errors.append(f"File in {program_code} is empty")

    if len(file_paths) == 0:
        errors.append(f"No files could be found in your program string: {program_code}")

    return errors

def get_base_errors() -> list:
    base_errors: list = [
        "Segmentation fault",
        "Illegal division by zero",
        "OOM",
        ["Killed", "Detected kill, maybe OOM or Signal?"]
    ]

    return base_errors

def check_for_base_errors(file_as_string: str) -> list: # pragma: no cover
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

def check_for_non_zero_exit_codes(file_as_string: str) -> list[str]:
    errors: list[str] = []
    for r in range(1, 255):
        special_exit_codes = get_exit_codes()
        search_for_exit_code = "Exit-Code: " + str(r) + ","
        if search_for_exit_code in file_as_string:
            _error: str = "Non-zero exit-code detected: " + str(r)
            if str(r) in special_exit_codes:
                _error += " (May mean " + special_exit_codes[str(r)] + ", unless you used that exit code yourself or it was part of any of your used libraries or programs)"
            errors.append(_error)
    return errors

def get_python_errors() -> list[list[str]]:
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

def get_first_line_of_file_that_contains_string(i: str, s: str) -> str: # pragma: no cover
    if not os.path.exists(i):
        print_debug(f"File {i} not found")
        return ""

    f: str = get_file_as_string(i)

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

def check_for_python_errors(i: str, file_as_string: str) -> list[str]: # pragma: no cover
    errors: list[str] = []

    for search_array in get_python_errors():
        search_for_string = search_array[0]
        search_for_error = search_array[1]

        if search_for_string in file_as_string:
            error_line = get_first_line_of_file_that_contains_string(i, search_for_string)
            if error_line:
                errors.append(error_line)
            else:
                errors.append(search_for_error)

    return errors

def get_errors_from_outfile(i: str) -> list[str]:
    file_as_string = get_file_as_string(i)

    program_code = get_program_code_from_out_file(i)
    file_paths = find_file_paths(program_code)

    first_line: str = get_first_line_of_file(file_paths)

    errors: list[str] = []

    if "Result: None" in file_as_string: # pragma: no cover
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

        new_errors = check_for_python_errors(i, file_as_string)
        for n in new_errors:
            errors.append(n)

    return errors

def print_outfile_analyzed(stdout_path: str) -> None:
    errors = get_errors_from_outfile(stdout_path)

    _strs: list[str] = []
    j: int = 0

    if len(errors): # pragma: no cover
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

    if len(_strs): # pragma: no cover
        try:
            with open(f'{get_current_run_folder()}/evaluation_errors.log', mode="a+", encoding="utf-8") as error_file:
                error_file.write(out_files_string)
        except Exception as e: # pragma: no cover
            print_debug(f"Error occurred while writing to evaluation_errors.log: {e}")

        print_red(out_files_string)

def get_parameters_from_outfile(stdout_path: str) -> Union[None, str]:
    try:
        with open(stdout_path, mode='r', encoding="utf-8") as file: # pragma: no cover
            for line in file:
                if line.lower().startswith("parameters: "):
                    params = line.split(":", 1)[1].strip()
                    params = json.loads(params)
                    return params
    except FileNotFoundError:
        original_print(f"get_parameters_from_outfile: The file '{stdout_path}' was not found.")
    except Exception as e: # pragma: no cover
        print(f"get_parameters_from_outfile: There was an error: {e}")

    return None

def get_hostname_from_outfile(stdout_path: str) -> Union[str, None]:
    try:
        with open(stdout_path, mode='r', encoding="utf-8") as file:
            for line in file:
                if line.lower().startswith("hostname: "):
                    hostname = line.split(":", 1)[1].strip()
                    return hostname
        return None # pragma: no cover
    except FileNotFoundError:
        original_print(f"The file {stdout_path} was not found.")
        return None
    except Exception as e: # pragma: no cover
        print(f"There was an error: {e}")
        return None

def mark_trial_as_failed(_trial: Any) -> None:
    print_debug(f"Marking trial {_trial} as failed")
    try:
        _trial.mark_failed()
    except ValueError as e:
        print_debug(f"mark_trial_as_failed error: {e}")

def mark_trial_as_completed(_trial: Any) -> None:
    print_debug(f"Marking trial {_trial} as completed")
    _trial.mark_completed(unsafe=True)

def finish_previous_jobs(new_msgs: list[str]) -> None:
    global random_steps
    global ax_client
    global JOBS_FINISHED

    this_jobs_finished = 0

    print_debug(f"jobs in finish_previous_jobs: {global_vars['jobs']}")

    for job, trial_index in global_vars["jobs"][:]:
        # Poll if any jobs completed
        # Local and debug jobs don't run until .result() is called.
        if job is None: # pragma: no cover
            print_debug(f"finish_previous_jobs: job {job} is None")
            continue

        print_debug(f"finish_previous_jobs: single job {job}")

        if job.done() or type(job) in [LocalJob, DebugJob]:
            try:
                result = job.result()
                raw_result = result
                result = result["result"]
                this_jobs_finished += 1

                if ax_client:
                    _trial = ax_client.get_trial(trial_index)

                    if result != VAL_IF_NOTHING_FOUND:
                        ax_client.complete_trial(trial_index=trial_index, raw_data=raw_result)

                        #count_done_jobs(1)
                        try:
                            progressbar_description([f"new result: {result}"])
                            mark_trial_as_completed(_trial)
                            succeeded_jobs(1)
                            update_progress_bar(progress_bar, 1)
                        except Exception as e: # pragma: no cover
                            print(f"ERROR in line {get_line_info()}: {e}")
                    else:
                        if job:
                            try:
                                progressbar_description(["job_failed"])
                                ax_client.log_trial_failure(trial_index=trial_index)
                            except Exception as e: # pragma: no cover
                                print(f"ERROR in line {get_line_info()}: {e}")
                            job.cancel()
                            mark_trial_as_failed(_trial)
                            orchestrate_job(job, trial_index)
                        failed_jobs(1)
                else: # pragma: no cover
                    print_red("ax_client could not be found or used")
                    my_exit(9)
                global_vars["jobs"].remove((job, trial_index))
            except (FileNotFoundError, submitit.core.utils.UncompletedJobError, ax.exceptions.core.UserInputError) as error: # pragma: no cover
                if "None for metric" in str(error):
                    print_red(f"\n⚠ It seems like the program that was about to be run didn't have 'RESULT: <NUMBER>' in it's output string.\nError: {error}")
                else:
                    print_red(f"\n⚠ {error}")
                if job:
                    try:
                        progressbar_description(["job_failed"])
                        if ax_client:
                            _trial = ax_client.get_trial(trial_index)
                            ax_client.log_trial_failure(trial_index=trial_index)
                            mark_trial_as_failed(_trial)
                        else:
                            print_red("ax_client failed")
                            my_exit(9)
                    except Exception as e: # pragma: no cover
                        print(f"ERROR in line {get_line_info()}: {e}")
                    job.cancel()
                    orchestrate_job(job, trial_index)
                failed_jobs(1)
                this_jobs_finished += 1
                global_vars["jobs"].remove((job, trial_index))
            save_checkpoint()
            save_pd_csv()
        else: # pragma: no cover
            print_debug(f"finish_previous_jobs: job was neither done, nor LocalJob nor DebugJob, but {job}")

    if this_jobs_finished == 1:
        progressbar_description([*new_msgs, f"finished {this_jobs_finished} job"])
    elif this_jobs_finished > 0: # pragma: no cover
        progressbar_description([*new_msgs, f"finished {this_jobs_finished} jobs"])

    JOBS_FINISHED += this_jobs_finished

    clean_completed_jobs()

def check_orchestrator(stdout_path: str, trial_index: int) -> list: # pragma: no cover
    behavs: list = []

    if orchestrator and "errors" in orchestrator:
        try:
            stdout = Path(stdout_path).read_text("UTF-8")
        except FileNotFoundError:
            orchestrate_todo_copy = ORCHESTRATE_TODO
            if stdout_path not in orchestrate_todo_copy.keys():
                ORCHESTRATE_TODO[stdout_path] = trial_index
                print_red(f"File not found: {stdout_path}, will try again later")
            else:
                print_red(f"File not found: {stdout_path}, not trying again")

            return None

        for oc in orchestrator["errors"]:
            #name = oc["name"]
            match_strings = oc["match_strings"]
            behavior = oc["behavior"]

            for match_string in match_strings:
                if match_string.lower() in stdout.lower():
                    if behavior not in behavs:
                        behavs.append(behavior)

    return behavs

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
    for todo_stdout_file in orchestrate_todo_copy.keys(): # pragma: no cover
        old_behavs = check_orchestrator(todo_stdout_file, ORCHESTRATE_TODO[todo_stdout_file])
        if old_behavs is not None:
            del ORCHESTRATE_TODO[todo_stdout_file]

def is_already_in_defective_nodes(hostname: str) -> bool: # pragma: no cover
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
    except Exception as e: # pragma: no cover
        print_red(f"is_already_in_defective_nodes: Error reading the file {file_path}: {e}")
        return False

    return False

def orchestrator_start_trial(params_from_out_file: str, trial_index: int) -> None: # pragma: no cover
    global global_vars

    if executor and ax_client:
        new_job = executor.submit(evaluate, params_from_out_file)
        submitted_jobs(1)

        _trial = ax_client.get_trial(trial_index)

        try:
            _trial.mark_staged(unsafe=True)
        except Exception as e:
            print_debug(f"orchestrator_start_trial: error {e}")
        _trial.mark_running(unsafe=True, no_runner_required=True)

        global_vars["jobs"].append((new_job, trial_index))
    else:
        print_red("executor or ax_client could not be found properly")
        my_exit(9)

def handle_exclude_node(stdout_path: str, hostname_from_out_file: Union[None, str]) -> None: # pragma: no cover
    if hostname_from_out_file:
        if not is_already_in_defective_nodes(hostname_from_out_file):
            print_yellow(f"ExcludeNode was triggered for node {hostname_from_out_file}")
            count_defective_nodes(None, hostname_from_out_file)
        else:
            print_yellow(f"ExcludeNode was triggered for node {hostname_from_out_file}, but it was already in defective nodes and won't be added again")
    else:
        print_red(f"Cannot do ExcludeNode because the host could not be determined from {stdout_path}")

def handle_restart(stdout_path: str, trial_index: int) -> None: # pragma: no cover
    params_from_out_file = get_parameters_from_outfile(stdout_path)
    if params_from_out_file:
        orchestrator_start_trial(params_from_out_file, trial_index)
    else:
        print(f"Could not determine parameters from outfile {stdout_path} for restarting job")

def handle_restart_on_different_node(stdout_path: str, hostname_from_out_file: Union[None, str], trial_index: int) -> None: # pragma: no cover
    if hostname_from_out_file:
        if not is_already_in_defective_nodes(hostname_from_out_file):
            print_yellow(f"RestartOnDifferentNode was triggered for node {hostname_from_out_file}. Adding node to defective hosts list and restarting on another host.")
            count_defective_nodes(None, hostname_from_out_file)
        else:
            print_yellow(f"RestartOnDifferentNode was triggered for node {hostname_from_out_file}, but it was already in defective nodes. Job will only be resubmitted.")
        handle_restart(stdout_path, trial_index)
    else:
        print_red(f"Cannot do RestartOnDifferentNode because the host could not be determined from {stdout_path}")

def handle_exclude_node_and_restart_all(stdout_path: str, hostname_from_out_file: Union[None, str]) -> None: # pragma: no cover
    if hostname_from_out_file:
        if not is_already_in_defective_nodes(hostname_from_out_file):
            print_yellow(f"ExcludeNodeAndRestartAll not yet fully implemented. Adding {hostname_from_out_file} to unavailable hosts.")
            count_defective_nodes(None, hostname_from_out_file)
        else:
            print_yellow(f"ExcludeNodeAndRestartAll was triggered for node {hostname_from_out_file}, but it was already in defective nodes and won't be added again.")
    else:
        print_red(f"Cannot do ExcludeNodeAndRestartAll because the host could not be determined from {stdout_path}")

def _orchestrate(stdout_path: str, trial_index: int) -> None: # pragma: no cover
    behavs = check_orchestrator(stdout_path, trial_index)

    if not behavs:
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

def write_continue_run_uuid_to_file() -> bool:
    if args.continue_previous_job:
        continue_dir = args.continue_previous_job

        try:

            with open(f'{continue_dir}/state_files/run_uuid', mode='r', encoding='utf-8') as f:
                continue_from_uuid = f.readline()

                file_path: str = f"{get_current_run_folder()}/state_files/uuid_of_continued_run"

                makedirs(os.path.dirname(file_path))

                with open(file_path, 'w', encoding="utf-8") as file:
                    file.write(continue_from_uuid)

                return True
        except Exception as e: # pragma: no cover
            print(f"write_continue_run_uuid_to_file: An error occurred: {e}")

    return False

def write_run_uuid_to_file() -> bool:
    try:
        file_path: str = f"{get_current_run_folder()}/state_files/run_uuid"

        makedirs(os.path.dirname(file_path))

        with open(file_path, 'w', encoding="utf-8") as file:
            file.write(run_uuid)

        return True
    except Exception as e: # pragma: no cover
        print(f"write_run_uuid_to_file: An error occurred: {e}")

    return False # pragma: no cover

def save_state_files() -> None:
    global global_vars

    state_files_folder: str = f"{get_current_run_folder()}/state_files/"

    makedirs(state_files_folder)

    with open(f'{state_files_folder}/joined_run_program', mode='w', encoding="utf-8") as f:
        original_print(global_vars["joined_run_program"], file=f)

    with open(f'{state_files_folder}/experiment_name', mode='w', encoding="utf-8") as f:
        original_print(global_vars["experiment_name"], file=f)

    with open(f'{state_files_folder}/mem_gb', mode='w', encoding='utf-8') as f:
        original_print(global_vars["mem_gb"], file=f)

    with open(f'{state_files_folder}/max_eval', mode='w', encoding='utf-8') as f:
        original_print(max_eval, file=f)

    with open(f'{state_files_folder}/gpus', mode='w', encoding='utf-8') as f:
        original_print(args.gpus, file=f)

    with open(f'{state_files_folder}/time', mode='w', encoding='utf-8') as f:
        original_print(global_vars["_time"], file=f)

    with open(f'{state_files_folder}/env', mode='a', encoding="utf-8") as f:
        env: dict = dict(os.environ)
        for key in env:
            original_print(str(key) + " = " + str(env[key]), file=f)

    with open(f'{state_files_folder}/run.sh', mode='w', encoding='utf-8') as f:
        original_print("omniopt '" + " ".join(sys.argv[1:]), file=f)

def submit_job(parameters: dict) -> Union[None, Job[dict[Any, Any]]]:
    try:
        if executor:
            new_job = executor.submit(evaluate, parameters)
            submitted_jobs(1)
            return new_job

        print_red("executor could not be found") # pragma: no cover
        my_exit(9) # pragma: no cover
    except Exception as e: # pragma: no cover
        print_debug(f"Error while trying to submit job: {e}")
        raise

    return None # pragma: no cover

def execute_evaluation(_params: list) -> Optional[int]:
    global global_vars

    print_debug(f"execute_evaluation({_params})")
    trial_index, parameters, trial_counter, next_nr_steps, phase = _params
    if ax_client:
        _trial = ax_client.get_trial(trial_index)

        # Helper function for trial stage marking with exception handling
        def mark_trial_stage(stage: str, error_msg: str) -> None:
            try:
                getattr(_trial, stage)()
            except Exception as e: # pragma: no cover
                print_debug(f"execute_evaluation({_params}): {error_msg} with error: {e}")

        mark_trial_stage("mark_staged", "Marking the trial as staged failed")

        new_job = None

        try:
            initialize_job_environment()
            new_job = submit_job(parameters)

            global_vars["jobs"].append((new_job, trial_index))
            if is_slurm_job() and not args.force_local_execution: # pragma: no cover
                _sleep(1)

            mark_trial_stage("mark_running", "Marking the trial as running failed")
            trial_counter += 1

            update_progress()
        except submitit.core.utils.FailedJobError as error: # pragma: no cover
            handle_failed_job(error, trial_index, new_job)
            trial_counter += 1
        except (SignalUSR, SignalINT, SignalCONT): # pragma: no cover
            handle_exit_signal()
        except Exception as e: # pragma: no cover
            handle_generic_error(e)

        add_to_phase_counter(phase, 1)
        return trial_counter

    print_red("Failed to get ax_client") # pragma: no cover
    my_exit(9) # pragma: no cover

    return None # pragma: no cover

def initialize_job_environment() -> None:
    progressbar_description(["starting new job"])
    set_sbatch_environment()
    exclude_defective_nodes()

def set_sbatch_environment() -> None:
    if args.reservation: # pragma: no cover
        os.environ['SBATCH_RESERVATION'] = args.reservation
    if args.account: # pragma: no cover
        os.environ['SBATCH_ACCOUNT'] = args.account

def exclude_defective_nodes() -> None:
    excluded_string: str = ",".join(count_defective_nodes())
    if len(excluded_string) > 1: # pragma: no cover
        if executor:
            executor.update_parameters(exclude=excluded_string)
        else:
            print_red("executor could not be found")
            my_exit(9)

def handle_failed_job(error: Union[None, Exception, str], trial_index: int, new_job: Job) -> None: # pragma: no cover
    if "QOSMinGRES" in str(error) and args.gpus == 0:
        print_red("\n⚠ It seems like, on the chosen partition, you need at least one GPU. Use --gpus=1 (or more) as parameter.")
    else:
        print_red(f"\n⚠ FAILED: {error}")

    try:
        cancel_failed_job(trial_index, new_job)
    except Exception as e: # pragma: no cover
        print_red(f"\n⚠ Cancelling failed job FAILED: {e}")

def cancel_failed_job(trial_index: int, new_job: Job) -> None: # pragma: no cover
    print_debug("Trying to cancel job that failed")
    if new_job:
        try:
            if ax_client:
                ax_client.log_trial_failure(trial_index=trial_index)
            else:
                print_red("ax_client not defined")
                my_exit(9)
        except Exception as e: # pragma: no cover
            print(f"ERROR in line {get_line_info()}: {e}")
        new_job.cancel()
        print_debug("Cancelled failed job")

        global_vars["jobs"].remove((new_job, trial_index))
        print_debug("Removed failed job")
        save_checkpoint()
        save_pd_csv()
    else:
        print_debug("cancel_failed_job: new_job was undefined")

def update_progress() -> None:
    progressbar_description(["started new job"])

def handle_exit_signal() -> None: # pragma: no cover
    print_red("\n⚠ Detected signal. Will exit.")
    end_program(RESULT_CSV_FILE, False, 1)

def handle_generic_error(e: Union[Exception, str]) -> None: # pragma: no cover
    tb = traceback.format_exc()
    print(tb)
    print_red(f"\n⚠ Starting job failed with error: {e}")

def succeeded_jobs(nr: int = 0) -> int:
    state_files_folder = f"{get_current_run_folder()}/state_files/"

    makedirs(state_files_folder)

    return append_and_read(f'{get_current_run_folder()}/state_files/succeeded_jobs', nr)

def show_debug_table_for_break_run_search(_name: str, _max_eval: int, _progress_bar: Any, _ret: Any) -> None: # pragma: no cover
    table = Table(show_header=True, header_style="bold", title=f"break_run_search for {_name}")

    headers = ["Variable", "Value"]
    table.add_column(headers[0])
    table.add_column(headers[1])

    rows = [
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

def break_run_search(_name: str, _max_eval: int, _progress_bar: Any) -> bool:
    _ret = False

    _counted_done_jobs = count_done_jobs()

    conditions = [
        (lambda: succeeded_jobs() >= _max_eval + 1, f"1. succeeded_jobs() {succeeded_jobs()} >= _max_eval {_max_eval} + 1"),
        (lambda: submitted_jobs() >= _progress_bar.total + 1, f"2. submitted_jobs() {submitted_jobs()} >= _progress_bar.total {_progress_bar.total} + 1"),
        (lambda: _counted_done_jobs >= _max_eval, f"3. _counted_done_jobs {_counted_done_jobs} >= _max_eval {_max_eval}"),
        (lambda: _counted_done_jobs >= max_eval, f"3. _counted_done_jobs {_counted_done_jobs} >= max_eval {max_eval}"),
        (lambda: submitted_jobs() >= _max_eval + 1, f"4. submitted_jobs() {submitted_jobs()} > _max_eval {_max_eval} + 1"),
        (lambda: submitted_jobs() >= max_eval + 1, f"4. submitted_jobs() {submitted_jobs()} > max_eval {max_eval} + 1"),
        (lambda: 0 >= abs(_counted_done_jobs - _max_eval - NR_INSERTED_JOBS), f"5. 0 >= abs(_counted_done_jobs {_counted_done_jobs} - _max_eval {_max_eval} - NR_INSERTED_JOBS {NR_INSERTED_JOBS})")
    ]

    for condition_func, debug_msg in conditions:
        if condition_func():
            print_debug(f"breaking {_name}: {debug_msg}")
            _ret = True

    if args.verbose: # pragma: no cover
        show_debug_table_for_break_run_search(_name, _max_eval, _progress_bar, _ret)

    return _ret

def _get_last_and_avg_times() -> Union[Tuple[None, None], Tuple[float, float]]:
    """Returns the last and average times from TIME_NEXT_TRIALS_TOOK, or None if empty."""
    if len(TIME_NEXT_TRIALS_TOOK) == 0:
        return None, None
    last_time = TIME_NEXT_TRIALS_TOOK[-1]
    avg_time = sum(TIME_NEXT_TRIALS_TOOK) / len(TIME_NEXT_TRIALS_TOOK)
    return last_time, avg_time

def _calculate_nr_of_jobs_to_get(simulated_jobs: int, currently_running_jobs: int) -> int:
    """Calculates the number of jobs to retrieve."""
    return min(
        max_eval + simulated_jobs - count_done_jobs(),
        max_eval + simulated_jobs - submitted_jobs(),
        num_parallel_jobs - currently_running_jobs
    )

def _get_trials_message(nr_of_jobs_to_get: int, last_time: Union[float, int, None], avg_time: Union[int, float, None], force_local_execution: bool) -> str:
    """Generates the appropriate message for the number of trials being retrieved."""
    base_msg = f"getting {nr_of_jobs_to_get} trials "

    if SYSTEM_HAS_SBATCH and not force_local_execution: # pragma: no cover
        if last_time:
            return f"{base_msg}(last/avg {last_time:.2f}s/{avg_time:.2f}s)"
        return base_msg

    return f"{base_msg}(no sbatch)" + (f", last/avg {last_time:.2f}s/{avg_time:.2f}s" if last_time else "(no sbatch)")

def get_parallelism_schedule_description() -> str:
    try:
        # Retrieve parallelism settings
        if ax_client:
            max_parallelism_settings = ax_client.get_max_parallelism()

            if not max_parallelism_settings: # pragma: no cover
                return "No parallelism settings available."

            # Build human-readable descriptions
            descriptions = []
            for num_trials, max_parallelism in max_parallelism_settings:
                if num_trials == -1:
                    trial_text = "all remaining trials"
                else:
                    trial_text = f"{num_trials} trials"

                if max_parallelism == -1: # pragma: no cover
                    parallelism_text = "any number of trials can be run in parallel"
                else:
                    parallelism_text = f"up to {max_parallelism} trials can be run in parallel"

                descriptions.append(f"For {trial_text}, {parallelism_text}.")

            # Join descriptions into a single string
            human_readable_output: str = "\n".join(descriptions)
            return human_readable_output

        print_red("Error defining ax_client") # pragma: no cover
        sys.exit(9) # pragma: no cover

    except Exception as e: # pragma: no cover
        return f"An error occurred while processing parallelism schedule: {str(e)}"

@disable_logs
def _fetch_next_trials(nr_of_jobs_to_get: int) -> Optional[Tuple[dict[int, Any], bool]]:
    """Attempts to fetch the next trials using the ax_client."""
    try:
        print_debug(f"_fetch_next_trials({nr_of_jobs_to_get}), get_parallelism_schedule_description: {get_parallelism_schedule_description()}")
        #trial_index_to_param, optimization_complete = ax_client.get_next_trial(max_trials=nr_of_jobs_to_get)
        #return trial_index_to_param, optimization_complete

        trials_dict: dict = {}

        try:
            if ax_client:
                params, trial_index = ax_client.get_next_trial(force=True)

                trials_dict[trial_index] = params
            else: # pragma: no cover
                print_red("ax_client was not defined")
                my_exit(9)
        except (ax.exceptions.core.SearchSpaceExhausted, ax.exceptions.generation_strategy.GenerationStrategyRepeatedPoints, ax.exceptions.generation_strategy.MaxParallelismReachedException) as e: # pragma: no cover
            print_red("\n⚠Error 8: " + str(e))

        return trials_dict, False
    except np.linalg.LinAlgError as e: # pragma: no cover
        _handle_linalg_error(e)
        my_exit(242)

    return None # pragma: no cover

def _handle_linalg_error(error: Union[None, str, Exception]) -> None: # pragma: no cover
    """Handles the np.linalg.LinAlgError based on the model being used."""
    if args.model and args.model.upper() in ["THOMPSON", "EMPIRICAL_BAYES_THOMPSON"]:
        print_red(f"Error: {error}. This may happen because the THOMPSON model is used. Try another one.")
    else:
        print_red(f"Error: {error}")

def _get_next_trials(nr_of_jobs_to_get: int) -> Tuple[dict, bool]:
    global global_vars

    finish_previous_jobs(["finishing jobs (_get_next_trials)"])

    if break_run_search("_get_next_trials", max_eval, progress_bar) or nr_of_jobs_to_get == 0:
        return {}, True

    last_ax_client_time, ax_client_time_avg = _get_last_and_avg_times()

    # Message handling
    message = _get_trials_message(
        nr_of_jobs_to_get,
        last_ax_client_time,
        ax_client_time_avg,
        args.force_local_execution
    )
    progressbar_description([message])

    # Fetching the next trials
    start_time: float = time.time()
    trial_index_to_param, optimization_complete = _fetch_next_trials(nr_of_jobs_to_get)
    end_time: float = time.time()

    # Log and update timing
    TIME_NEXT_TRIALS_TOOK.append(end_time - start_time)
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

def get_next_nr_steps(_num_parallel_jobs: int, _max_eval: int) -> int: # pragma: no cover
    if not SYSTEM_HAS_SBATCH:
        return 1

    simulated_nr_inserted_jobs = get_nr_of_imported_jobs()

    requested = min(_num_parallel_jobs - len(global_vars["jobs"]), _max_eval + simulated_nr_inserted_jobs - submitted_jobs(), max_eval + simulated_nr_inserted_jobs - count_done_jobs())

    return requested

def check_max_parallelism_arg(possible_values: list) -> bool:
    if args.max_parallelism in possible_values or helpers.looks_like_int(args.max_parallelism):
        return True
    return False

def _get_max_parallelism() -> int: # pragma: no cover
    possible_values: list = [None, "None", "none", "max_eval", "num_parallel_jobs", "twice_max_eval", "twice_num_parallel_jobs", "max_eval_times_thousand_plus_thousand"]

    ret: int = 0

    if check_max_parallelism_arg(possible_values):
        if args.max_parallelism == "max_eval":
            ret = max_eval
        if args.max_parallelism == "num_parallel_jobs":
            ret = args.num_parallel_jobs
        if args.max_parallelism == "twice_max_eval":
            ret = 2 * max_eval
        if args.max_parallelism == "twice_num_parallel_jobs":
            ret = 2 * args.num_parallel_jobs
        if args.max_parallelism == "max_eval_times_thousand_plus_thousand":
            ret = 1000 * max_eval + 1000
        if helpers.looks_like_int(args.max_parallelism):
            ret = int(args.max_parallelism)
    else:
        print_red(f"Invalid --max_parallelism value. Must be one of those: {', '.join(possible_values)}")

    return ret

def create_systematic_step(model: Any) -> Any:
    """Creates a generation step for Bayesian optimization."""
    return GenerationStep(
        model=model,
        num_trials=-1,
        max_parallelism=_get_max_parallelism(),
        model_gen_kwargs={'enforce_num_arms': False},
        should_deduplicate=args.should_deduplicate
    )

def create_random_generation_step() -> int:
    """Creates a generation step for random models."""
    return GenerationStep(
        model=Models.SOBOL,
        num_trials=max(num_parallel_jobs, random_steps),
        min_trials_observed=min(max_eval, random_steps),
        max_parallelism=_get_max_parallelism(),
        #enforce_num_trials=True,
        model_kwargs={"seed": args.seed},
        model_gen_kwargs={'enforce_num_arms': False},
        should_deduplicate=args.should_deduplicate
    )

def select_model(model_arg: Any) -> Any:
    """Selects the model based on user input or defaults to BOTORCH_MODULAR."""
    available_models = list(Models.__members__.keys())
    chosen_model = Models.BOTORCH_MODULAR

    if model_arg:
        model_upper = str(model_arg).upper()
        if model_upper in available_models:
            chosen_model = Models.__members__[model_upper]
        else: # pragma: no cover
            print_red(f"⚠ Cannot use {model_arg}. Available models are: {', '.join(available_models)}. Using BOTORCH_MODULAR instead.")

        if model_arg.lower() != "factorial" and args.gridsearch:
            print_red("Gridsearch only really works when you chose the FACTORIAL model.")

    return chosen_model

def get_generation_strategy() -> Any:
    global random_steps

    # Initialize steps for the generation strategy
    steps: list = []

    # Get the number of imported jobs and update max evaluations
    num_imported_jobs: int = get_nr_of_imported_jobs()
    set_max_eval(max_eval + num_imported_jobs)

    # Initialize random_steps if None
    random_steps = random_steps or 0

    # Set max_eval if it's None
    if max_eval is None: # pragma: no cover
        set_max_eval(max(1, random_steps))

    # Add a random generation step if conditions are met
    if random_steps >= 1 and num_imported_jobs < random_steps:
        steps.append(create_random_generation_step())

    # Choose a model for the non-random step
    chosen_non_random_model = select_model(args.model)

    # Append the Bayesian optimization step
    steps.append(create_systematic_step(chosen_non_random_model))

    # Create and return the GenerationStrategy
    return GenerationStrategy(steps=steps)

def create_and_execute_next_runs(next_nr_steps: int, phase: str, _max_eval: int, _progress_bar: Any) -> int:
    global random_steps

    if next_nr_steps == 0:
        return 0

    trial_index_to_param = None

    done_optimizing: bool = False

    try:
        nr_of_jobs_to_get = _calculate_nr_of_jobs_to_get(get_nr_of_imported_jobs(), len(global_vars["jobs"]))

        results = []

        for _ in range(nr_of_jobs_to_get + 1):
            trial_index_to_param, optimization_complete = _get_next_trials(1)

            if optimization_complete:
                done_optimizing = True
                continue

            if trial_index_to_param:
                i = 1
                with ThreadPoolExecutor() as con_exe:
                    for trial_index, parameters in trial_index_to_param.items():
                        if break_run_search("create_and_execute_next_runs", _max_eval, _progress_bar): # pragma: no cover
                            break

                        while len(global_vars["jobs"]) > num_parallel_jobs: # pragma: no cover
                            finish_previous_jobs(["finishing previous jobs"])

                            if break_run_search("create_and_execute_next_runs", _max_eval, _progress_bar):
                                break

                            if is_slurm_job() and not args.force_local_execution:
                                _sleep(5)

                        if (JOBS_FINISHED - NR_INSERTED_JOBS) >= _max_eval: # pragma: no cover
                            break

                        if not break_run_search("create_and_execute_next_runs", _max_eval, _progress_bar):
                            progressbar_description(["starting parameter set"])

                            _args = [
                                trial_index,
                                parameters,
                                i,
                                next_nr_steps,
                                phase
                            ]

                            results.append(con_exe.submit(execute_evaluation, _args))

                            i += 1
                        else: # pragma: no cover
                            break

        for r in results:
            r.result()

        finish_previous_jobs(["finishing jobs after starting them"])

        if done_optimizing:
            end_program(RESULT_CSV_FILE, False, 0)
    except TypeError as e: # pragma: no cover
        print_red(f"Error 1: {e}")
        return 0
    except botorch.exceptions.errors.InputDataError as e: # pragma: no cover
        print_red(f"Error 2: {e}")
        return 0
    except ax.exceptions.core.DataRequiredError as e: # pragma: no cover
        if "transform requires non-empty data" in str(e) and args.num_random_steps == 0:
            print_red(f"Error 3: {e} This may happen when there are no random_steps, but you tried to get a model anyway. Increase --num_random_steps to at least 1 to continue.")
            die_no_random_steps()
        else:
            print_debug(f"Error 4: {e}")
            return 0
    except RuntimeError as e: # pragma: no cover
        print_red("\n⚠ Error 5: " + str(e))
    except botorch.exceptions.errors.ModelFittingError as e: # pragma: no cover
        print_red("\n⚠ Error 6: " + str(e))
        end_program(RESULT_CSV_FILE, False, 1)
    except (ax.exceptions.core.SearchSpaceExhausted, ax.exceptions.generation_strategy.GenerationStrategyRepeatedPoints) as e: # pragma: no cover
        print_red("\n⚠ Error 7" + str(e))
        end_program(RESULT_CSV_FILE, False, 87)

    num_new_keys = 0

    try:
        if trial_index_to_param:
            num_new_keys = len(trial_index_to_param.keys())
    except Exception: # pragma: no cover
        pass

    return num_new_keys

def get_number_of_steps(_max_eval: int) -> Tuple[int, int]:
    _random_steps = args.num_random_steps

    already_done_random_steps = get_random_steps_from_prev_job()

    _random_steps = args.num_random_steps - already_done_random_steps

    if _random_steps > _max_eval:
        print_yellow(f"You have less --max_eval {_max_eval} than --num_random_steps {_random_steps}. Switched both.")
        _random_steps, _max_eval = _max_eval, _random_steps

    if _random_steps < num_parallel_jobs and SYSTEM_HAS_SBATCH: # pragma: no cover
        old_random_steps = _random_steps
        _random_steps = num_parallel_jobs
        original_print(f"_random_steps {old_random_steps} is smaller than num_parallel_jobs {num_parallel_jobs}. --num_random_steps will be ignored and set to num_parallel_jobs ({num_parallel_jobs}) to not have idle workers in the beginning.")

    if _random_steps > _max_eval: # pragma: no cover
        set_max_eval(_random_steps)

    original_second_steps = _max_eval - _random_steps
    second_step_steps = max(0, original_second_steps)
    if second_step_steps != original_second_steps: # pragma: no cover
        original_print(f"? original_second_steps: {original_second_steps} = max_eval {_max_eval} - _random_steps {_random_steps}")
    if second_step_steps == 0:
        print_red("This is basically a random search. Increase --max_eval or reduce --num_random_steps")

    second_step_steps = second_step_steps - already_done_random_steps

    if args.continue_previous_job:
        second_step_steps = _max_eval

    return _random_steps, second_step_steps

def set_global_executor() -> None:
    global executor

    log_folder: str = f'{get_current_run_folder()}/single_runs/%j'

    if args.force_local_execution:
        executor = LocalExecutor(folder=log_folder)
    else:
        executor = AutoExecutor(folder=log_folder)

    # TODO: The following settings can be in submitit's executor.update_parameters, set but aren't currently utilized because I am not sure of the defaults:
    # 'nodes': <class 'int'>
    # 'gpus_per_node': <class 'int'>
    # 'tasks_per_node': <class 'int'>
    # Should they just be None by default if not set in the argparser? No, submitit fails if gpu related stuff is None

    if executor:
        executor.update_parameters(
            name=f'{global_vars["experiment_name"]}_{run_uuid}_{str(uuid.uuid4())}',
            timeout_min=args.worker_timeout,
            slurm_gres=f"gpu:{args.gpus}",
            cpus_per_task=args.cpus_per_task,
            nodes=args.nodes_per_job,
            stderr_to_stdout=args.stderr_to_stdout,
            mem_gb=args.mem_gb,
            slurm_signal_delay_s=args.slurm_signal_delay_s,
            slurm_use_srun=args.slurm_use_srun,
            exclude=args.exclude
        )

        print_debug(f"""
    executor.update_parameters(
        name={f'{global_vars["experiment_name"]}_{run_uuid}_{str(uuid.uuid4())}'}
        timeout_min={args.worker_timeout}
        "slurm_gres={f"gpu:{args.gpus}"}
        "cpus_per_task={args.cpus_per_task}
        nodes={args.nodes_per_job}
        stderr_to_stdout={args.stderr_to_stdout}
        mem_gb={args.mem_gb}
        slurm_signal_delay_s={args.slurm_signal_delay_s}
        "slurm_use_srun={args.slurm_use_srun}
        exclude={args.exclude}
    )
"""
        )

        if args.exclude: # pragma: no cover
            print_yellow(f"Excluding the following nodes: {args.exclude}")
    else: # pragma: no cover
        print_red("executor could not be found")
        my_exit(9)

def execute_nvidia_smi() -> None: # pragma: no cover
    if not IS_NVIDIA_SMI_SYSTEM:
        print_debug("Cannot find nvidia-smi. Cannot take GPU logs")
        return

    while True:
        try:
            host = socket.gethostname()

            if NVIDIA_SMI_LOGS_BASE and host:
                _file = NVIDIA_SMI_LOGS_BASE + "_" + host + ".csv"
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
        except Exception as e: # pragma: no cover
            print(f"execute_nvidia_smi: An error occurred: {e}")
        if is_slurm_job() and not args.force_local_execution: # pragma: no cover
            _sleep(10)

def start_nvidia_smi_thread() -> None: # pragma: no cover
    if IS_NVIDIA_SMI_SYSTEM:
        nvidia_smi_thread = threading.Thread(target=execute_nvidia_smi, daemon=True)
        nvidia_smi_thread.start()

def run_search(_progress_bar: Any) -> bool:
    global NR_OF_0_RESULTS

    NR_OF_0_RESULTS = 0

    log_what_needs_to_be_logged()
    write_process_info()

    while submitted_jobs() <= max_eval:
        log_what_needs_to_be_logged()
        wait_for_jobs_to_complete(num_parallel_jobs)

        finish_previous_jobs([])

        if break_run_search("run_search", max_eval, _progress_bar):
            break

        if (JOBS_FINISHED - NR_INSERTED_JOBS) >= max_eval:
            break

        next_nr_steps: int = get_next_nr_steps(num_parallel_jobs, max_eval)

        nr_of_items: int = 0

        if next_nr_steps:
            progressbar_description([f"trying to get {next_nr_steps} next steps (current done: {count_done_jobs()}, max: {max_eval})"])

            nr_of_items = create_and_execute_next_runs(next_nr_steps, "systematic", max_eval, _progress_bar)

            progressbar_description([f"got {nr_of_items}, requested {next_nr_steps}"])

        _debug_worker_creation(f"{int(time.time())}, {len(global_vars['jobs'])}, {nr_of_items}, {next_nr_steps}")

        finish_previous_jobs(["finishing previous jobs"])

        if is_slurm_job() and not args.force_local_execution: # pragma: no cover
            _sleep(1)

        if nr_of_items == 0 and len(global_vars["jobs"]) == 0:
            _wrn = f"found {NR_OF_0_RESULTS} zero-jobs (max: {args.max_nr_of_zero_results})"
            NR_OF_0_RESULTS += 1
            progressbar_description([_wrn])
            print_debug(_wrn)
        else:
            NR_OF_0_RESULTS = 0

        if not args.disable_search_space_exhaustion_detection and NR_OF_0_RESULTS >= args.max_nr_of_zero_results:
            _wrn = f"NR_OF_0_RESULTS {NR_OF_0_RESULTS} >= {args.max_nr_of_zero_results}"

            print_debug(_wrn)
            progressbar_description([_wrn])

            raise SearchSpaceExhausted("Search space exhausted")
        log_what_needs_to_be_logged()

    #wait_for_jobs_to_complete(2)

    while len(global_vars["jobs"]): # pragma: no cover
        wait_for_jobs_to_complete(1)
        finish_previous_jobs([f"waiting for jobs ({len(global_vars['jobs'])} left)"])

        if is_slurm_job() and not args.force_local_execution:
            _sleep(1)

    log_what_needs_to_be_logged()
    return False

def wait_for_jobs_to_complete(_num_parallel_jobs: int) -> None: # pragma: no cover
    if SYSTEM_HAS_SBATCH: # pragma: no cover
        while len(global_vars["jobs"]) > _num_parallel_jobs:
            print_debug(f"Waiting for jobs to finish since it equals or exceeds the num_random_steps ({_num_parallel_jobs}), currently, len(global_vars['jobs']) = {len(global_vars['jobs'])}")
            progressbar_description([f"waiting for old jobs to finish ({len(global_vars['jobs'])} left)"])
            if is_slurm_job() and not args.force_local_execution:
                _sleep(5)

            finish_previous_jobs([f"waiting for jobs ({len(global_vars['jobs'])} left)"])

            clean_completed_jobs()

def human_readable_generation_strategy() -> Optional[str]:
    if ax_client:
        generation_strategy_str = str(ax_client.generation_strategy)

        _pattern: str = r'\[(.*?)\]'

        match = re.search(_pattern, generation_strategy_str)

        if match:
            content = match.group(1)
            return content

    return None

def die_orchestrator_exit_code_206(_test: bool) -> None: # pragma: no cover
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
                valid_behaviours: list = ["ExcludeNodeAndRestartAll", "RestartOnDifferentNode", "ExcludeNode", "Restart"]

                for x in data["errors"]: # pragma: no cover
                    if not isinstance(x, dict):
                        print_red(f"Entry is not of type dict but {type(x)}")
                        die_orchestrator_exit_code_206(_test)

                    if set(x.keys()) != set(valid_keys):
                        print_red(f"{x.keys()} does not match {valid_keys}")
                        die_orchestrator_exit_code_206(_test)

                    if x["behavior"] not in valid_behaviours:
                        print_red(f"behavior-entry {x['behavior']} is not in valid_behaviours: {', '.join(valid_behaviours)}")
                        die_orchestrator_exit_code_206(_test)

                    if not isinstance(x["name"], str):
                        print_red(f"name-entry is not string but {type(x['name'])}")
                        die_orchestrator_exit_code_206(_test)

                    if not isinstance(x["match_strings"], list):
                        print_red(f"name-entry is not list but {type(x['match_strings'])}")
                        die_orchestrator_exit_code_206(_test)

                    for y in x["match_strings"]:
                        if not isinstance(y, str):
                            print_red("x['match_strings'] is not a string but {type(x['match_strings'])}")
                            die_orchestrator_exit_code_206(_test)

                return data
            except Exception as e: # pragma: no cover
                print(f"Error while parse_experiment_parameters({_f}): {e}")
    else:
        print_red(f"{_f} could not be found")

    return None

def set_orchestrator() -> None:
    global orchestrator

    if args.orchestrator_file:
        if SYSTEM_HAS_SBATCH: # pragma: no cover
            orchestrator = parse_orchestrator_file(args.orchestrator_file, False)
        else:
            print_yellow("--orchestrator_file will be ignored on non-sbatch-systems.")

def die_no_random_steps() -> None:
    my_exit(233)

def check_if_has_random_steps() -> None:
    if (not args.continue_previous_job and not args.load_previous_job_data and "--continue" not in sys.argv) and (args.num_random_steps == 0 or not args.num_random_steps):
        print_red("You have no random steps set. This is only allowed in continued jobs. To start, you need either some random steps, or a continued run.")
        die_no_random_steps()

def add_exclude_to_defective_nodes() -> None:
    if args.exclude: # pragma: no cover
        entries = [entry.strip() for entry in args.exclude.split(',')]

        for entry in entries:
            count_defective_nodes(None, entry)

def check_max_eval(_max_eval: int) -> None:
    if not _max_eval: # pragma: no cover
        print_red("--max_eval needs to be set!")
        my_exit(19)

def parse_parameters() -> Union[Tuple[Any | None, Any | None], Tuple[Any | None, Any | None]]:
    experiment_parameters = None
    cli_params_experiment_parameters = None
    if args.parameter:
        experiment_parameters = parse_experiment_parameters()
        cli_params_experiment_parameters = experiment_parameters
    return experiment_parameters, cli_params_experiment_parameters

def main() -> None:
    global RESULT_CSV_FILE, ax_client, global_vars, max_eval
    global NVIDIA_SMI_LOGS_BASE
    global LOGFILE_DEBUG_GET_NEXT_TRIALS, random_steps
    check_if_has_random_steps()

    log_worker_creation()

    original_print("./omniopt " + " ".join(sys.argv[1:]))
    check_slurm_job_id()

    set_run_folder()
    RESULT_CSV_FILE = create_folder_and_file(get_current_run_folder())

    if os.getenv("CI"):
        data_dict: dict = {
            "param1": "value1",
            "param2": "value2",
            "param3": "value3"
        }

        error_description: str = "Some error occurred during execution (this is not a real error!)."

        write_failed_logs(data_dict, error_description)

    save_state_files()
    write_run_uuid_to_file()

    handle_maximize_argument()
    print_run_info()

    initialize_nvidia_logs()
    write_ui_url_if_present()

    LOGFILE_DEBUG_GET_NEXT_TRIALS = f'{get_current_run_folder()}/get_next_trials.csv'
    experiment_parameters, cli_params_experiment_parameters = parse_parameters()

    disable_logging()
    check_max_eval(max_eval)

    random_steps, second_step_steps = get_number_of_steps(max_eval)
    add_exclude_to_defective_nodes()
    handle_random_steps()

    gs = get_generation_strategy()
    initialize_ax_client(gs)

    minimize_or_maximize: bool = not args.maximize
    ax_client, experiment_parameters, experiment_args = get_experiment_parameters([
        args.continue_previous_job,
        args.seed,
        args.experiment_constraints,
        args.parameter,
        cli_params_experiment_parameters,
        experiment_parameters,
        minimize_or_maximize
    ])

    set_orchestrator()
    print_generation_strategy()

    checkpoint_parameters_filepath = f"{get_current_run_folder()}/state_files/checkpoint.json.parameters.json"
    save_experiment_parameters(checkpoint_parameters_filepath, experiment_parameters)
    print_overview_tables(experiment_parameters, experiment_args)

    try:
        set_global_executor()
    except ModuleNotFoundError as e: # pragma: no cover
        print_red(f"set_global_executor() failed with error {e}. It may help if you can delete and re-install the virtual Environment containing the OmniOpt2 modules.")
        sys.exit(244)

    load_existing_job_data_into_ax_client()
    original_print(f"Run-Program: {global_vars['joined_run_program']}")

    save_global_vars()
    write_process_info()

    start_live_share_background_job()

    write_continue_run_uuid_to_file()

    disable_tqdm = args.disable_tqdm or ci_env

    run_search_with_progress_bar(disable_tqdm)

    wait_for_jobs_to_complete(0)
    end_program(RESULT_CSV_FILE)

def log_worker_creation() -> None:
    _debug_worker_creation("time, nr_workers, got, requested, phase")

def set_run_folder() -> None:
    global CURRENT_RUN_FOLDER
    RUN_FOLDER_NUMBER: int = 0
    CURRENT_RUN_FOLDER = f"{args.run_dir}/{global_vars['experiment_name']}/{RUN_FOLDER_NUMBER}"

    while os.path.exists(f"{CURRENT_RUN_FOLDER}"):
        RUN_FOLDER_NUMBER += 1
        CURRENT_RUN_FOLDER = f"{args.run_dir}/{global_vars['experiment_name']}/{RUN_FOLDER_NUMBER}"

def handle_maximize_argument() -> None:
    if args.maximize: # pragma: no cover
        print_red("--maximize is not fully supported yet!")

def print_run_info() -> None:
    print(f"[yellow]Run-folder[/yellow]: [underline]{get_current_run_folder()}[/underline]")
    if args.continue_previous_job:
        print(f"[yellow]Continuation from {args.continue_previous_job}[/yellow]")

def initialize_nvidia_logs() -> None:
    global NVIDIA_SMI_LOGS_BASE
    NVIDIA_SMI_LOGS_BASE = f'{get_current_run_folder()}/gpu_usage_'

def write_ui_url_if_present() -> None:
    if args.ui_url:
        with open(f"{get_current_run_folder()}/ui_url.txt", mode="a", encoding="utf-8") as myfile:
            myfile.write(decode_if_base64(args.ui_url))

def handle_random_steps() -> None:
    global random_steps
    if args.parameter and args.continue_previous_job and random_steps <= 0: # pragma: no cover
        print(f"A parameter has been reset, but the earlier job already had its random phase. To look at the new search space, {args.num_random_steps} random steps will be executed.")
        random_steps = args.num_random_steps

def initialize_ax_client(gs: Any) -> None:
    global ax_client
    ax_client = AxClient(
        verbose_logging=args.verbose,
        enforce_sequential_optimization=args.enforce_sequential_optimization,
        generation_strategy=gs
    )

    ax_client = cast(AxClient, ax_client)

def print_generation_strategy() -> None:
    gs_hr = human_readable_generation_strategy()
    if gs_hr:
        print(f"Generation strategy: {gs_hr}")

def save_experiment_parameters(filepath: str, experiment_parameters: dict) -> None:
    with open(filepath, mode="w", encoding="utf-8") as outfile:
        json.dump(experiment_parameters, outfile, cls=NpEncoder)

def run_search_with_progress_bar(disable_tqdm: bool) -> None:
    with tqdm(total=max_eval, disable=disable_tqdm, ascii="░▒█") as _progress_bar:
        write_process_info()
        global progress_bar
        progress_bar = _progress_bar

        progressbar_description(["Started OmniOpt2 run..."])
        update_progress_bar(progress_bar, count_done_jobs())

        run_search(progress_bar)

        wait_for_jobs_to_complete(num_parallel_jobs)

def complex_tests(_program_name: str, wanted_stderr: str, wanted_exit_code: int, wanted_signal: Union[int, None], res_is_none: bool = False) -> int:
    print_yellow(f"Test suite: {_program_name}")

    nr_errors: int = 0

    program_path: str = f"./.tests/test_wronggoing_stuff.bin/bin/{_program_name}"

    if not os.path.exists(program_path): # pragma: no cover
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
        stdout_stderr_exit_code_signal = execute_bash_code(program_string_with_params)

        print("stdout_stderr_exit_code_signal:")
        print(stdout_stderr_exit_code_signal)

        stdout = stdout_stderr_exit_code_signal[0]
        stderr = stdout_stderr_exit_code_signal[1]
        exit_code = stdout_stderr_exit_code_signal[2]
        _signal = stdout_stderr_exit_code_signal[3]

        res = get_result(stdout)

        if res_is_none:
            nr_errors += is_equal(f"{_program_name} res is None", None, res)
        else:
            nr_errors += is_equal(f"{_program_name} res type is nr", True, isinstance(res, (float, int, list)))
        nr_errors += is_equal(f"{_program_name} stderr", True, wanted_stderr in stderr)
        nr_errors += is_equal(f"{_program_name} exit-code ", exit_code, wanted_exit_code)
        nr_errors += is_equal(f"{_program_name} signal", _signal, wanted_signal)

        return nr_errors
    except Exception as e: # pragma: no cover
        print_red(f"Error complex_tests: {e}")

        return 1

@wrapper_print_debug
def get_files_in_dir(mypath: str) -> list:
    print_debug("get_files_in_dir")
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    return [mypath + "/" + s for s in onlyfiles]

@wrapper_print_debug
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
            if os.path.exists(i): # pragma: no cover
                print("Missing {i} in find_file_paths string!")
                nr_errors += 1

    return nr_errors

@wrapper_print_debug
def run_tests() -> None:
    print_red("This should be red")
    print_yellow("This should be yellow")
    print_green("This should be green")

    global global_vars

    print(f"Printing test from current line {get_line_info()}")

    nr_errors: int = 0

    try:
        ie = is_equal('get_max_column_value(".tests/_plot_example_runs/ten_params/0/IDONTEVENEXIST/results.csv", "result", -123)', str(get_min_column_value(".tests/_plot_example_runs/ten_params/0/IDONTEVENEXIST/results.csv", "result", -123)), '-123')

        if not ie: # pragma: no cover
            nr_errors += 1
    except FileNotFoundError:
        pass
    except Exception as e: # pragma: no cover
        print(f"get_max_column_value on a non-existing file path excepted with another exception than FileNotFoundError (only acceptable one!). Error: {e}")
        nr_errors += 1

    non_rounded_lower, non_rounded_upper = round_lower_and_upper_if_type_is_int("float", -123.4, 123.4)
    nr_errors += is_equal("non_rounded_lower", non_rounded_lower, -123.4)
    nr_errors += is_equal("non_rounded_upper", non_rounded_upper, 123.4)

    rounded_lower, rounded_upper = round_lower_and_upper_if_type_is_int("int", -123.4, 123.4)
    nr_errors += is_equal("rounded_lower", rounded_lower, -124)
    nr_errors += is_equal("rounded_upper", rounded_upper, 124)

    nr_errors += is_equal('get_max_column_value(".tests/_plot_example_runs/ten_params/0/results.csv", "result", -123)', str(get_min_column_value(".tests/_plot_example_runs/ten_params/0/results.csv", "result", -123)), '17143005390319.627')
    nr_errors += is_equal('get_max_column_value(".tests/_plot_example_runs/ten_params/0/results.csv", "result", -123)', str(get_max_column_value(".tests/_plot_example_runs/ten_params/0/results.csv", "result", -123)), '9.865416064838896e+29')

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

    nr_errors += is_equal(f'check_for_non_zero_exit_codes("{nonzerodebug}")', check_for_non_zero_exit_codes(nonzerodebug), ["Non-zero exit-code detected: 159.  (May mean " + get_exit_codes()[str(159)] + ", unless you used that exit code yourself or it was part of any of your used libraries or programs)"])

    nr_errors += is_equal('state_from_job("")', state_from_job(''), "None")

    nr_errors += is_equal('print_image_to_cli("", "")', print_image_to_cli("", 1200), False)
    nr_errors += is_equal('print_image_to_cli(".tools/slimer.png", 200)', print_image_to_cli(".tools/slimer.png", 200), True)

    _check_for_basic_string_errors_example_str: str = """
    Exec format error
    """

    nr_errors += is_equal('check_for_basic_string_errors("_check_for_basic_string_errors_example_str", "", [], "")', check_for_basic_string_errors(_check_for_basic_string_errors_example_str, "", [], ""), [f"Was the program compiled for the wrong platform? Current system is {platform.machine()}", "No files could be found in your program string: "])

    nr_errors += is_equal("get_old_result_by_params('', {})", get_old_result_by_params('', {}), None)
    nr_errors += is_equal("get_old_result_by_params('.tests/_plot_example_runs/empty_resultsfile/0/results.csv', {})", get_old_result_by_params('.tests/_plot_example_runs/empty_resultsfile/0/results.csv', {}), None)

    nr_errors += is_equal('state_from_job("state=\"FINISHED\")', state_from_job('state="FINISHED"'), "finished")

    nr_errors += is_equal('state_from_job("state=\"FINISHED\")', state_from_job('state="FINISHED"'), "finished")

    nr_errors += is_equal('load_data_from_existing_run_folders("/dev/i/dont/exist/0")', load_data_from_existing_run_folders("/dev/i/dont/exist/0"), None)

    nr_errors += is_equal('load_data_from_existing_run_folders(["/dev/i/dont/exist/0", "/dev/i/dont/exist/1"])', load_data_from_existing_run_folders(["/dev/i/dont/exist/0", "/dev/i/dont/exist/1"]), None)

    nr_errors += is_equal('get_first_line_of_file_that_contains_string("IDONTEXIST", "HALLO")', get_first_line_of_file_that_contains_string("IDONTEXIST", "HALLO"), "")

    nr_errors += is_equal('compare_parameters("x", "y")', compare_parameters("x", "y"), '')

    nr_errors += is_equal('extract_info("OO-Info: SLURM_JOB_ID: 123")', json.dumps(extract_info("OO-Info: SLURM_JOB_ID: 123")), '[["OO_Info_SLURM_JOB_ID"], ["123"]]')

    if not SYSTEM_HAS_SBATCH or args.run_tests_that_fail_on_taurus:
        nr_errors += complex_tests("signal_but_has_output", "Killed", 137, None) # Doesnt show Killed on taurus
        nr_errors += complex_tests("signal", "Killed", 137, None, True) # Doesnt show Killed on taurus
    else: # pragma: no cover
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

    equal: list = [
        ["helpers.convert_string_to_number('123.123')", 123.123],
        ["helpers.convert_string_to_number('1')", 1],
        ["helpers.convert_string_to_number('-1')", -1],
        ["helpers.convert_string_to_number(None)", None],
        ["get_result(None)", None],
        ["parse_parameter_type_error(None)", None],
        ["parse_parameter_type_error(\"Value for parameter xxx: bla is of type <class 'int'>, expected <class 'float'>.\")", example_parse_parameter_type_error_result],
        ["get_hostname_from_outfile(None)", None],
        ["get_result(123)", None],
        ["get_result('RESULT: 10')", [10.0]],
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
            json.dumps({'result': 123.0})
    )

    nr_errors += is_equal(
            "evaluate({'x': -0.05})",
            json.dumps(evaluate({'x': -0.05})),
            json.dumps({'result': -0.05})
    )

    #complex_tests (_program_name, wanted_stderr, wanted_exit_code, wanted_signal, res_is_none=False):
    _complex_tests: list = [
        ["simple_ok", "hallo", 0, None],
        ["divide_by_0", 'Illegal division by zero at ./.tests/test_wronggoing_stuff.bin/bin/divide_by_0 line 3.\n', 255, None, True],
        ["result_but_exit_code_stdout_stderr", "stderr", 5, None],
        ["exit_code_no_output", "", 5, None, True],
        ["exit_code_stdout", "STDERR", 5, None, False],
        ["no_chmod_x", "Permission denied", 126, None, True],
        ["exit_code_stdout_stderr", "This has stderr", 5, None, True],
        ["module_not_found", "ModuleNotFoundError", 1, None, True]
    ]

    for _item in _complex_tests:
        nr_errors += complex_tests(*_item)

    find_path_res = test_find_paths("ls")
    if find_path_res: # pragma: no cover
        is_equal("test_find_paths failed", True, False)
        nr_errors += find_path_res

    orchestrator_yaml: str = ".tests/example_orchestrator_config.yaml"

    if os.path.exists(orchestrator_yaml):
        _is: str = json.dumps(parse_orchestrator_file(orchestrator_yaml, True))
        should_be: str = '{"errors": [{"name": "GPUDisconnected", "match_strings": ["AssertionError: ``AmpOptimizerWrapper`` is only available"], "behavior": "ExcludeNode"}, {"name": "Timeout", "match_strings": ["Timeout"], "behavior": "RestartOnDifferentNode"}, {"name": "StorageError", "match_strings": ["Read/Write failure"], "behavior": "ExcludeNodeAndRestartAll"}]}'
        nr_errors += is_equal(f"parse_orchestrator_file({orchestrator_yaml})", should_be, _is)
    else: # pragma: no cover
        nr_errors += is_equal(".tests/example_orchestrator_config.yaml exists", True, False)

    _example_csv_file: str = ".gui/_share_test_case/test_user/ClusteredStatisticalTestDriftDetectionMethod_NOAAWeather/0/results.csv"
    _best_results_from_example_file_minimize: str = json.dumps(get_best_params_from_csv(_example_csv_file, False))
    _expected_best_result_minimize: str = json.dumps(json.loads('{"result": "0.6951756801409847", "parameters": {"arm_name": "392_0", "trial_status": "COMPLETED", "generation_method": "BoTorch", "n_samples":  "905", "confidence": "0.1", "feature_proportion": "0.049534662817342145",  "n_clusters": "3"}}'))

    nr_errors += is_equal(f"Testing get_best_params_from_csv('{_example_csv_file}', False)", _expected_best_result_minimize, _best_results_from_example_file_minimize)

    _best_results_from_example_file_maximize: str = json.dumps(get_best_params_from_csv(_example_csv_file, True))
    _expected_best_result_maximize: str = json.dumps(json.loads('{"result": "0.7404449829276352", "parameters": {"arm_name": "132_0", "trial_status": "COMPLETED", "generation_method": "BoTorch", "n_samples": "391", "confidence": "0.001", "feature_proportion": "0.022059224931466673", "n_clusters": "4"}}'))

    nr_errors += is_equal(f"Testing get_best_params_from_csv('{_example_csv_file}', True)", _expected_best_result_maximize, _best_results_from_example_file_maximize)

    _print_best_result(_example_csv_file, False, False)

    nr_errors += is_equal("get_workers_string()", get_workers_string(), "")

    nr_errors += is_equal("check_file_info('/dev/i/dont/exist')", check_file_info('/dev/i/dont/exist'), "")

    nr_errors += is_equal(
        "get_parameters_from_outfile()",
        get_parameters_from_outfile(""),
        None
    )

    nr_errors += is_equal("calculate_moo(None)", calculate_moo(None), VAL_IF_NOTHING_FOUND)
    nr_errors += is_equal("calculate_moo([])", calculate_moo([]), VAL_IF_NOTHING_FOUND)

    nr_errors += is_equal("calculate_signed_harmonic_distance(None)", calculate_signed_harmonic_distance(None), 0)
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

    my_exit(nr_errors)

def live_share_background(interval: int) -> None:
    if not args.live_share: # pragma: no cover
        return

    while True:
        live_share()
        time.sleep(interval)

def start_live_share_background_job() -> None:
    if not args.live_share: # pragma: no cover
        return

    live_share()

    interval: int = 10
    thread = threading.Thread(target=live_share_background, args=(interval,), daemon=True)
    thread.start()

@wrapper_print_debug
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
            except (SignalUSR, SignalINT, SignalCONT, KeyboardInterrupt): # pragma: no cover
                print_red("\n⚠ You pressed CTRL+C or got a signal. Optimization stopped.")

                end_program(RESULT_CSV_FILE, False, 1)
            except SearchSpaceExhausted:
                _get_perc: int = abs(int(((count_done_jobs() - NR_INSERTED_JOBS) / max_eval) * 100))

                if _get_perc < 100:
                    print_red(f"\nIt seems like the search space was exhausted. "
                        f"You were able to get {_get_perc}% of the jobs you requested "
                        f"(got: {count_done_jobs() - NR_INSERTED_JOBS}, "
                        f"requested: {max_eval}) after main ran"
                    )

                if _get_perc != 100:
                    end_program(RESULT_CSV_FILE, True, 87)
                else: # pragma: no cover
                    end_program(RESULT_CSV_FILE, True)

if __name__ == "__main__":
    main_outside()
