#!/bin/env python3

# Idee: gridsearch implementieren, i.e. range -> choice mit allen werten zwischen low und up
# Geht, aber was ist mit continued runs?

import sys
import platform

ORCHESTRATE_TODO = {}

class SignalUSR (Exception):
    pass

class SignalINT (Exception):
    pass

class SignalCONT (Exception):
    pass

try:
    import traceback
    import time
except ModuleNotFoundError as e:
    print(f"ModuleNotFoundError: {e}")
    sys.exit(31)

def my_exit(_code=0):
    tb = traceback.format_exc()

    try:
        print_debug(f"Exiting with error code {_code}. Traceback: {tb}")
    except NameError:
        print(f"Exiting with error code {_code}. Traceback: {tb}")

    if (is_slurm_job() and not args.force_local_execution) and not (args.show_sixel_scatter or args.show_sixel_general or args.show_sixel_trial_index_result or args.show_sixel_graphics):
        _sleep(5)

    print("Exit-Code: " + str(_code))
    sys.exit(_code)

class SearchSpaceExhausted (Exception):
    pass

NR_INSERTED_JOBS = 0
changed_grid_search_params = {}
executor = None
LAST_CPU_MEM_TIME = None

SUPPORTED_MODELS = [
    "SOBOL",
    "GPEI",
    "FACTORIAL",
    "SAASBO",
    "FULLYBAYESIAN",
    "LEGACY_BOTORCH",
    "BOTORCH_MODULAR",
    "UNIFORM",
    "BO_MIXED"
]

NR_OF_0_RESULTS = 0
original_print = print

orchestrator = None
double_hashes = []
missing_results = []
already_inserted_param_hashes = {}
already_inserted_param_data = []

console = None

try:
    from rich.console import Console

    console = Console(
        force_terminal=True,
        force_interactive=True,
        soft_wrap=True,
        color_system="256"
    )

    with console.status("[bold green]Loading yaml...") as status:
        import yaml
    with console.status("[bold green]Loading psutil...") as status:
        import psutil
    with console.status("[bold green]Loading uuid...") as status:
        import uuid
    with console.status("[bold green]Loading inspect...") as status:
        import inspect
        from inspect import currentframe, getframeinfo
    with console.status("[bold green]Loading tokenize...") as status:
        import tokenize
    with console.status("[bold green]Loading os...") as status:
        import os
    with console.status("[bold green]Loading threading...") as status:
        import threading
    with console.status("[bold green]Loading shutil...") as status:
        import shutil
    with console.status("[bold green]Loading math...") as status:
        import math
    with console.status("[bold green]Loading json...") as status:
        import json
    with console.status("[bold green]Loading itertools...") as status:
        from itertools import combinations
    with console.status("[bold green]Loading importlib...") as status:
        import importlib.util
    with console.status("[bold green]Loading signal...") as status:
        import signal
    with console.status("[bold green]Loading datetime...") as status:
        import datetime
    with console.status("[bold green]Loading difflib...") as status:
        import difflib
    with console.status("[bold green]Loading warnings...") as status:
        import warnings
    with console.status("[bold green]Loading pandas...") as status:
        import pandas as pd
    with console.status("[bold green]Loading pathlib...") as status:
        from pathlib import Path
    with console.status("[bold green]Loading os...") as status:
        from os import listdir
        from os.path import isfile, join
    with console.status("[bold green]Loading re...") as status:
        import re
    with console.status("[bold green]Loading socket...") as status:
        import socket
    with console.status("[bold green]Loading stat...") as status:
        import stat
    with console.status("[bold green]Loading pwd...") as status:
        import pwd
    with console.status("[bold green]Loading base64...") as status:
        import base64
    with console.status("[bold green]Loading argparse...") as status:
        import argparse
    with console.status("[bold green]Loading rich_argparse...") as status:
        from rich_argparse import RichHelpFormatter
    with console.status("[bold green]Loading pformat...") as status:
        from pprint import pformat
    with console.status("[bold green]Loading sixel...") as status:
        import sixel
    with console.status("[bold green]Loading PIL...") as status:
        from PIL import Image

    #with console.status("[bold green]Importing rich tracebacks...") as status:
    #    #from rich.traceback import install
    #    #install(show_locals=True)

    with console.status("[bold green]Loading rich.table...") as status:
        from rich.table import Table
    with console.status("[bold green]Loading print from rich...") as status:
        from rich import print
    with console.status("[bold green]Loading csv...") as status:
        import csv
    with console.status("[bold green]Loading rich.pretty...") as status:
        from rich.pretty import pprint
    with console.status("[bold green]Loading subprocess...") as status:
        import subprocess
    with console.status("[bold green]Loading logging...") as status:
        import logging
        logging.basicConfig(level=logging.ERROR)
    with console.status("[bold green]Loading tqdm...") as status:
        from tqdm import tqdm
except ModuleNotFoundError as e:
    original_print(f"Base modules could not be loaded: {e}")
    my_exit(31)
except SignalINT:
    print("\n⚠ Signal INT was detected. Exiting with 128 + 2.")
    my_exit(128 + 2)
except SignalUSR:
    print("\n⚠ Signal USR was detected. Exiting with 128 + 10.")
    my_exit(128 + 10)
except SignalCONT:
    print("\n⚠ Signal CONT was detected. Exiting with 128 + 18.")
    my_exit(128 + 18)
except KeyboardInterrupt:
    print("\n⚠ You pressed CTRL+C. Program execution halted.")
    my_exit(0)

process = psutil.Process(os.getpid())

global_vars = {}

VAL_IF_NOTHING_FOUND = 99999999999999999999999999999999999999999999999999999999999
NO_RESULT = "{:.0e}".format(VAL_IF_NOTHING_FOUND)

global_vars["jobs"] = []
global_vars["_time"] = None
global_vars["mem_gb"] = None
global_vars["num_parallel_jobs"] = None
global_vars["parameter_names"] = []

# max_eval usw. in unterordner
# grid ausblenden
PD_CSV_FILENAME = "results.csv"
worker_percentage_usage = []
IS_IN_EVALUATE = False
END_PROGRAM_RAN = False
ALREADY_SHOWN_WORKER_USAGE_OVER_TIME = False
ax_client = None
TIME_NEXT_TRIALS_TOOK = []
CURRENT_RUN_FOLDER = None
RUN_FOLDER_NUMBER = 0
args = None
RESULT_CSV_FILE = None
SHOWN_END_TABLE = False
max_eval = None
random_steps = None
progress_bar = None
SUM_OF_VALUES_FOR_TQDM = 0

main_pid = os.getpid()

run_uuid = uuid.uuid4()

def set_max_eval(new_max_eval):
    global max_eval

    #import traceback
    #print(f"set_max_eval(new_max_eval: {new_max_eval})")
    #traceback.print_stack()

    max_eval = new_max_eval

def log_what_needs_to_be_logged():
    if "write_worker_usage" in globals():
        try:
            write_worker_usage()
        except Exception:
            pass

    if "write_process_info" in globals():
        try:
            write_process_info()
        except Exception:
            pass

    if "log_nr_of_workers" in globals():
        try:
            log_nr_of_workers()
        except Exception:
            pass

def get_nesting_level(caller_frame):
    filename, caller_lineno, _, _, _ = inspect.getframeinfo(caller_frame)
    with open(filename, encoding="utf-8") as f:
        indentation_level = 0
        for token_record in tokenize.generate_tokens(f.readline):
            token_type, _, (token_lineno, _), _, _ = token_record
            if token_lineno > caller_lineno:
                pass
            elif token_type == tokenize.INDENT:
                indentation_level += 1
            elif token_type == tokenize.DEDENT:
                indentation_level -= 1
        return indentation_level

def _debug(msg, _lvl=0, ee=None):
    if _lvl > 3:
        original_print(f"Cannot write _debug, error: {ee}")
        sys.exit(193)

    try:
        with open(logfile, mode='a', encoding="utf-8") as f:
            original_print(msg, file=f)
    except FileNotFoundError:
        print_red("It seems like the run's folder was deleted during the run. Cannot continue.")
        sys.exit(99) # generalized code for run folder deleted during run
    except Exception as e:
        original_print("_debug: Error trying to write log file: " + str(e))

        _debug(msg, _lvl + 1, e)

def get_functions_stack_array():
    stack = inspect.stack()
    function_names = []
    for frame_info in stack[1:]:
        if str(frame_info.function) != "<module>" and str(frame_info.function) != "print_debug":
            if frame_info.function != "wrapper":
                function_names.insert(0, f"{frame_info.function} ({frame_info.lineno})")
    return "Function stack: " + (" -> ".join(function_names) + ":")

def print_debug(msg):
    time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #nl = get_nesting_level(inspect.currentframe().f_back)
    #_tabs = "\t" * nl
    _tabs = ""
    msg = f"{time_str}:{_tabs}{msg}"
    if args.debug:
        print(msg)

    try:
        _debug(f"{time_str}: {get_functions_stack_array()}")
    except Exception:
        pass
    _debug(msg)

def write_process_info():
    global LAST_CPU_MEM_TIME

    if LAST_CPU_MEM_TIME is None:
        process.cpu_percent(interval=0.0)
        LAST_CPU_MEM_TIME = time.time()
    elif abs(LAST_CPU_MEM_TIME - time.time()) < 1:
        pass
    else:
        try:
            cpu_usage = process.cpu_percent(interval=1)
            ram_usage = process.memory_info().rss / (1024 * 1024)  # in MB

            print_debug(f"CPU Usage: {cpu_usage}%, RAM Usage: {ram_usage} MB")
        except Exception as e:
            print_debug(f"Error retrieving process information: {str(e)}")
        LAST_CPU_MEM_TIME = None

def get_line_info():
    return (inspect.stack()[1][1], ":", inspect.stack()[1][2], ":", inspect.stack()[1][3])

#print(f"sys.path: {sys.path}")

with console.status("[bold green]Loading helpers-module...") as status:
    script_dir = os.path.dirname(os.path.realpath(__file__))
    helpers_file = f"{script_dir}/.helpers.py"
    spec = importlib.util.spec_from_file_location(
        name="helpers",
        location=helpers_file,
    )
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)

def print_image_to_cli(image_path, width):
    print("")
    try:
        # Einlesen des Bildes um die Dimensionen zu erhalten
        image = Image.open(image_path)
        original_width, original_height = image.size

        # Berechnen der proportionalen Höhe
        height = int((original_height / original_width) * width)

        # Erstellen des SixelConverters mit den neuen Dimensionen
        sixel_converter = sixel.converter.SixelConverter(image_path, w=width, h=height)

        # Schreiben der Ausgabe in sys.stdout
        sixel_converter.write(sys.stdout)
        _sleep(2)
    except Exception as e:
        print_debug(
            f"Error converting and resizing image: "
            f"{str(e)}, width: {width}, image_path: {image_path}"
        )

with console.status("[bold green]Defining creating .logs dir if it doesn't exist...") as status:
    LOG_DIR = ".logs"
    try:
        Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        original_print(f"Could not create logs for {os.path.abspath(LOG_DIR)}: " + str(e))

with console.status("[bold green]Defining variables...") as status:
    LOG_I = 0
    logfile = f'{LOG_DIR}/{LOG_I}'
    logfile_nr_workers = f'{LOG_DIR}/{LOG_I}_nr_workers'
    while os.path.exists(logfile):
        LOG_I = LOG_I + 1
        logfile = f'{LOG_DIR}/{LOG_I}'

    logfile_nr_workers = f'{LOG_DIR}/{LOG_I}_nr_workers'
    logfile_progressbar = f'{LOG_DIR}/{LOG_I}_progressbar'
    logfile_worker_creation_logs = f'{LOG_DIR}/{LOG_I}_worker_creation_logs'
    logfile_trial_index_to_param_logs = f'{LOG_DIR}/{LOG_I}_trial_index_to_param_logs'
    LOGFILE_DEBUG_GET_NEXT_TRIALS = None

    NVIDIA_SMI_LOGS_BASE = None

def log_message_to_file(_logfile, message, _lvl=0, ee=None):
    assert _logfile is not None, "Logfile path must be provided."
    assert message is not None, "Message to log must be provided."

    if _lvl > 3:
        original_print(f"Cannot write _debug, error: {ee}")
        return

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

def _log_trial_index_to_param(trial_index, _lvl=0, ee=None):
    log_message_to_file(logfile_trial_index_to_param_logs, trial_index, _lvl, ee)

def _debug_worker_creation(msg, _lvl=0, ee=None):
    log_message_to_file(logfile_worker_creation_logs, msg, _lvl, ee)

def append_to_nvidia_smi_logs(_file, _host, result, _lvl=0, ee=None):
    log_message_to_file(_file, result, _lvl, ee)

def _debug_get_next_trials(msg, _lvl=0, ee=None):
    log_message_to_file(LOGFILE_DEBUG_GET_NEXT_TRIALS, msg, _lvl, ee)

def _debug_progressbar(msg, _lvl=0, ee=None):
    log_message_to_file(logfile_progressbar, msg, _lvl, ee)

def print_red(text):
    helpers.print_color("red", text)

    if CURRENT_RUN_FOLDER:
        try:
            with open(f"{CURRENT_RUN_FOLDER}/oo_errors.txt", mode="a", encoding="utf-8") as myfile:
                myfile.write(text)
        except FileNotFoundError as e:
            print_red(
                f"Error: {e}. This may mean that the {CURRENT_RUN_FOLDER} was deleted during the run."
            )
            sys.exit(99)

def print_green(text):
    helpers.print_color("green", text)

def print_yellow(text):
    helpers.print_color("yellow", text)

def add_to_phase_counter(phase, nr=0, run_folder=""):
    if run_folder == "":
        run_folder = CURRENT_RUN_FOLDER
    return append_and_read(f'{run_folder}/state_files/phase_{phase}_steps', nr)

parser = argparse.ArgumentParser(
    prog="omniopt",
    description='A hyperparameter optimizer for slurmbased HPC-systems',
    epilog="Example:\n\n./omniopt --partition=alpha --experiment_name=neural_network --mem_gb=1 --time=60 --worker_timeout=60 --max_eval=500 --num_parallel_jobs=500 --gpus=0 --follow --run_program=bHMgJyUoYXNkYXNkKSc= --parameter epochs range 0 10 int --parameter epochs range 0 10 int",
    formatter_class=RichHelpFormatter
)

required = parser.add_argument_group('Required arguments', "These options have to be set")
required_but_choice = parser.add_argument_group('Required arguments that allow a choice', "Of these arguments, one has to be set to continue.")
optional = parser.add_argument_group('Optional', "These options are optional")
slurm = parser.add_argument_group('Slurm', "Parameters related to Slurm")
installing = parser.add_argument_group('Installing', "Parameters related to installing")
debug = parser.add_argument_group('Debug', "These options are mainly useful for debugging")

required.add_argument('--num_parallel_jobs', help='Number of parallel slurm jobs (default: 20)', type=int, default=20)
required.add_argument('--num_random_steps', help='Number of random steps to start with', type=int, default=20)
required.add_argument('--max_eval', help='Maximum number of evaluations', type=int)
required.add_argument('--worker_timeout', help='Timeout for slurm jobs (i.e. for each single point to be optimized)', type=int, default=30)
required.add_argument('--run_program', action='append', nargs='+', help='A program that should be run. Use, for example, $x for the parameter named x.', type=str)
required.add_argument('--experiment_name', help='Name of the experiment.', type=str)
required.add_argument('--mem_gb', help='Amount of RAM for each worker in GB (default: 1GB)', type=float, default=1)

required_but_choice.add_argument('--parameter', action='append', nargs='+', help="Experiment parameters in the formats (options in round brackets are optional): <NAME> range <LOWER BOUND> <UPPER BOUND> (<INT, FLOAT>) -- OR -- <NAME> fixed <VALUE> -- OR -- <NAME> choice <Comma-seperated list of values>", default=None)
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
optional.add_argument('--root_venv_dir', help=f'Where to install your modules to ($root_venv_dir/.omniax_..., default: {os.getenv("HOME")}', default=os.getenv("HOME"), type=str)
optional.add_argument('--exclude', help='A comma seperated list of values of excluded nodes (taurusi8009,taurusi8010)', default=None, type=str)
optional.add_argument('--main_process_gb', help='Amount of RAM for the main process in GB (default: 1GB)', type=float, default=4)
optional.add_argument('--max_nr_of_zero_results', help='Max. nr of successive zero results by ax_client.get_next_trials() before the search space is seen as exhausted. Default is 20', type=int, default=20)
optional.add_argument('--disable_search_space_exhaustion_detection', help='Disables automatic search space reduction detection', action='store_true', default=False)
optional.add_argument('--abbreviate_job_names', help='Abbreviate pending job names (r = running, p = pending, u = unknown, c = cancelling)', action='store_true', default=False)
optional.add_argument('--orchestrator_file', help='An orchestrator file', default=None, type=str)

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
slurm.add_argument('--tasks_per_node', help='ntasks', type=int, default=1)

installing.add_argument('--run_mode', help='Either local or docker', default="local", type=str)

debug.add_argument('--verbose', help='Verbose logging', action='store_true', default=False)
debug.add_argument('--debug', help='Enable debugging', action='store_true', default=False)
debug.add_argument('--no_sleep', help='Disables sleeping for fast job generation (not to be used on HPC)', action='store_true', default=False)
debug.add_argument('--tests', help='Run simple internal tests', action='store_true', default=False)
debug.add_argument('--show_worker_percentage_table_at_end', help='Show a table of percentage of usage of max worker over time', action='store_true', default=False)
debug.add_argument('--auto_exclude_defective_hosts', help='Run a Test if you can allocate a GPU on each node and if not, exclude it since the GPU driver seems to be broken somehow.', action='store_true', default=False)

args = parser.parse_args()

if args.model and str(args.model).upper() not in SUPPORTED_MODELS:
    print(f"Unspported model {args.model}. Cannot continue. Valid models are {', '.join(SUPPORTED_MODELS)}")
    my_exit(203)

if args.num_parallel_jobs:
    num_parallel_jobs = args.num_parallel_jobs

def decode_if_base64(input_str):
    try:
        decoded_bytes = base64.b64decode(input_str)
        decoded_str = decoded_bytes.decode('utf-8')
        return decoded_str
    except Exception:
        return input_str

def get_file_as_string(f):
    datafile = ""
    if not os.path.exists(f):
        print_debug(f"{f} not found!")
        return ""

    with open(f, encoding="utf-8") as _f:
        datafile = _f.readlines()

    return "\n".join(datafile)

global_vars["joined_run_program"] = ""
if not args.continue_previous_job:
    if args.run_program:
        global_vars["joined_run_program"] = " ".join(args.run_program[0])
        global_vars["joined_run_program"] = decode_if_base64(global_vars["joined_run_program"])
else:
    prev_job_folder = args.continue_previous_job
    prev_job_file = prev_job_folder + "/state_files/joined_run_program"
    if os.path.exists(prev_job_file):
        global_vars["joined_run_program"] = get_file_as_string(prev_job_file)
    else:
        print(f"The previous job file {prev_job_file} could not be found. You may forgot to add the run number at the end.")
        my_exit(44)

global_vars["experiment_name"] = args.experiment_name

def load_global_vars(_file):
    if not os.path.exists(_file):
        print(f"You've tried to continue a non-existing job: {_file}")
        sys.exit(44)
    try:
        global global_vars
        with open(_file, encoding="utf-8") as f:
            global_vars = json.load(f)
    except Exception as e:
        print("Error while loading old global_vars: " + str(e) + ", trying to load " + str(_file))
        my_exit(44)

if not args.tests:
    if args.continue_previous_job:
        load_global_vars(f"{args.continue_previous_job}/state_files/global_vars.json")

    if args.parameter is None and args.continue_previous_job is None:
        original_print("Either --parameter or --continue_previous_job is required. Both were not found.")
        my_exit(19)
    elif not args.run_program and not args.continue_previous_job:
        print("--run_program needs to be defined when --continue_previous_job is not set")
        my_exit(19)
    elif not global_vars["experiment_name"] and not args.continue_previous_job:
        print("--experiment_name needs to be defined when --continue_previous_job is not set")
        my_exit(19)
    elif args.continue_previous_job:
        if not os.path.exists(args.continue_previous_job):
            print_red(f"The previous job folder {args.continue_previous_job} could not be found!")
            my_exit(19)

        if not global_vars["experiment_name"]:
            exp_name_file = f"{args.continue_previous_job}/experiment_name"
            if os.path.exists(exp_name_file):
                global_vars["experiment_name"] = get_file_as_string(exp_name_file).strip()
            else:
                original_print(f"{exp_name_file} not found, and no --experiment_name given. Cannot continue.")
                my_exit(19)

    if not args.mem_gb:
        print("--mem_gb needs to be set")
        my_exit(19)

    if not args.time:
        if not args.continue_previous_job:
            print("--time needs to be set")
        else:
            time_file = args.continue_previous_job + "/state_files/time"
            if os.path.exists(time_file):
                TIME_FILE_CONTENTS = get_file_as_string(time_file).strip()
                if TIME_FILE_CONTENTS.isdigit():
                    global_vars["_time"] = int(TIME_FILE_CONTENTS)
                    print(f"Using old run's --time: {global_vars['_time']}")
                else:
                    print(f"Time-setting: The contents of {time_file} do not contain a single number")
            else:
                print(f"neither --time nor file {time_file} found")
                my_exit(19)
    else:
        global_vars["_time"] = args.time

    if not global_vars["_time"]:
        print("Missing --time parameter. Cannot continue.")
        sys.exit(19)

    if not args.mem_gb:
        if not args.continue_previous_job:
            print("--mem_gb needs to be set")
        else:
            mem_gb_file = args.continue_previous_job + "/state_files/mem_gb"
            if os.path.exists(mem_gb_file):
                mem_gb_file_contents = get_file_as_string(mem_gb_file).strip()
                if mem_gb_file_contents.isdigit():
                    mem_gb = int(mem_gb_file_contents)
                    print(f"Using old run's --mem_gb: {mem_gb}")
                else:
                    print(f"mem_gb-setting: The contents of {mem_gb_file} do not contain a single number")
            else:
                print(f"neither --mem_gb nor file {mem_gb_file} found")
                my_exit(19)
    else:
        mem_gb = int(args.mem_gb)

    if args.continue_previous_job and not args.gpus:
        gpus_file = args.continue_previous_job + "/state_files/gpus"
        if os.path.exists(gpus_file):
            GPUS_FILE_CONTENTS = get_file_as_string(gpus_file).strip()
            if GPUS_FILE_CONTENTS.isdigit():
                gpus = int(GPUS_FILE_CONTENTS)
                print(f"Using old run's --gpus: {gpus}")
            else:
                print(f"gpus-setting: The contents of {gpus_file} do not contain a single number")
        else:
            print(f"neither --gpus nor file {gpus_file} found")
            my_exit(19)
    else:
        set_max_eval(args.max_eval)

    if not args.max_eval:
        if not args.continue_previous_job:
            print("--max_eval needs to be set")
        else:
            max_eval_file = args.continue_previous_job + "/state_files/max_eval"
            if os.path.exists(max_eval_file):
                MAX_EVAL_FILE_CONTENTS = get_file_as_string(max_eval_file).strip()
                if MAX_EVAL_FILE_CONTENTS.isdigit():
                    set_max_eval(int(MAX_EVAL_FILE_CONTENTS))
                    print(f"Using old run's --max_eval: {max_eval}")
                else:
                    print(f"max_eval-setting: The contents of {max_eval_file} do not contain a single number")
            else:
                print(f"neither --max_eval nor file {max_eval_file} found")
                my_exit(19)
    else:
        set_max_eval(args.max_eval)

        if max_eval <= 0:
            print_red("--max_eval must be larger than 0")
            my_exit(19)

def print_debug_get_next_trials(got, requested, _line):
    time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"{time_str}, {got}, {requested}"

    _debug_get_next_trials(msg)

def print_debug_progressbar(msg):
    time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"{time_str}: {msg}"

    _debug_progressbar(msg)

def receive_usr_signal_one(signum, stack):
    raise SignalUSR(f"USR1-signal received ({signum})")

def receive_usr_signal_int(signum, stack):
    raise SignalINT(f"INT-signal received ({signum})")

def receive_signal_cont(signum, stack):
    raise SignalCONT(f"CONT-signal received ({signum})")

signal.signal(signal.SIGUSR1, receive_usr_signal_one)
signal.signal(signal.SIGUSR2, receive_usr_signal_one)
signal.signal(signal.SIGINT, receive_usr_signal_int)
signal.signal(signal.SIGTERM, receive_usr_signal_int)
signal.signal(signal.SIGCONT, receive_signal_cont)

def is_executable_in_path(executable_name):
    print_debug(f"is_executable_in_path({executable_name})")
    for path in os.environ.get('PATH', '').split(':'):
        executable_path = os.path.join(path, executable_name)
        if os.path.exists(executable_path) and os.access(executable_path, os.X_OK):
            return True
    return False

SYSTEM_HAS_SBATCH = False
IS_NVIDIA_SMI_SYSTEM = False

if is_executable_in_path("sbatch"):
    SYSTEM_HAS_SBATCH = True
if is_executable_in_path("nvidia-smi"):
    IS_NVIDIA_SMI_SYSTEM = True

if not SYSTEM_HAS_SBATCH:
    num_parallel_jobs = 1

def save_global_vars():
    state_files_folder = f"{CURRENT_RUN_FOLDER}/state_files"
    if not os.path.exists(state_files_folder):
        os.makedirs(state_files_folder)

    with open(f'{state_files_folder}/global_vars.json', mode="w", encoding="utf-8") as f:
        json.dump(global_vars, f)

def check_slurm_job_id():
    print_debug("check_slurm_job_id()")
    if SYSTEM_HAS_SBATCH:
        slurm_job_id = os.environ.get('SLURM_JOB_ID')
        if slurm_job_id is not None and not slurm_job_id.isdigit():
            print_red("Not a valid SLURM_JOB_ID.")
        elif slurm_job_id is None:
            print_red(
                "You are on a system that has SLURM available, but you are not running the main-script in a Slurm-Environment. "
                "This may cause the system to slow down for all other users. It is recommended you run the main script in a Slurm job."
            )

def create_folder_and_file(folder):
    print_debug(f"create_folder_and_file({folder})")

    try:
        if not os.path.exists(folder):
            os.makedirs(folder)
    except FileExistsError:
        print_red(f"create_folder_and_file({folder}) failed, because the folder already existed. Cannot continue.")
        sys.exit(13)

    file_path = os.path.join(folder, "results.csv")

    #with open(file_path, mode='w', encoding='utf-8') as file:
    #    pass

    return file_path

def sort_numerically_or_alphabetically(arr):
    try:
        # Check if all elements can be converted to numbers
        numbers = [float(item) for item in arr]
        # If successful, order them numerically
        sorted_arr = sorted(numbers)
    except ValueError:
        # If there's an error, order them alphabetically
        sorted_arr = sorted(arr)

    return sorted_arr

def get_program_code_from_out_file(f):
    if not os.path.exists(f):
        original_print(f"{f} not found")
        return ""

    fs = get_file_as_string(f)

    for line in fs.split("\n"):
        if "Program-Code:" in line:
            return line

    return ""

def get_max_column_value(pd_csv, column, _default):
    """
    Reads the CSV file and returns the maximum value in the specified column.

    :param pd_csv: The path to the CSV file.
    :param column: The column name for which the maximum value is to be found.
    :return: The maximum value in the specified column.
    """
    assert isinstance(pd_csv, str), "pd_csv must be a string"
    assert isinstance(column, str), "column must be a string"

    if not os.path.exists(pd_csv):
        raise FileNotFoundError(f"CSV file {pd_csv} not found")

    try:
        df = pd.read_csv(pd_csv, float_precision='round_trip')
        if column not in df.columns:
            print_red(f"Cannot load data from {pd_csv}: column {column} does not exist")
            return _default
        max_value = df[column].max()
        return max_value
    except Exception as e:
        print_red(f"Error while getting max value from column {column}: {str(e)}")
        raise

def get_min_column_value(pd_csv, column, _default):
    """
    Reads the CSV file and returns the minimum value in the specified column.

    :param pd_csv: The path to the CSV file.
    :param column: The column name for which the minimum value is to be found.
    :return: The minimum value in the specified column.
    """
    assert isinstance(pd_csv, str), "pd_csv must be a string"
    assert isinstance(column, str), "column must be a string"

    if not os.path.exists(pd_csv):
        raise FileNotFoundError(f"CSV file {pd_csv} not found")

    try:
        df = pd.read_csv(pd_csv, float_precision='round_trip')
        if column not in df.columns:
            print_red(f"Cannot load data from {pd_csv}: column {column} does not exist")
            return _default
        min_value = df[column].min()
        return min_value
    except Exception as e:
        print_red(f"Error while getting min value from column {column}: {str(e)}")
        raise

def get_ret_value_from_pd_csv(pd_csv, _type, _column, _default):
    if os.path.exists(pd_csv):
        if _type == "lower":
            _old_min_col = get_min_column_value(pd_csv, _column, _default)
            if _old_min_col:
                found_in_file = True

            if found_in_file and _default > _old_min_col:
                ret_val = _old_min_col
            else:
                ret_val = _default
        else:
            _old_max_col = get_max_column_value(pd_csv, _column, _default)
            if _old_max_col:
                found_in_file = True

            if found_in_file and _default < _old_max_col:
                ret_val = _old_max_col
            else:
                ret_val = _default
    else:
        print_red(f"{pd_csv} was not found")

    return ret_val

def get_bound_if_prev_data(_type, _column, _default):
    ret_val = _default

    found_in_file = False

    if args.load_previous_job_data and len(args.load_previous_job_data):
        prev_runs = helpers.flatten_extend(args.load_previous_job_data)
        for prev_run in prev_runs:
            pd_csv = f"{prev_run}/{PD_CSV_FILENAME}"

            ret_val = get_ret_value_from_pd_csv(pd_csv, _type, _column, _default)
    if args.continue_previous_job:
        pd_csv = f"{args.continue_previous_job}/{PD_CSV_FILENAME}"

        ret_val = get_ret_value_from_pd_csv(pd_csv, _type, _column, _default)

    return round(ret_val, 4), found_in_file

def parse_experiment_parameters():
    global global_vars
    global changed_grid_search_params

    params = []
    param_names = []

    i = 0

    search_space_reduction_warning = False

    valid_types = ["range", "fixed", "choice"]
    invalid_names = ["start_time", "end_time", "run_time", "program_string", "result", "exit_code", "signal"]

    while args.parameter and i < len(args.parameter):
        this_args = args.parameter[i]
        j = 0
        while j < len(this_args):
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

            if param_type == "range":
                j, params, search_space_reduction_warning = parse_range_param(params, j, this_args, name, search_space_reduction_warning)
            elif param_type == "fixed":
                j, params, search_space_reduction_warning = parse_fixed_param(params, j, this_args, name, search_space_reduction_warning)
            elif param_type == "choice":
                j, params, search_space_reduction_warning = parse_choice_param(params, j, this_args, name, search_space_reduction_warning)
            else:
                print_red(f"⚠ Parameter type '{param_type}' not yet implemented.")
                my_exit(181)
        i += 1

    if search_space_reduction_warning:
        print_red("⚠ Search space reduction is not currently supported on continued runs or runs that have previous data.")

    return params

def check_factorial_range():
    if args.model and args.model == "FACTORIAL":
        print_red("\n⚠ --model FACTORIAL cannot be used with range parameter")
        my_exit(181)

def check_if_range_types_are_invalid(value_type, valid_value_types):
    if value_type not in valid_value_types:
        valid_value_types_string = ", ".join(valid_value_types)
        print_red(f"⚠ {value_type} is not a valid value type. Valid types for range are: {valid_value_types_string}")
        my_exit(181)

def check_range_params_length(this_args):
    if len(this_args) != 5 and len(this_args) != 4:
        print_red("\n⚠ --parameter for type range must have 4 (or 5, the last one being optional and float by default) parameters: <NAME> range <START> <END> (<TYPE (int or float)>)")
        my_exit(181)

def die_181_if_lower_and_upper_bound_equal_zero(lower_bound, upper_bound):
    if upper_bound == lower_bound:
        if lower_bound == 0:
            print_red(f"⚠ Lower bound and upper bound are equal: {lower_bound}, cannot automatically fix this, because they -0 = +0 (usually a quickfix would be to set lower_bound = -upper_bound)")
            my_exit(181)
        print_red(f"⚠ Lower bound and upper bound are equal: {lower_bound}, setting lower_bound = -upper_bound")
        lower_bound = -upper_bound

def switch_lower_and_upper_if_needed(name, lower_bound, upper_bound):
    if lower_bound > upper_bound:
        print_yellow(f"⚠ Lower bound ({lower_bound}) was larger than upper bound ({upper_bound}) for parameter '{name}'. Switched them.")
        upper_bound, lower_bound = lower_bound, upper_bound

    return lower_bound, upper_bound

def round_lower_and_upper_if_type_is_int(value_type, lower_bound, upper_bound):
    if value_type == "int":
        if not helpers.looks_like_int(lower_bound):
            print_yellow(f"⚠ {value_type} can only contain integers. You chose {lower_bound}. Will be rounded down to {math.floor(lower_bound)}.")
            lower_bound = math.floor(lower_bound)

        if not helpers.looks_like_int(upper_bound):
            print_yellow(f"⚠ {value_type} can only contain integers. You chose {upper_bound}. Will be rounded up to {math.ceil(upper_bound)}.")
            upper_bound = math.ceil(upper_bound)

    return lower_bound, upper_bound

def parse_range_param(params, j, this_args, name, search_space_reduction_warning):
    check_factorial_range()

    check_range_params_length(this_args)

    try:
        lower_bound = float(this_args[j + 2])
    except Exception:
        print_red(f"\n⚠ {this_args[j + 2]} is not a number")
        my_exit(181)

    try:
        upper_bound = float(this_args[j + 3])
    except Exception:
        print_red(f"\n⚠ {this_args[j + 3]} is not a number")
        my_exit(181)

    die_181_if_lower_and_upper_bound_equal_zero(lower_bound, upper_bound)

    lower_bound, upper_bound = switch_lower_and_upper_if_needed(name, lower_bound, upper_bound)

    skip = 5

    try:
        value_type = this_args[j + 4]
    except Exception:
        value_type = "float"
        skip = 4

    valid_value_types = ["int", "float"]

    check_if_range_types_are_invalid(value_type, valid_value_types)

    old_lower_bound = lower_bound
    old_upper_bound = upper_bound

    lower_bound, upper_bound = round_lower_and_upper_if_type_is_int(value_type, lower_bound, upper_bound)

    lower_bound, found_lower_bound_in_file = get_bound_if_prev_data("lower", name, lower_bound)
    upper_bound, found_upper_bound_in_file = get_bound_if_prev_data("upper", name, upper_bound)

    if value_type == "int":
        lower_bound = math.floor(lower_bound)
        upper_bound = math.ceil(upper_bound)

    if args.continue_previous_job:
        if old_lower_bound != lower_bound and found_lower_bound_in_file and old_lower_bound < lower_bound:
            print_yellow(f"⚠ previous jobs contained smaller values for the parameter {name} than are currently possible. The lower bound will be set from {old_lower_bound} to {lower_bound}")
            search_space_reduction_warning = True

        if old_upper_bound != upper_bound and found_upper_bound_in_file and old_upper_bound > upper_bound:
            print_yellow(f"⚠ previous jobs contained larger values for the parameter {name} than are currently possible. The upper bound will be set from {old_upper_bound} to {upper_bound}")
            search_space_reduction_warning = True

    param = {
        "name": name,
        "type": "range",
        "bounds": [lower_bound, upper_bound],
        "value_type": value_type
    }

    if args.gridsearch:
        values = np.linspace(lower_bound, upper_bound, args.max_eval, endpoint=True).tolist()

        if value_type == "int":
            values = [int(value) for value in values]
            changed_grid_search_params[name] = f"Gridsearch from {helpers.to_int_when_possible(lower_bound)} to {helpers.to_int_when_possible(upper_bound)} ({args.max_eval} steps, int)"
        else:
            changed_grid_search_params[name] = f"Gridsearch from {helpers.to_int_when_possible(lower_bound)} to {helpers.to_int_when_possible(upper_bound)} ({args.max_eval} steps)"

        values = sorted(set(values))
        values = [str(helpers.to_int_when_possible(value)) for value in values]

        param = {
            "name": name,
            "type": "choice",
            "is_ordered": True,
            "values": values
        }

    global_vars["parameter_names"].append(name)

    params.append(param)

    j += skip

    return j, params, search_space_reduction_warning

def parse_fixed_param(params, j, this_args, name, search_space_reduction_warning):
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

def parse_choice_param(params, j, this_args, name, search_space_reduction_warning):
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

def replace_parameters_in_string(parameters, input_string):
    try:
        for param_item in parameters:
            input_string = input_string.replace(f"${param_item}", str(parameters[param_item]))
            input_string = input_string.replace(f"$({param_item})", str(parameters[param_item]))

            input_string = input_string.replace(f"%{param_item}", str(parameters[param_item]))
            input_string = input_string.replace(f"%({param_item})", str(parameters[param_item]))

        return input_string
    except Exception as e:
        print_red(f"\n⚠ Error: {e}")
        return None

def execute_bash_code(code):
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

def get_result(input_string):
    if input_string is None:
        print("get_result: Input-String is None")
        return None

    if not isinstance(input_string, str):
        print_debug(f"get_result: Type of input_string is not string, but {type(input_string)}")
        return None

    try:
        pattern = r'\s*RESULT:\s*(-?\d+(?:\.\d+)?)'

        match = re.search(pattern, input_string)

        if match:
            result_number = float(match.group(1))
            return result_number
        return None

    except Exception as e:
        print(f"Error extracting the RESULT-string: {e}")
        return None

def add_to_csv(file_path, heading, data_line):
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

def find_file_paths(_text):
    file_paths = []

    if isinstance(_text, str):
        words = _text.split()

        for word in words:
            if os.path.exists(word):
                file_paths.append(word)

        return file_paths

    return []

def check_file_info(file_path):
    if not os.path.exists(file_path):
        print(f"The file {file_path} does not exist.")
        return ""

    if not os.access(file_path, os.R_OK):
        print(f"The file {file_path} is not readable.")
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
    string += f"Status-Change-Time: {status_change_time}\n"
    string += f"Size: {size} Bytes\n"
    string += f"Permissions: {permissions}\n"
    string += f"Owner: {username}\n"
    string += f"Last access: {access_time}\n"
    string += f"Last modification: {modification_time}\n"

    string += f"Hostname: {socket.gethostname()}"

    return string

def find_file_paths_and_print_infos(_text, program_code):
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

def write_failed_logs(data_dict, error_description=""):
    assert isinstance(data_dict, dict), "The parameter must be a dictionary."
    assert isinstance(error_description, str), "The error_description must be a string."

    headers = list(data_dict.keys())
    data = [list(data_dict.values())]

    if error_description:
        headers.append('error_description')
        for row in data:
            row.append(error_description)

    failed_logs_dir = os.path.join(CURRENT_RUN_FOLDER, 'failed_logs')

    data_file_path = os.path.join(failed_logs_dir, 'parameters.csv')
    header_file_path = os.path.join(failed_logs_dir, 'headers.csv')

    try:
        # Create directories if they do not exist
        if not os.path.exists(failed_logs_dir):
            os.makedirs(failed_logs_dir)
            print_debug(f"Directory created: {failed_logs_dir}")

        # Write headers if the file does not exist
        if not os.path.exists(header_file_path):
            try:
                with open(header_file_path, mode='w', encoding='utf-8', newline='') as header_file:
                    writer = csv.writer(header_file)
                    writer.writerow(headers)
                    print_debug(f"Header file created with headers: {headers}")
            except Exception as e:
                print_red(f"Failed to write header file: {e}")
        else:
            print_debug("Header file already exists, skipping header writing.")

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

def test_gpu_before_evaluate(return_in_case_of_error):
    if SYSTEM_HAS_SBATCH and args.gpus >= 1 and args.auto_exclude_defective_hosts:
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

def evaluate(parameters):
    global IS_IN_EVALUATE

    start_nvidia_smi_thread()

    return_in_case_of_error = {"result": VAL_IF_NOTHING_FOUND}

    if args.maximize:
        return_in_case_of_error = {"result": -VAL_IF_NOTHING_FOUND}

    _test_gpu = test_gpu_before_evaluate(return_in_case_of_error)

    if _test_gpu is not None:
        return _test_gpu

    IS_IN_EVALUATE = True

    parameters = {k: (int(v) if isinstance(v, (int, float, str)) and re.fullmatch(r'^\d+(\.0+)?$', str(v)) else v) for k, v in parameters.items()}

    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    signal.signal(signal.SIGUSR2, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGQUIT, signal.SIG_IGN)

    try:
        original_print(f"Parameters: {json.dumps(parameters)}")

        parameters_keys = list(parameters.keys())
        parameters_values = list(parameters.values())

        program_string_with_params = replace_parameters_in_string(parameters, global_vars["joined_run_program"])

        program_string_with_params = program_string_with_params.replace('\r', ' ').replace('\n', ' ')

        string = find_file_paths_and_print_infos(program_string_with_params, program_string_with_params)

        original_print("Debug-Infos:", string)

        original_print(program_string_with_params)

        start_time = int(time.time())

        stdout_stderr_exit_code_signal = execute_bash_code(program_string_with_params)

        end_time = int(time.time())

        stdout = stdout_stderr_exit_code_signal[0]
        exit_code = stdout_stderr_exit_code_signal[2]
        _signal = stdout_stderr_exit_code_signal[3]

        run_time = end_time - start_time

        original_print("stdout:")
        original_print(stdout)

        result = get_result(stdout)

        original_print(f"Result: {result}")

        headline = ["start_time", "end_time", "run_time", "program_string", *parameters_keys, "result", "exit_code", "signal", "hostname"]
        values = [start_time, end_time, run_time, program_string_with_params, *parameters_values, result, exit_code, _signal, socket.gethostname()]

        original_print(f"EXIT_CODE: {exit_code}")

        headline = ['None' if element is None else element for element in headline]
        values = ['None' if element is None else element for element in values]

        if CURRENT_RUN_FOLDER is not None and os.path.exists(CURRENT_RUN_FOLDER):
            add_to_csv(f"{CURRENT_RUN_FOLDER}/job_infos.csv", headline, values)
        else:
            print_debug(f"evaluate: CURRENT_RUN_FOLDER {CURRENT_RUN_FOLDER} could not be found")

        if isinstance(result, (int, float)):
            IS_IN_EVALUATE = False
            return {"result": float(result)}

        write_failed_logs(parameters, "No Result")
    except SignalUSR:
        print("\n⚠ USR1-Signal was sent. Cancelling evaluation.")
        write_failed_logs(parameters, "USR1-signal")
    except SignalCONT:
        print("\n⚠ CONT-Signal was sent. Cancelling evaluation.")
        write_failed_logs(parameters, "CONT-signal")
    except SignalINT:
        print("\n⚠ INT-Signal was sent. Cancelling evaluation.")
        write_failed_logs(parameters, "INT-signal")

    IS_IN_EVALUATE = False

    return return_in_case_of_error

try:
    if not args.tests:
        with console.status("[bold green]Loading torch...") as status:
            import torch
        with console.status("[bold green]Loading numpy...") as status:
            import numpy as np
        with console.status("[bold green]Loading ax...") as status:
            import ax.modelbridge.generation_node
            import ax
            from ax.service.ax_client import AxClient, ObjectiveProperties
            import ax.exceptions.core
            from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
            from ax.modelbridge.registry import Models
            from ax.storage.json_store.save import save_experiment
            from ax.storage.json_store.load import load_experiment
        with console.status("[bold green]Loading botorch...") as status:
            import botorch
        with console.status("[bold green]Loading submitit...") as status:
            import submitit
            from submitit import LocalJob, DebugJob
except ModuleNotFoundError as e:
    original_print(f"Base modules could not be loaded: {e}")
    my_exit(31)
except SignalINT:
    print("\n⚠ Signal INT was detected. Exiting with 128 + 2.")
    my_exit(128 + 2)
except SignalUSR:
    print("\n⚠ Signal USR was detected. Exiting with 128 + 10.")
    my_exit(128 + 10)
except SignalCONT:
    print("\n⚠ Signal CONT was detected. Exiting with 128 + 18.")
    my_exit(128 + 18)
except KeyboardInterrupt:
    print("\n⚠ You pressed CTRL+C. Program execution halted.")
    my_exit(0)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def disable_logging():
    if args.verbose:
        return

    logging.basicConfig(level=logging.ERROR)
    logging.getLogger().setLevel(logging.ERROR)

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

    for _module in modules:
        logging.getLogger(_module).setLevel(logging.ERROR)

        for _cat in categories:
            warnings.filterwarnings("ignore", category=_cat)
            warnings.filterwarnings("ignore", category=_cat, module=_module)

def display_failed_jobs_table():
    _console = Console()

    failed_jobs_folder = f"{CURRENT_RUN_FOLDER}/failed_logs"
    header_file = os.path.join(failed_jobs_folder, "headers.csv")
    parameters_file = os.path.join(failed_jobs_folder, "parameters.csv")

    # Assert the existence of the folder and files
    if not os.path.exists(failed_jobs_folder):
        print_debug(f"Failed jobs {failed_jobs_folder} folder does not exist.")
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
    except Exception as e:
        print_red(f"Error: {str(e)}")

def plot_command(_command, tmp_file, _width=1300):
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
        print_image_to_cli(tmp_file, _width)
    else:
        print_debug(f"{tmp_file} not found, error: {error}")

def replace_string_with_params(input_string, params):
    try:
        assert isinstance(input_string, str), "Input string must be a string"
        replaced_string = input_string
        i = 0
        for param in params:
            #print(f"param: {param}, type: {type(param)}")
            replaced_string = replaced_string.replace(f"%{i}", param)
            i += 1
        return replaced_string
    except AssertionError as e:
        error_text = f"Error in replace_string_with_params: {e}"
        print(error_text)
        raise

def print_best_result(csv_file_path):
    global global_vars
    global SHOWN_END_TABLE

    try:
        best_params = get_best_params(csv_file_path)

        best_result = None

        if best_params and "result" in best_params:
            best_result = best_params["result"]
        else:
            best_result = NO_RESULT

        if str(best_result) == NO_RESULT or best_result is None or best_result == "None":
            print_red("Best result could not be determined")
            return 87

        total_str = f"total: {count_done_jobs() - NR_INSERTED_JOBS}"

        if NR_INSERTED_JOBS:
            total_str += f" + inserted jobs: {NR_INSERTED_JOBS}"

        failed_error_str = ""
        if failed_jobs() >= 1:
            failed_error_str = f", failed: {failed_jobs()}"

        table = Table(show_header=True, header_style="bold", title=f"Best parameter ({total_str}{failed_error_str}):")

        k = 0
        for key in best_params["parameters"].keys():
            if k > 2:
                table.add_column(key)
            k += 1

        table.add_column("result")

        row_without_result = [str(helpers.to_int_when_possible(best_params["parameters"][key])) for key in best_params["parameters"].keys()]
        row = [*row_without_result, str(best_result)][3:]

        table.add_row(*row)

        console.print(table)

        with console.capture() as capture:
            console.print(table)
        table_str = capture.get()

        with open(f'{CURRENT_RUN_FOLDER}/best_result.txt', mode="w", encoding="utf-8") as text_file:
            text_file.write(table_str)

        _pd_csv = f"{CURRENT_RUN_FOLDER}/{PD_CSV_FILENAME}"

        show_sixel_graphics(_pd_csv)

        SHOWN_END_TABLE = True
    except Exception as e:
        tb = traceback.format_exc()
        print_red(f"[print_best_result] Error during print_best_result: {e}, tb: {tb}")

    return -1

def get_plot_types(x_y_combinations):
    plot_types = []

    if args.show_sixel_trial_index_result:
        plot_types.append(
            {
                "type": "trial_index_result",
                "min_done_jobs": 2
            }
        )

    if args.show_sixel_scatter:
        plot_types.append(
            {
                "type": "scatter",
                "params": "--bubblesize=50 --allow_axes %0 --allow_axes %1",
                "iterate_through": x_y_combinations,
                "dpi": 76,
                "filename": "plot_%0_%1_%2" # omit file ending
            }
        )

    if args.show_sixel_general:
        plot_types.append(
            {
                "type": "general"
            }
        )

    return plot_types

def plot_params_to_cli(_command, plot, _tmp, plot_type, tmp_file, _width):
    if "params" in plot.keys():
        if "iterate_through" in plot.keys():
            iterate_through = plot["iterate_through"]
            if len(iterate_through):
                for j in range(0, len(iterate_through)):
                    this_iteration = iterate_through[j]
                    _iterated_command = _command + " " + replace_string_with_params(plot["params"], [this_iteration[0], this_iteration[1]])

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
                    plot_command(_iterated_command, tmp_file, _width)
    else:
        _command += f" --save_to_file={tmp_file} "
        plot_command(_command, tmp_file, _width)

def show_sixel_graphics(_pd_csv):
    _show_sixel_graphics = args.show_sixel_scatter or args.show_sixel_general or args.show_sixel_scatter or args.show_sixel_trial_index_result

    if os.path.exists(_pd_csv) and _show_sixel_graphics:
        x_y_combinations = list(combinations(global_vars["parameter_names"], 2))

        plot_types = get_plot_types(x_y_combinations)

        for plot in plot_types:
            plot_type = plot["type"]
            min_done_jobs = 1

            if "min_done_jobs" in plot:
                min_done_jobs = plot["min_done_jobs"]

            if count_done_jobs() < min_done_jobs:
                print_debug(
                    f"Cannot plot {plot_type}, because it needs {min_done_jobs}, but you only have {count_done_jobs()} jobs done"
                )
                continue

            try:
                _tmp = f"{CURRENT_RUN_FOLDER}/plots/"
                _width = 1200

                if "width" in plot:
                    _width = plot["width"]

                if not os.path.exists(_tmp):
                    os.makedirs(_tmp)

                j = 0
                _fn = plot_type

                if "filename" in plot:
                    _fn = plot['filename']

                tmp_file = f"{_tmp}/{_fn}.png"

                while os.path.exists(tmp_file):
                    j += 1
                    tmp_file = f"{_tmp}/{_fn}_{j}.png"

                maindir = os.path.dirname(os.path.realpath(__file__))

                _command = f"bash {maindir}/omniopt_plot --run_dir {CURRENT_RUN_FOLDER} --plot_type={plot_type}"

                if "dpi" in plot:
                    _command += " --dpi=" + str(plot["dpi"])

                plot_params_to_cli(_command, plot, _tmp, plot_type, tmp_file, _width)
            except Exception as e:
                tb = traceback.format_exc()
                print_red(f"Error trying to print {plot_type} to to CLI: {e}, {tb}")
                print_debug(f"Error trying to print {plot_type} to to CLI: {e}")

def show_end_table_and_save_end_files(csv_file_path):
    print_debug(f"show_end_table_and_save_end_files({csv_file_path})")

    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    signal.signal(signal.SIGUSR2, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGQUIT, signal.SIG_IGN)

    global ALREADY_SHOWN_WORKER_USAGE_OVER_TIME
    global global_vars

    if SHOWN_END_TABLE:
        print("End table already shown, not doing it again")
        return -1

    _exit = 0

    display_failed_jobs_table()

    best_result_exit = print_best_result(csv_file_path)

    if best_result_exit > 0:
        _exit = best_result_exit

    if args.show_worker_percentage_table_at_end and len(worker_percentage_usage) and not ALREADY_SHOWN_WORKER_USAGE_OVER_TIME:
        ALREADY_SHOWN_WORKER_USAGE_OVER_TIME = True

        table = Table(header_style="bold", title="Worker usage over time:")
        columns = ["Time", "Nr. workers", "Max. nr. workers", "%"]
        for column in columns:
            table.add_column(column)
        for row in worker_percentage_usage:
            table.add_row(str(row["time"]), str(row["nr_current_workers"]), str(row["num_parallel_jobs"]), f'{row["percentage"]}%', style='bright_green')
        console.print(table)

    return _exit

def write_worker_usage():
    if len(worker_percentage_usage):
        csv_filename = f'{CURRENT_RUN_FOLDER}/worker_usage.csv'

        csv_columns = ['time', 'num_parallel_jobs', 'nr_current_workers', 'percentage']

        with open(csv_filename, mode='w', encoding="utf-8", newline='') as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            for row in worker_percentage_usage:
                csv_writer.writerow(row)
    else:
        if is_slurm_job():
            print_debug("worker_percentage_usage seems to be empty. Not writing worker_usage.csv")

def end_program(csv_file_path, _force=False, exit_code=None):
    global global_vars
    global END_PROGRAM_RAN

    if os.getpid() != main_pid:
        print_debug("returning from end_program, because it can only run in the main thread, not any forks")
        return

    if IS_IN_EVALUATE and not _force:
        print_debug("IS_IN_EVALUATE true, returning end_program")
        return

    if END_PROGRAM_RAN and not _force:
        print_debug("[end_program] END_PROGRAM_RAN was true. Returning.")
        return

    END_PROGRAM_RAN = True

    _exit = 0

    try:
        if CURRENT_RUN_FOLDER is None:
            print_debug("[end_program] CURRENT_RUN_FOLDER was empty. Not running end-algorithm.")
            return

        if ax_client is None:
            print_debug("[end_program] ax_client was empty. Not running end-algorithm.")
            return

        if console is None:
            print_debug("[end_program] console was empty. Not running end-algorithm.")
            return

        new_exit = show_end_table_and_save_end_files(csv_file_path)
        if new_exit > 0:
            _exit = new_exit
    except (SignalUSR, SignalINT, SignalCONT, KeyboardInterrupt):
        print_red("\n⚠ You pressed CTRL+C or a signal was sent. Program execution halted.")
        print("\n⚠ KeyboardInterrupt signal was sent. Ending program will still run.")
        new_exit = show_end_table_and_save_end_files(csv_file_path)
        if new_exit > 0:
            _exit = new_exit
    except TypeError as e:
        print_red(f"\n⚠ The program has been halted without attaining any results. Error: {e}")

    for job, trial_index in global_vars["jobs"][:]:
        if job:
            try:
                _trial = ax_client.get_trial(trial_index)
                _trial.mark_abandoned()
                global_vars["jobs"].remove((job, trial_index))
            except Exception as e:
                print(f"ERROR in line {get_line_info()}: {e}")
            job.cancel()

    save_pd_csv()

    if exit_code:
        _exit = exit_code

    my_exit(_exit)

def save_checkpoint(trial_nr=0, ee=None):
    if trial_nr > 3:
        if ee:
            print("Error during saving checkpoint: " + str(ee))
        else:
            print("Error during saving checkpoint")
        return

    try:
        state_files_folder = f"{CURRENT_RUN_FOLDER}/state_files/"

        if not os.path.exists(state_files_folder):
            os.makedirs(state_files_folder)

        checkpoint_filepath = f'{state_files_folder}/checkpoint.json'
        ax_client.save_to_json_file(filepath=checkpoint_filepath)
    except Exception as e:
        save_checkpoint(trial_nr + 1, e)

def save_pd_csv():
    pd_csv = f'{CURRENT_RUN_FOLDER}/{PD_CSV_FILENAME}'
    pd_json = f'{CURRENT_RUN_FOLDER}/state_files/pd.json'

    state_files_folder = f"{CURRENT_RUN_FOLDER}/state_files/"

    if not os.path.exists(state_files_folder):
        os.makedirs(state_files_folder)

    if ax_client is None:
        return pd_csv

    try:
        pd_frame = ax_client.get_trials_data_frame()

        pd_frame.to_csv(pd_csv, index=False, float_format="%.30f")
        #pd_frame.to_json(pd_json)

        json_snapshot = ax_client.to_json_snapshot()

        with open(pd_json, mode='w', encoding="utf-8") as json_file:
            json.dump(json_snapshot, json_file, indent=4)

        save_experiment(ax_client.experiment, f"{CURRENT_RUN_FOLDER}/state_files/ax_client.experiment.json")

        #print_debug("pd.{csv,json} saved")
    except SignalUSR as e:
        raise SignalUSR(str(e)) from e
    except SignalCONT as e:
        raise SignalCONT(str(e)) from e
    except SignalINT as e:
        raise SignalINT(str(e)) from e
    except Exception as e:
        print_red(f"While saving all trials as a pandas-dataframe-csv, an error occured: {e}")

    return pd_csv

def get_tmp_file_from_json(experiment_args):
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

def compare_parameters(old_param_json, new_param_json):
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

        return "No differences found between the old and new parameters."

    except AssertionError as e:
        print(f"Assertion error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    return ""

def get_ax_param_representation(data):
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

    print("data:")
    pprint(data)
    helpers.dier(f"Unknown data range {data['type']}")

    return {} # only for linter, never reached because of die

def set_torch_device_to_experiment_args(experiment_args):
    torch_device = None
    try:
        cuda_is_available = torch.cuda.is_available()

        if not cuda_is_available or cuda_is_available == 0:
            print_yellow("No suitable CUDA devices found")
        else:
            if torch.cuda.device_count() >= 1:
                torch_device = torch.cuda.current_device()
                print_yellow(f"Using CUDA device {torch.cuda.get_device_name(0)}")
            else:
                print_yellow("No suitable CUDA devices found")
    except ModuleNotFoundError:
        print_red("Cannot load torch and thus, cannot use gpus")

    if torch_device:
        experiment_args["choose_generation_strategy_kwargs"]["torch_device"] = torch_device

    return experiment_args

def die_with_47_if_file_doesnt_exists(_file):
    if not os.path.exists(_file):
        print_red(f"Cannot find {_file}")
        my_exit(47)

def copy_state_files_from_previous_job(continue_previous_job):
    for state_file in ["submitted_jobs"]:
        old_state_file = f"{continue_previous_job}/state_files/{state_file}"
        new_state_file = f'{CURRENT_RUN_FOLDER}/state_files/{state_file}'
        die_with_47_if_file_doesnt_exists(old_state_file)

        if not os.path.exists(new_state_file):
            shutil.copy(old_state_file, new_state_file)

def get_experiment_parameters(_params):
    continue_previous_job, seed, experiment_constraints, parameter, cli_params_experiment_parameters, experiment_parameters, minimize_or_maximize = _params

    global ax_client

    experiment_args = None

    if continue_previous_job:
        print_debug(f"Load from checkpoint: {continue_previous_job}")

        checkpoint_file = continue_previous_job + "/state_files/checkpoint.json"
        checkpoint_parameters_filepath = continue_previous_job + "/state_files/checkpoint.json.parameters.json"

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

        original_ax_client_file = f"{CURRENT_RUN_FOLDER}/state_files/original_ax_client_before_loading_tmp_one.json"
        ax_client.save_to_json_file(filepath=original_ax_client_file)

        with open(original_ax_client_file, encoding="utf-8") as f:
            loaded_original_ax_client_json = json.load(f)
            original_generation_strategy = loaded_original_ax_client_json["generation_strategy"]

            if original_generation_strategy:
                experiment_parameters["generation_strategy"] = original_generation_strategy

        tmp_file_path = get_tmp_file_from_json(experiment_parameters)

        ax_client = AxClient.load_from_json_file(tmp_file_path)

        os.unlink(tmp_file_path)

        state_files_folder = f"{CURRENT_RUN_FOLDER}/state_files"

        checkpoint_filepath = f'{state_files_folder}/checkpoint.json'
        if not os.path.exists(state_files_folder):
            os.makedirs(state_files_folder)
        with open(checkpoint_filepath, mode="w", encoding="utf-8") as outfile:
            json.dump(experiment_parameters, outfile)

        if not os.path.exists(checkpoint_filepath):
            print_red(f"{checkpoint_filepath} not found. Cannot continue_previous_job without.")
            my_exit(47)

        with open(f'{CURRENT_RUN_FOLDER}/checkpoint_load_source', mode='w', encoding="utf-8") as f:
            print(f"Continuation from checkpoint {continue_previous_job}", file=f)
    else:
        experiment_args = {
            "name": global_vars["experiment_name"],
            "parameters": experiment_parameters,
            "objectives": {"result": ObjectiveProperties(minimize=minimize_or_maximize)},
            "choose_generation_strategy_kwargs": {
                "num_trials": max_eval,
                "num_initialization_trials": num_parallel_jobs,
                "max_parallelism_cap": num_parallel_jobs,
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
            ax_client.create_experiment(**experiment_args)
        except ValueError as error:
            print_red(f"An error has occured while creating the experiment: {error}")
            my_exit(49)
        except TypeError as error:
            print_red(f"An error has occured while creating the experiment: {error}. This is probably a bug in OmniOpt.")
            my_exit(49)

    return ax_client, experiment_parameters, experiment_args

def get_type_short(typename):
    if typename == "RangeParameter":
        return "range"

    if typename == "ChoiceParameter":
        return "choice"

    return typename

def parse_single_experiment_parameter_table(experiment_parameters):
    rows = []

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

            rows.append([str(param["name"]), get_type_short(_type), str(helpers.to_int_when_possible(_lower)), str(helpers.to_int_when_possible(_upper)), "", value_type])
        elif "fixed" in _type.lower():
            rows.append([str(param["name"]), get_type_short(_type), "", "", str(helpers.to_int_when_possible(param["value"])), ""])
        elif "choice" in _type.lower():
            values = param["values"]
            values = [str(helpers.to_int_when_possible(item)) for item in values]

            rows.append([str(param["name"]), get_type_short(_type), "", "", ", ".join(values), ""])
        else:
            print_red(f"Type {_type} is not yet implemented in the overview table.")
            my_exit(15)

    return rows

def print_overview_tables(experiment_parameters, experiment_args):
    if not experiment_parameters:
        print_red("Cannot determine experiment_parameters. No parameter table will be shown.")
        return

    if not experiment_parameters:
        print_red("Experiment parameters could not be determined for display")

    min_or_max = "minimize"
    if args.maximize:
        min_or_max = "maximize"

    with open(f"{CURRENT_RUN_FOLDER}/state_files/{min_or_max}", mode='w', encoding="utf-8") as f:
        print('The contents of this file do not matter. It is only relevant that it exists.', file=f)

    if "_type" in experiment_parameters:
        experiment_parameters = experiment_parameters["experiment"]["search_space"]["parameters"]

    rows = parse_single_experiment_parameter_table(experiment_parameters)

    table = Table(header_style="bold", title="Experiment parameters:")
    columns = ["Name", "Type", "Lower bound", "Upper bound", "Values", "Type"]

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

    with open(f"{CURRENT_RUN_FOLDER}/parameters.txt", mode="w", encoding="utf-8") as text_file:
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

        with open(f"{CURRENT_RUN_FOLDER}/constraints.txt", mode="w", encoding="utf-8") as text_file:
            text_file.write(table_str)

def check_equation(variables, equation):
    print_debug(f"check_equation({variables}, {equation})")

    _errors = []

    if not (">=" in equation or "<=" in equation):
        _errors.append(f"check_equation({variables}, {equation}): if not ('>=' in equation or '<=' in equation)")

    comparer_at_beginning = re.search("^\\s*((<=|>=)|(<=|>=))", equation)
    if comparer_at_beginning:
        _errors.append(f"The restraints {equation} contained comparision operator like <=, >= at at the beginning. This is not a valid equation.")

    comparer_at_end = re.search("((<=|>=)|(<=|>=))\\s*$", equation)
    if comparer_at_end:
        _errors.append(f"The restraints {equation} contained comparision operator like <=, >= at at the end. This is not a valid equation.")

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

    regex_pattern = r'\s+|(?=[+\-*\/()-])|(?<=[+\-*\/()-])'
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
                print("There is already one comparision operator! Cannot have more than one in an equation!")
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

def update_progress_bar(_progress_bar, nr):
    #import traceback
    #print(f"update_progress_bar(_progress_bar, {nr})")
    #traceback.print_stack()

    global SUM_OF_VALUES_FOR_TQDM

    SUM_OF_VALUES_FOR_TQDM += nr

    if SUM_OF_VALUES_FOR_TQDM > max_eval:
        print_debug(f"Reaching upper limit for tqdm: SUM_OF_VALUES_FOR_TQDM {SUM_OF_VALUES_FOR_TQDM} > max_eval {max_eval}")
        return

    _progress_bar.update(nr)

def progressbar_description(new_msgs=[]):
    desc = get_desc_progress_text(new_msgs)
    print_debug_progressbar(desc)
    progress_bar.set_description(desc)
    progress_bar.refresh()

def clean_completed_jobs():
    for job, trial_index in global_vars["jobs"][:]:
        if state_from_job(job) in ["completed", "early_stopped", "abandoned"]:
            global_vars["jobs"].remove((job, trial_index))

def get_old_result_by_params(file_path, params, float_tolerance=1e-6):
    """
    Open the CSV file and find the row where the subset of columns matching the keys in params have the same values.
    Return the value of the 'result' column from that row.

    :param file_path: The path to the CSV file.
    :param params: A dictionary of parameters with column names as keys and values to match.
    :param float_tolerance: The tolerance for comparing float values.
    :return: The value of the 'result' column from the matched row.
    """
    assert isinstance(file_path, str), "file_path must be a string"
    assert isinstance(params, dict), "params must be a dictionary"

    if not os.path.exists(file_path):
        print_red(f"{file_path} for getting old CSV results cannot be found")
        return None

    try:
        df = pd.read_csv(file_path, float_precision='round_trip')
    except Exception as e:
        raise RuntimeError(f"Failed to read the CSV file: {str(e)}") from e

    if 'result' not in df.columns:
        print_red(f"Error: Could not get old result for {params} in {file_path}")
        return None

    try:
        matching_rows = df
        print_debug(matching_rows)

        for param, value in params.items():
            if param in df.columns:
                if isinstance(value, float):
                    # Log current state before filtering
                    print_debug(f"Filtering for float parameter '{param}' with value '{value}' and tolerance '{float_tolerance}'")

                    is_close_array = np.isclose(matching_rows[param], value, atol=float_tolerance)
                    print_debug(is_close_array)

                    matching_rows = matching_rows[is_close_array]
                    print_debug(matching_rows)

                    assert not matching_rows.empty, f"No matching rows found for float parameter '{param}' with value '{value}'"
                else:
                    # Ensure consistent types for comparison
                    print_debug(f"Filtering for parameter '{param}' with value '{value}'")
                    if matching_rows[param].dtype == np.int64 and isinstance(value, str):
                        value = int(value)
                    elif matching_rows[param].dtype == np.float64 and isinstance(value, str):
                        value = float(value)

                    matching_rows = matching_rows[matching_rows[param] == value]
                    print_debug(matching_rows)

                    assert not matching_rows.empty, f"No matching rows found for parameter '{param}' with value '{value}'"
            else:
                print_debug(f"Parameter '{param}' not found in DataFrame columns")

        if matching_rows.empty:
            print_debug("No matching rows found after all filters applied")
            return None

        print_debug("Matching rows found")
        print_debug(matching_rows)
        return matching_rows
    except AssertionError as ae:
        print_red(f"Assertion error: {str(ae)}")
        raise
    except Exception as e:
        print_red(f"Error during filtering or extracting result: {str(e)}")
        raise

def load_existing_job_data_into_ax_client():
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

def parse_parameter_type_error(error_message):
    error_message = str(error_message)
    try:
        # Defining the regex pattern to match the required parts of the error message
        pattern = r"Value for parameter (?P<parameter_name>\w+): .*? is of type <class '(?P<current_type>\w+)'>, expected  <class '(?P<expected_type>\w+)'>."
        match = re.search(pattern, error_message)

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

def extract_headers_and_rows(data_list):
    try:
        if not data_list:
            return None, None

        # Extract headers from the first dictionary
        first_entry = data_list[0]
        headers = list(first_entry.keys())

        # Initialize rows list
        rows = []

        # Extract rows based on headers order
        for entry in data_list:
            row = [str(entry.get(header, None)) for header in headers]
            rows.append(row)

        return headers, rows
    except Exception as e:
        print(f"An error occured: {e}")
        return None, None

def simulate_load_data_from_existing_run_folders(_paths):
    _counter = 0

    path_idx = 0
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

            print_debug(f"trial_status_str: {trial_status_str}")

            if "COMPLETED".lower() not in str(trial_status_str).lower(): # or "MANUAL".lower() in str(trial_status_str).lower()):
                continue

            old_arm_parameter = old_trial.arm.parameters

            old_result_simple = None
            try:
                tmp_old_res = get_old_result_by_params(f"{this_path}/{PD_CSV_FILENAME}", old_arm_parameter)["result"]
                tmp_old_res_list = list(set(list(tmp_old_res)))

                if len(tmp_old_res_list) == 1:
                    print_debug(f"Got a list of length {len(tmp_old_res_list)}. This means the result was found properly and will be added.")
                    old_result_simple = float(tmp_old_res_list[0])
                else:
                    print_debug(
                        f"Got a list of length {len(tmp_old_res_list)}. Cannot add this to previous jobs."
                    )
                    old_result_simple = None
            except Exception:
                pass

            if old_result_simple and helpers.looks_like_number(old_result_simple) and str(old_result_simple) != "nan":
                _counter += 1

        path_idx += 1

    return _counter

def get_list_import_as_string(_brackets=True, _comma=False):
    _str = []

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

def insert_job_into_ax_client(old_arm_parameter, old_result, hashed_params_result):
    done_converting = False

    while not done_converting:
        try:
            new_old_trial = ax_client.attach_trial(old_arm_parameter)

            ax_client.complete_trial(trial_index=new_old_trial[1], raw_data=old_result)

            already_inserted_param_hashes[hashed_params_result] = 1

            done_converting = True
            save_pd_csv()
        except ax.exceptions.core.UnsupportedError as e:
            parsed_error = parse_parameter_type_error(e)

            if parsed_error["expected_type"] == "int" and type(old_arm_parameter[parsed_error["parameter_name"]]).__name__ != "int":
                print_yellow(f"⚠ converted parameter {parsed_error['parameter_name']} type {parsed_error['current_type']} to {parsed_error['expected_type']}")
                old_arm_parameter[parsed_error["parameter_name"]] = int(old_arm_parameter[parsed_error["parameter_name"]])
            elif parsed_error["expected_type"] == "float" and type(old_arm_parameter[parsed_error["parameter_name"]]).__name__ != "float":
                print_yellow(f"⚠ converted parameter {parsed_error['parameter_name']} type {parsed_error['current_type']} to {parsed_error['expected_type']}")
                old_arm_parameter[parsed_error["parameter_name"]] = float(old_arm_parameter[parsed_error["parameter_name"]])

def load_data_from_existing_run_folders(_paths):
    global already_inserted_param_hashes
    global already_inserted_param_data
    global double_hashes
    global missing_results

    #helpers.dier(help(ax_client.experiment.search_space))
    with console.status("[bold green]Loading existing jobs into ax_client...") as _status:
        path_idx = 0
        for this_path in _paths:
            if len(_paths) > 1:
                _status.update(f"[bold green]Loading existing jobs from {this_path} into ax_client (folder {path_idx + 1}{get_list_import_as_string(False, True)})...")
            else:
                _status.update(f"[bold green]Loading existing jobs from {this_path} into ax_client{get_list_import_as_string()}...")

            this_path_json = str(this_path) + "/state_files/ax_client.experiment.json"

            if not os.path.exists(this_path_json):
                print_red(f"{this_path_json} does not exist, cannot load data from it")
                return

            old_experiments = load_experiment(this_path_json)

            old_trials = old_experiments.trials

            trial_idx = 0
            for old_trial_index in old_trials:
                if len(_paths) > 1:
                    _status.update(f"[bold green]Loading existing jobs from {this_path} into ax_client (folder {path_idx + 1}/{len(_paths)}, trial {trial_idx + 1}/{len(old_trials)}{get_list_import_as_string(False, True)})...")
                else:
                    _status.update(f"[bold green]Loading existing jobs from {this_path} into ax_client (trial {trial_idx + 1}/{len(old_trials)}{get_list_import_as_string(False, True)})...")

                trial_idx += 1

                old_trial = old_trials[old_trial_index]
                trial_status = old_trial.status
                trial_status_str = trial_status.__repr__

                print_debug(f"trial_status_str: {trial_status_str}")

                if "COMPLETED" not in str(trial_status_str):
                    continue

                old_arm_parameter = old_trial.arm.parameters

                old_result_simple = None
                try:
                    tmp_old_res = get_old_result_by_params(f"{this_path}/{PD_CSV_FILENAME}", old_arm_parameter)["result"]
                    tmp_old_res_list = list(set(list(tmp_old_res)))

                    if len(tmp_old_res_list) == 1:
                        print_debug(f"Got a list of length {len(tmp_old_res_list)}. This means the result was found properly and will be added.")
                        old_result_simple = float(tmp_old_res_list[0])
                    else:
                        print_debug(f"Got a list of length {len(tmp_old_res_list)}. Cannot add this to previous jobs.")
                        old_result_simple = None
                except Exception:
                    pass

                hashed_params_result = pformat(old_arm_parameter) + "====" + pformat(old_result_simple)

                if old_result_simple and helpers.looks_like_number(old_result_simple) and str(old_result_simple) != "nan":
                    if hashed_params_result not in already_inserted_param_hashes.keys():
                        #print(f"ADDED: old_result_simple: {old_result_simple}, type: {type(old_result_simple)}")
                        old_result = {'result': old_result_simple}

                        insert_job_into_ax_client(old_arm_parameter, old_result, hashed_params_result)
                    else:
                        print_debug("Prevented inserting a double entry")
                        already_inserted_param_hashes[hashed_params_result] += 1

                        double_hashes.append(hashed_params_result)

                        old_arm_parameter_with_result = old_arm_parameter
                        old_arm_parameter_with_result["result"] = old_result_simple
                else:
                    print_debug("Prevent inserting a parameter set without result")

                    missing_results.append(hashed_params_result)

                    old_arm_parameter_with_result = old_arm_parameter
                    old_arm_parameter_with_result["result"] = old_result_simple
                    already_inserted_param_data.append(old_arm_parameter_with_result)

            path_idx += 1

    headers, rows = extract_headers_and_rows(already_inserted_param_data)

    if headers and rows:
        table = Table(show_header=True, header_style="bold", title="Duplicate parameters only inserted once or without result:")

        for header in headers:
            table.add_column(header)

        for row in rows:
            table.add_row(*row)

        console.print(table)

def print_outfile_analyzed(stdout_path):
    errors = get_errors_from_outfile(stdout_path)

    _strs = []
    j = 0

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

    out_files_string = "\n".join(_strs)

    if len(_strs):
        try:
            with open(f'{CURRENT_RUN_FOLDER}/evaluation_errors.log', mode="a+", encoding="utf-8") as error_file:
                error_file.write(out_files_string)
        except Exception as e:
            print_debug(f"Error occurred while writing to evaluation_errors.log: {e}")

        print_red(out_files_string)

def get_parameters_from_outfile(stdout_path):
    try:
        with open(stdout_path, mode='r', encoding="utf-8") as file:
            for line in file:
                if line.lower().startswith("parameters: "):
                    params = line.split(":", 1)[1].strip()
                    params = json.loads(params)
                    return params
        # Wenn keine passende Zeile gefunden wurde, gib None zurück
        return None
    except FileNotFoundError:
        original_print(f"The file {stdout_path} was not found.")
        return None
    except Exception as e:
        print(f"There was an error: {e}")
        return None

def get_hostname_from_outfile(stdout_path):
    try:
        with open(stdout_path, mode='r', encoding="utf-8") as file:
            for line in file:
                if line.lower().startswith("hostname: "):
                    hostname = line.split(":", 1)[1].strip()
                    return hostname
        # Wenn keine passende Zeile gefunden wurde, gib None zurück
        return None
    except FileNotFoundError:
        original_print(f"The file {stdout_path} was not found.")
        return None
    except Exception as e:
        print(f"There was an error: {e}")
        return None

def finish_previous_jobs(new_msgs):
    global random_steps
    global ax_client

    #print("jobs in finish_previous_jobs:")
    #print(jobs)

    jobs_finished = 0

    for job, trial_index in global_vars["jobs"][:]:
        # Poll if any jobs completed
        # Local and debug jobs don't run until .result() is called.
        if job is not None and (job.done() or type(job) in [LocalJob, DebugJob]):
            try:
                result = job.result()
                raw_result = result
                result = result["result"]
                jobs_finished += 1
                if result != VAL_IF_NOTHING_FOUND:
                    ax_client.complete_trial(trial_index=trial_index, raw_data=raw_result)

                    #count_done_jobs(1)

                    _trial = ax_client.get_trial(trial_index)
                    try:
                        progressbar_description([f"new result: {result}"])
                        _trial.mark_completed(unsafe=True)
                        succeeded_jobs(1)

                        #update_progress_bar(progress_bar, 1)
                        update_progress_bar(progress_bar, 1)
                    except Exception as e:
                        print(f"ERROR in line {get_line_info()}: {e}")
                else:
                    if job:
                        try:
                            progressbar_description(["job_failed"])

                            ax_client.log_trial_failure(trial_index=trial_index)
                        except Exception as e:
                            print(f"ERROR in line {get_line_info()}: {e}")
                        job.cancel()

                    failed_jobs(1)

                global_vars["jobs"].remove((job, trial_index))
            except (FileNotFoundError, submitit.core.utils.UncompletedJobError) as error:
                print_red(str(error))

                if job:
                    try:
                        progressbar_description(["job_failed"])
                        _trial = ax_client.get_trial(trial_index)
                        _trial.mark_failed()
                    except Exception as e:
                        print(f"ERROR in line {get_line_info()}: {e}")
                    job.cancel()

                failed_jobs(1)
                jobs_finished += 1

                global_vars["jobs"].remove((job, trial_index))
            except ax.exceptions.core.UserInputError as error:
                if "None for metric" in str(error):
                    print_red(f"\n⚠ It seems like the program that was about to be run didn't have 'RESULT: <NUMBER>' in it's output string.\nError: {error}")
                else:
                    print_red(f"\n⚠ {error}")

                if job:
                    try:
                        progressbar_description(["job_failed"])
                        ax_client.log_trial_failure(trial_index=trial_index)
                    except Exception as e:
                        print(f"ERROR in line {get_line_info()}: {e}")
                    job.cancel()

                failed_jobs(1)
                jobs_finished += 1

                global_vars["jobs"].remove((job, trial_index))

            if args.verbose:
                progressbar_description([f"saving checkpoints and {PD_CSV_FILENAME}"])
            save_checkpoint()
            save_pd_csv()
        else:
            pass

        orchestrate_job(job, trial_index)

    if jobs_finished == 1:
        progressbar_description([*new_msgs, f"finished {jobs_finished} job"])
    elif jobs_finished > 0:
        progressbar_description([*new_msgs, f"finished {jobs_finished} jobs"])

    clean_completed_jobs()

def orchestrate_job(job, trial_index):
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

    for todo_stdout_file in ORCHESTRATE_TODO.keys():
        old_behavs = check_orchestrator(todo_stdout_file, ORCHESTRATE_TODO[todo_stdout_file])
        if old_behavs is not None:
            del ORCHESTRATE_TODO[todo_stdout_file]

def _orchestrate(stdout_path, trial_index):
    behavs = check_orchestrator(stdout_path, trial_index)

    if behavs is not None and len(behavs):
        hostname_from_out_file = get_hostname_from_outfile(stdout_path)
        for behav in behavs:
            if behav == "ExcludeNode":
                if hostname_from_out_file:
                    if not is_already_in_defective_nodes(hostname_from_out_file):
                        print_yellow(f"ExcludeNode was triggered for node {hostname_from_out_file}")
                        count_defective_nodes(None, hostname_from_out_file)
                    else:
                        print_yellow(f"ExcludeNode was triggered for node {hostname_from_out_file}, but it was already in defective nodes and won't be added again")
                else:
                    print_red(f"Cannot do ExcludeNode because the host could not be determined from {stdout_path}")

            elif behav == "RestartOnDifferentNode":
                if hostname_from_out_file:
                    if not is_already_in_defective_nodes(hostname_from_out_file):
                        print_yellow(f"RestartOnDifferentNode was triggered for node {hostname_from_out_file}. Will add the node to the defective hosts list and restart it to schedule it on another host.")
                        count_defective_nodes(None, hostname_from_out_file)
                    else:
                        print_yellow(f"RestartOnDifferentNode was triggered for node {hostname_from_out_file}, but it was already in defective nodes and won't be added again. The job only will be resubmitted.")

                    params_from_out_file = get_parameters_from_outfile(stdout_path)
                    if params_from_out_file:
                        new_job = executor.submit(evaluate, params_from_out_file)
                        submitted_jobs(1)

                        _trial = ax_client.get_trial(trial_index)

                        _trial.mark_staged(unsafe=True)
                        _trial.mark_running(unsafe=True, no_runner_required=True)

                        global_vars["jobs"].append((new_job, trial_index))
                    else:
                        print(f"Could not determine parameters from outfile {stdout_path} for restarting job")

                else:
                    print_red(f"Cannot do RestartOnDifferentNode because the host could not be determined from {stdout_path}")

            elif behav == "ExcludeNodeAndRestartAll":
                if hostname_from_out_file:
                    if not is_already_in_defective_nodes(hostname_from_out_file):
                        print_yellow(f"ExcludeNodeAndRestartAll not yet fully implemented. Will only add {hostname_from_out_file} to unavailable hosts and not currently restart the job")
                        count_defective_nodes(None, hostname_from_out_file)
                    else:
                        print_yellow(f"ExcludeNodeAndRestartAll was triggered for node {hostname_from_out_file}, but it was already in defective nodes and won't be added again")
                else:
                    print_red(f"Cannot do ExcludeNodeAndRestartAll because the host could not be determined from {stdout_path}")

            else:
                print_red(f"Orchestrator: {behav} not yet implemented!")
                sys.exit(210)

def check_orchestrator(stdout_path, trial_index):
    behavs = []

    if orchestrator and "errors" in orchestrator:
        try:
            stdout = Path(stdout_path).read_text("UTF-8")
        except FileNotFoundError:
            if stdout_path not in ORCHESTRATE_TODO.keys():
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

def state_from_job(job):
    job_string = f'{job}'
    match = re.search(r'state="([^"]+)"', job_string)

    state = None

    if match:
        state = match.group(1).lower()
    else:
        state = f"{state}"

    return state

def get_workers_string():
    string = ""

    string_keys = []
    string_values = []

    stats = {}

    for job, _ in global_vars["jobs"][:]:
        state = state_from_job(job)

        if state not in stats.keys():
            stats[state] = 0
        stats[state] += 1

    for key in stats.keys():
        if args.abbreviate_job_names:
            string_keys.append(key.lower()[0])
        else:
            string_keys.append(key.lower())
        string_values.append(str(stats[key]))

    if len(string_keys) and len(string_values):
        _keys = "/".join(string_keys)
        _values = "/".join(string_values)

        if len(_keys):
            nr_current_workers = len(global_vars["jobs"])
            percentage = round((nr_current_workers / num_parallel_jobs) * 100)
            string = f"{_keys} {_values} ({percentage}%/{num_parallel_jobs})"

    return string

def get_best_params_str():
    if count_done_jobs() >= 0:
        best_params = get_best_params(RESULT_CSV_FILE)
        if best_params and "result" in best_params:
            best_result = best_params["result"]
            if isinstance(best_result, (int, float)) or helpers.looks_like_float(best_result):
                best_result_int_if_possible = helpers.to_int_when_possible(float(best_result))

                if str(best_result) != NO_RESULT and best_result is not None:
                    return f"best result: {best_result_int_if_possible}"
    return ""

def get_desc_progress_text(new_msgs=[]):
    global global_vars
    global random_steps
    global max_eval

    desc = ""

    in_brackets = []

    if failed_jobs():
        in_brackets.append(f"{helpers.bcolors.red}failed jobs: {failed_jobs()}{helpers.bcolors.endc}")

    current_model = get_current_model()

    in_brackets.append(f"{current_model}")

    this_time = time.time()

    best_params_str = get_best_params_str()
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

        if len(worker_percentage_usage) == 0 or worker_percentage_usage[len(worker_percentage_usage) - 1] != this_values:
            worker_percentage_usage.append(this_values)

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

    def capitalized_string(s):
        return s[0].upper() + s[1:] if s else ""
    desc = capitalized_string(desc)

    return desc

def is_slurm_job():
    if os.environ.get('SLURM_JOB_ID') is not None:
        return True
    return False

def _sleep(t: int):
    if not args.no_sleep:
        time.sleep(t)

def save_state_files():
    global global_vars

    state_files_folder = f"{CURRENT_RUN_FOLDER}/state_files/"

    if not os.path.exists(state_files_folder):
        os.makedirs(state_files_folder)

    with open(f'{state_files_folder}/joined_run_program', mode='w', encoding="utf-8") as f:
        print(global_vars["joined_run_program"], file=f)

    with open(f'{state_files_folder}/experiment_name', mode='w', encoding="utf-8") as f:
        print(global_vars["experiment_name"], file=f)

    with open(f'{state_files_folder}/mem_gb', mode='w', encoding='utf-8') as f:
        print(global_vars["mem_gb"], file=f)

    with open(f'{state_files_folder}/max_eval', mode='w', encoding='utf-8') as f:
        print(max_eval, file=f)

    with open(f'{state_files_folder}/gpus', mode='w', encoding='utf-8') as f:
        print(args.gpus, file=f)

    with open(f'{state_files_folder}/time', mode='w', encoding='utf-8') as f:
        print(global_vars["_time"], file=f)

    with open(f'{state_files_folder}/env', mode='a', encoding="utf-8") as f:
        env = dict(os.environ)
        for key in env:
            print(str(key) + " = " + str(env[key]), file=f)

    with open(f'{state_files_folder}/run.sh', mode='w', encoding='utf-8') as f:
        print("omniopt '" + " ".join(sys.argv[1:]), file=f)

def execute_evaluation(_params):
    global global_vars
    global IS_IN_EVALUATE

    trial_index, parameters, trial_counter, next_nr_steps, phase = _params

    _trial = ax_client.get_trial(trial_index)

    try:
        _trial.mark_staged()
    except Exception:
        #print(e)
        pass
    new_job = None
    try:
        progressbar_description([f"starting new job ({trial_counter}/{next_nr_steps})"])

        if args.reservation:
            os.environ['SBATCH_RESERVATION'] = args.reservation

        if args.account:
            os.environ['SBATCH_ACCOUNT'] = args.account

        excluded_string = ",".join(count_defective_nodes())
        if len(excluded_string) > 1:
            executor.update_parameters(
                    exclude=excluded_string
            )

        #try:
        new_job = executor.submit(evaluate, parameters)
        submitted_jobs(1)
        #except TypeError as e:
        #    print_red(f"Error while trying to submit job: {e}")

        global_vars["jobs"].append((new_job, trial_index))
        if is_slurm_job() and not args.force_local_execution:
            _sleep(1)

        try:
            _trial.mark_running(no_runner_required=True)
        except Exception:
            #print(f"ERROR in line {get_line_info()}: {e}")
            pass
        trial_counter += 1

        progressbar_description([f"started new job ({trial_counter - 1}/{next_nr_steps})"])
    except submitit.core.utils.FailedJobError as error:
        if "QOSMinGRES" in str(error) and args.gpus == 0:
            print_red("\n⚠ It seems like, on the chosen partition, you need at least one GPU. Use --gpus=1 (or more) as parameter.")
        else:
            print_red(f"\n⚠ FAILED: {error}")

        try:
            print_debug("Trying to cancel job that failed")
            if new_job:
                try:
                    ax_client.log_trial_failure(trial_index=trial_index)
                except Exception as e:
                    print(f"ERROR in line {get_line_info()}: {e}")
                new_job.cancel()
                print_debug("Cancelled failed job")

            global_vars["jobs"].remove((new_job, trial_index))
            print_debug("Removed failed job")

            #update_progress_bar(1)

            save_checkpoint()
            save_pd_csv()
            trial_counter += 1
        except Exception as e:
            print_red(f"\n⚠ Cancelling failed job FAILED: {e}")
    except (SignalUSR, SignalINT, SignalCONT):
        print_red("\n⚠ Detected signal. Will exit.")
        IS_IN_EVALUATE = False
        end_program(RESULT_CSV_FILE, 1)
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        print_red(f"\n⚠ Starting job failed with error: {e}")

    finish_previous_jobs([])

    add_to_phase_counter(phase, 1)

    return trial_counter

def get_current_model():
    global ax_client

    if ax_client:
        gs_model = ax_client.generation_strategy.model

        if gs_model:
            return str(gs_model.model)

    return "initializing"

def _get_next_trials():
    global global_vars

    finish_previous_jobs(["finishing jobs (_get_next_trials)"])

    if break_run_search("_get_next_trials", max_eval, progress_bar):
        return False

    last_ax_client_time = None
    ax_client_time_avg = None
    if len(TIME_NEXT_TRIALS_TOOK):
        last_ax_client_time = TIME_NEXT_TRIALS_TOOK[len(TIME_NEXT_TRIALS_TOOK) - 1]
        ax_client_time_avg = sum(TIME_NEXT_TRIALS_TOOK) / len(TIME_NEXT_TRIALS_TOOK)

    new_msgs = []

    currently_running_jobs = len(global_vars["jobs"])
    real_num_parallel_jobs = num_parallel_jobs - currently_running_jobs

    if real_num_parallel_jobs == 0:
        return None

    base_msg = f"getting {real_num_parallel_jobs} trials "

    if SYSTEM_HAS_SBATCH:
        if last_ax_client_time:
            new_msgs.append(f"{base_msg}(last/avg {last_ax_client_time:.2f}s/{ax_client_time_avg:.2f}s)")
        else:
            new_msgs.append(f"{base_msg}")
    else:
        real_num_parallel_jobs = 1

        if last_ax_client_time:
            new_msgs.append(f"{base_msg}(no sbatch, last/avg {last_ax_client_time:.2f}s/{ax_client_time_avg:.2f}s)")
        else:
            new_msgs.append(f"{base_msg}(no sbatch)")

    progressbar_description(new_msgs)

    trial_index_to_param = None

    get_next_trials_time_start = time.time()
    try:
        trial_index_to_param, _ = ax_client.get_next_trials(
            max_trials=real_num_parallel_jobs
        )
    except np.linalg.LinAlgError as e:
        if args.model and args.model.upper() in ["THOMPSON", "EMPIRICAL_BAYES_THOMPSON"]:
            print_red(f"Error: {e}. This may happen because you have the THOMPSON model used. Try another one.")
        else:
            print_red(f"Error: {e}")
        sys.exit(242)

    print_debug_get_next_trials(len(trial_index_to_param.items()), real_num_parallel_jobs, getframeinfo(currentframe()).lineno)

    get_next_trials_time_end = time.time()

    _ax_took = get_next_trials_time_end - get_next_trials_time_start

    TIME_NEXT_TRIALS_TOOK.append(_ax_took)

    _log_trial_index_to_param(trial_index_to_param)

    return trial_index_to_param

def get_next_nr_steps(_num_parallel_jobs, _max_eval):
    if not SYSTEM_HAS_SBATCH:
        return 1

    requested = min(_num_parallel_jobs - len(global_vars["jobs"]), _max_eval - submitted_jobs())

    return requested

def get_nr_of_imported_jobs():
    nr_jobs = 0

    if args.load_previous_job_data:
        for this_path in args.load_previous_job_data:
            nr_jobs += simulate_load_data_from_existing_run_folders(this_path)

    if args.continue_previous_job:
        nr_jobs += simulate_load_data_from_existing_run_folders([args.continue_previous_job])

    return nr_jobs

def get_generation_strategy(_num_parallel_jobs, seed, _max_eval):
    global random_steps

    _steps = []

    nr_of_imported_jobs = get_nr_of_imported_jobs()

    set_max_eval(_max_eval + nr_of_imported_jobs)

    if random_steps is None:
        random_steps = 0

    if _max_eval is None:
        set_max_eval(max(1, random_steps))

    if random_steps >= 1:
        # TODO: nicht, wenn continue_previous_job und bereits random_steps schritte erfolgt
        # 1. Initialization step (does not require pre-existing data and is well-suited for
        # initial sampling of the search space)

        #print(f"!!! get_generation_strategy: random_steps == {random_steps}")
        #min_trials_observed=max(min(0, _max_eval, random_steps), random_steps + NR_INSERTED_JOBS),

        _steps.append(
            GenerationStep(
                model=Models.SOBOL,
                num_trials=max(_num_parallel_jobs, random_steps),
                min_trials_observed=min(_max_eval, random_steps),
                max_parallelism=_num_parallel_jobs, # Max parallelism for this step
                enforce_num_trials=True,
                model_kwargs={"seed": seed}, # Any kwargs you want passed into the model
                model_gen_kwargs={'enforce_num_arms': True}, # Any kwargs you want passed to `modelbridge.gen`
            )
        )

    chosen_non_random_model = Models.BOTORCH_MODULAR

    available_models = list(Models.__members__.keys())

    if args.model:
        if str(args.model).upper() in available_models:
            chosen_non_random_model = Models.__members__[str(args.model).upper()]
        else:
            print_red(f"⚠ Cannot use {args.model}. Available models are: {', '.join(available_models)}. Using BOTORCH_MODULAR instead.")

        if args.model.lower() != "FACTORIAL" and args.gridsearch:
            print_red("Gridsearch only really works when you chose the FACTORIAL model.")

    # 2. Bayesian optimization step (requires data obtained from previous phase and learns
    # from all data available at the time of each new candidate generation call)

    #print("get_generation_strategy: Second step")

    #print(f"_steps.append(")
    #print(f"    GenerationStep(")
    #print(f"        model={chosen_non_random_model},")
    #print(f"        num_trials=-1, # No limitation on how many trials should be produced from this step")
    #print(f"        max_parallelism={_num_parallel_jobs} * 2, # Max parallelism for this step")
    #print(f"        #model_kwargs=seed: {seed}, # Any kwargs you want passed into the model")
    #print(f"        #enforce_num_trials=True,")
    #print(f"        model_gen_kwargs='enforce_num_arms': True, # Any kwargs you want passed to `modelbridge.gen`")
    #print(f"        # More on parallelism vs. required samples in BayesOpt:")
    #print(f"        # https://ax.dev/docs/bayesopt.html#tradeoff-between-parallelism-and-total-number-of-trials")
    #print(f"    )")
    #print(f")")

    #_nr_trials = _max_eval - random_steps + nr_of_imported_jobs

    #if _nr_trials <= 0:
    #    _nr_trials = -1

    _nr_trials = -1
    #print(f"_nr_trials: {_nr_trials}")

    _steps.append(
        GenerationStep(
            model=chosen_non_random_model,
            num_trials=_nr_trials, # No limitation on how many trials should be produced from this step
            max_parallelism=_num_parallel_jobs * 2, # Max parallelism for this step
            #model_kwargs={"seed": seed}, # Any kwargs you want passed into the model
            #enforce_num_trials=True,
            model_gen_kwargs={'enforce_num_arms': True}, # Any kwargs you want passed to `modelbridge.gen`
            # More on parallelism vs. required samples in BayesOpt:
            # https://ax.dev/docs/bayesopt.html#tradeoff-between-parallelism-and-total-number-of-trials
        )
    )

    gs = GenerationStrategy(
        steps=_steps
    )

    return gs

def create_and_execute_next_runs(next_nr_steps, phase, _max_eval, _progress_bar):
    global random_steps

    if next_nr_steps == 0:
        return 0

    trial_index_to_param = None
    try:
        try:
            trial_index_to_param = _get_next_trials()

            if trial_index_to_param:
                i = 1
                for trial_index, parameters in trial_index_to_param.items():
                    if break_run_search("create_and_execute_next_runs", _max_eval, _progress_bar):
                        break

                    while len(global_vars["jobs"]) > num_parallel_jobs:
                        finish_previous_jobs(["finishing prev jobs"])

                        if break_run_search("create_and_execute_next_runs", _max_eval, _progress_bar):
                            break

                        if is_slurm_job() and not args.force_local_execution:
                            _sleep(5)

                    if not break_run_search("create_and_execute_next_runs", _max_eval, _progress_bar):
                        progressbar_description([f"starting parameter set ({i}/{next_nr_steps})"])
                        execute_evaluation([
                            trial_index,
                            parameters,
                            i,
                            next_nr_steps,
                            phase
                        ])
                        i += 1
                    else:
                        break
        except botorch.exceptions.errors.InputDataError as e:
            print_red(f"Error 1: {e}")
            return 0
        except ax.exceptions.core.DataRequiredError as e:
            if "transform requires non-empty data" in str(e) and args.num_random_steps == 0:
                print_red(f"Error 5: {e} This may happen when there are no random_steps, but you tried to get a model anyway. Increase --num_random_steps to at least 1 to continue.")
                sys.exit(233)
            else:
                print_red(f"Error 2: {e}")
                return 0
    except RuntimeError as e:
        print_red("\n⚠ " + str(e))
    except (
        botorch.exceptions.errors.ModelFittingError,
        ax.exceptions.core.SearchSpaceExhausted,
        ax.exceptions.core.DataRequiredError,
        botorch.exceptions.errors.InputDataError
    ) as e:
        print_red("\n⚠ " + str(e))
        end_program(RESULT_CSV_FILE, 1)

    num_new_keys = 0

    try:
        num_new_keys = len(trial_index_to_param.keys())
    except Exception:
        pass

    return num_new_keys

def get_random_steps_from_prev_job():
    if not args.continue_previous_job:
        return count_sobol_steps()

    prev_step_file = args.continue_previous_job + "/state_files/phase_random_steps"

    if not os.path.exists(prev_step_file):
        return count_sobol_steps()

    return add_to_phase_counter("random", count_sobol_steps() + _count_sobol_steps(f"{args.continue_previous_job}/results.csv"), args.continue_previous_job)

def get_number_of_steps(_max_eval):
    _random_steps = args.num_random_steps

    already_done_random_steps = get_random_steps_from_prev_job()

    _random_steps = args.num_random_steps - already_done_random_steps

    if _random_steps > _max_eval:
        print_yellow(f"You have less --max_eval {_max_eval} than --num_random_steps {_random_steps}. Switched both.")
        _random_steps, _max_eval = _max_eval, _random_steps

    if _random_steps < num_parallel_jobs and SYSTEM_HAS_SBATCH:
        old_random_steps = _random_steps
        _random_steps = num_parallel_jobs
        original_print(f"_random_steps {old_random_steps} is smaller than num_parallel_jobs {num_parallel_jobs}. --num_random_steps will be ignored and set to num_parallel_jobs ({num_parallel_jobs}) to not have idle workers in the beginning.")

    if _random_steps > _max_eval:
        set_max_eval(_random_steps)

    original_second_steps = _max_eval - _random_steps
    second_step_steps = max(0, original_second_steps)
    if second_step_steps != original_second_steps:
        original_print(f"? original_second_steps: {original_second_steps} = max_eval {_max_eval} - _random_steps {_random_steps}")
    if second_step_steps == 0:
        print_red("This is basically a random search. Increase --max_eval or reduce --num_random_steps")

    second_step_steps = second_step_steps - already_done_random_steps

    if args.continue_previous_job:
        second_step_steps = _max_eval

    return _random_steps, second_step_steps

def get_executor():
    log_folder = f'{CURRENT_RUN_FOLDER}/single_runs/%j'
    global executor

    if args.force_local_execution:
        executor = submitit.LocalExecutor(folder=log_folder)
    else:
        executor = submitit.AutoExecutor(folder=log_folder)

    # 'nodes': <class 'int'>, 'gpus_per_node': <class 'int'>, 'tasks_per_node': <class 'int'>

    executor.update_parameters(
        name=f'{global_vars["experiment_name"]}_{run_uuid}',
        timeout_min=args.worker_timeout,
        tasks_per_node=args.tasks_per_node,
        slurm_gres=f"gpu:{args.gpus}",
        cpus_per_task=args.cpus_per_task,
        nodes=args.nodes_per_job,
        stderr_to_stdout=args.stderr_to_stdout,
        mem_gb=args.mem_gb,
        slurm_signal_delay_s=args.slurm_signal_delay_s,
        slurm_use_srun=args.slurm_use_srun,
        exclude=args.exclude
    )

    if args.exclude:
        print_yellow(f"Excluding the following nodes: {args.exclude}")

def append_and_read(file, nr=0):
    try:
        with open(file, mode='a+', encoding="utf-8") as f:
            f.seek(0)  # Setze den Dateizeiger auf den Anfang der Datei
            anzahl_zeilen = len(f.readlines())

            if nr == 1:
                f.write('1\n')

        return anzahl_zeilen

    except FileNotFoundError as e:
        original_print(f"File not found: {e}")
    except (SignalUSR, SignalINT, SignalCONT):
        append_and_read(file, nr)
    except OSError as e:
        print_red(f"OSError: {e}. This may happen on unstable file systems.")
        sys.exit(199)
    except Exception as e:
        print(f"Error editing the file: {e}")

    return 0

def failed_jobs(nr=0):
    state_files_folder = f"{CURRENT_RUN_FOLDER}/state_files/"

    if not os.path.exists(state_files_folder):
        os.makedirs(state_files_folder)

    return append_and_read(f'{CURRENT_RUN_FOLDER}/state_files/failed_jobs', nr)

def get_steps_from_prev_job(prev_job, nr=0):
    state_files_folder = f"{CURRENT_RUN_FOLDER}/state_files/"

    if not os.path.exists(state_files_folder):
        os.makedirs(state_files_folder)

    return append_and_read(f"{prev_job}/state_files/submitted_jobs", nr)

def succeeded_jobs(nr=0):
    state_files_folder = f"{CURRENT_RUN_FOLDER}/state_files/"

    if not os.path.exists(state_files_folder):
        os.makedirs(state_files_folder)

    return append_and_read(f'{CURRENT_RUN_FOLDER}/state_files/succeeded_jobs', nr)

def submitted_jobs(nr=0):
    state_files_folder = f"{CURRENT_RUN_FOLDER}/state_files/"

    if not os.path.exists(state_files_folder):
        os.makedirs(state_files_folder)

    return append_and_read(f'{CURRENT_RUN_FOLDER}/state_files/submitted_jobs', nr)

def count_done_jobs():
    csv_file_path = save_pd_csv()

    return _count_done_jobs(csv_file_path)

def _count_done_jobs(csv_file_path):
    results = 0

    if not os.path.exists(csv_file_path):
        return results

    df = None

    __debug = f"_count_done_jobs({csv_file_path})\n"
    with open(csv_file_path, mode='r', encoding="utf-8") as fin:
        __debug += fin.read()

    _err = False

    try:
        df = pd.read_csv(csv_file_path, index_col=0, float_precision='round_trip')
        df.dropna(subset=["result"], inplace=True)
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
        return 1

    assert df is not None, "DataFrame should not be None after reading CSV file"
    assert "generation_method" in df.columns, "'generation_method' column must be present in the DataFrame"

    completed_rows = df[df["trial_status"] == "COMPLETED"]
    completed_rows_count = len(completed_rows)

    return completed_rows_count

def count_sobol_steps():
    csv_file_path = save_pd_csv()

    return _count_sobol_steps(csv_file_path)

def _count_sobol_steps(csv_file_path):
    sobol_count = 0

    if not os.path.exists(csv_file_path):
        return sobol_count

    df = None

    _err = False

    try:
        df = pd.read_csv(csv_file_path, index_col=0, float_precision='round_trip')
        df.dropna(subset=["result"], inplace=True)
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
    assert "generation_method" in df.columns, "'generation_method' column must be present in the DataFrame"

    sobol_rows = df[df["generation_method"] == "Sobol"]
    sobol_count = len(sobol_rows)

    return sobol_count

def execute_nvidia_smi():
    if not IS_NVIDIA_SMI_SYSTEM:
        print_debug("Cannot find nvidia-smi. Cannot take GPU logs")
        return

    while True:
        try:
            host = socket.gethostname()

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
        except Exception as e:
            print(f"An error occurred: {e}")
        if is_slurm_job() and not args.force_local_execution:
            _sleep(10)

def start_nvidia_smi_thread():
    if IS_NVIDIA_SMI_SYSTEM:
        nvidia_smi_thread = threading.Thread(target=execute_nvidia_smi, daemon=True)
        nvidia_smi_thread.start()
        return nvidia_smi_thread
    return None

def break_run_search(_name, _max_eval, _progress_bar):
    if succeeded_jobs() > _max_eval:
        print_debug(f"breaking {_name}: succeeded_jobs() {succeeded_jobs()} > max_eval {_max_eval}")
        return True

    if _progress_bar.total < submitted_jobs():
        print_debug(f"breaking {_name}: _progress_bar.total {_progress_bar.total} <= submitted_jobs() {submitted_jobs()}")
        return True

    if count_done_jobs() >= _max_eval:
        print_debug(f"breaking {_name}: count_done_jobs() {count_done_jobs()} > max_eval {_max_eval}")
        return True

    if submitted_jobs() > _max_eval:
        print_debug(f"breaking {_name}: submitted_jobs() {submitted_jobs()} > max_eval {_max_eval}")
        return True

    if abs(count_done_jobs() - _max_eval - NR_INSERTED_JOBS) <= 0:
        print_debug(f"breaking {_name}: if abs(count_done_jobs() {count_done_jobs()} - max_eval {_max_eval} - NR_INSERTED_JOBS {NR_INSERTED_JOBS}) <= 0")
        return True

    return False

def run_search(_progress_bar):
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

        next_nr_steps = get_next_nr_steps(num_parallel_jobs, max_eval)

        nr_of_items = 0

        if next_nr_steps:
            progressbar_description([f"trying to get {next_nr_steps} next steps (current done: {count_done_jobs()}, max: {max_eval})"])

            nr_of_items = create_and_execute_next_runs(next_nr_steps, "systematic", max_eval, _progress_bar)

            progressbar_description([f"got {nr_of_items}, requested {next_nr_steps}"])

        _debug_worker_creation(f"{int(time.time())}, {len(global_vars['jobs'])}, {nr_of_items}, {next_nr_steps}")

        finish_previous_jobs(["finishing prev jobs"])

        if is_slurm_job() and not args.force_local_execution:
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

    wait_for_jobs_to_complete(num_parallel_jobs)

    while len(global_vars["jobs"]):
        finish_previous_jobs([f"waiting for jobs ({len(global_vars['jobs'])} left)"])
        if is_slurm_job() and not args.force_local_execution:
            _sleep(1)

    log_what_needs_to_be_logged()
    return False

def wait_for_jobs_to_complete(_num_parallel_jobs):
    if SYSTEM_HAS_SBATCH:
        while len(global_vars["jobs"]) > _num_parallel_jobs:
            progressbar_description([f"waiting for old jobs to finish ({len(global_vars['jobs'])} left)"])
            if is_slurm_job() and not args.force_local_execution:
                _sleep(5)
            clean_completed_jobs()

def print_logo():
    original_print("""
   ---------
  (OmniOpt2!)
   ---------
          \\/
         /\\_/\\
        ( o.o )
         > ^ <  ,"",
         ( " ) :
          (|)""
""")

def is_already_in_defective_nodes(hostname):
    file_path = os.path.join(CURRENT_RUN_FOLDER, "state_files", "defective_nodes")

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if not os.path.isfile(file_path):
        print_red(f"Error: The file {file_path} does not exist.")
        return False

    try:
        # Datei öffnen und Zeilen durchsuchen
        with open(file_path, mode="r", encoding="utf-8") as file:
            for line in file:
                # Zeilenenden entfernen und auf Übereinstimmung prüfen
                if line.strip() == hostname:
                    return True
    except Exception as e:
        print_red(f"Error reading the file {file_path}: {e}")
        return False

    # Wenn keine Übereinstimmung gefunden wurde
    return False

def count_defective_nodes(file_path=None, entry=None):
    """
    Diese Funktion nimmt optional einen Dateipfad und einen Eintrag entgegen.
    Sie öffnet die Datei, erstellt sie, wenn sie nicht existiert,
    prüft, ob der Eintrag bereits als einzelne Zeile in der Datei vorhanden ist,
    und fügt ihn am Ende hinzu, wenn dies nicht der Fall ist.
    Schließlich gibt sie eine sortierte Liste aller eindeutigen Einträge in der Datei zurück.
    Wenn keine Argumente übergeben werden, wird der Dateipfad auf
    '{CURRENT_RUN_FOLDER}/state_files/defective_nodes' gesetzt und kein Eintrag hinzugefügt.
    """
    # Standardpfad für die Datei, wenn keiner angegeben ist
    if file_path is None:
        file_path = os.path.join(CURRENT_RUN_FOLDER, "state_files", "defective_nodes")

    # Sicherstellen, dass das Verzeichnis existiert
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    try:
        # Öffnen der Datei im Modus 'a+' (Anhängen und Lesen)
        with open(file_path, mode='a+', encoding="utf-8") as file:
            file.seek(0)  # Zurück zum Anfang der Datei
            lines = file.readlines()

            # Entfernen von Zeilenumbrüchen und Erstellen einer Liste der Einträge
            entries = [line.strip() for line in lines]

            # Prüfen, ob der Eintrag nicht None und nicht bereits vorhanden ist
            if entry is not None and entry not in entries:
                file.write(entry + '\n')
                entries.append(entry)

        # Zurückgeben der sortierten, eindeutigen Liste der Einträge
        return sorted(set(entries))

    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")
        return []

def human_readable_generation_strategy():
    generation_strategy_str = str(ax_client.generation_strategy)

    pattern = r'\[(.*?)\]'

    match = re.search(pattern, generation_strategy_str)

    if match:
        content = match.group(1)
        return content

    return None

def parse_orchestrator_file(_f):
    if os.path.exists(_f):
        with open(_f, mode='r', encoding="utf-8") as file:
            try:
                data = yaml.safe_load(file)

                if "errors" not in data:
                    print_red(f"{_f} file does not contain key 'errors'")
                    sys.exit(206)

                valid_keys = ['name', 'match_strings', 'behavior']
                valid_behaviours = ["ExcludeNodeAndRestartAll", "RestartOnDifferentNode", "ExcludeNode"]

                for x in data["errors"]:
                    if not isinstance(x, dict):
                        print_red(f"Entry is not of type dict but {type(x)}")
                        sys.exit(206)

                    if set(x.keys()) != set(valid_keys):
                        print_red(f"{x.keys()} does not match {valid_keys}")
                        sys.exit(206)

                    if x["behavior"] not in valid_behaviours:
                        print_red(f"behavior-entry {x['behavior']} is not in valid_behaviours: {', '.join(valid_behaviours)}")
                        sys.exit(206)

                    if not isinstance(x["name"], str):
                        print_red(f"name-entry is not string but {type(x['name'])}")
                        sys.exit(206)

                    if not isinstance(x["match_strings"], list):
                        print_red(f"name-entry is not list but {type(x['match_strings'])}")
                        sys.exit(206)

                    for y in x["match_strings"]:
                        if not isinstance(y, str):
                            print_red("x['match_strings'] is not a string but {type(x['match_strings'])}")
                            sys.exit(206)

                #helpers.dier(data)

                return data
            except Exception as e:
                print(f"Error while parse_experiment_parameters({_f}): {e}")

                return None
    else:
        print_red(f"{_f} could not be found")

        return None

def set_orchestrator():
    global orchestrator

    if args.orchestrator_file:
        if SYSTEM_HAS_SBATCH:
            orchestrator = parse_orchestrator_file(args.orchestrator_file)
        else:
            print_yellow("--orchestrator_file will be ignored on non-sbatch-systems.")

def check_if_has_random_steps():
    if (not args.continue_previous_job and not args.load_previous_job_data and "--continue" not in sys.argv) and (args.num_random_steps == 0 or not args.num_random_steps):
        print_red("You have no random steps set. This is only allowed in continued jobs. To start, you need either some random steps, or a continued run.")
        my_exit(233)

def add_exclude_to_defective_nodes():
    if args.exclude:
        entries = [entry.strip() for entry in args.exclude.split(',')]

        for entry in entries:
            count_defective_nodes(None, entry)

def check_max_eval(_max_eval):
    if not _max_eval:
        print_red("--max_eval needs to be set!")
        sys.exit(19)

def main():
    print_debug("main()")

    print_logo()

    global RESULT_CSV_FILE
    global ax_client
    global global_vars
    global max_eval
    global global_vars
    global RUN_FOLDER_NUMBER
    global CURRENT_RUN_FOLDER
    global NVIDIA_SMI_LOGS_BASE
    global LOGFILE_DEBUG_GET_NEXT_TRIALS
    global random_steps

    check_if_has_random_steps()

    _debug_worker_creation("time, nr_workers, got, requested, phase")

    original_print("./omniopt " + " ".join(sys.argv[1:]))

    check_slurm_job_id()

    CURRENT_RUN_FOLDER = f"{args.run_dir}/{global_vars['experiment_name']}/{RUN_FOLDER_NUMBER}"
    while os.path.exists(f"{CURRENT_RUN_FOLDER}"):
        CURRENT_RUN_FOLDER = f"{args.run_dir}/{global_vars['experiment_name']}/{RUN_FOLDER_NUMBER}"
        RUN_FOLDER_NUMBER = RUN_FOLDER_NUMBER + 1

    RESULT_CSV_FILE = create_folder_and_file(f"{CURRENT_RUN_FOLDER}")

    save_state_files()

    print(f"[yellow]Run-folder[/yellow]: [underline]{CURRENT_RUN_FOLDER}[/underline]")
    if args.continue_previous_job:
        print(f"[yellow]Continuation from {args.continue_previous_job}[/yellow]")

    NVIDIA_SMI_LOGS_BASE = f'{CURRENT_RUN_FOLDER}/gpu_usage_'

    if args.ui_url:
        with open(f"{CURRENT_RUN_FOLDER}/ui_url.txt", mode="a", encoding="utf-8") as myfile:
            myfile.write(decode_if_base64(args.ui_url))

    LOGFILE_DEBUG_GET_NEXT_TRIALS = f'{CURRENT_RUN_FOLDER}/get_next_trials.csv'

    experiment_parameters = None
    cli_params_experiment_parameters = None
    checkpoint_parameters_filepath = f"{CURRENT_RUN_FOLDER}/state_files/checkpoint.json.parameters.json"

    if args.parameter:
        experiment_parameters = parse_experiment_parameters()
        cli_params_experiment_parameters = experiment_parameters

    disable_logging()

    check_max_eval(max_eval)

    random_steps, second_step_steps = get_number_of_steps(max_eval)

    add_exclude_to_defective_nodes()

    if args.parameter and len(args.parameter) and args.continue_previous_job and random_steps <= 0:
        print(f"A parameter has been reset, but the earlier job already had it's random phase. To look at the new search space, {args.num_random_steps} random steps will be executed.")
        random_steps = args.num_random_steps

    gs = get_generation_strategy(num_parallel_jobs, args.seed, args.max_eval)

    ax_client = AxClient(
        verbose_logging=args.verbose,
        enforce_sequential_optimization=args.enforce_sequential_optimization,
        generation_strategy=gs
    )

    minimize_or_maximize = not args.maximize

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

    gs_hr = human_readable_generation_strategy()
    if gs_hr:
        print(f"Generation strategy: {gs_hr}")

    with open(checkpoint_parameters_filepath, mode="w", encoding="utf-8") as outfile:
        json.dump(experiment_parameters, outfile, cls=NpEncoder)

    print_overview_tables(experiment_parameters, experiment_args)

    get_executor()

    load_existing_job_data_into_ax_client()

    original_print(f"Run-Program: {global_vars['joined_run_program']}")

    max_nr_steps = second_step_steps
    if count_done_jobs() < random_steps:
        max_nr_steps = (random_steps - count_done_jobs()) + second_step_steps
        set_max_eval(max_nr_steps)

    prev_steps_nr = 0

    if args.continue_previous_job:
        prev_steps_nr = get_steps_from_prev_job(args.continue_previous_job)

        max_nr_steps = prev_steps_nr + max_nr_steps

    save_global_vars()

    write_process_info()

    with tqdm(total=max_eval, disable=False) as _progress_bar:
        write_process_info()
        global progress_bar
        progress_bar = _progress_bar

        progressbar_description(["Started OmniOpt2 run..."])

        update_progress_bar(progress_bar, count_done_jobs())

        run_search(progress_bar)

        wait_for_jobs_to_complete(num_parallel_jobs)

    end_program(RESULT_CSV_FILE)

def _unidiff_output(expected, actual):
    """
    Helper function. Returns a string containing the unified diff of two multiline strings.
    """

    expected = expected.splitlines(1)
    actual = actual.splitlines(1)

    diff = difflib.unified_diff(expected, actual)

    return ''.join(diff)

def print_diff(i, o):
    if isinstance(i, str):
        print("Should be:", i.strip())
    else:
        print("Should be:", i)

    if isinstance(o, str):
        print("Is:", o.strip())
    else:
        print("Is:", o)
    if isinstance(i, str) or isinstance(o, str):
        print("Diff:", _unidiff_output(json.dumps(i), json.dumps(o)))

def is_equal(n, i, o):
    r = _is_equal(n, i, o)

    if r:
        print_diff(i, o)

    return r

def is_not_equal(n, i, o):
    r = _is_not_equal(n, i, o)

    if r:
        print_diff(i, o)

    return r

def _is_not_equal(name, _input, output):
    _equal_types = [
        int, str, float, bool
    ]
    for equal_type in _equal_types:
        if isinstance(_input, equal_type) and isinstance(output, equal_type) and _input == output:
            print_red(f"Failed test (1): {name}")
            return 1

    if isinstance(_input, bool) and _input == output:
        print_red(f"Failed test (2): {name}")
        return 1

    if not (output is not None and _input is not None):
        print_red(f"Failed test (3): {name}")
        return 1

    print_green(f"Test OK: {name}")
    return 0

def _is_equal(name, _input, output):
    _equal_types = [
        int, str, float, bool
    ]
    for equal_type in _equal_types:
        if type(_input) is equal_type and type(output) and _input != output:
            print_red(f"Failed test (1): {name}")
            return 1

    if type(_input) is not type(output):
        print_red(f"Failed test (4): {name}")
        return 1

    if isinstance(_input, bool) and _input != output:
        print_red(f"Failed test (6): {name}")
        return 1

    if (output is None and _input is not None) or (output is not None and _input is None):
        print_red(f"Failed test (7): {name}")
        return 1

    print_green(f"Test OK: {name}")
    return 0

def complex_tests(_program_name, wanted_stderr, wanted_exit_code, wanted_signal, res_is_none=False):
    print_yellow(f"Test suite: {_program_name}")

    nr_errors = 0

    program_path = f"./.tests/test_wronggoing_stuff.bin/bin/{_program_name}"

    if not os.path.exists(program_path):
        print_red(f"Program path {program_path} not found!")
        my_exit(18)

    program_path_with_program = f"{program_path}"

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

    stdout_stderr_exit_code_signal = execute_bash_code(program_string_with_params)

    stdout = stdout_stderr_exit_code_signal[0]
    stderr = stdout_stderr_exit_code_signal[1]
    exit_code = stdout_stderr_exit_code_signal[2]
    _signal = stdout_stderr_exit_code_signal[3]

    res = get_result(stdout)

    if res_is_none:
        nr_errors += is_equal(f"{_program_name} res is None", None, res)
    else:
        nr_errors += is_equal(f"{_program_name} res type is nr", True, isinstance(res, (float, int)))
    nr_errors += is_equal(f"{_program_name} stderr", True, wanted_stderr in stderr)
    nr_errors += is_equal(f"{_program_name} exit-code ", exit_code, wanted_exit_code)
    nr_errors += is_equal(f"{_program_name} signal", _signal, wanted_signal)

    return nr_errors

def get_files_in_dir(mypath):
    print_debug("get_files_in_dir")
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    return [mypath + "/" + s for s in onlyfiles]

def test_find_paths(program_code):
    nr_errors = 0

    files = [
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

    text = " -- && !!  ".join(files)

    string = find_file_paths_and_print_infos(text, program_code)

    for i in files:
        if i not in string:
            if os.path.exists(i):
                print("Missing {i} in find_file_paths string!")
                nr_errors += 1

    return nr_errors

def run_tests():
    global global_vars

    print_image_to_cli(".tools/slimer.png", 300)
    print_image_to_cli(".tools/slimer2.png", 300)

    nr_errors = 0

    nr_errors += is_not_equal("nr equal string", 1, "1")

    nr_errors += is_equal("nr equal nr", 1, 1)
    nr_errors += is_not_equal("unequal strings", "hallo", "welt")

    nr_errors += is_equal("get_result(None)", get_result(None), None)
    nr_errors += is_equal("get_result(123)", get_result(123), None)
    nr_errors += is_equal("get_result('RESULT: 10')", get_result('RESULT: 10'), 10.0)
    nr_errors += is_equal("helpers.looks_like_float(10)", helpers.looks_like_float(10), True)
    nr_errors += is_equal("helpers.looks_like_float('hallo')", helpers.looks_like_float('hallo'), False)
    nr_errors += is_equal("helpers.looks_like_int('hallo')", helpers.looks_like_int('hallo'), False)
    nr_errors += is_equal("helpers.looks_like_int('1')", helpers.looks_like_int('1'), True)
    nr_errors += is_equal("helpers.looks_like_int(False)", helpers.looks_like_int(False), False)
    nr_errors += is_equal("helpers.looks_like_int(True)", helpers.looks_like_int(True), False)
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

    nr_errors += is_equal(
        "_count_sobol_steps('/etc/idontexist')",
        _count_sobol_steps("/etc/idontexist"),
        0
    )

    nr_errors += is_equal(
        "_count_done_jobs('/etc/idontexist')",
        _count_done_jobs("/etc/idontexist"),
        0
    )

    nr_errors += is_equal(
        "get_program_code_from_out_file('/etc/doesntexist')",
        get_program_code_from_out_file("/etc/doesntexist"),
        ""
    )

    nr_errors += is_equal("get_type_short('RangeParameter')", get_type_short("RangeParameter"), "range")
    nr_errors += is_equal(
        "get_type_short('ChoiceParameter')",
        get_type_short("ChoiceParameter"),
        "choice"
    )
    nr_errors += is_equal(
        "create_and_execute_next_runs(0, None, None, None)",
        create_and_execute_next_runs(0, None, None, None),
        0
    )

    #complex_tests (_program_name, wanted_stderr, wanted_exit_code, wanted_signal, res_is_none=False):
    nr_errors += complex_tests("simple_ok", "hallo", 0, None)
    nr_errors += complex_tests(
        "divide_by_0",
        'Illegal division by zero at ./.tests/test_wronggoing_stuff.bin/bin/divide_by_0 line 3.\n',
        255,
        None,
        True
    )
    #nr_errors += complex_tests("result_but_exit_code_stdout_stderr", "stderr", 5, None)
    #nr_errors += complex_tests("signal_but_has_output", "Killed", 137, None) # Doesnt show Killed on taurus
    nr_errors += complex_tests("exit_code_no_output", "", 5, None, True)
    nr_errors += complex_tests("exit_code_stdout", "STDERR", 5, None, False)
    nr_errors += complex_tests("no_chmod_x", "Permission denied", 126, None, True)
    #nr_errors += complex_tests("signal", "Killed", 137, None, True) # Doesnt show Killed on taurus
    nr_errors += complex_tests("exit_code_stdout_stderr", "This has stderr", 5, None, True)
    nr_errors += complex_tests("module_not_found", "ModuleNotFoundError", 1, None, True)

    find_path_res = test_find_paths("ls")
    if find_path_res:
        is_equal("test_find_paths failed", True, False)
        nr_errors += find_path_res

    my_exit(nr_errors)

def get_first_line_of_file_that_contains_string(i, s):
    if not os.path.exists(i):
        print_debug(f"File {i} not found")
        return ""

    f = get_file_as_string(i)

    lines = ""
    get_lines_until_end = False

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

def get_python_errors():
    synerr = "Python syntax error detected. Check log file."

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

def get_exit_codes():
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

def get_base_errors():
    base_errors = [
        "Segmentation fault",
        "Illegal division by zero",
        "OOM",
        ["Killed", "Detected kill, maybe OOM or Signal?"]
    ]

    return base_errors

def get_first_line_of_file(file_paths):
    first_line = ""
    if len(file_paths):
        first_file_as_string = ""
        try:
            first_file_as_string = get_file_as_string(file_paths[0])
            if isinstance(first_file_as_string, str) and first_file_as_string.strip().isprintable():
                first_line = first_file_as_string.split('\n')[0]
        except UnicodeDecodeError:
            pass

        if first_file_as_string == "":
            first_line = "#!/bin/bash"

    return first_line

def check_for_basic_string_errors(file_as_string, first_line, file_paths, program_code):
    errors = []

    if first_line and isinstance(first_line, str) and first_line.isprintable() and not first_line.startswith("#!"):
        errors.append("First line does not seem to be a shebang line: " + first_line)

    if "Permission denied" in file_as_string and "/bin/sh" in file_as_string:
        errors.append("Log file contains 'Permission denied'. Did you try to run the script without chmod +x?")

    if "Exec format error" in file_as_string:
        current_platform = platform.machine()
        file_output = ""

        if len(file_paths):
            file_result = execute_bash_code("file " + file_paths[0])
            if len(file_result) and isinstance(file_result[0], str):
                file_output = ", " + file_result[0].strip()

        errors.append(f"Was the program compiled for the wrong platform? Current system is {current_platform}{file_output}")

    if "/bin/sh" in file_as_string and "not found" in file_as_string:
        errors.append("Wrong path? File not found")

    if len(file_paths) and os.stat(file_paths[0]).st_size == 0:
        errors.append(f"File in {program_code} is empty")

    if len(file_paths) == 0:
        errors.append(f"No files could be found in your program string: {program_code}")

    return errors

def check_for_python_errors(i, file_as_string):
    errors = []

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

def check_for_non_zero_exit_codes(file_as_string):
    errors = []
    for r in range(1, 255):
        special_exit_codes = get_exit_codes()
        search_for_exit_code = "Exit-Code: " + str(r) + ","
        if search_for_exit_code in file_as_string:
            _error = "Non-zero exit-code detected: " + str(r)
            if str(r) in special_exit_codes:
                _error += " (May mean " + special_exit_codes[str(r)] + ", unless you used that exit code yourself or it was part of any of your used libraries or programs)"
            errors.append(_error)
    return errors

def check_for_base_errors(file_as_string):
    errors = []
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

def get_errors_from_outfile(i):
    file_as_string = get_file_as_string(i)

    program_code = get_program_code_from_out_file(i)
    file_paths = find_file_paths(program_code)

    first_line = get_first_line_of_file(file_paths)

    errors = []

    if "Result: None" in file_as_string:
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

def log_nr_of_workers():
    try:
        write_process_info()
    except Exception as e:
        print_debug(f"log_nr_of_workers: failed to write_process_info: {e}")

    if "jobs" not in global_vars:
        print_debug("log_nr_of_workers: Could not find jobs in global_vars")
        return

    nr_of_workers = len(global_vars["jobs"])

    if not nr_of_workers:
        return

    try:
        with open(logfile_nr_workers, mode='a+', encoding="utf-8") as f:
            f.write(str(nr_of_workers) + "\n")
    except FileNotFoundError:
        print_red(f"It seems like the folder for writing {logfile_nr_workers} was deleted during the run. Cannot continue.")
        sys.exit(99)
    except OSError as e:
        print_red(f"Tried writing log_nr_of_workers to file {logfile_nr_workers}, but failed with error {e}. This may mean that the file system you are running on is instable. OmniOpt probably cannot do anything about it.")
        sys.exit(199)

def get_best_params(csv_file_path):
    results = {
        "result": None,
        "parameters": {}
    }

    if not os.path.exists(csv_file_path):
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

        if isinstance(this_line_result, str) and re.match(r'^-?\d+(?:\.\d+)$', this_line_result) is not None:
            this_line_result = float(this_line_result)

        if type(this_line_result) in [float, int]:
            if best_result is None:
                best_line = this_line
                best_result = this_line_result
            elif args.maximize and this_line_result >= best_result:
                best_line = this_line
                best_result = this_line_result
            elif not args.maximize and this_line_result <= best_result:
                best_line = this_line
                best_result = this_line_result

    if best_line is None:
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
                results["result"] = "{:f}".format(best_line[i]) if type(best_line[i]) in [int, float] else best_line[i]
            else:
                results["parameters"][col] = "{:f}".format(best_line[i]) if type(best_line[i]) in [int, float] else best_line[i]

    return results

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if args.tests:
            print_red("This should be red")
            print_yellow("This should be yellow")
            print_green("This should be green")

            run_tests()
        else:
            try:
                main()
            except (SignalUSR, SignalINT, SignalCONT, KeyboardInterrupt):
                print_red("\n⚠ You pressed CTRL+C or got a signal. Optimization stopped.")
                IS_IN_EVALUATE = False

                end_program(RESULT_CSV_FILE, 1)
            except SearchSpaceExhausted:
                _get_perc = abs(int(((count_done_jobs() - NR_INSERTED_JOBS) / max_eval) * 100))

                if _get_perc < 100:
                    print_red(f"\nIt seems like the search space was exhausted. "
                        f"You were able to get {_get_perc}% of the jobs you requested "
                        f"(got: {count_done_jobs() - NR_INSERTED_JOBS}, "
                        f"requested: {max_eval}) after main ran"
                    )

                if _get_perc != 100:
                    end_program(RESULT_CSV_FILE, 1, 87)
                else:
                    end_program(RESULT_CSV_FILE, 1)
