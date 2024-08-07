#!/bin/env python3

# Idee: gridsearch implementieren, i.e. range -> choice mit allen werten zwischen low und up
# Geht, aber was ist mit continued runs?

import sys

try:
    import traceback
    import time
except ModuleNotFoundError as e:
    print(f"ModuleNotFoundError: {e}")
    sys.exit(31)

def my_exit(_code=0):
    tb = traceback.format_exc()
    print_debug(f"Exiting with error code {_code}. Traceback: {tb}")

    time.sleep(5)

    print("Exit-Code: " + str(_code))
    sys.exit(_code)

class searchSpaceExhausted (Exception):
    pass

nr_inserted_jobs = 0
changed_grid_search_params = {}
last_cpu_mem_time = None

search_space_exhausted = False
SUPPORTED_MODELS = [
    "SOBOL",
    "GPEI",
    "FACTORIAL", # ValueError: RangeParameter(name='float_param', parameter_type=FLOAT, range=[-5.0, 5.0]) not ChoiceParameter or FixedParameter
    "SAASBO",
    "FULLYBAYESIAN",
    "LEGACY_BOTORCH",
    "BOTORCH_MODULAR",
    "UNIFORM",
    "BO_MIXED"
]

nr_of_0_results = 0
original_print = print

double_hashes = []
missing_results = []
already_inserted_param_hashes = {}
already_inserted_param_data = []

console = None
try:
    from rich.console import Console
    console = Console(force_terminal=True, force_interactive=True, soft_wrap=True, color_system="256")
except ModuleNotFoundError as e:
    print(f"Error loading module rich")

try:
    with console.status("[bold green]Loading functools...") as status:
        import functools
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
        from shutil import which
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
        from datetime import datetime, timezone
        import datetime
    with console.status("[bold green]Loading tzlocal...") as status:
        from tzlocal import get_localzone
    with console.status("[bold green]Loading platform...") as status:
        import platform
    with console.status("[bold green]Loading difflib...") as status:
        import difflib
    with console.status("[bold green]Loading warnings...") as status:
        import warnings
    with console.status("[bold green]Loading pandas...") as status:
        import pandas as pd
    with console.status("[bold green]Loading random...") as status:
        import random
    with console.status("[bold green]Loading pathlib...") as status:
        from pathlib import Path
    with console.status("[bold green]Loading glob...") as status:
        import glob
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
        from sixel import converter
    with console.status("[bold green]Loading PIL...") as status:
        from PIL import Image

    #with console.status("[bold green]Importing rich tracebacks...") as status:
    #    #from rich.traceback import install
    #    #install(show_locals=True)

    with console.status("[bold green]Loading rich.table...") as status:
        from rich.table import Table
    with console.status("[bold green]Loading print from rich...") as status:
        from rich import print
    with console.status("[bold green]Loading rich.progress...") as status:
        from rich.progress import track
        from rich.progress import BarColumn, Progress, TextColumn, TaskProgressColumn, TimeRemainingColumn, Column
    with console.status("[bold green]Loading csv...") as status:
        import csv
    with console.status("[bold green]Loading rich.pretty...") as status:
        from rich.pretty import pprint
    with console.status("[bold green]Loading pprint...") as status:
        from pprint import pprint
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
except KeyboardInterrupt:
    original_print("You cancelled loading the basic modules")
    my_exit(31)
except signalINT:
    print("\n⚠ Signal INT was detected. Exiting with 128 + 2.")
    my_exit(128 + 2)
except (signalUSR):
    print("\n⚠ Signal USR was detected. Exiting with 128 + 10.")
    my_exit(128 + 10)
except signalCONT:
    print("\n⚠ Signal CONT was detected. Exiting with 128 + 18.")
    my_exit(128 + 18)
except (KeyboardInterrupt) as e:
    print("\n⚠ You pressed CTRL+C. Program execution halted.")
    my_exit(0)

process = psutil.Process(os.getpid())

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

global_vars = {}

val_if_nothing_found = 99999999999999999999999999999999999999999999999999999999999
NO_RESULT = "{:.0e}".format(val_if_nothing_found)

global_vars["jobs"] = []
global_vars["_time"] = None
global_vars["mem_gb"] = None
global_vars["num_parallel_jobs"] = None
global_vars["parameter_names"] = []

# max_eval usw. in unterordner
# grid ausblenden
pd_csv_filename = "results.csv"
worker_percentage_usage = []
is_in_evaluate = False
end_program_ran = False
already_shown_worker_usage_over_time = False
ax_client = None
time_get_next_trials_took = []
current_run_folder = None
run_folder_number = 0
args = None
result_csv_file = None
shown_end_table = False
max_eval = None
random_steps = None
progress_bar = None
searching_for = None
sum_of_values_for_tqdm = 0

main_pid = os.getpid()

run_uuid = uuid.uuid4()

log_stack = []

def set_max_eval (new_max_eval):
    global max_eval

    #import traceback
    #print(f"set_max_eval(new_max_eval: {new_max_eval})")
    #traceback.print_stack()

    max_eval = new_max_eval

class LogEntry:
    def __init__(self, function_name, start_time, caller):
        self.function_name = function_name
        self.start_time = start_time
        self.caller = caller
        self.end_time = None
        self.duration = None
        self.sub_calls = []

    def end(self, end_time):
        self.end_time = end_time
        self.duration = end_time - self.start_time

    def to_dict(self):
        return {
            "function_name": self.function_name,
            "start_time": self.start_time,
            "caller": self.caller,
            "end_time": self.end_time,
            "duration": self.duration,
            "sub_calls": [sub_call.to_dict() for sub_call in self.sub_calls]
        }

def log_what_needs_to_be_logged ():
    if "write_worker_usage" in globals():
        try:
            write_worker_usage()
        except:
            pass

    if "write_process_info" in globals():
        try:
            write_process_info()
        except:
            pass

    if "log_nr_of_workers" in globals():
        try:
            log_nr_of_workers()
        except:
            pass

def sort_log_entries(log_entry):
    log_entry.sub_calls.sort(key=lambda x: x.duration, reverse=True)
    for sub_call in log_entry.sub_calls:
        sort_log_entries(sub_call)

def save_logs_to_json(filename):
    root_entries = [entry for entry in log_stack if entry.caller == "<module>"]
    try:
        for root_entry in root_entries:
            sort_log_entries(root_entry)
    except:
        pass

    sorted_entries = sorted(root_entries, key=lambda x: x.duration, reverse=True)
    with open(filename, 'w') as log_file:
        json.dump([entry.to_dict() for entry in sorted_entries], log_file, indent=4)

def get_nesting_level(caller_frame):
    filename, caller_lineno, _, _, _ = inspect.getframeinfo(caller_frame)
    with open(filename) as f:
        indentation_level = 0
        for token_record in tokenize.generate_tokens(f.readline):
            token_type, _, (token_lineno, _), _, _ = token_record
            if token_lineno > caller_lineno:
                break
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
        with open(logfile, 'a') as f:
            original_print(msg, file=f)
    except FileNotFoundError:
        print_red(f"It seems like the run's folder was deleted during the run. Cannot continue.")
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
    nl = get_nesting_level(inspect.currentframe().f_back)
    _tabs = ("\t" * nl)
    msg = f"{time_str}:{_tabs}{msg}"
    if args.debug:
        print(msg)

    try:
        _debug(f"{time_str}: {get_functions_stack_array()}")
    except:
        pass
    _debug(msg)

def write_process_info():
    global last_cpu_mem_time

    if last_cpu_mem_time is None:
        process.cpu_percent(interval=0.0)
        last_cpu_mem_time = time.time()
    elif abs(last_cpu_mem_time - time.time()) < 1:
        pass
    else:
        try:
            cpu_usage = process.cpu_percent(interval=1)
            ram_usage = process.memory_info().rss / (1024 * 1024)  # in MB

            print_debug(f"CPU Usage: {cpu_usage}%")
            print_debug(f"RAM Usage: {ram_usage} MB")
        except Exception as e:
            print_debug(f"Error retrieving process information: {str(e)}")
        last_cpu_mem_time = None

def getLineInfo():
    return(inspect.stack()[1][1],":",inspect.stack()[1][2],":",
          inspect.stack()[1][3])

#print(f"sys.path: {sys.path}")

with console.status("[bold green]Loading helpers-module...") as status:
    script_dir = os.path.dirname(os.path.realpath(__file__))
    helpers_file = f"{script_dir}/.helpers.py"
    spec = importlib.util.spec_from_file_location(
        name="helpers",
        location=helpers_file,
    )
    my_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(my_module)

def print_image_to_cli(image_path, width):
    print("")
    try:
        # Einlesen des Bildes um die Dimensionen zu erhalten
        image = Image.open(image_path)
        original_width, original_height = image.size

        # Berechnen der proportionalen Höhe
        height = int((original_height / original_width) * width)

        # Erstellen des SixelConverters mit den neuen Dimensionen
        sixel_converter = converter.SixelConverter(image_path, w=width, h=height)

        # Schreiben der Ausgabe in sys.stdout
        sixel_converter.write(sys.stdout)
        time.sleep(2)
    except Exception as e:
        print_debug(f"Error converting and resizing image: {str(e)}, width: {width}, image_path: {image_path}")

def datetime_from_string(input_string):
    return datetime.datetime.fromtimestamp(input_string)

def get_timezone_offset_seconds():
    # Get the current time in the local timezone
    local_tz = get_localzone()
    local_time = datetime.datetime.now(local_tz)

    # Get the offset of the local timezone from UTC in seconds
    offset = local_time.utcoffset().total_seconds()

    return offset

with console.status("[bold green]Defining creating .logs dir if it doesn't exist...") as status:
    log_dir = ".logs"
    try:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        original_print(f"Could not create logs for {os.path.abspath(log_dir)}: " + str(e))

with console.status("[bold green]Defining variables...") as status:
    log_dir = ".logs"
    log_i = 0
    logfile = f'{log_dir}/{log_i}'
    logfile_linewise = f'{log_dir}/{log_i}_linewise'
    logfile_nr_workers = f'{log_dir}/{log_i}_nr_workers'
    while os.path.exists(logfile):
        log_i = log_i + 1
        logfile = f'{log_dir}/{log_i}'

    logfile_nr_workers = f'{log_dir}/{log_i}_nr_workers'
    logfile_linewise = f'{log_dir}/{log_i}_linewise'
    logfile_progressbar = f'{log_dir}/{log_i}_progressbar'
    logfile_worker_creation_logs = f'{log_dir}/{log_i}_worker_creation_logs'
    logfile_trial_index_to_param_logs = f'{log_dir}/{log_i}_trial_index_to_param_logs'
    logfile_debug_get_next_trials = None

    nvidia_smi_logs_base = None

def log_message_to_file(logfile, message, _lvl=0, ee=None):
    assert logfile is not None, "Logfile path must be provided."
    assert message is not None, "Message to log must be provided."

    if _lvl > 3:
        original_print(f"Cannot write _debug, error: {ee}")
        return

    try:
        with open(logfile, 'a') as f:
            #original_print(f"========= {time.time()} =========", file=f)
            original_print(message, file=f)
    except FileNotFoundError:
        print_red("It seems like the run's folder was deleted during the run. Cannot continue.")
        sys.exit(99) # generalized code for run folder deleted during run
    except Exception as e:
        original_print(f"Error trying to write log file: {e}")
        log_message_to_file(logfile, message, _lvl + 1, e)

def _log_trial_index_to_param(trial_index, _lvl=0, ee=None):
    log_message_to_file(logfile_trial_index_to_param_logs, trial_index, _lvl, ee)

def _debug_worker_creation(msg, _lvl=0, ee=None):
    log_message_to_file(logfile_worker_creation_logs, msg, _lvl, ee)

def append_to_nvidia_smi_logs(_file, _host, result, _lvl=0, ee=None):
    log_message_to_file(_file, result, _lvl, ee)

def _debug_get_next_trials(msg, _lvl=0, ee=None):
    log_message_to_file(logfile_debug_get_next_trials, msg, _lvl, ee)

def _debug_progressbar(msg, _lvl=0, ee=None):
    log_message_to_file(logfile_progressbar, msg, _lvl, ee)

def add_to_phase_counter(phase, nr=0, run_folder=""):
    global current_run_folder

    if run_folder == "":
        run_folder = current_run_folder
    return append_and_read(f'{run_folder}/state_files/phase_{phase}_steps', nr)


class REMatcher(object):
    def __init__(self, matchstring):
        self.matchstring = matchstring

    def match(self,regexp):
        self.rematch = re.match(regexp, self.matchstring)
        return bool(self.rematch)

    def group(self,i):
        return self.rematch.group(i)

def dier(msg):
    pprint(msg)
    my_exit(10)

parser = argparse.ArgumentParser(
    prog="omniopt",
    description='A hyperparameter optimizer for slurmbased HPC-systems',
    epilog="Example:\n\n./omniopt --partition=alpha --experiment_name=neural_network --mem_gb=1 --time=60 --worker_timeout=60 --max_eval=500 --num_parallel_jobs=500 --gpus=0 --follow --run_program=bHMgJyUoYXNkYXNkKSc= --parameter epochs range 0 10 int --parameter epochs range 0 10 int",
    formatter_class=RichHelpFormatter
)

required = parser.add_argument_group('Required arguments', "These options have to be set")
required_but_choice = parser.add_argument_group('Required arguments that allow a choice', "Of these arguments, one has to be set to continue.")
optional = parser.add_argument_group('Optional', "These options are optional")
experimental = parser.add_argument_group('Experimental', "Experimental parameters")
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
optional.add_argument('--exclude', help=f'A comma seperated list of values of excluded nodes (taurusi8009,taurusi8010)', default=None, type=str)
optional.add_argument('--main_process_gb', help='Amount of RAM for the main process in GB (default: 1GB)', type=float, default=4)
optional.add_argument('--max_nr_of_zero_results', help='Max. nr of successive zero results by ax_client.get_next_trials() before the search space is seen as exhausted. Default is 20', type=int, default=20)
optional.add_argument('--disable_search_space_exhaustion_detection', help='Disables automatic search space reduction detection', action='store_true', default=False)
optional.add_argument('--abbreviate_job_names', help='Abbreviate pending job names (r = running, p = pending, u = unknown, c = cancelling)', action='store_true', default=False)

experimental.add_argument('--experimental', help='Do some stuff not well tested yet.', action='store_true', default=False)
experimental.add_argument('--auto_execute_suggestions', help='Automatically run again with suggested parameters (NOT FOR SLURM YET!)', action='store_true', default=False)
experimental.add_argument('--auto_execute_counter', help='(Will automatically be set)', type=int, default=0)
experimental.add_argument('--max_auto_execute', help='How many nested jobs should be done', type=int, default=3)
experimental.add_argument('--show_parameter_suggestions', help='Show suggestions for possible promising parameter space changes.', action='store_true', default=False)
experimental.add_argument('--maximizer', help='Value to expand search space for suggestions (default: 2), calculation is point [+-] maximizer * abs(point)', type=float, default=2)

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
debug.add_argument('--evaluate_to_random_value', help='Evaluate to random values', action='store_true', default=False)
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
    except Exception as e:
        return input_str

def get_file_as_string(f):
    datafile = ""
    if not os.path.exists(f):
        print_red(f"{f} not found!")
    else:
        with open(f) as f:
            datafile = f.readlines()

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
        with open(_file) as f:
            global_vars = json.load(f)
    except Exception as e:
        print("Error while loading old global_vars: " + str(e) + ", trying to load " + str(_file))
        my_exit(44)

if not args.tests:
    if args.continue_previous_job:
        load_global_vars(f"{args.continue_previous_job}/state_files/global_vars.json")

    if args.parameter is None and args.continue_previous_job is None:
        print("Either --parameter or --continue_previous_job is required. Both were not found.")
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
                print(f"{exp_name_file} not found, and no --experiment_name given. Cannot continue.")
                my_exit(19)

    if not args.mem_gb:
        print(f"--mem_gb needs to be set")
        my_exit(19)

    if not args.time:
        if not args.continue_previous_job:
            print(f"--time needs to be set")
        else:
            time_file = args.continue_previous_job + "/state_files/time"
            if os.path.exists(time_file):
                time_file_contents = get_file_as_string(time_file).strip()
                if time_file_contents.isdigit():
                    global_vars["_time"] = int(time_file_contents)
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
            print(f"--mem_gb needs to be set")
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
            gpus_file_contents = get_file_as_string(gpus_file).strip()
            if gpus_file_contents.isdigit():
                gpus = int(gpus_file_contents)
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
            print(f"--max_eval needs to be set")
        else:
            max_eval_file = args.continue_previous_job + "/state_files/max_eval"
            if os.path.exists(max_eval_file):
                max_eval_file_contents = get_file_as_string(max_eval_file).strip()
                if max_eval_file_contents.isdigit():
                    set_max_eval(int(max_eval_file_contents))
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

class signalUSR (Exception):
    pass

class signalINT (Exception):
    pass

class signalCONT (Exception):
    pass

def receive_usr_signal_one(signum, stack):
    raise signalUSR(f"USR1-signal received ({signum})")

def receive_usr_signal_int(signum, stack):
    raise signalINT(f"INT-signal received ({signum})")

def receive_signal_cont(signum, stack):
    raise signalCONT(f"CONT-signal received ({signum})")

signal.signal(signal.SIGUSR1, receive_usr_signal_one)
signal.signal(signal.SIGUSR2, receive_usr_signal_one)
signal.signal(signal.SIGINT, receive_usr_signal_int)
signal.signal(signal.SIGTERM, receive_usr_signal_int)
signal.signal(signal.SIGCONT, receive_signal_cont)


class bcolors:
    header = '\033[95m'
    blue = '\033[94m'
    cyan = '\033[96m'
    green = '\033[92m'
    warning = '\033[93m'
    red = '\033[91m'
    endc = '\033[0m'
    bold = '\033[1m'
    underline = '\033[4m'
    yellow = '\033[33m'

def print_color(color, text):
    color_codes = {
        "header": bcolors.header,
        "blue": bcolors.blue,
        "cyan": bcolors.cyan,
        "green": bcolors.green,
        "warning": bcolors.warning,
        "red": bcolors.red,
        "bold": bcolors.bold,
        "underline": bcolors.underline,
        "yellow": bcolors.yellow,
    }
    end_color = bcolors.endc

    try:
        assert color in color_codes, f"Color '{color}' is not supported."
        original_print(f"{color_codes[color]}{text}{end_color}")
    except AssertionError as e:
        print(f"Error: {e}")
        original_print(text)

def print_red(text):
    print_color("red", text)

    if current_run_folder:
        try:
            with open(f"{current_run_folder}/oo_errors.txt", "a") as myfile:
                myfile.write(text)
        except FileNotFoundError as e:
            print_red(f"Error: {e}. This may mean that the {current_run_folder} was deleted during the run.")
            sys.exit(99)

def print_green(text):
    print_color("green", text)

def print_yellow(text):
    print_color("yellow", text)

def is_executable_in_path(executable_name):
    print_debug(f"is_executable_in_path({executable_name})")
    for path in os.environ.get('PATH', '').split(':'):
        executable_path = os.path.join(path, executable_name)
        if os.path.exists(executable_path) and os.access(executable_path, os.X_OK):
            return True
    return False

system_has_sbatch = False
is_nvidia_smi_system = False

if is_executable_in_path("sbatch"):
    system_has_sbatch = True
if is_executable_in_path("nvidia-smi"):
    is_nvidia_smi_system = True

if not system_has_sbatch:
    num_parallel_jobs = 1

def save_global_vars():
    state_files_folder = f"{current_run_folder}/state_files"
    if not os.path.exists(state_files_folder):
        os.makedirs(state_files_folder)
    
    with open(f'{state_files_folder}/global_vars.json', "w") as f:
        json.dump(global_vars, f)

def check_slurm_job_id():
    print_debug("check_slurm_job_id()")
    if system_has_sbatch:
        slurm_job_id = os.environ.get('SLURM_JOB_ID')
        if slurm_job_id is not None and not slurm_job_id.isdigit():
            print_red("Not a valid SLURM_JOB_ID.")
        elif slurm_job_id is None:
            print_red("You are on a system that has SLURM available, but you are not running the main-script in a Slurm-Environment. " +
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

    file_path = os.path.join(folder, f"results.csv")

    with open(file_path, 'w') as file:
        pass

    return file_path

def sort_numerically_or_alphabetically(arr):
    print_debug("sort_numerically_or_alphabetically")
    try:
        # Check if all elements can be converted to numbers
        numbers = [float(item) for item in arr]
        # If successful, order them numerically
        sorted_arr = sorted(numbers)
    except ValueError:
        # If there's an error, order them alphabetically
        sorted_arr = sorted(arr)

    return sorted_arr

def looks_like_number (x):
    return looks_like_float(x) or looks_like_int(x)

def looks_like_float(x):
    print_debug(f"looks_like_float({x})")
    if isinstance(x, (int, float)):
        return True  # int and float types are directly considered as floats
    elif isinstance(x, str):
        try:
            float(x)  # Try converting string to float
            return True
        except ValueError:
            return False  # If conversion fails, it's not a float-like string
    return False  # If x is neither str, int, nor float, it's not float-like

def looks_like_int(x):
    print_debug(f"looks_like_int({x})")
    if isinstance(x, int):
        return True
    elif isinstance(x, float):
        return x.is_integer()
    elif isinstance(x, str):
        return bool(re.match(r'^\d+$', x))
    else:
        return False

def get_program_code_from_out_file(f):
    print_debug(f"get_program_code_from_out_file({f})")
    if not os.path.exists(f):
        print(f"{f} not found")
    else:
        fs = get_file_as_string(f)

        for line in fs.split("\n"):
            if "Program-Code:" in line:
                return line

def get_max_column_value(pd_csv, column, _default):
    """
    Reads the CSV file and returns the maximum value in the specified column.

    :param pd_csv: The path to the CSV file.
    :param column: The column name for which the maximum value is to be found.
    :return: The maximum value in the specified column.
    """
    assert_condition(isinstance(pd_csv, str), "pd_csv must be a string")
    assert_condition(isinstance(column, str), "column must be a string")

    if not os.path.exists(pd_csv):
        raise FileNotFoundError(f"CSV file {pd_csv} not found")

    try:
        df = pd.read_csv(pd_csv, float_precision='round_trip')
        if not column in df.columns:
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
    assert_condition(isinstance(pd_csv, str), "pd_csv must be a string")
    assert_condition(isinstance(column, str), "column must be a string")

    if not os.path.exists(pd_csv):
        raise FileNotFoundError(f"CSV file {pd_csv} not found")

    try:
        df = pd.read_csv(pd_csv, float_precision='round_trip')
        if not column in df.columns:
            print_red(f"Cannot load data from {pd_csv}: column {column} does not exist")
            return _default
        min_value = df[column].min()
        return min_value
    except Exception as e:
        print_red(f"Error while getting min value from column {column}: {str(e)}")
        raise

def flatten_extend(matrix):
    flat_list = []
    for row in matrix:
        flat_list.extend(row)
    return flat_list

def get_bound_if_prev_data(_type, _column, _default):
    ret_val = _default

    found_in_file = False

    if args.load_previous_job_data and len(args.load_previous_job_data):
        prev_runs = flatten_extend(args.load_previous_job_data)
        for prev_run in prev_runs:
            pd_csv = f"{prev_run}/{pd_csv_filename}"

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

    if args.continue_previous_job:
        pd_csv = f"{args.continue_previous_job}/{pd_csv_filename}"
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

    return round(ret_val, 4), found_in_file

def parse_experiment_parameters(args):
    global global_vars
    print_debug("parse_experiment_parameters")

    #if args.continue_previous_job and len(args.parameter):
    #    print_red("Cannot use --parameter when using --continue_previous_job. Parameters must stay the same.")
    #    my_exit(53)

    params = []

    param_names = []

    i = 0

    search_space_reduction_warning = False

    while args.parameter and i < len(args.parameter):
        this_args = args.parameter[i]
        j = 0
        while j < len(this_args):
            name = this_args[j]

            invalid_names = ["start_time", "end_time", "run_time", "program_string", "result", "exit_code", "signal"]

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
                print_red(f"Not enough arguments for --parameter")
                my_exit(181)


            valid_types = ["range", "fixed", "choice"]

            if param_type not in valid_types:
                valid_types_string = ', '.join(valid_types)
                print_red(f"\n⚠ Invalid type {param_type}, valid types are: {valid_types_string}")
                my_exit(181)

            if param_type == "range":
                if args.model and args.model == "FACTORIAL":
                    print_red(f"\n⚠ --model FACTORIAL cannot be used with range parameter")
                    my_exit(181)

                if len(this_args) != 5 and len(this_args) != 4:
                    print_red(f"\n⚠ --parameter for type range must have 4 (or 5, the last one being optional and float by default) parameters: <NAME> range <START> <END> (<TYPE (int or float)>)")
                    my_exit(181)

                try:
                    lower_bound = float(this_args[j + 2])
                except:
                    print_red(f"\n⚠ {this_args[j + 2]} is not a number")
                    my_exit(181)

                try:
                    upper_bound = float(this_args[j + 3])
                except:
                    print_red(f"\n⚠ {this_args[j + 3]} is not a number")
                    my_exit(181)

                if upper_bound == lower_bound:
                    if lower_bound == 0:
                        print_red(f"⚠ Lower bound and upper bound are equal: {lower_bound}, cannot automatically fix this, because they -0 = +0 (usually a quickfix would be to set lower_bound = -upper_bound)")
                        my_exit(181)
                    print_red(f"⚠ Lower bound and upper bound are equal: {lower_bound}, setting lower_bound = -upper_bound")
                    lower_bound = -upper_bound

                if lower_bound > upper_bound:
                    print_yellow(f"⚠ Lower bound ({lower_bound}) was larger than upper bound ({upper_bound}) for parameter '{name}'. Switched them.")
                    tmp = upper_bound
                    upper_bound = lower_bound
                    lower_bound = tmp

                skip = 5

                try:
                    value_type = this_args[j + 4]
                except:
                    value_type = "float"
                    skip = 4

                valid_value_types = ["int", "float"]

                if value_type not in valid_value_types:
                    valid_value_types_string = ", ".join(valid_value_types)
                    print_red(f"⚠ {value_type} is not a valid value type. Valid types for range are: {valid_value_types_string}")
                    my_exit(181)

                if value_type == "int":
                    if not looks_like_int(lower_bound):
                        print_yellow(f"⚠ {value_type} can only contain integers. You chose {lower_bound}. Will be rounded down to {math.floor(lower_bound)}.")
                        lower_bound = math.floor(lower_bound)

                    if not looks_like_int(upper_bound):
                        print_yellow(f"⚠ {value_type} can only contain integers. You chose {upper_bound}. Will be rounded up to {math.ceil(upper_bound)}.")
                        upper_bound = math.ceil(upper_bound)

                old_lower_bound = lower_bound
                old_upper_bound = upper_bound

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
                
                if args.gridsearch:
                    global changed_grid_search_params

                    values = np.linspace(lower_bound, upper_bound, args.max_eval, endpoint=True).tolist()

                    if value_type == "int":
                        values = [int(value) for value in values]
                        changed_grid_search_params[name] = f"Gridsearch from {to_int_when_possible(lower_bound)} to {to_int_when_possible(upper_bound)} ({args.max_eval} steps, int)"
                    else:
                        changed_grid_search_params[name] = f"Gridsearch from {to_int_when_possible(lower_bound)} to {to_int_when_possible(upper_bound)} ({args.max_eval} steps)"

                    values = sorted(set(values))
                    values = [str(to_int_when_possible(value)) for value in values]

                    param = {
                        "name": name,
                        "type": "choice",
                        "is_ordered": True,
                        "values": values
                    }
                else:
                    param = {
                        "name": name,
                        "type": param_type,
                        "bounds": [lower_bound, upper_bound],
                        "value_type": value_type
                    }

                global_vars["parameter_names"].append(name)

                params.append(param)

                j += skip
            elif param_type == "fixed":
                if len(this_args) != 3:
                    print_red(f"⚠ --parameter for type fixed must have 3 parameters: <NAME> range <VALUE>")
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
            elif param_type == "choice":
                if len(this_args) != 3:
                    print_red(f"⚠ --parameter for type choice must have 3 parameters: <NAME> choice <VALUE,VALUE,VALUE,...>")
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
            else:
                print_red(f"⚠ Parameter type '{param_type}' not yet implemented.")
                my_exit(181)
        i += 1

    if search_space_reduction_warning:
        print_red(f"⚠ Search space reduction is not currently supported on continued runs or runs that have previous data.")

    return params

def replace_parameters_in_string(parameters, input_string):
    print_debug("replace_parameters_in_string")
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
    print_debug(f"execute_bash_code({code})")
    try:
        result = subprocess.run(code, shell=True, check=True, text=True, capture_output=True)

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
    print_debug("get_result(input_string)")
    if input_string is None:
        print("Input-String is None")
        return None

    if not isinstance(input_string, str):
        print(f"Type of input_string is not string, but {type(input_string)}")
        return None

    try:
        pattern = r'\s*RESULT:\s*(-?\d+(?:\.\d+)?)'

        match = re.search(pattern, input_string)

        if match:
            result_number = float(match.group(1))
            return result_number
        else:
            return None

    except Exception as e:
        print(f"Error extracting the RESULT-string: {e}")
        return None

def add_to_csv(file_path, heading, data_line):
    print_debug(f"add_to_csv({file_path}, {heading}, {data_line})")
    is_empty = os.path.getsize(file_path) == 0 if os.path.exists(file_path) else True

    data_line = [to_int_when_possible(x) for x in data_line]

    with open(file_path, 'a+', newline='') as file:
        csv_writer = csv.writer(file)

        if is_empty:
            csv_writer.writerow(heading)

        # desc += " (best loss: " + '{:f}'.format(best_result) + ")"
        data_line = ["{:.20f}".format(x) if type(x) == float else x for x in data_line]
        csv_writer.writerow(data_line)

def make_strings_equal_length(str1, str2):
    print_debug("make_strings_equal_length")
    length_difference = len(str1) - len(str2)

    if length_difference > 0:
        str2 = str2 + ' ' * length_difference
    elif length_difference < 0:
        str2 = str2[:len(str1)]

    return str1, str2

def find_file_paths(_text):
    print_debug("find_file_paths(_text)")
    file_paths = []

    if type(_text) == str:
        words = _text.split()

        for word in words:
            if os.path.exists(word):
                file_paths.append(word)

        return file_paths

    return []

def check_file_info(file_path):
    print_debug("check_file_info")
    if not os.path.exists(file_path):
        print(f"The file {file_path} does not exist.")
        return

    if not os.access(file_path, os.R_OK):
        print(f"The file {file_path} is not readable.")

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
    string += f"Size: {size} Bytes\n"
    string += f"Permissions: {permissions}\n"
    string += f"Owner: {username}\n"
    string += f"Last access: {access_time}\n"
    string += f"Last modification: {modification_time}\n"

    string += f"Hostname: {socket.gethostname()}"

    return string

def find_file_paths_and_print_infos(_text, program_code):
    print_debug("find_file_paths_and_print_infos(_text, {program_code}")
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

def write_data_and_headers(data_dict, error_description=""):
    assert isinstance(data_dict, dict), "The parameter must be a dictionary."
    assert isinstance(error_description, str), "The error_description must be a string."

    headers = list(data_dict.keys())
    data = [list(data_dict.values())]

    if error_description:
        headers.append('error_description')
        for row in data:
            row.append(error_description)

    global current_run_folder

    failed_logs_dir = os.path.join(current_run_folder, 'failed_logs')

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
                with open(header_file_path, mode='w', newline='') as header_file:
                    writer = csv.writer(header_file)
                    writer.writerow(headers)
                    print_debug(f"Header file created with headers: {headers}")
            except Exception as e:
                print_red(f"Failed to write header file: {e}")
        else:
            print_debug("Header file already exists, skipping header writing.")

        # Append data to the data file
        try:
            with open(data_file_path, mode='a', newline='') as data_file:
                writer = csv.writer(data_file)
                writer.writerows(data)
                print_debug(f"Data appended to file: {data_file_path}")
        except Exception as e:
            print_red(f"Failed to append data to file: {e}")

    except Exception as e:
        print_red(f"Unexpected error: {e}")


def evaluate(parameters):
    global global_vars
    global is_in_evaluate

    nvidia_smi_thread = start_nvidia_smi_thread()

    return_in_case_of_error = {"result": val_if_nothing_found}

    if args.maximize:
        return_in_case_of_error = {"result": -val_if_nothing_found}

    if system_has_sbatch and args.gpus >= 1 and args.auto_exclude_defective_hosts:
        try:
            for i in range(torch.cuda.device_count()):
                tmp = torch.cuda.get_device_properties(i).name
        except RuntimeError:
            print(f"Node {socket.gethostname()} was detected as faulty. It should have had a GPU, but there is an error initializing the CUDA driver. Adding this node to the --exclude list.")
            count_defective_nodes(None, socket.gethostname())
            return return_in_case_of_error
        except:
            pass

    is_in_evaluate = True
    if args.evaluate_to_random_value:
        rand_res = random.uniform(0, 1)
        is_in_evaluate = False
        return {"result": float(rand_res)}

    parameters = {k: (int(v) if isinstance(v, (int, float, str)) and re.fullmatch(r'^\d+(\.0+)?$', str(v)) else v) for k, v in parameters.items()}

    print_debug(f"evaluate with parameters {parameters}")
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    signal.signal(signal.SIGUSR2, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGQUIT, signal.SIG_IGN)

    try:
        original_print("parameters:", parameters)

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
        stderr = stdout_stderr_exit_code_signal[1]
        exit_code = stdout_stderr_exit_code_signal[2]
        _signal = stdout_stderr_exit_code_signal[3]

        run_time = end_time - start_time

        original_print("stdout:")
        original_print(stdout)

        result = get_result(stdout)

        original_print(f"Result: {result}")

        headline = ["start_time", "end_time", "run_time", "program_string", *parameters_keys, "result", "exit_code", "signal", "hostname"]
        values = [start_time, end_time, run_time, program_string_with_params,  *parameters_values, result, exit_code, _signal, socket.gethostname()]

        original_print(f"EXIT_CODE: {exit_code}")

        headline = ['None' if element is None else element for element in headline]
        values = ['None' if element is None else element for element in values]

        add_to_csv(f"{current_run_folder}/job_infos.csv", headline, values)

        if type(result) == int:
            is_in_evaluate = False
            return {"result": int(result)}
        elif type(result) == float:
            is_in_evaluate = False
            return {"result": float(result)}
        else:
            is_in_evaluate = False
            write_data_and_headers(parameters, "No Result")
            return return_in_case_of_error
    except signalUSR:
        print("\n⚠ USR1-Signal was sent. Cancelling evaluation.")
        is_in_evaluate = False
        write_data_and_headers(parameters, "USR1-signal")
        return return_in_case_of_error
    except signalCONT:
        print("\n⚠ CONT-Signal was sent. Cancelling evaluation.")
        is_in_evaluate = False
        write_data_and_headers(parameters, "CONT-signal")
        return return_in_case_of_error
    except signalINT:
        print("\n⚠ INT-Signal was sent. Cancelling evaluation.")
        write_data_and_headers(parameters, "INT-signal")
        is_in_evaluate = False
        return return_in_case_of_error

try:
    if not args.tests:
        with console.status("[bold green]Loading torch...") as status:
            import torch
        with console.status("[bold green]Loading numpy...") as status:
            import numpy as np
        with console.status("[bold green]Loading ax...") as status:
            try:
                import ax.modelbridge.generation_node
                import ax
                from ax.service.ax_client import AxClient, ObjectiveProperties
                import ax.exceptions.core
                from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
                from ax.modelbridge.registry import ModelRegistryBase, Models
                from ax.storage.json_store.save import save_experiment
                from ax.storage.json_store.load import load_experiment
                from ax.service.utils.report_utils import exp_to_df
            except ModuleNotFoundError as e:
                print_red("\n⚠ ax could not be loaded. Did you create and load the virtual environment properly?")
                my_exit(31)
            except KeyboardInterrupt:
                print_red("\n⚠ You pressed CTRL+C. Program execution halted.")
                my_exit(31)

        with console.status("[bold green]Loading botorch...") as status:
            try:
                import botorch
            except ModuleNotFoundError as e:
                print_red("\n⚠ ax could not be loaded. Did you create and load the virtual environment properly?")
                my_exit(31)
            except KeyboardInterrupt:
                print_red("\n⚠ You pressed CTRL+C. Program execution halted.")
                my_exit(31)

        with console.status("[bold green]Loading submitit...") as status:
            try:
                import submitit
                from submitit import AutoExecutor, LocalJob, DebugJob
            except:
                print_red("\n⚠ submitit could not be loaded. Did you create and load the virtual environment properly?")
                my_exit(31)
except ModuleNotFoundError as e:
    original_print(f"Base modules could not be loaded: {e}")
    my_exit(31)
except KeyboardInterrupt:
    original_print("You cancelled loading the basic modules")
    my_exit(31)
except signalINT:
    print("\n⚠ Signal INT was detected. Exiting with 128 + 2.")
    my_exit(128 + 2)
except (signalUSR):
    print("\n⚠ Signal USR was detected. Exiting with 128 + 10.")
    my_exit(128 + 10)
except signalCONT:
    print("\n⚠ Signal CONT was detected. Exiting with 128 + 18.")
    my_exit(128 + 18)
except (KeyboardInterrupt) as e:
    print("\n⚠ You pressed CTRL+C. Program execution halted.")
    my_exit(0)

def disable_logging():
    print_debug("disable_logging()")
    logging.basicConfig(level=logging.ERROR)

    logging.getLogger("ax").setLevel(logging.ERROR)
    logging.getLogger("ax.modelbridge").setLevel(logging.ERROR)
    logging.getLogger("ax.modelbridge.torch").setLevel(logging.ERROR)
    logging.getLogger("ax.models.torch.botorch_modular.acquisition").setLevel(logging.ERROR)
    logging.getLogger("ax.modelbridge.transforms.standardize_y").setLevel(logging.ERROR)
    logging.getLogger("ax.modelbridge.torch").setLevel(logging.ERROR)
    logging.getLogger("ax.service.utils.instantiation").setLevel(logging.ERROR)
    logging.getLogger("ax.modelbridge.dispatch_utils").setLevel(logging.ERROR)
    logging.getLogger("ax.modelbridge.cross_validation").setLevel(logging.ERROR)
    logging.getLogger("ax.core.experiment").setLevel(logging.ERROR)
    logging.getLogger("ax.modelbridge.base").setLevel(logging.ERROR)

    logging.getLogger().setLevel(logging.ERROR)
    logging.getLogger("ax").setLevel(logging.ERROR)
    logging.getLogger("ax.service").setLevel(logging.ERROR)
    logging.getLogger("ax.service.utils").setLevel(logging.ERROR)
    logging.getLogger("ax.service.utils.report_utils").setLevel(logging.ERROR)

    categories = [FutureWarning, RuntimeWarning, UserWarning, Warning]
    modules = [
        "ax",
        "ax.core.data",
        "ax.service.utils.report_utils",
        "ax.modelbridge"
        "ax.modelbridge.transforms.standardize_y",
        "botorch.models.utils.assorted",
        "ax.modelbridge.torch",
        "ax.models.torch.botorch_modular.acquisition",
        "ax.modelbridge.cross_validation",
        "ax.service.utils.best_point",
        "torch.autograd",
        "torch.autograd.__init__",
        "botorch.optim.fit",
        "ax.core.parameter",
        "ax.service.utils.report_utils",
        "ax.modelbridge.transforms.int_to_float",
        "ax.modelbridge.cross_validation",
        "ax.service.utils.best_point",
        "ax.modelbridge.transforms.int_to_float",
        "ax.modelbridge.dispatch_utils",
        "ax.service.utils.instantiation",
        "botorch.optim.optimize",
        "linear_operator.utils.cholesky"
    ]

    for _module in modules:
        for _cat in categories:
            warnings.filterwarnings("ignore", category=_cat)
            warnings.filterwarnings("ignore", category=_cat, module=_module)


    print_debug("disable_logging() done")

def display_failed_jobs_table():
    global current_run_folder
    console = Console()
    
    failed_jobs_folder = f"{current_run_folder}/failed_logs"
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
        with open(header_file, mode='r') as file:
            reader = csv.reader(file)
            headers = next(reader)
            print_debug(f"Headers: {headers}")
        
        with open(parameters_file, mode='r') as file:
            reader = csv.reader(file)
            parameters = [row for row in reader]
            print_debug(f"Parameters: {parameters}")
            
        # Create the table
        table = Table(show_header=True, header_style="bold red", title="Failed Jobs parameters:")
        
        for header in headers:
            table.add_column(header)
        
        added_rows = set()

        for parameter_set in parameters:
            row = [str(to_int_when_possible(value)) for value in parameter_set]
            row_tuple = tuple(row)  # Convert to tuple for set operations
            if row_tuple not in added_rows:
                table.add_row(*row, style='red')
                added_rows.add(row_tuple)
        
        # Print the table to the console
        console.print(table)
    except Exception as e:
        print_red(f"Error: {str(e)}")

def plot_command(_command, tmp_file, _width=1300):
    show_sixel_graphics = args.show_sixel_scatter or args.show_sixel_general or args.show_sixel_scatter
    if not show_sixel_graphics:
        return

    if os.getenv("DISABLE_SIXEL_GRAPHICS"):
        return

    print_debug(f"command: {_command}")

    my_env = os.environ.copy()
    my_env["DONT_INSTALL_MODULES"] = "1"
    my_env["DONT_SHOW_DONT_INSTALL_MESSAGE"] = "1"

    process = subprocess.Popen(_command.split(), stdout=subprocess.PIPE, env=my_env)
    output, error = process.communicate()

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

def print_best_result(csv_file_path, result_column):
    global current_run_folder

    try:
        best_params = get_best_params(csv_file_path, result_column)

        best_result = None

        if best_params and "result" in best_params:
            best_result = best_params["result"]
        else:
            best_result = NO_RESULT

        if str(best_result) == NO_RESULT or best_result is None or best_result == "None":
            print_red("Best result could not be determined")
            return 87
        else:
            total_str = f"total: {count_done_jobs() - nr_inserted_jobs}"

            if nr_inserted_jobs:
                total_str += f" + nr_inserted_jobs: {nr_inserted_jobs}"

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

            row_without_result = [str(to_int_when_possible(best_params["parameters"][key])) for key in best_params["parameters"].keys()]
            row = [*row_without_result, str(best_result)][3:]

            table.add_row(*row)

            console.print(table)

            with console.capture() as capture:
                console.print(table)
            table_str = capture.get()

            with open(f'{current_run_folder}/best_result.txt', "w") as text_file:
                text_file.write(table_str)

            _pd_csv = f"{current_run_folder}/{pd_csv_filename}"

            global global_vars

            x_y_combinations = list(combinations(global_vars["parameter_names"], 2))

            show_sixel_graphics = args.show_sixel_scatter or args.show_sixel_general or args.show_sixel_scatter or args.show_sixel_trial_index_result

            if os.path.exists(_pd_csv) and show_sixel_graphics and not os.getenv("DISABLE_SIXEL_GRAPHICS"):
                if args.show_sixel_trial_index_result:
                    plot_types = [
                        {
                            "type": "trial_index_result",
                            "min_done_jobs": 2
                        }
                    ]

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

                for plot in plot_types:
                    plot_type = plot["type"]
                    min_done_jobs = 1

                    if "min_done_jobs" in plot:
                        min_done_jobs = plot["min_done_jobs"]

                    if count_done_jobs() < min_done_jobs:
                        print_debug(f"Cannot plot {plot_type}, because it needs {min_done_jobs}, but you only have {count_done_jobs()} jobs done")
                        continue

                    try:
                        _tmp = f"{current_run_folder}/plots/"
                        _width = 1200

                        if "width" in plot:
                            _width = plot["width"]

                        if not os.path.exists(_tmp):
                            os.makedirs(_tmp)

                        j = 0
                        tmp_file = f"{_tmp}/{plot_type}.png"
                        _fn = ""
                        _p = []
                        if "filename" in plot:
                            _fn = plot['filename']

                        while os.path.exists(tmp_file):
                            j += 1
                            tmp_file = f"{_tmp}/{plot_type}_{j}.png"

                        maindir = os.path.dirname(os.path.realpath(__file__))

                        _command = f"bash {maindir}/omniopt_plot --run_dir {current_run_folder} --plot_type={plot_type}"
                        if "dpi" in plot:
                            _command += " --dpi=" + str(plot["dpi"])

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
                                        _p = []
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
                                        #original_print(f"iterated_command: >>{_iterated_command}<<")
                                        plot_command(_iterated_command, tmp_file, _width)
                        else:
                            _command += f" --save_to_file={tmp_file} "
                            plot_command(_command, tmp_file, _width)
                    except Exception as e:
                        tb = traceback.format_exc()
                        print_red(f"Error trying to print {plot_type} to to CLI: {e}, {tb}")
                        print_debug(f"Error trying to print {plot_type} to to CLI: {e}")
            else:
                print_debug(f"{_pd_csv} not found")

        shown_end_table = True
    except Exception as e:
        tb = traceback.format_exc()
        print_red(f"[print_best_result] Error during print_best_result: {e}, tb: {tb}")

    return None

def show_end_table_and_save_end_files(csv_file_path, result_column):
    print_debug(f"show_end_table_and_save_end_files({csv_file_path}, {result_column})")

    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    signal.signal(signal.SIGUSR2, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGQUIT, signal.SIG_IGN)

    disable_logging()

    global ax_client
    global console
    global shown_end_table
    global args
    global already_shown_worker_usage_over_time
    global global_vars

    if shown_end_table:
        print("End table already shown, not doing it again")
        return

    _exit = 0

    display_failed_jobs_table()

    best_result_exit = print_best_result(csv_file_path, result_column)

    if best_result_exit:
        _exit = best_result_exit

    if args.show_worker_percentage_table_at_end and len(worker_percentage_usage) and not already_shown_worker_usage_over_time:
        already_shown_worker_usage_over_time = True

        table = Table(header_style="bold", title="Worker usage over time:")
        columns = ["Time", "Nr. workers", "Max. nr. workers", "%"]
        for column in columns:
            table.add_column(column)
        for row in worker_percentage_usage:
            table.add_row(str(row["time"]), str(row["nr_current_workers"]), str(row["num_parallel_jobs"]), f'{row["percentage"]}%', style='bright_green')
        console.print(table)

    return _exit

def write_worker_usage():
    global current_run_folder

    if len(worker_percentage_usage):
        csv_filename = f'{current_run_folder}/worker_usage.csv'

        csv_columns = ['time', 'num_parallel_jobs', 'nr_current_workers', 'percentage']

        with open(csv_filename, 'w', newline='') as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            for row in worker_percentage_usage:
                csv_writer.writerow(row)
    else:
        if is_slurm_job():
            print_debug(f"worker_percentage_usage seems to be empty. Not writing worker_usage.csv")

def end_program(csv_file_path, result_column="result", _force=False, exit_code=None):
    global global_vars
    global ax_client
    global console
    global end_program_ran

    if os.getpid() != main_pid:
        print_debug("returning from end_program, because it can only run in the main thread, not any forks")
        return

    if is_in_evaluate and not force:
        print_debug("is_in_evaluate true, returning end_program")
        return

    if end_program_ran and not force:
        print_debug("[end_program] end_program_ran was true. Returning.")
        return

    end_program_ran = True

    _exit = 0

    save_logs_to_json(f'{current_run_folder}/function_logs.json')

    try:
        if current_run_folder is None:
            print_debug("[end_program] current_run_folder was empty. Not running end-algorithm.")
            return

        if ax_client is None:
            print_debug("[end_program] ax_client was empty. Not running end-algorithm.")
            return

        if console is None:
            print_debug("[end_program] console was empty. Not running end-algorithm.")
            return

        _exit = show_end_table_and_save_end_files(csv_file_path, result_column)
    except (signalUSR, signalINT, signalCONT, KeyboardInterrupt) as e:
        print_red("\n⚠ You pressed CTRL+C or a signal was sent. Program execution halted.")
        print("\n⚠ KeyboardInterrupt signal was sent. Ending program will still run.")
        _exit = show_end_table_and_save_end_files(csv_file_path, result_column)
    except TypeError as e:
        print_red(f"\n⚠ The program has been halted without attaining any results. Error: {e}")

    for job, trial_index in global_vars["jobs"][:]:
        if job:
            try:
                _trial = ax_client.get_trial(trial_index)
                _trial.mark_abandoned()
                global_vars["jobs"].remove((job, trial_index))
            except Exception as e:
                print(f"ERROR in line {getLineInfo()}: {e}")
            job.cancel()

    save_pd_csv()

    pd_csv = f'{current_run_folder}/{pd_csv_filename}'
    try:
        find_promising_bubbles(pd_csv)
    except Exception as e:
        print_debug("Error trying to find promising bubbles: " + str(e))

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
        print_debug(f"save_checkpoint(trial_nr: {trial_nr}, ee: {ee})")
        global ax_client

        state_files_folder = f"{current_run_folder}/state_files/"

        if not os.path.exists(state_files_folder):
            os.makedirs(state_files_folder)

        checkpoint_filepath = f'{state_files_folder}/checkpoint.json'
        ax_client.save_to_json_file(filepath=checkpoint_filepath)

        print_debug("Checkpoint saved")
    except Exception as e:
        save_checkpoint(trial_nr + 1, e)

def to_int_when_possible(val):
    # Überprüfung, ob der Wert ein Integer ist oder ein Float, der eine ganze Zahl sein könnte
    if type(val) == int or (type(val) == float and val.is_integer()) or (type(val) == str and val.isdigit()):
        return int(val)

    # Überprüfung auf nicht-numerische Zeichenketten
    if type(val) == str and re.match(r'^-?\d+(?:\.\d+)?$', val) is None:
        return val

    try:
        # Versuche, den Wert als Float zu interpretieren
        val = float(val)
        # Bestimmen der Anzahl der Dezimalstellen, um die Genauigkeit der Ausgabe zu steuern
        if '.' in str(val):
            decimal_places = len(str(val).split('.')[1])
            # Formatieren des Floats mit der exakten Anzahl der Dezimalstellen, ohne wissenschaftliche Notation
            formatted_value = format(val, f'.{decimal_places}f').rstrip('0').rstrip('.')
            return formatted_value if formatted_value else '0'
        else:
            return str(int(val))
    except:
        # Falls ein Fehler auftritt, gebe den ursprünglichen Wert zurück
        return val

def save_pd_csv():
    global ax_client
    #print_debug("save_pd_csv()")

    pd_csv = f'{current_run_folder}/{pd_csv_filename}'
    pd_json = f'{current_run_folder}/state_files/pd.json'

    state_files_folder = f"{current_run_folder}/state_files/"

    if not os.path.exists(state_files_folder):
        os.makedirs(state_files_folder)

    if ax_client is None:
        return pd_csv

    try:
        pd_frame = ax_client.get_trials_data_frame()

        pd_frame.to_csv(pd_csv, index=False, float_format="%.30f")
        #pd_frame.to_json(pd_json)

        json_snapshot = ax_client.to_json_snapshot()

        with open(pd_json, 'w') as json_file:
            json.dump(json_snapshot, json_file, indent=4)

        save_experiment(ax_client.experiment, f"{current_run_folder}/state_files/ax_client.experiment.json")

        #print_debug("pd.{csv,json} saved")
    except signalUSR as e:
        raise signalUSR(str(e))
    except signalCONT as e:
        raise signalCONT(str(e))
    except signalINT as e:
        raise signalINT(str(e))
    except Exception as e:
        print_red(f"While saving all trials as a pandas-dataframe-csv, an error occured: {e}")

    return pd_csv

def get_tmp_file_from_json(experiment_args):
    k = 0
    p = "/tmp/0"
    while os.path.exists(f"/tmp/{k}"):
        k = k + 1

    try:
        with open(f'/tmp/{k}', "w") as f:
            json.dump(experiment_args, f)
    except PermissionError as e:
        print_red(f"Error writing '{k}' in get_tmp_file_from_json: {e}")

    return f"/tmp/{k}"

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
        else:
            return "No differences found between the old and new parameters."

    except AssertionError as e:
        print(f"Assertion error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

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
    elif data["type"] == "choice":
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

    else:
        print("data:")
        pprint(data)
        dier(f"Unknown data range {data['type']}")

def get_experiment_parameters(ax_client, continue_previous_job, seed, experiment_constraints, parameter, cli_params_experiment_parameters, experiment_parameters, minimize_or_maximize):
    experiment_args = None
    if continue_previous_job:
        print_debug(f"Load from checkpoint: {continue_previous_job}")

        checkpoint_file = continue_previous_job + "/state_files/checkpoint.json"
        checkpoint_parameters_filepath = continue_previous_job + "/state_files/checkpoint.json.parameters.json"

        if not os.path.exists(checkpoint_file):
            print_red(f"{checkpoint_file} not found")
            my_exit(47)

        try:
            f = open(checkpoint_file)
            experiment_parameters = json.load(f)
            f.close()

            with open(checkpoint_file) as f:
                experiment_parameters = json.load(f)

            cuda_is_available = torch.cuda.is_available()

            if not cuda_is_available or cuda_is_available == 0:
                print_yellow("No suitable CUDA devices found")
            else:
                if torch.cuda.device_count() >= 1:
                    torch_device = torch.cuda.current_device()
                    if "choose_generation_strategy_kwargs" not in experiment_parameters:
                        experiment_parameters["choose_generation_strategy_kwargs"] = {}
                    experiment_parameters["choose_generation_strategy_kwargs"]["torch_device"] = torch_device
                    print_yellow(f"Using CUDA device {torch.cuda.get_device_name(0)}")
                else:
                    if args.verbose:
                        print_yellow("No suitable CUDA devices found")
        except json.decoder.JSONDecodeError as e:
            print_red(f"Error parsing checkpoint_file {checkpoint_file}")
            my_exit(47)

        if not os.path.exists(checkpoint_parameters_filepath):
            print_red(f"Cannot find {checkpoint_parameters_filepath}")
            my_exit(47)

        done_jobs_file = f"{continue_previous_job}/state_files/submitted_jobs"
        done_jobs_file_dest = f'{current_run_folder}/state_files/submitted_jobs'
        if not os.path.exists(done_jobs_file):
            print_red(f"Cannot find {done_jobs_file}")
            my_exit(47)

        if not os.path.exists(done_jobs_file_dest):
            shutil.copy(done_jobs_file, done_jobs_file_dest)

        submitted_jobs_file = f"{continue_previous_job}/state_files/submitted_jobs"
        submitted_jobs_file_dest = f'{current_run_folder}/state_files/submitted_jobs'
        if not os.path.exists(submitted_jobs_file):
            print_red(f"Cannot find {submitted_jobs_file}")
            my_exit(47)

        if not os.path.exists(submitted_jobs_file_dest):
            shutil.copy(submitted_jobs_file, submitted_jobs_file_dest)

        if parameter:
            for _item in cli_params_experiment_parameters:
                _replaced = False
                for _item_id_to_overwrite in range(0, len(experiment_parameters["experiment"]["search_space"]["parameters"])):
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

        original_ax_client_file = f"{current_run_folder}/state_files/original_ax_client_before_loading_tmp_one.json"
        ax_client.save_to_json_file(filepath=original_ax_client_file)

        with open(original_ax_client_file) as f:
            loaded_original_ax_client_json = json.load(f)
            original_generation_strategy = loaded_original_ax_client_json["generation_strategy"]

            if original_generation_strategy:
                experiment_parameters["generation_strategy"] = original_generation_strategy

        tmp_file_path = get_tmp_file_from_json(experiment_parameters)

        ax_client = (AxClient.load_from_json_file(tmp_file_path))
        #dier(human_readable_generation_strategy(ax_client))
        #dier(tmp_file_path)

        os.unlink(tmp_file_path)

        state_files_folder = f"{current_run_folder}/state_files"

        checkpoint_filepath = f'{state_files_folder}/checkpoint.json'
        if not os.path.exists(state_files_folder):
            os.makedirs(state_files_folder)
        with open(checkpoint_filepath, "w") as outfile:
            json.dump(experiment_parameters, outfile)

        if not os.path.exists(checkpoint_filepath):
            print_red(f"{checkpoint_filepath} not found. Cannot continue_previous_job without.")
            my_exit(47)

        with open(f'{current_run_folder}/checkpoint_load_source', 'w') as f:
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

        try:
            cuda_is_available = torch.cuda.is_available()

            if not cuda_is_available or cuda_is_available == 0:
                print_yellow("No suitable CUDA devices found")
            else:
                if torch.cuda.device_count() >= 1:
                    torch_device = torch.cuda.current_device()
                    experiment_args["choose_generation_strategy_kwargs"]["torch_device"] = torch_device
                    print_yellow(f"Using CUDA device {torch.cuda.get_device_name(0)}")
                else:
                    if args.verbose:
                        print_yellow("No suitable CUDA devices found")
        except ModuleNotFoundError:
            print_red("Cannot load torch and thus, cannot use gpus")

        if experiment_constraints and len(experiment_constraints):
            experiment_args["parameter_constraints"] = []
            for l in range(0, len(experiment_constraints)):
                constraints_string = " ".join(experiment_constraints[l])

                variables = [item['name'] for item in experiment_parameters]

                equation = check_equation(variables, constraints_string)

                if equation:
                    experiment_args["parameter_constraints"].append(constraints_string)
                else:
                    print_red(f"Experiment constraint '{constraints_string}' is invalid and will be ignored.")

        try:
            experiment = ax_client.create_experiment(**experiment_args)
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

def print_overview_tables(experiment_parameters, experiment_args):
    if not experiment_parameters:
        print_red("Cannot determine experiment_parameters. No parameter table will be shown.")
        return

    print_debug("print_overview_tables(experiment_parameters, experiment_args)")
    global args

    if not experiment_parameters:
        print_red("Experiment parameters could not be determined for display")

    min_or_max = "minimize"
    if args.maximize:
        min_or_max = "maximize"

    with open(f"{current_run_folder}/state_files/{min_or_max}", 'w') as f:
        print('The contents of this file do not matter. It is only relevant that it exists.', file=f)

    rows = []

    if "_type" in experiment_parameters:
        experiment_parameters = experiment_parameters["experiment"]["search_space"]["parameters"]

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

            rows.append([str(param["name"]), get_type_short(_type), str(to_int_when_possible(_lower)), str(to_int_when_possible(_upper)), "", value_type])
        elif "fixed" in _type.lower():
            rows.append([str(param["name"]), get_type_short(_type), "", "", str(to_int_when_possible(param["value"])), ""])
        elif "choice" in _type.lower():
            values = param["values"]
            values = [str(to_int_when_possible(item)) for item in values]

            rows.append([str(param["name"]), get_type_short(_type), "", "", ", ".join(values), ""])
        else:
            print_red(f"Type {_type} is not yet implemented in the overview table.")
            my_exit(15)

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

    with open(f"{current_run_folder}/parameters.txt", "w") as text_file:
        text_file.write(table_str)

    if not experiment_args is None and "parameter_constraints" in experiment_args and len(experiment_args["parameter_constraints"]):
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

        with open(f"{current_run_folder}/constraints.txt", "w") as text_file:
            text_file.write(table_str)

def check_equation(variables, equation):
    print_debug(f"check_equation({variables}, {equation})")
    if not (">=" in equation or "<=" in equation):
        print_debug(f"check_equation({variables}, {equation}): if not ('>=' in equation or '<=' in equation)")
        return False

    comparer_at_beginning = re.search("^\\s*((<=|>=)|(<=|>=))", equation)
    if comparer_at_beginning:
        print(f"The restraints {equation} contained comparision operator like <=, >= at at the beginning. This is not a valid equation.")
        return False

    comparer_at_end = re.search("((<=|>=)|(<=|>=))\\s*$", equation)
    if comparer_at_end:
        print(f"The restraints {equation} contained comparision operator like <=, >= at at the end. This is not a valid equation.")
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
    else:
        return False

    return equation

def update_progress_bar (progress_bar, nr):
    #import traceback
    #print(f"update_progress_bar(progress_bar, {nr})")
    #traceback.print_stack()

    global sum_of_values_for_tqdm

    sum_of_values_for_tqdm += nr

    if sum_of_values_for_tqdm > max_eval:
        #print(f"Reaching upper limit for tqdm: sum_of_values_for_tqdm {sum_of_values_for_tqdm} > max_eval {max_eval}")
        return

    progress_bar.update(nr)

def progressbar_description(new_msgs=[]):
    global result_csv_file
    global random_steps
    global searching_for
    global progress_bar

    desc = get_desc_progress_text(new_msgs)
    print_debug_progressbar(desc)
    progress_bar.set_description(desc)
    progress_bar.refresh()

def clean_completed_jobs():
    global jobs
    for job, trial_index in global_vars["jobs"][:]:
        if state_from_job(job) in ["completed", "early_stopped", "abandoned"]:
            global_vars["jobs"].remove((job, trial_index))

def dataframe_dier():
    pd_frame = ax_client.get_trials_data_frame()
    dier(pd_frame)

def assert_condition(condition, error_text):
    if not condition:
        raise AssertionError(error_text)

def get_old_result_by_params(file_path, params, float_tolerance=1e-6):
    """
    Open the CSV file and find the row where the subset of columns matching the keys in params have the same values.
    Return the value of the 'result' column from that row.

    :param file_path: The path to the CSV file.
    :param params: A dictionary of parameters with column names as keys and values to match.
    :param float_tolerance: The tolerance for comparing float values.
    :return: The value of the 'result' column from the matched row.
    """
    assert_condition(isinstance(file_path, str), "file_path must be a string")
    assert_condition(isinstance(params, dict), "params must be a dictionary")
    
    if not os.path.exists(file_path):
        print_red(f"{file_path} for getting old CSV results cannot be found")
        return None
    
    try:
        df = pd.read_csv(file_path, float_precision='round_trip')
    except Exception as e:
        raise RuntimeError(f"Failed to read the CSV file: {str(e)}")
    
    if not 'result' in df.columns:
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
        else:
            print_debug("Matching rows found")
            print_debug(matching_rows)
            return matching_rows
        
        result_value = matching_rows['result'].values[0]
        return result_value
    except AssertionError as ae:
        print_red(f"Assertion error: {str(ae)}")
        raise
    except Exception as e:
        print_red(f"Error during filtering or extracting result: {str(e)}")
        raise

def load_existing_job_data_into_ax_client(args):
    global nr_inserted_jobs

    if args.load_previous_job_data:
        for this_path in args.load_previous_job_data:
            load_data_from_existing_run_folders(args, this_path)

    #if args.continue_previous_job:
    #    load_data_from_existing_run_folders(args, [args.continue_previous_job])

    if len(already_inserted_param_hashes.keys()):
        if len(missing_results):
            print(f"Missing results: {len(missing_results)}")
            #nr_inserted_jobs += len(double_hashes)

        if len(double_hashes):
            print(f"Double parameters not inserted: {len(double_hashes)}")
            #nr_inserted_jobs += len(double_hashes)

        if len(double_hashes) - len(already_inserted_param_hashes.keys()):
            print(f"Restored trials: {len(already_inserted_param_hashes.keys())}")
            nr_inserted_jobs += len(already_inserted_param_hashes.keys())
    else:
        nr_of_imported_jobs = get_nr_of_imported_jobs(args)
        nr_inserted_jobs += nr_of_imported_jobs

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
    except AssertionError as e:
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


def simulate_load_data_from_existing_run_folders(args, _paths):
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

            if not ("COMPLETED".lower() in str(trial_status_str).lower()): # or "MANUAL".lower() in str(trial_status_str).lower()):
                continue

            old_arm_parameter = old_trial.arm.parameters

            old_result_simple = None
            try:
                tmp_old_res = get_old_result_by_params(f"{this_path}/{pd_csv_filename}", old_arm_parameter)["result"]
                tmp_old_res_list = list(set(list(tmp_old_res)))

                if len(tmp_old_res_list) == 1:
                    print_debug(f"Got a list of length {len(tmp_old_res_list)}. This means the result was found properly and will be added.")
                    old_result_simple = float(tmp_old_res_list[0])
                else:
                    print_debug(f"Got a list of length {len(tmp_old_res_list)}. Cannot add this to previous jobs.")
                    old_result_simple = None
            except:
                pass

            if old_result_simple and looks_like_number(old_result_simple) and str(old_result_simple) != "nan":
                _counter += 1
                
        path_idx += 1

    return _counter

def get_list_import_as_string (_brackets=True, _comma=False):
    _str = []

    if len(double_hashes):
        _str.append(f"double hashes: {len(double_hashes)}")

    if len(missing_results):
        _str.append(f"missing_results: {len(missing_results)}")

    if len(_str):
        if(_brackets):
            if _comma:
                return ", (" + (", ".join(_str)) + ")"
            return " (" + (", ".join(_str)) + ")"
        else:
            if _comma:
                return ", " + (", ".join(_str))
            return ", ".join(_str)

    return ""

def load_data_from_existing_run_folders(args, _paths):
    #return
    global already_inserted_param_hashes
    global already_inserted_param_data
    global double_hashes
    global missing_results

    #dier(help(ax_client.experiment.search_space))
    with console.status(f"[bold green]Loading existing jobs into ax_client...") as status:
        path_idx = 0
        for this_path in _paths:
            if len(_paths) > 1:
                status.update(f"[bold green]Loading existing jobs from {this_path} into ax_client (folder {path_idx + 1}{get_list_import_as_string(False, True)})...")
            else:
                status.update(f"[bold green]Loading existing jobs from {this_path} into ax_client{get_list_import_as_string()}...")

            this_path_json = str(this_path) + "/state_files/ax_client.experiment.json"

            if not os.path.exists(this_path_json):
                print_red(f"{this_path_json} does not exist, cannot load data from it")
                return

            old_experiments = load_experiment(this_path_json)

            old_trials = old_experiments.trials
            #dataframe_dier()

            trial_idx = 0
            for old_trial_index in old_trials:
                if len(_paths) > 1:
                    status.update(f"[bold green]Loading existing jobs from {this_path} into ax_client (folder {path_idx + 1}/{len(_paths)}, trial {trial_idx + 1}/{len(old_trials)}{get_list_import_as_string(False, True)})...")
                else:
                    status.update(f"[bold green]Loading existing jobs from {this_path} into ax_client (trial {trial_idx + 1}/{len(old_trials)}{get_list_import_as_string(False, True)})...")

                trial_idx += 1

                old_trial = old_trials[old_trial_index]
                trial_status = old_trial.status
                trial_status_str = trial_status.__repr__

                print_debug(f"trial_status_str: {trial_status_str}")

                if not "COMPLETED" in str(trial_status_str):
                    continue

                old_arm_parameter = old_trial.arm.parameters

                old_result_simple = None
                try:
                    tmp_old_res = get_old_result_by_params(f"{this_path}/{pd_csv_filename}", old_arm_parameter)["result"]
                    tmp_old_res_list = list(set(list(tmp_old_res)))

                    if len(tmp_old_res_list) == 1:
                        print_debug(f"Got a list of length {len(tmp_old_res_list)}. This means the result was found properly and will be added.")
                        old_result_simple = float(tmp_old_res_list[0])
                    else:
                        print_debug(f"Got a list of length {len(tmp_old_res_list)}. Cannot add this to previous jobs.")
                        old_result_simple = None
                except:
                    pass

                hashed_params_result = pformat(old_arm_parameter) + "====" + pformat(old_result_simple)

                if old_result_simple and looks_like_number(old_result_simple) and str(old_result_simple) != "nan":
                    if hashed_params_result not in already_inserted_param_hashes.keys():
                        #print(f"ADDED: old_result_simple: {old_result_simple}, type: {type(old_result_simple)}")
                        old_result = {'result': old_result_simple}

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
                                #elif parsed_error["expected_type"] == "str" and type(old_arm_parameter[parsed_error["parameter_name"]]).__name__ != "str":
                                #    print_yellow(f"⚠ converted parameter {parsed_error['parameter_name']} type {parsed_error['current_type']} to {parsed_error['expected_type']}")
                                #    old_arm_parameter[parsed_error["parameter_name"]] = str(old_arm_parameter[parsed_error["parameter_name"]])
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

def print_outfile_analyzed(job):
    stdout_path = str(job.paths.stdout.resolve())
    stderr_path = str(job.paths.stderr.resolve())
    errors = get_errors_from_outfile(stdout_path)
    #errors.append(...get_errors_from_outfile(stderr_path))

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
            with open(f'{current_run_folder}/evaluation_errors.log', "a+") as error_file:
                error_file.write(out_files_string)
        except Exception as e:
            print_debug(f"Error occurred while writing to evaluation_errors.log: {e}")

        print_red(out_files_string)

def finish_previous_jobs(args, new_msgs):
    print_debug(f"finish_previous_jobs(args, {new_msgs})")

    global result_csv_file
    global random_steps
    global jobs
    global ax_client
    global progress_bar

    #print("jobs in finish_previous_jobs:")
    #print(jobs)

    jobs_finished = 0

    for job, trial_index in global_vars["jobs"][:]:
        # Poll if any jobs completed
        # Local and debug jobs don't run until .result() is called.
        if not job is None and (job.done() or type(job) in [LocalJob, DebugJob]):
            #print(job.result())
            try:
                result = job.result()
                raw_result = result
                result = result["result"]
                #print_debug(f"Got job result: {result}")
                jobs_finished += 1
                if result != val_if_nothing_found:
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
                        print(f"ERROR in line {getLineInfo()}: {e}")
                    print_outfile_analyzed(job)
                else:
                    if job:
                        try:
                            progressbar_description([f"job_failed"])

                            ax_client.log_trial_failure(trial_index=trial_index)
                            print_outfile_analyzed(job)
                        except Exception as e:
                            print(f"ERROR in line {getLineInfo()}: {e}")
                        job.cancel()

                    failed_jobs(1)

                global_vars["jobs"].remove((job, trial_index))
            except FileNotFoundError as error:
                print_red(str(error))

                if job:
                    try:
                        progressbar_description([f"job_failed"])
                        _trial = ax_client.get_trial(trial_index)
                        _trial.mark_failed()
                        print_outfile_analyzed(job)
                    except Exception as e:
                        print(f"ERROR in line {getLineInfo()}: {e}")
                    job.cancel()

                failed_jobs(1)
                jobs_finished += 1

                global_vars["jobs"].remove((job, trial_index))
            except submitit.core.utils.UncompletedJobError as error:
                print_red(str(error))

                if job:
                    try:
                        progressbar_description([f"job_failed"])
                        _trial = ax_client.get_trial(trial_index)
                        _trial.mark_failed()
                        print_outfile_analyzed(job)
                    except Exception as e:
                        print(f"ERROR in line {getLineInfo()}: {e}")
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
                        progressbar_description([f"job_failed"])
                        ax_client.log_trial_failure(trial_index=trial_index)
                        print_outfile_analyzed(job)
                    except Exception as e:
                        print(f"ERROR in line {getLineInfo()}: {e}")
                    job.cancel()

                failed_jobs(1)
                jobs_finished += 1

                global_vars["jobs"].remove((job, trial_index))

            if args.verbose:
                progressbar_description([f"saving checkpoints and {pd_csv_filename}"])
            save_checkpoint()
            save_pd_csv()
        else:
            pass

    if jobs_finished == 1:
        progressbar_description([*new_msgs, f"finished {jobs_finished} job"])
    else:
        if jobs_finished:
            progressbar_description([*new_msgs, f"finished {jobs_finished} jobs"])

    clean_completed_jobs()

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
    global jobs

    string = ""

    string_keys = []
    string_values = []

    stats = {}

    for job, trial_index in global_vars["jobs"][:]:
        state = state_from_job(job)

        if not state in stats.keys():
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
            percentage = round((nr_current_workers/num_parallel_jobs) * 100)
            string = f"{_keys} {_values} ({percentage}%/{num_parallel_jobs})"

    return string

def get_desc_progress_text(new_msgs=[]):
    global global_vars
    global result_csv_file
    global random_steps
    global max_eval

    desc = ""

    in_brackets = []

    if failed_jobs():
        in_brackets.append(f"{bcolors.red}failed jobs: {failed_jobs()}{bcolors.endc}")

    current_model = get_current_model()

    in_brackets.append(f"{current_model}")

    best_params = None

    this_time = time.time()

    if count_done_jobs() >= 0:
        best_params = get_best_params(result_csv_file, "result")
        if best_params and "result" in best_params:
            best_result = best_params["result"]
            if type(best_result) == float or type(best_result) == int or looks_like_float(best_result):
                best_result_int_if_possible = to_int_when_possible(float(best_result))

                if str(best_result) != NO_RESULT and best_result is not None:
                    in_brackets.append(f"best result: {best_result_int_if_possible}")

        if is_slurm_job():
            nr_current_workers = len(global_vars["jobs"])
            percentage = round((nr_current_workers/num_parallel_jobs) * 100)

            this_values = {
                "nr_current_workers": nr_current_workers,
                "num_parallel_jobs": num_parallel_jobs,
                "percentage": percentage,
                "time": this_time
            }

            if len(worker_percentage_usage) == 0 or worker_percentage_usage[len(worker_percentage_usage) - 1] != this_values:
                worker_percentage_usage.append(this_values)

    #in_brackets.append(f"jobs {count_done_jobs()}/{max_eval}")

    if args.verbose_tqdm:
        if submitted_jobs():
            in_brackets.append(f"total submitted: {submitted_jobs()}")

        if max_eval:
            in_brackets.append(f"max_eval: {max_eval}")

    if system_has_sbatch:
        workers_strings = get_workers_string()
        if workers_strings:
            in_brackets.append(workers_strings)

    if len(new_msgs):
        for new_msg in new_msgs:
            if(new_msg):
                in_brackets.append(new_msg)

    if len(in_brackets):
        in_brackets_clean = []

        for item in in_brackets:
            if item:
                in_brackets_clean.append(item)

        if in_brackets_clean:
            desc += f"{', '.join(in_brackets_clean)}"

    capitalized_string = lambda s: s[0].upper() + s[1:] if s else ''
    desc = capitalized_string(desc)

    return desc

def is_slurm_job():
    if os.environ.get('SLURM_JOB_ID') is not None:
        return True
    return False

def _sleep(args, t):
    if not args.no_sleep:
        print_debug(f"Sleeping {t} second(s) before continuation")
        time.sleep(t)

def save_state_files(args):
    global current_run_folder
    global global_vars

    state_files_folder = f"{current_run_folder}/state_files/"

    if not os.path.exists(state_files_folder):
        os.makedirs(state_files_folder)

    with open(f'{state_files_folder}/joined_run_program', 'w') as f:
        print(global_vars["joined_run_program"], file=f)

    with open(f'{state_files_folder}/experiment_name', 'w') as f:
        print(global_vars["experiment_name"], file=f)

    with open(f'{state_files_folder}/mem_gb', 'w') as f:
        print(global_vars["mem_gb"], file=f)

    with open(f'{state_files_folder}/max_eval', 'w') as f:
        print(max_eval, file=f)

    with open(f'{state_files_folder}/gpus', 'w') as f:
        print(args.gpus, file=f)

    with open(f'{state_files_folder}/time', 'w') as f:
        print(global_vars["_time"], file=f)

    with open(f'{state_files_folder}/env', 'a') as f:
        env = dict(os.environ)
        for key in env:
            print(str(key) + " = " + str(env[key]), file=f)

    with open(f'{state_files_folder}/run.sh', 'w') as f:
        print("omniopt '" + " ".join(sys.argv[1:]), file=f)

def check_python_version():
    python_version = platform.python_version()
    supported_versions = ["3.8.10", "3.10.4", "3.11.2", "3.11.9", "3.9.2", "3.12.4"]
    if not python_version in supported_versions:
        print_yellow(f"Warning: Supported python versions are {', '.join(supported_versions)}, but you are running {python_version}. This may or may not cause problems. Just is just a warning.")

def execute_evaluation(args, trial_index_to_param, ax_client, trial_index, parameters, trial_counter, executor, next_nr_steps, phase):
    global global_vars
    global progress_bar

    _trial = ax_client.get_trial(trial_index)

    try:
        _trial.mark_staged()
    except Exception as e:
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
        _sleep(args, 1)

        try:
            _trial.mark_running(no_runner_required=True)
        except Exception as e:
            #print(f"ERROR in line {getLineInfo()}: {e}")
            pass
        trial_counter += 1

        progressbar_description([f"started new job ({phase}, {trial_counter - 1}/{next_nr_steps})"])
    except submitit.core.utils.FailedJobError as error:
        if "QOSMinGRES" in str(error) and args.gpus == 0:
            print_red(f"\n⚠ It seems like, on the chosen partition, you need at least one GPU. Use --gpus=1 (or more) as parameter.")
        else:
            print_red(f"\n⚠ FAILED: {error}")

        try:
            print_debug("Trying to cancel job that failed")
            if new_job:
                try:
                    ax_client.log_trial_failure(trial_index=trial_index)
                except Exception as e:
                    print(f"ERROR in line {getLineInfo()}: {e}")
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
    except (signalUSR, signalINT, signalCONT) as e:
        print_red(f"\n⚠ Detected signal. Will exit.")
        is_in_evaluate = False
        end_program(result_csv_file, "result", 1)
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        print_red(f"\n⚠ Starting job failed with error: {e}")

    finish_previous_jobs(args, [])

    add_to_phase_counter(phase, 1)

    return trial_counter

def get_current_model ():
    global ax_client

    if ax_client:
        gs_model = ax_client.generation_strategy.model

        if gs_model:
            return str(gs_model.model)

    return "initializing"

def _get_next_trials(args, ax_client):
    global global_vars

    finish_previous_jobs(args, ["finishing jobs (_get_next_trials)"])

    if break_run_search("_get_next_trials", max_eval, progress_bar):
        return

    last_ax_client_time = None
    ax_client_time_avg = None
    if len(time_get_next_trials_took):
        last_ax_client_time = time_get_next_trials_took[len(time_get_next_trials_took) - 1]
        ax_client_time_avg = sum(time_get_next_trials_took) / len(time_get_next_trials_took)

    new_msgs = []

    currently_running_jobs = len(global_vars["jobs"])
    real_num_parallel_jobs = num_parallel_jobs - currently_running_jobs

    if real_num_parallel_jobs == 0:
        return None

    base_msg = f"getting {real_num_parallel_jobs} trials "

    if system_has_sbatch:
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

        removed_duplicates = 0
    except np.linalg.LinAlgError as e:
        if args.model and args.model.upper() in ["THOMPSON", "EMPIRICAL_BAYES_THOMPSON"]:
            print_red(f"Error: {e}. This may happen because you have the THOMPSON model used. Try another one.")
        else:
            print_red(f"Error: {e}")
        sys.exit(242)

    print_debug_get_next_trials(len(trial_index_to_param.items()), real_num_parallel_jobs, getframeinfo(currentframe()).lineno)

    get_next_trials_time_end = time.time()

    _ax_took = get_next_trials_time_end - get_next_trials_time_start

    time_get_next_trials_took.append(_ax_took)

    _log_trial_index_to_param(trial_index_to_param)

    return trial_index_to_param

def get_next_nr_steps(num_parallel_jobs, max_eval):
    if not system_has_sbatch:
        return 1

    #return num_parallel_jobs

    requested = min(num_parallel_jobs - len(global_vars["jobs"]), max_eval - submitted_jobs())

    return requested

def get_nr_of_imported_jobs(args):
    nr_jobs = 0

    if args.load_previous_job_data:
        for this_path in args.load_previous_job_data:
            nr_jobs += simulate_load_data_from_existing_run_folders(args, this_path)

    if args.continue_previous_job:
        nr_jobs += simulate_load_data_from_existing_run_folders(args, [args.continue_previous_job])

    return nr_jobs

def get_generation_strategy(args, num_parallel_jobs, seed, max_eval):
    global random_steps

    _steps = []

    nr_of_imported_jobs = get_nr_of_imported_jobs(args)

    #random_steps = nr_of_imported_jobs

    set_max_eval(max_eval + nr_of_imported_jobs)

    if random_steps is None:
        random_steps = 0

    if max_eval is None:
        set_max_eval(max(1, random_steps))

    if random_steps >= 1: 
        # TODO: nicht, wenn continue_previous_job und bereits random_steps schritte erfolgt
        # 1. Initialization step (does not require pre-existing data and is well-suited for
        # initial sampling of the search space)

        #print(f"!!! get_generation_strategy: random_steps == {random_steps}")
        #min_trials_observed=max(min(0, max_eval, random_steps), random_steps + nr_inserted_jobs),

        _steps.append(
            GenerationStep(
                model=Models.SOBOL,
                num_trials=max(num_parallel_jobs, random_steps),
                min_trials_observed=min(max_eval, random_steps),
                max_parallelism=num_parallel_jobs,  # Max parallelism for this step
                enforce_num_trials=True,
                model_kwargs={"seed": seed},  # Any kwargs you want passed into the model
                model_gen_kwargs={'enforce_num_arms': True},  # Any kwargs you want passed to `modelbridge.gen`
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
            print_red(f"Gridsearch only really works when you chose the FACTORIAL model.")

    # 2. Bayesian optimization step (requires data obtained from previous phase and learns
    # from all data available at the time of each new candidate generation call)

    #print("get_generation_strategy: Second step")

    #print(f"_steps.append(")
    #print(f"    GenerationStep(")
    #print(f"        model={chosen_non_random_model},")
    #print(f"        num_trials=-1,  # No limitation on how many trials should be produced from this step")
    #print(f"        max_parallelism={num_parallel_jobs} * 2,  # Max parallelism for this step")
    #print(f"        #model_kwargs=seed: {seed},  # Any kwargs you want passed into the model")
    #print(f"        #enforce_num_trials=True,")
    #print(f"        model_gen_kwargs='enforce_num_arms': True,  # Any kwargs you want passed to `modelbridge.gen`")
    #print(f"        # More on parallelism vs. required samples in BayesOpt:")
    #print(f"        # https://ax.dev/docs/bayesopt.html#tradeoff-between-parallelism-and-total-number-of-trials")
    #print(f"    )")
    #print(f")")

    #_nr_trials = max_eval - random_steps + nr_of_imported_jobs

    #if _nr_trials <= 0:
    #    _nr_trials = -1

    _nr_trials = -1
    #print(f"_nr_trials: {_nr_trials}")

    _steps.append(
        GenerationStep(
            model=chosen_non_random_model,
            num_trials=_nr_trials,  # No limitation on how many trials should be produced from this step
            max_parallelism=num_parallel_jobs * 2,  # Max parallelism for this step
            #model_kwargs={"seed": seed},  # Any kwargs you want passed into the model
            #enforce_num_trials=True,
            model_gen_kwargs={'enforce_num_arms': True},  # Any kwargs you want passed to `modelbridge.gen`
            # More on parallelism vs. required samples in BayesOpt:
            # https://ax.dev/docs/bayesopt.html#tradeoff-between-parallelism-and-total-number-of-trials
        )
    )

    gs = GenerationStrategy(
        steps=_steps
    )

    return gs, _steps

def create_and_execute_next_runs(args, ax_client, next_nr_steps, executor, phase, max_eval, progress_bar):
    global random_steps

    #print(f"create_and_execute_next_runs, phase {phase}: next_nr_steps: {next_nr_steps}")
    if next_nr_steps == 0:
        return 0

    trial_index_to_param = None
    try:
        print_debug(f"trying to get trial_index_to_param")

        try:
            trial_index_to_param = _get_next_trials(args, ax_client)

            if trial_index_to_param:
                i = 1
                for trial_index, parameters in trial_index_to_param.items():
                    if break_run_search("create_and_execute_next_runs", max_eval, progress_bar):
                        break

                    while len(global_vars["jobs"]) > num_parallel_jobs:
                        finish_previous_jobs(args, ["finishing prev jobs"])

                        if break_run_search("create_and_execute_next_runs", max_eval, progress_bar):
                            break

                        time.sleep(5)

                    progressbar_description([f"starting parameter set ({i}/{next_nr_steps})"])
                    execute_evaluation(args, trial_index_to_param, ax_client, trial_index, parameters, i, executor, next_nr_steps, phase)
                    i += 1
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

        random_steps_left = count_done_jobs() - random_steps

        #if random_steps_left <= 0 and count_done_jobs() <= random_steps:
        #    return len(trial_index_to_param.keys())
    except RuntimeError as e:
        print_red("\n⚠ " + str(e))
    except (
        botorch.exceptions.errors.ModelFittingError,
        ax.exceptions.core.SearchSpaceExhausted,
        ax.exceptions.core.DataRequiredError,
        botorch.exceptions.errors.InputDataError
    ) as e:
        print_red("\n⚠ " + str(e))
        end_program(result_csv_file, "result", 1)

    num_new_keys = 0

    try:
        num_new_keys = len(trial_index_to_param.keys())
    except:
        pass

    return num_new_keys

def get_random_steps_from_prev_job(args):
    if not args.continue_previous_job:
        return count_sobol_steps()

    prev_step_file = args.continue_previous_job + "/state_files/phase_random_steps"

    if not os.path.exists(prev_step_file):
        return count_sobol_steps()

    return add_to_phase_counter("random", count_sobol_steps() + _count_sobol_steps(f"{args.continue_previous_job}/results.csv"), args.continue_previous_job)

def get_number_of_steps(args, max_eval):
    random_steps = args.num_random_steps

    already_done_random_steps = get_random_steps_from_prev_job(args)

    random_steps = random_steps - already_done_random_steps

    if random_steps > max_eval:
        print_yellow(f"You have less --max_eval {max_eval} than --num_random_steps {random_steps}. Switched both.")
        tmp = random_steps
        random_steps = max_eval
        max_eval = tmp

    if random_steps < num_parallel_jobs and system_has_sbatch:
        old_random_steps = random_steps
        random_steps = num_parallel_jobs
        original_print(f"random_steps {old_random_steps} is smaller than num_parallel_jobs {num_parallel_jobs}. --num_random_steps will be ignored and set to num_parallel_jobs ({num_parallel_jobs}) to not have idle workers in the beginning.")

    if random_steps > max_eval:
        set_max_eval(random_steps)

    original_second_steps = max_eval - random_steps
    second_step_steps = max(0, original_second_steps)
    if second_step_steps != original_second_steps:
        original_print(f"? original_second_steps: {original_second_steps} = max_eval {max_eval} - random_steps {random_steps}")
    if second_step_steps == 0:
        print_red("This is basically a random search. Increase --max_eval or reduce --num_random_steps")

    second_step_steps = second_step_steps - already_done_random_steps

    if args.continue_previous_job:
        second_step_steps = max_eval

    return random_steps, second_step_steps

def get_executor(args):
    global run_uuid

    log_folder = f'{current_run_folder}/single_runs/%j'
    executor = None
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

    return executor

def append_and_read(file, nr=0):
    try:
        with open(file, 'a+') as f:
            f.seek(0)  # Setze den Dateizeiger auf den Anfang der Datei
            anzahl_zeilen = len(f.readlines())

            if nr == 1:
                f.write('1\n')

        return anzahl_zeilen

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except (signalUSR, signalINT, signalCONT) as e:
        append_and_read(file, nr)
    except OSError as e:
        print_red(f"OSError: {e}. This may happen on unstable file systems.")
        sys.exit(199)
    except Exception as e:
        print(f"Error editing the file: {e}")

    return 0

def failed_jobs(nr=0):
    state_files_folder = f"{current_run_folder}/state_files/"

    if not os.path.exists(state_files_folder):
        os.makedirs(state_files_folder)

    return append_and_read(f'{current_run_folder}/state_files/failed_jobs', nr)

def get_steps_from_prev_job(prev_job, nr=0):
    state_files_folder = f"{current_run_folder}/state_files/"

    if not os.path.exists(state_files_folder):
        os.makedirs(state_files_folder)

    return append_and_read(f"{prev_job}/state_files/submitted_jobs", nr)

def succeeded_jobs(nr=0):
    state_files_folder = f"{current_run_folder}/state_files/"

    if not os.path.exists(state_files_folder):
        os.makedirs(state_files_folder)

    return append_and_read(f'{current_run_folder}/state_files/succeeded_jobs', nr)


def submitted_jobs(nr=0):
    state_files_folder = f"{current_run_folder}/state_files/"

    if not os.path.exists(state_files_folder):
        os.makedirs(state_files_folder)

    return append_and_read(f'{current_run_folder}/state_files/submitted_jobs', nr)

def count_done_jobs():
    csv_file_path = save_pd_csv()

    return _count_done_jobs(csv_file_path)

def _count_done_jobs(csv_file_path):
    results = 0

    if not os.path.exists(csv_file_path):
        print_yellow(f"CSV file does not exist at path: {csv_file_path}")
        return results

    df = None

    try:
        df = pd.read_csv(csv_file_path, index_col=0, float_precision='round_trip')
        df.dropna(subset=["result"], inplace=True)
    except KeyError as e:
        return 0
    except pd.errors.EmptyDataError as e:
        return 0
    except pd.errors.ParserError as e:
        print_red(f"Error reading CSV file 2: {str(e)}")
        return 0
    except UnicodeDecodeError as e:
        print_red(f"Error reading CSV file 3: {str(e)}")
        return 0
    except Exception as e:
        print_red(f"Error reading CSV file 4: {str(e)}")
        return 0

    assert_condition(df is not None, "DataFrame should not be None after reading CSV file")
    assert_condition("generation_method" in df.columns, "'generation_method' column must be present in the DataFrame")

    non_sobol_rows = df[df["generation_method"] != "Sobol"]
    non_sobol_rows_count = len(non_sobol_rows)

    return non_sobol_rows_count

def count_manual_steps():
    csv_file_path = save_pd_csv()

    return _count_manual_steps(csv_file_path)

def _count_manual_steps(csv_file_path):
    sobol_count = 0

    if not os.path.exists(csv_file_path):
        print_yellow(f"CSV file does not exist at path: {csv_file_path}")
        return sobol_count

    df = None

    try:
        df = pd.read_csv(csv_file_path, index_col=0, float_precision='round_trip')
        df.dropna(subset=["result"], inplace=True)
    except KeyError as e:
        return 0
    except pd.errors.EmptyDataError as e:
        return 0
    except pd.errors.ParserError as e:
        print_red(f"Error reading CSV file 2: {str(e)}")
        return 0
    except UnicodeDecodeError as e:
        print_red(f"Error reading CSV file 3: {str(e)}")
        return 0
    except Exception as e:
        print_red(f"Error reading CSV file 4: {str(e)}")
        return 0

    assert_condition(df is not None, "DataFrame should not be None after reading CSV file")
    assert_condition("generation_method" in df.columns, "'generation_method' column must be present in the DataFrame")

    sobol_rows = df[df["generation_method"] == "Manual"]
    sobol_count = len(sobol_rows)

    return sobol_count



def count_sobol_steps():
    csv_file_path = save_pd_csv()

    return _count_sobol_steps(csv_file_path)

def _count_sobol_steps(csv_file_path):
    sobol_count = 0

    if not os.path.exists(csv_file_path):
        print_yellow(f"CSV file does not exist at path: {csv_file_path}")
        return sobol_count

    df = None

    try:
        df = pd.read_csv(csv_file_path, index_col=0, float_precision='round_trip')
        df.dropna(subset=["result"], inplace=True)
    except KeyError as e:
        return 0
    except pd.errors.EmptyDataError as e:
        return 0
    except pd.errors.ParserError as e:
        print_red(f"Error reading CSV file 2: {str(e)}")
        return 0
    except UnicodeDecodeError as e:
        print_red(f"Error reading CSV file 3: {str(e)}")
        return 0
    except Exception as e:
        print_red(f"Error reading CSV file 4: {str(e)}")
        return 0

    assert_condition(df is not None, "DataFrame should not be None after reading CSV file")
    assert_condition("generation_method" in df.columns, "'generation_method' column must be present in the DataFrame")

    sobol_rows = df[df["generation_method"] == "Sobol"]
    sobol_count = len(sobol_rows)

    return sobol_count

def execute_nvidia_smi():
    if not is_nvidia_smi_system:
        print_debug(f"Cannot find nvidia-smi. Cannot take GPU logs")
        return

    while True:
        try:
            host = socket.gethostname()

            _file = nvidia_smi_logs_base + "_" + host + ".csv"
            noheader = ",noheader"

            result = subprocess.run([
                'nvidia-smi',
                '--query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used',
                f'--format=csv{noheader}'],
                capture_output=True,
                text=True
            )
            assert result.returncode == 0, "nvidia-smi execution failed"

            output = result.stdout

            output = output.rstrip('\n')

            if host and output:
                append_to_nvidia_smi_logs(_file, host, output)
        except Exception as e:
            print(f"An error occurred: {e}")
        time.sleep(10)

def start_nvidia_smi_thread():
    print_debug("start_nvidia_smi_thread()")

    if is_nvidia_smi_system:
        nvidia_smi_thread = threading.Thread(target=execute_nvidia_smi, daemon=True)
        nvidia_smi_thread.start()
        return nvidia_smi_thread
    return None

def break_run_search (_name, max_eval, progress_bar):
    if succeeded_jobs() > max_eval:
        print_debug(f"breaking {_name}: succeeded_jobs() {succeeded_jobs()} > max_eval {max_eval}")
        return True

    if progress_bar.total < submitted_jobs():
        print_debug(f"breaking {_name}: progress_bar.total {progress_bar.total} <= submitted_jobs() {submitted_jobs()}")
        return True

    if count_done_jobs() >= max_eval:
        print_debug(f"breaking {_name}: count_done_jobs() {count_done_jobs()} > max_eval {max_eval}")
        return True

    if submitted_jobs() > max_eval:
        print_debug(f"breaking {_name}: submitted_jobs() {submitted_jobs()} > max_eval {max_eval}")
        return True

    if abs(count_done_jobs() - max_eval - nr_inserted_jobs) <= 0:
        print_debug(f"breaking {_name}: if abs(count_done_jobs() {count_done_jobs()} - max_eval {max_eval} - nr_inserted_jobs {nr_inserted_jobs}) <= 0")
        return True

    return False

def run_search(args, max_nr_steps, executor, ax_client, progress_bar):
    global search_space_exhausted
    global nr_of_0_results

    log_what_needs_to_be_logged()

    nr_of_0_results = 0
    write_process_info()

    while (submitted_jobs() <= max_eval) and not search_space_exhausted:
        log_what_needs_to_be_logged()
        wait_for_jobs_to_complete(num_parallel_jobs)

        finish_previous_jobs(args, [])

        if break_run_search("run_search", max_eval, progress_bar):
            break

        next_nr_steps = get_next_nr_steps(num_parallel_jobs, max_eval)

        nr_of_items = 0

        if next_nr_steps:
            progressbar_description([f"trying to get {next_nr_steps} next steps"])
            nr_of_items = create_and_execute_next_runs(args, ax_client, next_nr_steps, executor, "systematic", max_eval, progress_bar)

            progressbar_description([f"got {nr_of_items}, requested {next_nr_steps}"])

        _debug_worker_creation(f"{int(time.time())}, {len(global_vars['jobs'])}, {nr_of_items}, {next_nr_steps}")

        finish_previous_jobs(args, ["finishing prev jobs"])

        _sleep(args, 1)

        if nr_of_items == 0 and len(global_vars["jobs"]) == 0:
            _wrn = f"found {nr_of_0_results} zero-jobs (max: {args.max_nr_of_zero_results})"
            nr_of_0_results += 1
            progressbar_description([_wrn])
            print_debug(_wrn)
        else:
            nr_of_0_results = 0

        if not args.disable_search_space_exhaustion_detection and nr_of_0_results >= args.max_nr_of_zero_results:
            _wrn = f"nr_of_0_results {nr_of_0_results} >= {args.max_nr_of_zero_results}"

            print_debug(_wrn)
            progressbar_description([_wrn])

            raise searchSpaceExhausted("Search space exhausted")
        log_what_needs_to_be_logged()


    wait_for_jobs_to_complete(num_parallel_jobs)

    while len(global_vars["jobs"]):
        finish_previous_jobs(args, [f"waiting for jobs ({len(global_vars['jobs'])} left)"])
        _sleep(args, 1)

    log_what_needs_to_be_logged()
    return False

def wait_for_jobs_to_complete (num_parallel_jobs):
    global global_vars

    if system_has_sbatch:
        while len(global_vars["jobs"]) > num_parallel_jobs:
            progressbar_description([f"waiting for old jobs to finish ({len(global_vars['jobs'])} left)"])
            time.sleep(5)
            clean_completed_jobs()

def finish_previous_jobs_random(args):
    try:
        while len(global_vars['jobs']):
            finish_previous_jobs(args, [f"waiting for jobs ({len(global_vars['jobs'])} left)"])
            _sleep(args, 1)
    except EOFError as e:
        print_red(f"Error {e}: This may mean that you are running on an unstable file system. Cannot continue.")
        sys.exit(199)

def print_logo():
    try:
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

    except Exception as e:
        print_green("OmniOpt2")

def show_strategy_message (random_steps, second_step_steps):
    second_step_steps_string = ""
    if second_step_steps:
        second_step_steps_string = f", followed by {second_step_steps} systematic steps"

    #if random_steps and random_steps > submitted_jobs():
    #    print(f"\nStarting random search for {random_steps} steps{second_step_steps_string}.")

def show_messages(args, random_steps, second_step_steps):
    global current_run_folder
    show_strategy_message(random_steps, second_step_steps)

    if args.auto_execute_suggestions and args.experimental and not current_run_folder.endswith("/0") and args.auto_execute_counter == 0:
        print_red(f"When you do a automatic parameter search space expansion, it's recommended that you have an empty run folder, i.e this run is runs/example/0. But this run is {current_run_folder}. This may not be what you want, when you want to plot the expansion of the search space with bash .tools/plot_gif_from_history runs/custom_run")

    if args.auto_execute_counter:
        if args.max_auto_execute:
            print(f"Auto-Executing step {args.auto_execute_counter}/max. {args.max_auto_execute}")
        else:
            print(f"Auto-Executing step {args.auto_execute_counter}")

def count_defective_nodes(file_path=None, entry=None):
    """
    Diese Funktion nimmt optional einen Dateipfad und einen Eintrag entgegen.
    Sie öffnet die Datei, erstellt sie, wenn sie nicht existiert,
    prüft, ob der Eintrag bereits als einzelne Zeile in der Datei vorhanden ist,
    und fügt ihn am Ende hinzu, wenn dies nicht der Fall ist.
    Schließlich gibt sie eine sortierte Liste aller eindeutigen Einträge in der Datei zurück.
    Wenn keine Argumente übergeben werden, wird der Dateipfad auf
    '{current_run_folder}/state_files/defective_nodes' gesetzt und kein Eintrag hinzugefügt.
    """
    # Standardpfad für die Datei, wenn keiner angegeben ist
    if file_path is None:
        file_path = os.path.join(current_run_folder, "state_files", "defective_nodes")

    # Sicherstellen, dass das Verzeichnis existiert
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    try:
        # Öffnen der Datei im Modus 'a+' (Anhängen und Lesen)
        with open(file_path, 'a+') as file:
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

def human_readable_generation_strategy (ax_client):
    generation_strategy_str = str(ax_client.generation_strategy)

    pattern = r'\[(.*?)\]'

    match = re.search(pattern, generation_strategy_str)

    if match:
        content = match.group(1)
        return content

    return None

def main():
    print_debug("main()")

    print_logo()

    global args
    global result_csv_file
    global ax_client
    global global_vars
    global max_eval
    global global_vars
    global run_folder_number
    global current_run_folder
    global nvidia_smi_logs_base
    global logfile_debug_get_next_trials
    global search_space_exhausted
    global random_steps
    global second_step_steps
    global searching_for

    if (not args.continue_previous_job and not args.load_previous_job_data and not "--continue" in sys.argv) and (args.num_random_steps == 0 or not args.num_random_steps):
        print_red("You have no random steps set. This is only allowed in continued jobs. To start, you need either some random steps, or a continued run.")
        my_exit(233)

    _debug_worker_creation("time, nr_workers, got, requested, phase")

    original_print("./omniopt " + " ".join(sys.argv[1:]))

    check_slurm_job_id()

    current_run_folder = f"{args.run_dir}/{global_vars['experiment_name']}/{run_folder_number}"
    while os.path.exists(f"{current_run_folder}"):
        current_run_folder = f"{args.run_dir}/{global_vars['experiment_name']}/{run_folder_number}"
        run_folder_number = run_folder_number + 1

    result_csv_file = create_folder_and_file(f"{current_run_folder}")

    save_state_files(args)

    print(f"[yellow]Run-folder[/yellow]: [underline]{current_run_folder}[/underline]")
    if args.continue_previous_job:
        print(f"[yellow]Continuation from {args.continue_previous_job}[/yellow]")

    nvidia_smi_logs_base = f'{current_run_folder}/gpu_usage_'

    if args.ui_url:
        with open(f"{current_run_folder}/ui_url.txt", "a") as myfile:
            myfile.write(decode_if_base64(args.ui_url))

    logfile_debug_get_next_trials = f'{current_run_folder}/get_next_trials.csv'

    check_python_version()

    experiment_parameters = None
    cli_params_experiment_parameters = None
    checkpoint_parameters_filepath = f"{current_run_folder}/state_files/checkpoint.json.parameters.json"

    if args.parameter:
        experiment_parameters = parse_experiment_parameters(args)
        cli_params_experiment_parameters = experiment_parameters

    if not args.verbose:
        disable_logging()

    #set_max_eval(args.max_eval)

    if not max_eval:
        print_red("--max_eval needs to be set!")
        sys.exit(19)

    random_steps, second_step_steps = get_number_of_steps(args, max_eval)

    if args.exclude:
        entries = [entry.strip() for entry in args.exclude.split(',')]

        # Verarbeitung der Einträge
        for entry in entries:
            count_defective_nodes(None, entry)

    if args.parameter and len(args.parameter) and args.continue_previous_job and random_steps <= 0:
        print(f"A parameter has been reset, but the earlier job already had it's random phase. To look at the new search space, {args.num_random_steps} random steps will be executed.")
        random_steps = args.num_random_steps

    gs, gs_steps = get_generation_strategy(args, num_parallel_jobs, args.seed, args.max_eval)

    ax_client = AxClient(
        verbose_logging=args.verbose,
        enforce_sequential_optimization=args.enforce_sequential_optimization,
        generation_strategy=gs
    )

    minimize_or_maximize = not args.maximize

    experiment = None

    ax_client, experiment_parameters, experiment_args = get_experiment_parameters(
        ax_client, 
        args.continue_previous_job, 
        args.seed, 
        args.experiment_constraints, 
        args.parameter, 
        cli_params_experiment_parameters, 
        experiment_parameters, 
        minimize_or_maximize
    )

    #ax_client._set_generation_strategy(*gs_steps)

    gs_hr = human_readable_generation_strategy(ax_client)
    if gs_hr:
        print(f"Generation strategy: {gs_hr}")

    with open(checkpoint_parameters_filepath, "w") as outfile:
        json.dump(experiment_parameters, outfile, cls=NpEncoder)

    print_overview_tables(experiment_parameters, experiment_args)

    executor = get_executor(args)
    searching_for = "minimum" if not args.maximize else "maximum"

    load_existing_job_data_into_ax_client(args)

    initial_text = get_desc_progress_text()
    print(f"Searching {searching_for}")

    original_print(f"Run-Program: {global_vars['joined_run_program']}")

    show_messages(args, random_steps, second_step_steps)

    max_nr_steps = second_step_steps
    if count_done_jobs() < random_steps:
        max_nr_steps = (random_steps - count_done_jobs()) + second_step_steps
        set_max_eval(max_nr_steps)

    prev_steps_nr = 0

    if args.continue_previous_job:
        prev_steps_nr = get_steps_from_prev_job(args.continue_previous_job)

        max_nr_steps = prev_steps_nr + max_nr_steps
        #set_max_eval(max_nr_steps)

    #set_max_eval(max_eval + nr_inserted_jobs)
    save_global_vars()

    write_process_info()

    #max_tqdm_jobs = count_manual_steps() + max_eval
    #print(f"max_tqdm_jobs {max_tqdm_jobs} = count_manual_steps: {count_manual_steps()} + max_eval: {max_eval}")

    with tqdm(total=max_eval, disable=False) as _progress_bar:
        write_process_info()
        global progress_bar
        progress_bar = _progress_bar

        progressbar_description([f"Started OmniOpt2 run..."])

        update_progress_bar(progress_bar, count_done_jobs())

        run_search(args, max_eval, executor, ax_client, progress_bar)

        wait_for_jobs_to_complete(num_parallel_jobs)

    save_logs_to_json(f'{current_run_folder}/function_logs.json')
    end_program(result_csv_file)

def _unidiff_output(expected, actual):
    """
    Helper function. Returns a string containing the unified diff of two multiline strings.
    """

    expected = expected.splitlines(1)
    actual = actual.splitlines(1)

    diff=difflib.unified_diff(expected, actual)

    return ''.join(diff)

def print_diff(o, i):
    if type(i) == str:
        print("Should be:", i.strip())
    else:
        print("Should be:", i)

    if type(o) == str:
        print("Is:", o.strip())
    else:
        print("Is:", o)
    #if type(i) == str or type(o) == str:
    #    print("Diff:", _unidiff_output(json.dumps(i), json.dumps(o)))

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

def _is_not_equal(name, input, output):
    if type(input) == str or type(input) == int or type(input) == float:
        if input == output:
            print_red(f"Failed test: {name}")
            return 1
    elif type(input) == bool:
        if input == output:
            print_red(f"Failed test: {name}")
            return 1
    elif output is None or input is None:
        if input == output:
            print_red(f"Failed test: {name}")
            return 1
    else:
        print_red(f"Unknown data type for test {name}")
        my_exit(193)

    print_green(f"Test OK: {name}")
    return 0

def _is_equal(name, input, output):
    if type(input) != type(output):
        print_red(f"Failed test: {name}")
        return 1
    elif type(input) == str or type(input) == int or type(input) == float:
        if input != output:
            print_red(f"Failed test: {name}")
            return 1
    elif type(input) == bool:
        if input != output:
            print_red(f"Failed test: {name}")
            return 1
    elif output is None or input is None:
        if input != output:
            print_red(f"Failed test: {name}")
            return 1
    else:
        print_red(f"Unknown data type for test {name}")
        my_exit(192)

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

    nr_errors += is_equal(f"replace_parameters_in_string {_program_name}", program_string_with_params, f"{program_path_with_program} 1 2 3 45")

    stdout_stderr_exit_code_signal = execute_bash_code(program_string_with_params)

    stdout = stdout_stderr_exit_code_signal[0]
    stderr = stdout_stderr_exit_code_signal[1]
    exit_code = stdout_stderr_exit_code_signal[2]
    signal = stdout_stderr_exit_code_signal[3]

    res = get_result(stdout)

    if res_is_none:
        nr_errors += is_equal(f"{_program_name} res is None", None, res)
    else:
        nr_errors += is_equal(f"{_program_name} res type is nr", True, type(res) == int or type(res) == float)
    nr_errors += is_equal(f"{_program_name} stderr", True, wanted_stderr in stderr)
    nr_errors += is_equal(f"{_program_name} exit-code ", exit_code, wanted_exit_code)
    nr_errors += is_equal(f"{_program_name} signal", signal, wanted_signal)

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
        if not i in string:
            if os.path.exists(i):
                print("Missing {i} in find_file_paths string!")
                nr_errors += 1

    return nr_errors

def run_tests():
    print_image_to_cli(".tools/slimer.png", 300)
    print_image_to_cli(".tools/slimer2.png", 300)

    nr_errors = 0

    nr_errors += is_not_equal("nr equal string", 1, "1")
    nr_errors += is_equal("nr equal nr", 1, 1)
    nr_errors += is_not_equal("unequal strings", "hallo", "welt")

    #complex_tests (_program_name, wanted_stderr, wanted_exit_code, wanted_signal, res_is_none=False):
    nr_errors += complex_tests("simple_ok", "hallo", 0, None)
    nr_errors += complex_tests("divide_by_0", 'Illegal division by zero at ./.tests/test_wronggoing_stuff.bin/bin/divide_by_0 line 3.\n', 255, None, True)
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
        is_equal("test_find_paths failed", true, false)
        nr_errors += find_path_res

    my_exit(nr_errors)

def file_contains_text(f, t):
    print_debug("file_contains_text")
    datafile = get_file_as_string(f)

    found = False
    for line in datafile:
        if t in line:
            return True
    return False

def get_first_line_of_file_that_contains_string(i, s):
    print_debug("get_first_line_of_file_that_contains_string")
    if not os.path.exists(i):
        print(f"File {i} not found")
        return

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

    return None

def get_errors_from_outfile(i):
    print_debug(f"get_errors_from_outfile({i})")
    file_as_string = get_file_as_string(i)
    m = REMatcher(file_as_string)

    program_code = get_program_code_from_out_file(i)
    file_paths = find_file_paths(program_code)

    first_line = ""

    first_file_as_string = ""

    if len(file_paths):
        try:
            first_file_as_string = get_file_as_string(file_paths[0])
            if type(first_file_as_string) == str and first_file_as_string.strip().isprintable():
                first_line = first_file_as_string.split('\n')[0]
        except UnicodeDecodeError as e:
            pass

        if first_file_as_string == "":
            first_line = "#!/bin/bash"

    errors = []

    if "Result: None" in file_as_string:
        errors.append("Got no result.")

        if first_line and type(first_line) == str and first_line.isprintable() and not first_line.startswith("#!"):
            errors.append("First line does not seem to be a shebang line: " + first_line)

        if "Permission denied" in file_as_string and "/bin/sh" in file_as_string:
            errors.append("Log file contains 'Permission denied'. Did you try to run the script without chmod +x?")

        if "Exec format error" in file_as_string:
            current_platform = platform.machine()
            file_output = ""

            if len(file_paths):
                file_result = execute_bash_code("file " + file_paths[0])
                if len(file_result) and type(file_result[0]) == str:
                    file_output = ", " + file_result[0].strip()

            errors.append(f"Was the program compiled for the wrong platform? Current system is {current_platform}{file_output}")

        base_errors = [
            "Segmentation fault",
            "Illegal division by zero",
            "OOM",
            ["Killed", "Detected kill, maybe OOM or Signal?"]
        ]

        for err in base_errors:
            if type(err) == list:
                if err[0] in file_as_string:
                    errors.append(f"{err[0]} {err[1]}")
            elif type(err) == str:
                if err in file_as_string:
                    errors.append(f"{err} detected")
            else:
                print_red(f"Wrong type, should be list or string, is {type(err)}")

        if "Can't locate" in file_as_string and "@INC" in file_as_string:
            errors.append("Perl module not found")

        if "/bin/sh" in file_as_string and "not found" in file_as_string:
            errors.append("Wrong path? File not found")

        if "Datei oder Verzeichnis nicht gefunden" in file_as_string:
            errors.append("Wrong path? File not found")

        if len(file_paths) and os.stat(file_paths[0]).st_size == 0:
            errors.append(f"File in {program_code} is empty")

        if len(file_paths) == 0:
            errors.append(f"No files could be found in your program string: {program_code}")

        for r in range(1, 255):
            special_exit_codes = {
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
            search_for_exit_code = "Exit-Code: " + str(r) + ","
            if search_for_exit_code in file_as_string:
                _error = "Non-zero exit-code detected: " + str(r)
                if str(r) in special_exit_codes:
                    _error += " (May mean " + special_exit_codes[str(r)] + ", unless you used that exit code yourself or it was part of any of your used libraries or programs)"
                errors.append(_error)

        synerr = "Python syntax error detected. Check log file."

        search_for_python_errors = [
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

        for search_array in search_for_python_errors:
            search_for_string = search_array[0]
            search_for_error = search_array[1]

            if search_for_string in file_as_string:
                error_line = get_first_line_of_file_that_contains_string(i, search_for_string)
                if error_line:
                    errors.append(error_line)
                else:
                    errors.append(search_for_error)

    return errors

def find_files(directory, extension='.out'):
    files = []
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(extension):
                files.append(os.path.join(root, filename))
    return files

def analyze_out_files(rootdir, print_to_stdout=True):
    try:
        outfiles = find_files('{rootdir}/')
        # outfiles = glob.glob(f'{rootdir}/**/*.out', recursive=True)

        j = 0

        _strs = []

        for i in outfiles:
            errors = get_errors_from_outfile(i)

            if len(errors):
                if j == 0:
                    _strs.append("")
                _strs.append(f"Out file {i} contains potential errors:\n")
                program_code = get_program_code_from_out_file(i)
                if program_code:
                    _strs.append(program_code)

                for e in errors:
                    _strs.append(f"- {e}\n")

                _strs.append("\n")

                j = j + 1

        if print_to_stdout:
            print_red("\n".join(errors))

        return "\n".join(_strs)
    except Exception as e:
        print("error: " + str(e))
        return ""

def log_nr_of_workers():
    try:
        write_process_info()
    except Exception as e:
        print_debug(f"log_nr_of_workers: failed to write_process_info: {e}")

    if not "jobs" in global_vars:
        print_debug("log_nr_of_workers: Could not find jobs in global_vars")
        return

    nr_of_workers = len(global_vars["jobs"])

    if not nr_of_workers:
        return

    try:
        with open(logfile_nr_workers, 'a+') as f:
            f.write(str(nr_of_workers) + "\n")
    except FileNotFoundError:
        print_red(f"It seems like the folder for writing {logfile_nr_workers} was deleted during the run. Cannot continue.")
        sys.exit(99)
    except OSError as e:
        print_red(f"Tried writing log_nr_of_workers to file {logfile_nr_workers}, but failed with error {e}. This may mean that the file system you are running on is instable. OmniOpt probably cannot do anything about it.")
        sys.exit(199)

def get_best_params(csv_file_path, result_column):
    results = {
        result_column: None,
        "parameters": {}
    }

    if not os.path.exists(csv_file_path):
        return results

    df = None

    try:
        df = pd.read_csv(csv_file_path, index_col=0, float_precision='round_trip')
        df.dropna(subset=[result_column], inplace=True)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError, KeyError):
        return results

    cols = df.columns.tolist()
    nparray = df.to_numpy()

    best_line = None

    result_idx = cols.index(result_column)

    best_result = None

    for i in range(0, len(nparray)):
        this_line = nparray[i]
        this_line_result = this_line[result_idx]

        if type(this_line_result) == str and re.match(r'^-?\d+(?:\.\d+)$', this_line_result) is not None:
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
        if col not in ["start_time", "end_time", "hostname", "signal", "exit_code", "run_time", "program_string"]:
            if col == result_column:
                results[result_column] = "{:f}".format(best_line[i]) if type(best_line[i]) in [int, float] else best_line[i]
            else:
                results["parameters"][col] = "{:f}".format(best_line[i]) if type(best_line[i]) in [int, float] else best_line[i]

    return results

def is_near_boundary(value, min_value, max_value, threshold=0.1):
    """
    Überprüft, ob ein Wert nahe an einem der Ränder seiner Parametergrenzen liegt.
    
    :param value: Der zu überprüfende Wert
    :param min_value: Der minimale Grenzwert
    :param max_value: Der maximale Grenzwert
    :param threshold: Der Prozentsatz der Nähe zum Rand (Standard ist 10%)
    :return: True, wenn der Wert nahe am Rand ist, sonst False
    """
    range_value = max_value - min_value
    return abs(value - min_value) < threshold * range_value or abs(value - max_value) < threshold * range_value

def calculate_probability(value, min_value, max_value):
    """
    Berechnet die Wahrscheinlichkeit basierend auf der relativen Nähe des Werts zu den Grenzen.
    
    :param value: Der zu überprüfende Wert
    :param min_value: Der minimale Grenzwert
    :param max_value: Der maximale Grenzwert
    :return: Wahrscheinlichkeit, dass eine Erweiterung sinnvoll ist
    """
    range_value = max_value - min_value
    if abs(value - min_value) < abs(value - max_value):
        distance_to_boundary = abs(value - min_value)
    else:
        distance_to_boundary = abs(value - max_value)
        
    # Je näher am Rand, desto höher die Wahrscheinlichkeit
    probability = (1 - (distance_to_boundary / range_value)) * 100
    return round(probability, 2)

def find_promising_bubbles(pd_csv):
    """
    Findet vielversprechende Punkte (grüne Bubbles) am Rand des Parameterraums.
    
    :param csv_file: Der Pfad zur CSV-Datei
    """
    # CSV-Datei einlesen
    data = pd.read_csv(pd_csv, float_precision='round_trip')
    
    # Ignoriere irrelevante Spalten
    relevant_columns = [col for col in data.columns if col not in ['trial_index', 'arm_name', 'trial_status', 'generation_method', 'result']]
    
    # Ergebnisse sortieren (niedrigste Werte zuerst)
    sorted_data = data.sort_values(by='result')
    
    # Bestimmen der Parametergrenzen für jede relevante Spalte
    param_bounds = {col: (sorted_data[col].min(), sorted_data[col].max()) for col in relevant_columns}

    # Berechnung des automatischen result_thresholds
    min_result = sorted_data['result'].min()
    result_threshold = min_result + 0.2 * abs(min_result)
    
    # Vielversprechende Punkte finden
    promising_bubbles = set()
    probability_dict = {}
    
    for index, row in sorted_data.iterrows():
        if row['result'] > result_threshold:
            continue
        try:
            for col in relevant_columns:
                min_value, max_value = param_bounds[col]
                if is_near_boundary(float(row[col]), float(min_value), float(max_value)):
                    direction = 'negative' if abs(float(row[col]) - float(min_value)) < abs(float(row[col]) - float(max_value)) else 'positive'
                    probability = calculate_probability(float(row[col]), float(min_value), float(max_value))
                    key = (col, direction)
                    promising_bubbles.add(key)
                    probability_dict[key] = probability_dict.get(key, []) + [probability]
        except Exception as e:
            pass
    

    param_directions_strings = {}

    for point in promising_bubbles:
        param, direction = point

        smaller_or_larger = "smaller"
        if direction == "positive":
            smaller_or_larger = "larger"

        if not param in param_directions_strings.keys():
            param_directions_strings[param] = []

        if direction == "positive":
            param_directions_strings[param].append(f"larger {param} upper bound")
        else:
            param_directions_strings[param].append(f"smaller {param} lower bound")

    if len(param_directions_strings.keys()) and args.show_parameter_suggestions:
        print("\nParameter suggestions:\n")

        for param in param_directions_strings.keys():
            _start = f"- Suggestion for {param}: "
            print(_start + ", ".join(param_directions_strings[param]))

        print(f"--maximizer: {args.maximizer}")

        argv_copy = sys.argv

        argv_copy[0] = f"{script_dir}/omniopt"

        # --parameter float_param range -5 5 float
        # --parameter choice_param choice 1,2,4,8,16,hallo

        i = 0

        while i < len(argv_copy):
            param = argv_copy[i]
            if param == "--parameter":
                i = i + 1
                param = argv_copy[i]

                _name = param

                i = i + 1
                param = argv_copy[i]

                _type = param
                
                if _type == "range":
                    for this_changable_name, this_changable_direction in promising_bubbles:
                        if _name == this_changable_name:
                            i = i + 1
                            param = argv_copy[i]

                            if this_changable_direction == "negative":
                                new_lower_limit = float(argv_copy[i]) - abs(args.maximizer * abs(float(argv_copy[i])))

                                integerize = False

                                if argv_copy[i + 2] == "int":
                                    integerize = True

                                if integerize:
                                    new_lower_limit = round(new_lower_limit)
                                else:
                                    new_lower_limit = round(new_lower_limit, 2)

                                argv_copy[i] = str(new_lower_limit)
                            else:
                                new_upper_limit = float(argv_copy[i]) + abs(args.maximizer * abs(float(argv_copy[i])))

                                integerize = False

                                if argv_copy[i + 1] == "int":
                                    integerize = True

                                if integerize:
                                    new_upper_limit = round(new_upper_limit)
                                else:
                                    new_upper_limit = round(new_upper_limit, 2)

                                argv_copy[i] = str(new_upper_limit)

            i = i + 1

        argv_copy.append("--load_previous_job_data")
        argv_copy.append(f"{current_run_folder}/")

        if args.auto_execute_suggestions:
            argv_copy.append("--auto_execute_counter")
            argv_copy.append(str(args.auto_execute_counter + 1))

        argv_copy_string = " ".join(argv_copy)

        if "--parameter" in argv_copy_string:
            original_print("Given, you accept these suggestions, simply run this OmniOpt command:\n" + argv_copy_string + "\n")

        if args.experimental and args.auto_execute_suggestions:
            if system_has_sbatch:
                print_red("Warning: Auto-executing on systems with sbatch may not work as expected. The main worker may get killed with all subjobs")
            print("Auto-executing is not recommended")

            if args.auto_execute_counter is not None and args.max_auto_execute is not None and args.auto_execute_counter >= args.max_auto_execute:
                print("Too nested")
                my_exit(0)

            subprocess.run([*argv_copy])
        elif args.auto_execute_suggestions:
            print("Auto executing suggestions is so fucking experimental that you need the --experimental switch to be set")


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
            except (signalUSR, signalINT, signalCONT, KeyboardInterrupt) as e:
                print_red("\n⚠ You pressed CTRL+C or got a signal. Optimization stopped.")
                is_in_evaluate = False

                end_program(result_csv_file, "result", 1)
            except searchSpaceExhausted:
                _get_perc = abs(int(((count_done_jobs() - nr_inserted_jobs) / max_eval) * 100))

                if _get_perc < 100:
                    print_red(f"\nIt seems like the search space was exhausted. You were able to get {_get_perc}% of the jobs you requested (got: {count_done_jobs() - nr_inserted_jobs}, requested: {max_eval}) after main ran")

                if _get_perc != 100:
                    end_program(result_csv_file, "result", 1, 87)
                else:
                    end_program(result_csv_file, "result", 1)
