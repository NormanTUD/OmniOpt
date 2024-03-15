val_if_nothing_found = 99999999999999999999999999999999999999999999999999999999999
NO_RESULT = "{:.0e}".format(val_if_nothing_found)

ax_client = None
done_jobs = 0
failed_jobs = 0
jobs = []
end_program_ran = False
program_name = "OmniOpt"
current_run_folder = None
file_number = 0
folder_number = 0
args = None
result_csv_file = None
shown_end_table = False
max_eval = None
_time = None
mem_gb = None

class trainingDone (Exception):
    pass

try:
    from shutil import which
    import warnings
    import pandas as pd
    import random
    from pathlib import Path
    import glob
    import platform
    from os import listdir
    from os.path import isfile, join
    import re
    import socket
    import os
    import stat
    import pwd
    import base64
    import sys
    import argparse
    import time
    from pprint import pformat
    from pprint import pprint
except ModuleNotFoundError as e:
    print(f"Base modules could not be loaded: {e}")
    sys.exit(31)
except KeyboardInterrupt:
    print("You cancelled loading the basic modules")
    sys.exit(32)

try:
    Path("logs").mkdir(parents=True, exist_ok=True)
except:
    print("Could not create logs")

log_i = 0
logfile = f'logs/{log_i}'
logfile_nr_workers = f'logs/{log_i}_nr_workers'
while os.path.exists(logfile):
    log_i = log_i + 1
    logfile = f'logs/{log_i}'

logfile_nr_workers = f'logs/{log_i}_nr_workers'

def _debug (msg):
    try:
        with open(logfile, 'a') as f:
            print(msg, file=f)
    except:
        print("Error trying to write log file")

class REMatcher(object):
    def __init__(self, matchstring):
        self.matchstring = matchstring

    def match(self,regexp):
        self.rematch = re.match(regexp, self.matchstring)
        return bool(self.rematch)

    def group(self,i):
        return self.rematch.group(i)

def dier (msg):
    pprint(msg)
    sys.exit(10)

parser = argparse.ArgumentParser(
    prog="main",
    description='A hyperparameter optimizer for the HPC-system of the TU Dresden',
    epilog="Example:\n\n./main --partition=alpha --experiment_name=neural_network --mem_gb=1 --time=60 --worker_timeout=60 --max_eval=500 --num_parallel_jobs=500 --gpus=0 --follow --run_program=bHMgJyUoYXNkYXNkKSc= --parameter epochs range 0 10 int --parameter epochs range 0 10 int"
)

required = parser.add_argument_group('Required arguments', "These options have to be set")
required_but_choice = parser.add_argument_group('Required arguments that allow a choice', "Of these arguments, one has to be set to continue.")
optional = parser.add_argument_group('Optional', "These options are optional")
bash = parser.add_argument_group('Bash', "These options are for the main worker bash script, not the python script itself")
debug = parser.add_argument_group('Debug', "These options are mainly useful for debugging")

required.add_argument('--num_parallel_jobs', help='Number of parallel slurm jobs', type=int, required=True)
required.add_argument('--max_eval', help='Maximum number of evaluations', type=int)
required.add_argument('--worker_timeout', help='Timeout for slurm jobs (i.e. for each single point to be optimized)', type=int, required=True)
required.add_argument('--run_program', action='append', nargs='+', help='A program that should be run. Use, for example, $x for the parameter named x.', type=str)
required.add_argument('--experiment_name', help='Name of the experiment.', type=str)
required.add_argument('--mem_gb', help='Amount of RAM for each worker in GB (default: 1GB)', type=float, default=1)

required_but_choice.add_argument('--parameter', action='append', nargs='+', help="Experiment parameters in the formats (options in round brackets are optional): <NAME> range <LOWER BOUND> <UPPER BOUND> (<INT, FLOAT>) -- OR -- <NAME> fixed <VALUE> -- OR -- <NAME> choice <Comma-seperated list of values>", default=None)
required_but_choice.add_argument('--continue_previous_job', help="Continue from a previous checkpoint", type=str, default=None)

optional.add_argument('--cpus_per_task', help='CPUs per task', type=int, default=1)
optional.add_argument('--gpus', help='Number of GPUs', type=int, default=0)
optional.add_argument('--maximize', help='Maximize instead of minimize (which is default)', action='store_true', default=False)
optional.add_argument('--experiment_constraints', action="append", nargs="+", help='Constraints for parameters. Example: x + y <= 2.0', type=str)
optional.add_argument('--stderr_to_stdout', help='Redirect stderr to stdout for subjobs', action='store_true', default=False)
optional.add_argument('--run_dir', help='Directory, in which runs should be saved. Default: runs', default="runs", type=str)

bash.add_argument('--time', help='Time for the main job', default="", type=str)
bash.add_argument('--follow', help='Automatically follow log file of sbatch', action='store_true', default=False)
bash.add_argument('--partition', help='Partition to be run on', default="", type=str)
bash.add_argument('--reservation', help='Reservation', default="", type=str)

debug.add_argument('--verbose', help='Verbose logging', action='store_true', default=False)
debug.add_argument('--debug', help='Enable debugging', action='store_true', default=False)
debug.add_argument('--wait_until_ended', help='Wait until the program has ended', action='store_true', default=False)
debug.add_argument('--no_sleep', help='Disables sleeping for fast job generation (not to be used on HPC)', action='store_true', default=False)
debug.add_argument('--tests', help='Run simple internal tests', action='store_true', default=False)
debug.add_argument('--evaluate_to_random_value', help='Evaluate to random values', action='store_true', default=False)

args = parser.parse_args()

if args.follow and args.wait_until_ended:
    print("--follow and --wait_until_ended at the same time. May not be what you want.")

def decode_if_base64(input_str):
    try:
        decoded_bytes = base64.b64decode(input_str)
        decoded_str = decoded_bytes.decode('utf-8')
        return decoded_str
    except Exception as e:
        return input_str

def get_file_as_string (f):
    datafile = ""
    if not os.path.exists(f):
        print_color("red", f"{f} not found!")
    else:
        with open(f) as f:
            datafile = f.readlines()

    return "\n".join(datafile)

joined_run_program = ""
if not args.continue_previous_job:
    joined_run_program = " ".join(args.run_program[0])
    joined_run_program = decode_if_base64(joined_run_program)
else:
    prev_job_folder = args.continue_previous_job
    prev_job_file = prev_job_folder + "/joined_run_program"
    if os.path.exists(prev_job_file):
        joined_run_program = get_file_as_string(prev_job_file)
    else:
        print(f"{prev_job_file} could not be found")
        sys.exit(44)

experiment_name = args.experiment_name

if not args.tests:
    if args.parameter is None and args.continue_previous_job is None:
        print("Either --parameter or --continue_previous_job is required. Both were not found.")
        sys.exit(19)
    elif args.parameter is not None and args.continue_previous_job is not None:
        print("You cannot use --parameter and --continue_previous_job. You have to decide for one.");
        sys.exit(20)
    elif not args.run_program and not args.continue_previous_job:
        print("--run_program needs to be defined when --continue_previous_job is not set")
        sys.exit(42)
    elif not experiment_name and not args.continue_previous_job:
        print("--experiment_name needs to be defined when --continue_previous_job is not set")
        sys.exit(43)
    elif args.continue_previous_job:
        if not os.path.exists(args.continue_previous_job):
            print("red", f"{args.continue_previous_job} could not be found!")
            sys.exit(21)

        if not experiment_name:
            exp_name_file = f"{args.continue_previous_job}/experiment_name"
            if os.path.exists(exp_name_file):
                experiment_name = get_file_as_string(exp_name_file).strip()
            else:
                print(f"{exp_name_file} not found, and no --experiment_name given. Cannot continue.")
                sys.exit(46)

    if not args.mem_gb:
        print(f"--mem_gb needs to be set")
        sys.exit(48)

    if not args.time:
        if not args.continue_previous_job:
            print(f"--time needs to be set")
        else:
            time_file = args.continue_previous_job + "/time"
            if os.path.exists(time_file):
                time_file_contents = get_file_as_string(time_file).strip()
                if time_file_contents.isdigit():
                    _time = int(time_file_contents)
                    print(f"Using old run's --time: {_time}")
                else:
                    print(f"Time-setting: The contents of {time_file} do not contain a single number")
            else:
                print(f"neither --time nor file {time_file} found")
                sys.exit(1)
    else:
        _time = args.time

    if not args.mem_gb:
        if not args.continue_previous_job:
            print(f"--mem_gb needs to be set")
        else:
            mem_gb_file = args.continue_previous_job + "/mem_gb"
            if os.path.exists(mem_gb_file):
                mem_gb_file_contents = get_file_as_string(mem_gb_file).strip()
                if mem_gb_file_contents.isdigit():
                    mem_gb = int(mem_gb_file_contents)
                    print(f"Using old run's --mem_gb: {mem_gb}")
                else:
                    print(f"mem_gb-setting: The contents of {mem_gb_file} do not contain a single number")
            else:
                print(f"neither --mem_gb nor file {mem_gb_file} found")
                sys.exit(1)
    else:
        mem_gb = int(args.mem_gb)

    if args.continue_previous_job and not args.gpus:
        gpus_file = args.continue_previous_job + "/gpus"
        if os.path.exists(gpus_file):
            gpus_file_contents = get_file_as_string(gpus_file).strip()
            if gpus_file_contents.isdigit():
                gpus = int(gpus_file_contents)
                print(f"Using old run's --gpus: {gpus}")
            else:
                print(f"gpus-setting: The contents of {gpus_file} do not contain a single number")
        else:
            print(f"neither --gpus nor file {gpus_file} found")
            sys.exit(1)
    else:
        max_eval = args.max_eval

    if not args.max_eval:
        if not args.continue_previous_job:
            print(f"--max_eval needs to be set")
        else:
            max_eval_file = args.continue_previous_job + "/max_eval"
            if os.path.exists(max_eval_file):
                max_eval_file_contents = get_file_as_string(max_eval_file).strip()
                if max_eval_file_contents.isdigit():
                    max_eval = int(max_eval_file_contents)
                    print(f"Using old run's --max_eval: {max_eval}")
                else:
                    print(f"max_eval-setting: The contents of {max_eval_file} do not contain a single number")
            else:
                print(f"neither --max_eval nor file {max_eval_file} found")
                sys.exit(1)
    else:
        max_eval = args.max_eval

    if max_eval <= 0:
        print_color("red", "--max_eval must be larger than 0")
        sys.exit(39)

def print_debug (msg):
    if args.debug:
        print(msg)

    _debug(msg)

try:
    import os
    import socket
    import json
    import signal
    from tqdm import tqdm
except ModuleNotFoundError as e:
    print(f"Error loading module: {e}")
    sys.exit(24)

class signalUSR (Exception):
    pass

class signalINT (Exception):
    pass

def receive_usr_signal_one (signum, stack):
    raise signalUSR(f"USR1-signal received ({signum})")

def receive_usr_signal_int (signum, stack):
    raise signalINT(f"INT-signal received ({signum})")

signal.signal(signal.SIGUSR1, receive_usr_signal_one)
signal.signal(signal.SIGUSR2, receive_usr_signal_one)
signal.signal(signal.SIGINT, receive_usr_signal_int)
signal.signal(signal.SIGTERM, receive_usr_signal_int)
signal.signal(signal.SIGQUIT, receive_usr_signal_int)

script_dir = os.path.dirname(os.path.realpath(__file__))
helpers_file = f"{script_dir}/.helpers.py"
import importlib.util
spec = importlib.util.spec_from_file_location(
    name="helpers",
    location=helpers_file,
)
my_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(my_module)

try:
    from rich.console import Console
    console = Console(force_terminal=True, force_interactive=True)
    with console.status("[bold green]Importing rich, time, csv, re, argparse, subprocess and logging...") as status:
        #from rich.traceback import install
        #install(show_locals=True)

        from rich.table import Table
        from rich import print
        from rich.progress import Progress

        import time
        import csv
        import argparse
        from rich.pretty import pprint
        import subprocess

        import logging
        import warnings
        logging.basicConfig(level=logging.ERROR)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
except ModuleNotFoundError as e:
    print(f"Error: {e}")
    sys.exit(20)
except (signalUSR, signalINT, KeyboardInterrupt) as e:
    print("\n:warning: You pressed CTRL+C or signal was sent. Program execution halted.")
    sys.exit(0)

def print_color (color, text):
    print(f"[{color}]{text}[/{color}]")

def is_executable_in_path(executable_name):
    print_debug("is_executable_in_path")
    for path in os.environ.get('PATH', '').split(':'):
        executable_path = os.path.join(path, executable_name)
        if os.path.exists(executable_path) and os.access(executable_path, os.X_OK):
            return True
    return False

def check_slurm_job_id():
    print_debug("check_slurm_job_id")
    if is_executable_in_path('sbatch'):
        slurm_job_id = os.environ.get('SLURM_JOB_ID')
        if slurm_job_id is not None and not slurm_job_id.isdigit():
            print_color("red", "Not a valid SLURM_JOB_ID.")
        elif slurm_job_id is None:
            print_color("red", "You are on a system that has SLURM available, but you are not running the main-script in a Slurm-Environment. " +
                "This may cause the system to slow down for all other users. It is recommended uou run the main script in a Slurm job."
            )

def create_folder_and_file (folder, extension):
    print_debug("create_folder_and_file")
    global file_number

    if not os.path.exists(folder):
        os.makedirs(folder)

    while True:
        filename = os.path.join(folder, f"{file_number}.{extension}")

        if not os.path.exists(filename):
            with open(filename, 'w') as file:
                pass
            return filename

        file_number += 1

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


def looks_like_int(x):
    print_debug("looks_like_int")
    if isinstance(x, int):
        return True
    elif isinstance(x, float):
        return x.is_integer()
    elif isinstance(x, str):
        return bool(re.match(r'^\d+$', x))
    else:
        return False

def get_program_code_from_out_file (f):
    print_debug("get_program_code_from_out_file")
    if not os.path.exists(f):
        print(f"{f} not found")
    else:
        fs = get_file_as_string(f)

        for line in fs.split("\n"):
            if "Program-Code:" in line:
                return line

def parse_experiment_parameters(args):
    print_debug("parse_experiment_parameters")
    params = []

    param_names = []

    i = 0

    while i < len(args):
        this_args = args[i]
        j = 0
        while j < len(this_args):
            name = this_args[j]

            invalid_names = ["start_time", "end_time", "run_time", "program_string", "result", "exit_code", "signal"]

            if name in invalid_names:
                print_color("red", f"\n:warning: Name for argument no. {j} is invalid: {name}. Invalid names are: {', '.join(invalid_names)}")
                sys.exit(18)

            if name in param_names:
                print_color("red", f"\n:warning: Parameter name '{name}' is not unique. Names for parameters must be unique!")
                sys.exit(1)

            param_names.append(name)

            param_type = this_args[j + 1]

            valid_types = ["range", "fixed", "choice"]

            if param_type not in valid_types:
                valid_types_string = ', '.join(valid_types)
                print_color("red", f"\n:warning: Invalid type {param_type}, valid types are: {valid_types_string}")
                sys.exit(3)

            if param_type == "range":
                if len(this_args) != 5 and len(this_args) != 4:
                    print_color("red", f"\n:warning: --parameter for type range must have 4 (or 5, the last one being optional and float by default) parameters: <NAME> range <START> <END> (<TYPE (int or float)>)");
                    sys.exit(9)

                try:
                    lower_bound = float(this_args[j + 2])
                except:
                    print_color("red", f"\n:warning: {this_args[j + 2]} is not a number")
                    sys.exit(4)

                try:
                    upper_bound = float(this_args[j + 3])
                except:
                    print_color("red", f"\n:warning: {this_args[j + 3]} is not a number")
                    sys.exit(5)

                if upper_bound == lower_bound:
                    print_color("red", f"Lower bound and upper bound are equal: {lower_bound}")
                    sys.exit(13)

                if lower_bound > upper_bound:
                    print_color("yellow", f"Lower bound ({lower_bound}) was larger than upper bound ({upper_bound}) for parameter '{name}'. Switched them.")
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
                    print_color("red", f"\n:warning: {value_type} is not a valid value type. Valid types for range are: {valid_value_types_string}")
                    sys.exit(8)

                if value_type == "int":
                    if not looks_like_int(lower_bound):
                        print_color("red", f"\n:warning: {value_type} can only contain integers. You chose {lower_bound}")
                        sys.exit(37)

                    if not looks_like_int(upper_bound):
                        print_color("red", f"\n:warning: {value_type} can only contain integers. You chose {upper_bound}")
                        sys.exit(38)


                param = {
                    "name": name,
                    "type": param_type,
                    "bounds": [lower_bound, upper_bound],
                    "value_type": value_type
                }

                params.append(param)

                j += skip
            elif param_type == "fixed":
                if len(this_args) != 3:
                    print_color("red", f"\n:warning: --parameter for type fixed must have 3 parameters: <NAME> range <VALUE>");
                    sys.exit(11)

                value = this_args[j + 2]

                value = value.replace('\r', ' ').replace('\n', ' ')

                param = {
                    "name": name,
                    "type": "fixed",
                    "value": value
                }

                params.append(param)

                j += 3
            elif param_type == "choice":
                if len(this_args) != 3:
                    print_color("red", f"\n:warning: --parameter for type choice must have 3 parameters: <NAME> choice <VALUE,VALUE,VALUE,...>");
                    sys.exit(11)

                values = re.split(r'\s*,\s*', str(this_args[j + 2]))

                values[:] = [x for x in values if x != ""]

                values = sort_numerically_or_alphabetically(values)

                param = {
                    "name": name,
                    "type": "choice",
                    "is_ordered": True,
                    "values": values
                }

                params.append(param)

                j += 3
            else:
                print_color("red", f"\n:warning: Parameter type {param_type} not yet implemented.");
                sys.exit(14)
        i += 1

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
        print_color("red", f"\n:warning: Error: {e}")
        return None

def execute_bash_code(code):
    print_debug("execute_bash_code")
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

def get_result (input_string):
    print_debug("get_result")
    if input_string is None:
        print("Input-String is None")
        return None

    if not isinstance(input_string, str):
        print(f"Type of input_string is not string, but {type(input_string)}")
        return None

    try:
        pattern = r'RESULT:\s*(-?\d+(?:\.\d+)?)'

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
    print_debug("add_to_csv")
    is_empty = os.path.getsize(file_path) == 0 if os.path.exists(file_path) else True

    with open(file_path, 'a', newline='') as file:
        csv_writer = csv.writer(file)

        if is_empty:
            csv_writer.writerow(heading)

        # desc += " (best loss: " + '{:f}'.format(best_result) + ")"
        data_line = ["{:.20f}".format(x) if type(x) == int or type(x) == float else x for x in data_line]
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
    print_debug("find_file_paths")
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

def find_file_paths_and_print_infos (_text, program_code):
    print_debug("find_file_paths_and_print_infos")
    file_paths = find_file_paths(_text)

    string = "";

    string += "\n========\nDEBUG INFOS START:\n"

    if not args.tests:
        print("Program-Code: " + program_code)
    if file_paths:
        for file_path in file_paths:
            string += "\n"
            string += check_file_info(file_path)
    string += "\n========\nDEBUG INFOS END\n"

    return string

def evaluate(parameters):
    if args.evaluate_to_random_value:
        rand_res = random.uniform(0, 1)
        return {"result": float(rand_res)}

    print_debug(f"evaluate with parameters {parameters}")
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    signal.signal(signal.SIGUSR2, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGQUIT, signal.SIG_IGN)

    return_in_case_of_error = {"result": val_if_nothing_found}

    if args.maximize:
        return_in_case_of_error = {"result": -val_if_nothing_found}

    try:
        print("parameters:", parameters)

        parameters_keys = list(parameters.keys())
        parameters_values = list(parameters.values())

        program_string_with_params = replace_parameters_in_string(parameters, joined_run_program)

        program_string_with_params = program_string_with_params.replace('\r', ' ').replace('\n', ' ')

        string = find_file_paths_and_print_infos(program_string_with_params, program_string_with_params)

        print("Debug-Infos:", string)

        print_color("green", program_string_with_params)

        start_time = int(time.time())

        stdout_stderr_exit_code_signal = execute_bash_code(program_string_with_params)

        end_time = int(time.time())

        stdout = stdout_stderr_exit_code_signal[0]
        stderr = stdout_stderr_exit_code_signal[1]
        exit_code = stdout_stderr_exit_code_signal[2]
        _signal = stdout_stderr_exit_code_signal[3]

        run_time = end_time - start_time

        print("stdout:")
        print(stdout)

        result = get_result(stdout)

        print(f"Result: {result}")

        headline = ["start_time", "end_time", "run_time", "program_string", *parameters_keys, "result", "exit_code", "signal", "hostname"];
        values = [start_time, end_time, run_time, program_string_with_params,  *parameters_values, result, exit_code, _signal, socket.gethostname()];

        headline = ['None' if element is None else element for element in headline]
        values = ['None' if element is None else element for element in values]

        add_to_csv(result_csv_file, headline, values)

        if type(result) == int:
            return {"result": int(result)}
        elif type(result) == float:
            return {"result": float(result)}
        else:
            return return_in_case_of_error
    except signalUSR:
        print("\n:warning: USR1-Signal was sent. Cancelling evaluation.")
        return return_in_case_of_error
    except signalINT:
        print("\n:warning: INT-Signal was sent. Cancelling evaluation.")
        return return_in_case_of_error

try:
    if not args.tests:
        with console.status("[bold green]Importing ax...") as status:
            try:
                import ax
                from ax.service.ax_client import AxClient, ObjectiveProperties
                import ax.exceptions.core
                from ax.modelbridge.dispatch_utils import choose_generation_strategy
                from ax.storage.json_store.save import save_experiment
                from ax.service.utils.report_utils import exp_to_df
            except ModuleNotFoundError as e:
                print_color("red", "\n:warning: ax could not be loaded. Did you create and load the virtual environment properly?")
                sys.exit(33)
            except KeyboardInterrupt:
                print_color("red", "\n:warning: You pressed CTRL+C. Program execution halted.")
                sys.exit(34)

        with console.status("[bold green]Importing botorch...") as status:
            try:
                import botorch
            except ModuleNotFoundError as e:
                print_color("red", "\n:warning: ax could not be loaded. Did you create and load the virtual environment properly?")
                sys.exit(35)
            except KeyboardInterrupt:
                print_color("red", "\n:warning: You pressed CTRL+C. Program execution halted.")
                sys.exit(36)

        with console.status("[bold green]Importing submitit...") as status:
            try:
                import submitit
                from submitit import AutoExecutor, LocalJob, DebugJob
            except:
                print_color("red", "\n:warning: submitit could not be loaded. Did you create and load the virtual environment properly?")
                sys.exit(7)
except (signalUSR, signalINT, KeyboardInterrupt) as e:
    print("\n:warning: signal was sent or CTRL-c pressed. Cancelling loading ax. Stopped loading program.")
    sys.exit(0)

def disable_logging ():
    print_debug("disable_logging")
    logging.basicConfig(level=logging.ERROR)

    logging.getLogger("ax").setLevel(logging.ERROR)
    logging.getLogger("ax.modelbridge").setLevel(logging.ERROR)
    logging.getLogger("ax.modelbridge.torch").setLevel(logging.ERROR)
    logging.getLogger("ax.models.torch.botorch_modular.acquisition").setLevel(logging.ERROR)
    logging.getLogger("ax.modelbridge.transforms.standardize_y").setLevel(logging.ERROR)
    logging.getLogger("ax.modelbridge.torch").setLevel(logging.ERROR)
    logging.getLogger("ax.models.torch.botorch_modular.acquisition").setLevel(logging.ERROR)
    logging.getLogger("ax.service.utils.instantiation").setLevel(logging.ERROR)
    logging.getLogger("ax.modelbridge.dispatch_utils").setLevel(logging.ERROR)
    logging.getLogger("ax.modelbridge.cross_validation").setLevel(logging.ERROR)

    warnings.filterwarnings("ignore", category=Warning, module="ax.modelbridge.dispatch_utils")
    warnings.filterwarnings("ignore", category=Warning, module="ax.service.utils.instantiation")

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="botorch.optim.optimize")
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="linear_operator.utils.cholesky")
    warnings.filterwarnings("ignore", category=FutureWarning, module="ax.core.data")

    warnings.filterwarnings("ignore", category=UserWarning, module="ax.modelbridge.transforms.standardize_y")
    warnings.filterwarnings("ignore", category=UserWarning, module="botorch.models.utils.assorted")
    warnings.filterwarnings("ignore", category=UserWarning, module="ax.modelbridge.torch")
    warnings.filterwarnings("ignore", category=UserWarning, module="ax.models.torch.botorch_modular.acquisition")
    warnings.filterwarnings("ignore", category=UserWarning, module="ax.modelbridge.cross_validation")
    warnings.filterwarnings("ignore", category=Warning, module="ax.modelbridge.cross_validation")
    warnings.filterwarnings("ignore", category=Warning, module="ax.modelbridge")
    warnings.filterwarnings("ignore", category=Warning, module="ax")
    warnings.filterwarnings("ignore", category=UserWarning, module="ax.service.utils.best_point")
    warnings.filterwarnings("ignore", category=Warning, module="ax.service.utils.best_point")
    warnings.filterwarnings("ignore", category=UserWarning, module="ax.service.utils.report_utils")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.autograd")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.autograd.__init__")
    warnings.filterwarnings("ignore", category=UserWarning, module="botorch.optim.fit")
    warnings.filterwarnings("ignore", category=UserWarning, module="ax.core.parameter")
    warnings.filterwarnings("ignore", category=Warning, module="ax.modelbridge.transforms.int_to_float")
    warnings.filterwarnings("ignore", category=UserWarning, module="ax.modelbridge.transforms.int_to_float")

def show_end_table_and_save_end_files (csv_file_path, result_column):
    print_debug("show_end_table_and_save_end_files")

    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    signal.signal(signal.SIGUSR2, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGQUIT, signal.SIG_IGN)

    disable_logging()

    global ax_client
    global console
    global current_run_folder
    global shown_end_table

    if shown_end_table:
        print("End table already shown, not doing it again")
        return

    print_debug("[show_end_table_and_save_end_files] Ignoring warnings")
    warnings.filterwarnings("ignore", category=UserWarning, module="ax.service.utils.report_utils")

    print_debug("[show_end_table_and_save_end_files] Getting best params")

    exit = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            #best_parameters, (means, covariances) = ax_client.get_best_parameters() # TODO get_best_params nutzen

            best_params = get_best_params(csv_file_path, result_column)

            print_debug("[show_end_table_and_save_end_files] Got best params")
            best_result = best_params["result"]

            if str(best_result) == NO_RESULT or best_result is None or best_result == "None":
                table_str = "Best result could not be determined"
                print_color("red", table_str)
                exit = 1
            else:
                print_debug("[show_end_table_and_save_end_files] Creating table")
                table = Table(show_header=True, header_style="bold", title="Best parameter:")

                for key in best_params["parameters"].keys():
                    table.add_column(key)

                print_debug("[show_end_table_and_save_end_files] Add last column to table")
                table.add_column("result")

                print_debug("[show_end_table_and_save_end_files] Defining rows")
                row_without_result = [str(best_params["parameters"][key]) for key in best_params["parameters"].keys()];
                row = [*row_without_result, str(best_result)]

                print_debug("[show_end_table_and_save_end_files] Adding rows to table")
                table.add_row(*row)

                print_debug("[show_end_table_and_save_end_files] Printing table")
                console.print(table)

                print_debug("[show_end_table_and_save_end_files] Capturing table")

                with console.capture() as capture:
                    console.print(table)
                table_str = capture.get()

            print_debug("[show_end_table_and_save_end_files] Printing captured table to file")
            with open(f"{current_run_folder}/best_result.txt", "w") as text_file:
                text_file.write(table_str)

            print_debug("[show_end_table_and_save_end_files] Setting shown_end_table = true")
            shown_end_table = True
        except Exception as e:
            print(f"[show_end_table_and_save_end_files] Error during show_end_table_and_save_end_files: {e}")

    sys.exit(exit)

def end_program (csv_file_path, result_column="result"):
    print_debug("[end_program] end_program started")

    global current_run_folder
    analyze_out_files(current_run_folder)

    global end_program_ran

    if end_program_ran:
        print_debug("[end_program] end_program_ran was true. Returning.")
        return

    end_program_ran = True
    print_debug("[end_program] Setting end_program_ran = True")

    global ax_client
    global console

    exit = 0

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

        print_debug("[end_program] Calling show_end_table_and_save_end_files")
        exit = show_end_table_and_save_end_files (csv_file_path, result_column)
        print_debug("[end_program] show_end_table_and_save_end_files called")
    except (signalUSR, signalINT, KeyboardInterrupt) as e:
        print_color("red", "\n:warning: You pressed CTRL+C or a signal was sent. Program execution halted.")
        print("\n:warning: KeyboardInterrupt signal was sent. Ending program will still run.")
        print_debug("[end_program] Calling show_end_table_and_save_end_files (in KeyboardInterrupt)")
        exit = show_end_table_and_save_end_files (csv_file_path, result_column)
        print_debug("[end_program] show_end_table_and_save_end_files called (in KeyboardInterrupt)")
    except TypeError:
        print_color("red", "\n:warning: The program has been halted without attaining any results.")

    pd_csv = f'{current_run_folder}/pd.csv'
    print_debug(f"[end_program] Trying to save file to {pd_csv}")

    save_pd_csv()

    for job, trial_index in jobs[:]:
        job.cancel()

    sys.exit(exit)

def save_checkpoint ():
    print_debug("save_checkpoint")
    global current_run_folder
    global ax_client

    checkpoint_filepath = f"{current_run_folder}/checkpoint.json"
    ax_client.save_to_json_file(filepath=checkpoint_filepath)

    print_debug("Checkpoint saved")

def save_pd_csv ():
    print_debug("save_pd_csv")
    global current_run_folder
    global ax_client

    pd_csv = f'{current_run_folder}/pd.csv'
    try:
        logger = logging.getLogger()
        logger.setLevel(logging.ERROR)

        logger = logging.getLogger("ax")
        logger.setLevel(logging.ERROR)

        logger = logging.getLogger("ax.service")
        logger.setLevel(logging.ERROR)

        logger = logging.getLogger("ax.service.utils")
        logger.setLevel(logging.ERROR)

        logger = logging.getLogger("ax.service.utils.report_utils")
        logger.setLevel(logging.ERROR)

        pd_frame = ax_client.get_trials_data_frame()
        pd_frame.to_csv(pd_csv, index=False)
        print_debug("pd.csv saved")
    except Exception as e:
        print_color("red", f"While saving all trials as a pandas-dataframe-csv, an error occured: {e}")

def print_overview_table (experiment_parameters):
    if not experiment_parameters:
        print_color("red", "Cannot determine experiment_parameters. No parameter table will be shown.")
        return

    print_debug("print_overview_table")
    global args
    global current_run_folder

    if not experiment_parameters:
        print_color("red", "Experiment parameters could not be determined for display")

    min_or_max = "minimize"
    if args.maximize:
        min_or_max = "maximize"

    with open(f"{current_run_folder}/{min_or_max}", 'w') as f:
        print('The contents of this file do not matter. It is only relevant that it exists.', file=f)

    rows = []

    for param in experiment_parameters:
        _type = str(param["type"])
        if _type == "range":
            rows.append([str(param["name"]), _type, str(param["bounds"][0]), str(param["bounds"][1]), "", str(param["value_type"])])
        elif _type == "fixed":
            rows.append([str(param["name"]), _type, "", "", str(param["value"]), ""])
        elif _type == "choice":
            values = param["values"]
            values = [str(item) for item in values]

            rows.append([str(param["name"]), _type, "", "", ", ".join(values), ""])
        else:
            print_color("red", f"Type {_type} is not yet implemented in the overview table.");
            sys.exit(15)

    table = Table(header_style="bold", title="Experiment parameters:")
    columns = ["Name", "Type", "Lower bound", "Upper bound", "Value(s)", "Value-Type"]
    for column in columns:
        table.add_column(column)
    for row in rows:
        table.add_row(*row, style='bright_green')
    console.print(table)

    with console.capture() as capture:
        console.print(table)
    table_str = capture.get()

    with open(f"{current_run_folder}/parameters.txt", "w") as text_file:
        text_file.write(table_str)

def check_equation (variables, equation):
    print_debug("check_equation")
    if not (">=" in equation or "<=" in equation):
        return False

    comparer_in_middle = re.search("(^(<=|>=))|((<=|>=)$)", equation)
    if comparer_in_middle:
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
            print_color("red", f"constraint error: Invalid item {item}")
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

def finish_previous_jobs (progress_bar, jobs, result_csv_file, searching_for):
    print_debug("finish_previous_jobs")

    global ax_client
    global done_jobs
    global failed_jobs

    for job, trial_index in jobs[:]:
        # Poll if any jobs completed
        # Local and debug jobs don't run until .result() is called.
        if job.done() or type(job) in [LocalJob, DebugJob]:
            try:
                result = job.result()
                print_debug(f"Got job result: {result}")
                if result != val_if_nothing_found:
                    ax_client.complete_trial(trial_index=trial_index, raw_data=result)

                    done_jobs += 1
                else:
                    job.cancel()

                    failed_jobs += 1
            except submitit.core.utils.UncompletedJobError as error:
                print_color("red", str(error))

                job.cancel()

                failed_jobs += 1
            except ax.exceptions.core.UserInputError as error:
                if "None for metric" in str(error):
                    print_color("red", f"\n:warning: It seems like the program that was about to be run didn't have 'RESULT: <NUMBER>' in it's output string.\nError: {error}")
                else:
                    print_color("red", f"\n:warning: {error}")

                job.cancel()

                failed_jobs += 1

            jobs.remove((job, trial_index))

            progress_bar.update(1)

            save_checkpoint()
            save_pd_csv()

    progress_bar.set_description(get_desc_progress_bar(result_csv_file, searching_for))

def get_desc_progress_bar(result_csv_file, searching_for):
    global done_jobs
    global failed_jobs

    desc = f"Searching {searching_for}"
    
    in_brackets = []

    if failed_jobs:
        in_brackets.append(f"failed: {failed_jobs}")

    best_params = None

    if done_jobs:
        best_params = get_best_params(result_csv_file, "result")
        best_result = best_params["result"]

        if str(best_result) != NO_RESULT and best_result is not None:
            in_brackets.append('best result: {:f}'.format(float(best_result)))

    if len(in_brackets):
        desc += f" ({', '.join(in_brackets)})"

    return desc

def main ():
    print_debug("main")
    global args
    global file_number
    global folder_number
    global result_csv_file
    global current_run_folder
    global ax_client
    global jobs

    check_slurm_job_id()

    current_run_folder = f"{args.run_dir}/{experiment_name}/{folder_number}"
    while os.path.exists(f"{current_run_folder}"):
        current_run_folder = f"{args.run_dir}/{experiment_name}/{folder_number}"
        folder_number = folder_number + 1

    result_csv_file = create_folder_and_file(f"{current_run_folder}", "csv")

    with open(f'{current_run_folder}/joined_run_program', 'w') as f:
        print(joined_run_program, file=f)

    with open(f'{current_run_folder}/experiment_name', 'w') as f:
        print(experiment_name, file=f)

    with open(f'{current_run_folder}/mem_gb', 'w') as f:
        print(mem_gb, file=f)

    with open(f'{current_run_folder}/time', 'w') as f:
        print(time, file=f)

    with open(f'{current_run_folder}/max_eval', 'w') as f:
        print(max_eval, file=f)

    with open(f'{current_run_folder}/gpus', 'w') as f:
        print(args.gpus, file=f)

    with open(f'{current_run_folder}/time', 'w') as f:
        print(_time, file=f)

    with open(f"{current_run_folder}/env", 'a') as f:
        env = dict(os.environ)
        for key in env:
            print(str(key) + " = " + str(env[key]), file=f)

    with open(f"{current_run_folder}/run.sh", 'w') as f:
        print("bash run.sh '" + "' '".join(sys.argv[1:]) + "'", file=f)

    if args.continue_previous_job:
        print(f"[yellow]Continuation from {args.continue_previous_job}[/yellow]")
    print(f"[yellow]CSV-File[/yellow]: [underline]{result_csv_file}[/underline]")
    print_color("green", program_name)

    experiment_parameters = None
    checkpoint_filepath = f"{current_run_folder}/checkpoint.json.parameters.json"

    if args.parameter:
        experiment_parameters = parse_experiment_parameters(args.parameter)

        with open(checkpoint_filepath, "w") as outfile:
            json.dump(experiment_parameters, outfile)

    if not args.verbose:
        disable_logging()

    try:
        ax_client = AxClient(verbose_logging=args.verbose)

        minimize_or_maximize = not args.maximize

        experiment = None

        if args.continue_previous_job:
            print_debug(f"Load from checkpoint: {args.continue_previous_job}")

            checkpoint_file = args.continue_previous_job + "/checkpoint.json"
            if not os.path.exists(checkpoint_file):
                print_color("red", f"{checkpoint_file} not found")
                sys.exit(47)

            ax_client = (AxClient.load_from_json_file(checkpoint_file))

            checkpoint_params_file = args.continue_previous_job + "/checkpoint.json.parameters.json"

            if not os.path.exists(checkpoint_params_file):
                print_color(f"Cannot find {checkpoint_params_file}")
                sys.exit(49)

            f = open(checkpoint_params_file)
            experiment_parameters = json.load(f)
            f.close()

            with open(checkpoint_filepath, "w") as outfile:
                json.dump(experiment_parameters, outfile)

            if not os.path.exists(checkpoint_params_file):
                print_color("red", f"{checkpoint_params_file} not found. Cannot continue_previous_job without.")
                sys.exit(22)

            with open(f'{current_run_folder}/checkpoint_load_source', 'w') as f:
                print(f"Continuation from checkpoint {args.continue_previous_job}", file=f)
        else:
            experiment_args = {
                "name": experiment_name,
                "parameters": experiment_parameters,
                "objectives": {"result": ObjectiveProperties(minimize=minimize_or_maximize)},
                "choose_generation_strategy_kwargs": {
                    "num_trials": max_eval,
                    "num_initialization_trials": args.num_parallel_jobs,
                    "max_parallelism_override": args.num_parallel_jobs
                }
            }


            if args.experiment_constraints:
                constraints_string = " ".join(args.experiment_constraints[0])

                variables = [item['name'] for item in experiment_parameters]

                equation = check_equation(variables, constraints_string)

                if equation:
                    experiment_args["parameter_constraints"] = [constraints_string]
                    print_color("yellow", "--parameter_constraints is experimental!")
                else:
                    print_color("red", "Experiment constraints are invalid.")
                    sys.exit(28)

            try:
                experiment = ax_client.create_experiment(**experiment_args)
            except ValueError as error:
                print_color("red", f"An error has occured: {error}")
                sys.exit(29)

        print_overview_table(experiment_parameters)

        log_folder = f"{current_run_folder}/%j"
        executor = submitit.AutoExecutor(folder=log_folder)

        # 'nodes': <class 'int'>, 'gpus_per_node': <class 'int'>, 'tasks_per_node': <class 'int'>

        executor.update_parameters(
            name=experiment_name,
            timeout_min=args.worker_timeout,
            slurm_gres=f"gpu:{args.gpus}",
            cpus_per_task=args.cpus_per_task,
            stderr_to_stdout=args.stderr_to_stdout,
            mem_gb=args.mem_gb,
            slurm_signal_delay_s=30,
            slurm_use_srun=False
        )

        submitted_jobs = 0
        # Run until all the jobs have finished and our budget is used up.

        searching_for = "minimum"
        if args.maximize:
            searching_for = "maximum"

        _k = 0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with tqdm(total=max_eval, disable=False) as progress_bar:
                while submitted_jobs < max_eval or jobs:
                    log_nr_of_workers()

                    # Schedule new jobs if there is availablity
                    try:
                        new_jobs_needed = min(args.num_parallel_jobs - len(jobs), max_eval - submitted_jobs)
                        if done_jobs >= max_eval:
                            raise trainingDone("Training done")

                        calculated_max_trials = max(1, new_jobs_needed)

                        print_debug(f"Trying to get the next {calculated_max_trials} trials, one by one.")

                        for m in range(0, calculated_max_trials):
                            finish_previous_jobs(progress_bar, jobs, result_csv_file, searching_for)

                            try:
                                print_debug("Trying to get trial_index_to_param")
                                trial_index_to_param, _ = ax_client.get_next_trials(
                                    max_trials=1
                                )

                                print_debug(f"Got {len(trial_index_to_param.items())} new items (m = {m}, in range(0, {calculated_max_trials})).")

                                for trial_index, parameters in trial_index_to_param.items():
                                    new_job = None
                                    try:
                                        print_debug(f"Trying to start new job.")
                                        new_job = executor.submit(evaluate, parameters)
                                        print_debug(f"Increasing submitted_jobs by 1.")
                                        submitted_jobs += 1
                                        print_debug(f"Appending started job to jobs array")
                                        jobs.append((new_job, trial_index))
                                        if not args.no_sleep:
                                            print_debug(f"Sleeping one second before continuation")
                                            time.sleep(1)
                                        print_debug(f"Got new job and started it. Parameters: {parameters}")
                                    except submitit.core.utils.FailedJobError as error:
                                        if "QOSMinGRES" in str(error) and args.gpus == 0:
                                            print_color("red", f"\n:warning: It seems like, on the chosen partition, you need at least one GPU. Use --gpus=1 (or more) as parameter.")
                                        else:
                                            print_color("red", f"\n:warning: FAILED: {error}")

                                        try:
                                            print_debug("Trying to cancel job that failed")
                                            new_job.cancel()
                                            print_debug("Cancelled failed job")

                                            jobs.remove((new_job, trial_index))
                                            print_debug("Removed failed job")

                                            progress_bar.update(1)

                                            save_checkpoint()
                                            save_pd_csv()
                                        except Exception as e:
                                            print_color("red", f"\n:warning: Cancelling failed job FAILED: {e}")
                                    except (signalUSR, signalINT) as e:
                                        print_color("red", f"\n:warning: Detected signal. Will exit.")
                                        end_program(result_csv_file)
                                    except Exception as e:
                                        print_color("red", f"\n:warning: Starting job failed with error: {e}")
                                    except Exception as e:
                                        print_color("red", f"\n:warning: Starting job failed with error: {e}")
                            except RuntimeError as e:
                                print_color("red", "\n:warning: " + str(e))
                            except botorch.exceptions.errors.ModelFittingError as e:
                                print_color("red", "\n:warning: " + str(e))
                                end_program(result_csv_file)
                    except botorch.exceptions.errors.InputDataError as e:
                        print_color("red", f"Error: {e}")
                    except ax.exceptions.core.DataRequiredError:
                        print_color("red", f"Error: {e}")

                    if not args.no_sleep:
                        time.sleep(0.1)
        end_program(result_csv_file)
    except trainingDone as e:
        end_program(result_csv_file)
    except (signalUSR, signalINT, KeyboardInterrupt) as e:
        print_color("red", "\n:warning: You pressed CTRL+C or got a signal. Optimization stopped.")
        end_program(result_csv_file)

def _unidiff_output(expected, actual):
    """
    Helper function. Returns a string containing the unified diff of two multiline strings.
    """

    import difflib
    expected = expected.splitlines(1)
    actual = actual.splitlines(1)

    diff=difflib.unified_diff(expected, actual)

    return ''.join(diff)

def print_diff (o, i):
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

def is_equal (n, i, o):
    r = _is_equal(n, i, o)

    if r:
        print_diff(i, o)

    return r

def is_not_equal (n, i, o):
    r = _is_not_equal(n, i, o)

    if r:
        print_diff(i, o)

    return r

def _is_not_equal (name, input, output):
    if type(input) == str or type(input) == int or type(input) == float:
        if input == output:
            print_color("red", f"Failed test: {name}")
            return 1
    elif type(input) == bool:
        if input == output:
            print_color("red", f"Failed test: {name}")
            return 1
    elif output is None or input is None:
        if input == output:
            print_color("red", f"Failed test: {name}")
            return 1
    else:
        print_color("red", f"Unknown data type for test {name}")
        sys.exit(100)

    print_color("green", f"Test OK: {name}")
    return 0

def _is_equal (name, input, output):
    if type(input) != type(output):
        print_color("red", f"Failed test: {name}")
        return 1
    elif type(input) == str or type(input) == int or type(input) == float:
        if input != output:
            print_color("red", f"Failed test: {name}")
            return 1
    elif type(input) == bool:
        if input != output:
            print_color("red", f"Failed test: {name}")
            return 1
    elif output is None or input is None:
        if input != output:
            print_color("red", f"Failed test: {name}")
            return 1
    else:
        print_color("red", f"Unknown data type for test {name}")
        sys.exit(100)

    print_color("green", f"Test OK: {name}")
    return 0

def get_shebang (path):
    sb = None
    with open(path) as f:
        first_line = f.readline()

        if first_line.startswith("#!"):
            sb = first_line.replace("#!", "").strip()

            sb = re.sub(
                r".*/",
                "",
                sb
            )

    return sb

def complex_tests (program_name, wanted_stderr, wanted_exit_code, wanted_signal, res_is_none=False):
    print_color("yellow", f"Test suite: {program_name}")

    nr_errors = 0

    program_path = f"./.tests/test_wronggoing_stuff.bin/bin/{program_name}"

    if not os.path.exists(program_path):
        print_color("red", f"Program path {program_path} not found!")
        sys.exit(99)

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

    nr_errors += is_equal(f"replace_parameters_in_string {program_name}", program_string_with_params, f"{program_path_with_program} 1 2 3 45")

    stdout_stderr_exit_code_signal = execute_bash_code(program_string_with_params)

    stdout = stdout_stderr_exit_code_signal[0]
    stderr = stdout_stderr_exit_code_signal[1]
    exit_code = stdout_stderr_exit_code_signal[2]
    signal = stdout_stderr_exit_code_signal[3]

    res = get_result(stdout)

    if res_is_none:
        nr_errors += is_equal(f"{program_name} res is None", None, res)
    else:
        nr_errors += is_equal(f"{program_name} res type is nr", True, type(res) == int or type(res) == float)
    nr_errors += is_equal(f"{program_name} stderr", True, wanted_stderr in stderr)
    nr_errors += is_equal(f"{program_name} exit-code ", exit_code, wanted_exit_code)
    nr_errors += is_equal(f"{program_name} signal", signal, wanted_signal)

    return nr_errors

def get_files_in_dir (mypath):
    print_debug("get_files_in_dir")
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    return [mypath + "/" + s for s in onlyfiles]

def test_find_paths (program_code):
    nr_errors = 0

    files = [
            "main",
            ".main.py",
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

def run_tests ():
    nr_errors = 0

    nr_errors += is_not_equal("nr equal string", 1, "1")
    nr_errors += is_equal("nr equal nr", 1, 1)
    nr_errors += is_not_equal("unequal strings", "hallo", "welt")

    #complex_tests (program_name, wanted_stderr, wanted_exit_code, wanted_signal, res_is_none=False):
    nr_errors += complex_tests("simple_ok", "hallo", 0, None)
    nr_errors += complex_tests("divide_by_0", 'Illegal division by zero at ./.tests/test_wronggoing_stuff.bin/bin/divide_by_0 line 3.\n', 255, None, True)
    nr_errors += complex_tests("result_but_exit_code_stdout_stderr", "stderr", 5, None)
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

    sys.exit(nr_errors)

def file_contains_text(f, t):
    print_debug("file_contains_text")
    datafile = get_file_as_string(f)

    found = False
    for line in datafile:
        if t in line:
            return True
    return False

def get_first_line_of_file_that_contains_string (i, s):
    print_debug("get_first_line_of_file_that_contains_string")
    if not os.path.exists(i):
        print(f"File {i} not found")
        return

    f = get_file_as_string(i)

    for line in f.split("\n"):
        if s in line:
            return line
    return None

def get_errors_from_outfile (i):
    print_debug("get_errors_from_outfile")
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
                print_color(f"Wrong type, should be list or string, is {type(err)}")
                sys.exit(41)

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
            search_for_exit_code = "Exit-Code: " + str(r) + ","
            if search_for_exit_code in file_as_string:
                errors.append("Non-zero exit-code detected: " + str(r))

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
            ["error: unrecognized arguments", "Wrong argparse argument"]
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

def analyze_out_files (rootdir):
    print_debug("analyze_out_files")
    outfiles = glob.glob(f'{rootdir}/**/*.out', recursive=True)

    j = 0

    for i in outfiles:
        errors = get_errors_from_outfile(i)

        if len(errors):
            if j == 0:
                print("")
            print_color("yellow", f"Out file {i} contains potential errors:")
            program_code = get_program_code_from_out_file(i)
            print(program_code)
            for e in errors:
                print_color("red", f"- {e}")

            print("")

            j = j + 1

def is_tool(name):
    """Check whether `name` is on PATH and marked as executable."""

    return which(name) is not None

def get_current_workers ():
    if not is_tool("squeue"):
        return {}

    if not 'SLURM_JOB_ID' in os.environ:
        print_debug("Not running inside a slurm job. Returning empty list.")
        return {}

    command = 'squeue --me --format "%.20i %.100j" --noheader | sed -e "s#^\s*##g" -e "s#\s\s*#  #g"'

    current_jobs = {}
    output = execute_bash_code(command)[0]

    lines = output.split("\n")

    lines = list(filter(lambda item: item, lines))

    main_job_name = ""
    main_job_id = ""

    for line in lines:
        splitted = line.split("  ")
        job_id = splitted[0]
        job_name = splitted[1]

        main_job_name = job_name
        main_job_id = job_id

    if not main_job_id:
        print_debug("Could not determine current job id. Returning empty list.")
        return {}

    if not main_job_name:
        print_debug("Could not determine current job name. Returning empty list.")
        return {}

    for line in lines:
        splitted = line.split("  ")
        job_id = splitted[0]
        job_name = splitted[1]

        if job_name == main_job_name and job_id != main_job_id:
            current_jobs[job_id] = splitted[1]

    return current_jobs

def get_number_of_current_workers ():
    workers = get_current_workers()

    return len(workers.keys())

def log_nr_of_workers ():
    last_line = ""
    nr_of_workers = get_number_of_current_workers()

    if not nr_of_workers:
        return

    if os.path.exists(logfile_nr_workers):
        with open(logfile_nr_workers, 'r') as f:
            for line in f:
                pass
            last_line = line.strip()

    if (last_line.isnumeric() or last_line == "") and str(last_line) != str(nr_of_workers):
        with open(logfile_nr_workers, 'a+') as f:
            f.write(str(nr_of_workers) + "\n")

def get_best_params(csv_file_path, result_column):
    results = {
        result_column: None,
        "parameters": {}
    }

    if not os.path.exists(csv_file_path):
        return results

    df = None

    try:
        df = pd.read_csv(csv_file_path, index_col=0)
        df.dropna(subset=[result_column], inplace=True)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
        return results

    cols = df.columns.tolist()
    nparray = df.to_numpy()

    best_line = None

    result_idx = cols.index(result_column)

    best_result = None

    for i in range(0, len(nparray)):
        this_line = nparray[i]
        this_line_result = this_line[result_idx]

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

if __name__ == "__main__":
    with warnings.catch_warnings():
        if args.tests:
            #dier(get_best_params("runs/test_wronggoing_stuff/4/0.csv", "result"))
            #dier(add_to_csv("x.csv", ["hallo", "welt"], [1, 0.0000001, "hallo welt"]))

            run_tests()
        else:
            warnings.simplefilter("ignore")

            main()
