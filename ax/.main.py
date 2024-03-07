ax_client = None
jobs = []
end_program_ran = False
program_name = "OmniOpt"
current_run_folder = None
file_number = 0
folder_number = 0
args = None
result_csv_file = None
shown_end_table = False

try:
    import socket
    import os
    import stat
    import pwd
    import base64
    import re
    import sys
    import argparse
    import time
    from pprint import pprint
except ModuleNotFoundError as e:
    print(f"Base modules could not be loaded: {e}")
    sys.exit(31)
except KeyboardInterrupt:
    print("You cancelled loading the basic modules")
    sys.exit(32)

def dier (msg):
    pprint(msg)
    sys.exit(10)

parser = argparse.ArgumentParser(
    prog=program_name,
    description='A hyperparameter optimizer for the HPC-system of the TU Dresden',
    epilog="Example:\n\n./main --partition=alpha --experiment_name=asd --mem_gb=1 --time=60 --worker_timeout=60 --max_eval=500 --num_parallel_jobs=500 --gpus=0 --follow --run_program=bHMgJyUoYXNkYXNkKSc= --parameter asdasd range 0 10 float"
)

required = parser.add_argument_group('Required arguments', "These options have to be set")
required_but_choice = parser.add_argument_group('Required arguments that allow a choice', "Of these arguments, one has to be set to continue.")
optional = parser.add_argument_group('Optional', "These options are optional")
bash = parser.add_argument_group('Bash', "These options are for the main worker bash script, not the python script itself")
debug = parser.add_argument_group('Debug', "These options are mainly useful for debugging")

required.add_argument('--num_parallel_jobs', help='Number of parallel slurm jobs', type=int, required=True)
required.add_argument('--max_eval', help='Maximum number of evaluations', type=int, required=True)
required.add_argument('--worker_timeout', help='Timeout for slurm jobs (i.e. for each single point to be optimized)', type=int, required=True)
required.add_argument('--run_program', action='append', nargs='+', help='A program that should be run. Use, for example, $x for the parameter named x.', type=str, required=True)
required.add_argument('--experiment_name', help='Name of the experiment.', type=str, required=True)
required.add_argument('--mem_gb', help='Amount of RAM for each worker in GB (default: 1GB)', type=float, default=1)

required_but_choice.add_argument('--parameter', action='append', nargs='+', help="Experiment parameters in the formats (options in round brackets are optional): <NAME> range <LOWER BOUND> <UPPER BOUND> (<INT, FLOAT>) -- OR -- <NAME> fixed <VALUE> -- OR -- <NAME> choice <Comma-seperated list of values>", default=None)
required_but_choice.add_argument('--load_checkpoint', help="Path of a checkpoint to be loaded", type=str, default=None)

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

args = parser.parse_args()

def decode_if_base64(input_str):
    try:
        decoded_bytes = base64.b64decode(input_str)
        decoded_str = decoded_bytes.decode('utf-8')
        return decoded_str
    except Exception as e:
        return input_str

joined_run_program = " ".join(args.run_program[0])
joined_run_program = decode_if_base64(joined_run_program)

if args.parameter is None and args.load_checkpoint is None:
    print("Either --parameter or --load_checkpoint is required. Both were not found.")
    sys.exit(19)
elif args.parameter is not None and args.load_checkpoint is not None:
    print("You cannot use --parameter and --load_checkpoint. You have to decide for one.");
    sys.exit(20)
elif args.load_checkpoint:
    if not os.path.exists(args.load_checkpoint):
        print("red", f"{args.load_checkpoint} could not be found!")
        sys.exit(21)

def print_debug (msg):
    if args.debug:
        print(msg)

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

def find_helpers_path():
    python_path = os.getenv("PYTHONPATH", "").split(":")
    for path in python_path:
        helpers_path = os.path.join(path, ".helpers.py")
        if os.path.exists(helpers_path):
            return helpers_path
    return None

helpers_file = find_helpers_path()
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
except KeyboardInterrupt:
    print("\n:warning: You pressed CTRL+C. Program execution halted.")
    sys.exit(0)
except signalUSR:
    print("\n:warning: USR1-Signal was sent. Cancelling loading modules.")
    sys.exit(0)
except signalINT:
    print("\n:warning: INT signal was sent. Cancelling loading modules.")
    sys.exit(0)

def print_color (color, text):
    print(f"[{color}]{text}[/{color}]")

if args.max_eval <= 0:
    print_color("red", "--max_eval must be larger than 0")
    sys.exit(39)

def is_executable_in_path(executable_name):
    for path in os.environ.get('PATH', '').split(':'):
        executable_path = os.path.join(path, executable_name)
        if os.path.exists(executable_path) and os.access(executable_path, os.X_OK):
            return True
    return False

def check_slurm_job_id():
    if is_executable_in_path('sbatch'):
        slurm_job_id = os.environ.get('SLURM_JOB_ID')
        if slurm_job_id is not None and not slurm_job_id.isdigit():
            print_color("red", "Not a valid SLURM_JOB_ID.")
        elif slurm_job_id is None:
            print_color("red", "You are on a system that has SLURM available, but you are not running the main-script in a Slurm-Environment. " +
                "This may cause the system to slow down for all other users. It is recommended uou run the main script in a Slurm job."
            )

def create_folder_and_file (folder, extension):
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
    try:
        # Check if all elements can be converted to numbers
        numbers = [float(item) for item in arr]
        # If successful, order them numerically
        sorted_arr = sorted(numbers)
    except ValueError:
        # If there's an error, order them alphabetically
        sorted_arr = sorted(arr)

    return sorted_arr

import re

def looks_like_int(x):
    if isinstance(x, int):
        return True
    elif isinstance(x, float):
        return x.is_integer()
    elif isinstance(x, str):
        return bool(re.match(r'^\d+$', x))
    else:
        return False


def parse_experiment_parameters(args):
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
    is_empty = os.path.getsize(file_path) == 0 if os.path.exists(file_path) else True

    with open(file_path, 'a', newline='') as file:
        csv_writer = csv.writer(file)

        if is_empty:
            csv_writer.writerow(heading)

        csv_writer.writerow(data_line)

def make_strings_equal_length(str1, str2):
    length_difference = len(str1) - len(str2)

    if length_difference > 0:
        str2 = str2 + ' ' * length_difference
    elif length_difference < 0:
        str2 = str2[:len(str1)]

    return str1, str2

def find_file_paths(_text):
    file_paths = []

    words = _text.split()

    for word in words:
        if os.path.exists(word):
            file_paths.append(word)

    return file_paths

def check_file_info(file_path):
    if not os.path.exists(file_path):
        print(f"Die Datei {file_path} existiert nicht.")
        return

    if not os.access(file_path, os.R_OK):
        print(f"Die Datei {file_path} ist nicht lesbar.")
        return

    file_stat = os.stat(file_path)

    uid = file_stat.st_uid
    gid = file_stat.st_gid

    username = pwd.getpwuid(uid).pw_name

    size = file_stat.st_size
    permissions = stat.filemode(file_stat.st_mode)

    access_time = file_stat.st_atime
    modification_time = file_stat.st_mtime
    status_change_time = file_stat.st_ctime

    # Gruppennamen für die Benutzergruppen abrufen
    string = ""

    string += f"Datei: {file_path}\n"
    string += f"Größe: {size} Bytes\n"
    string += f"Berechtigungen: {permissions}\n"
    string += f"Besitzer: {username}\n"
    string += f"Letzter Zugriff: {access_time}\n"
    string += f"Letzte Änderung: {modification_time}\n"
    string += f"Letzter Statuswechsel: {status_change_time}\n"

    string += f"Hostname: {socket.gethostname()}"

    return string

def find_file_paths_and_print_infos (_text):
    file_paths = find_file_paths(_text)

    string = "";

    if file_paths:
        string += "\n========\nDEBUG INFOS START:\n"
        for file_path in file_paths:
            string += "\n"
            string += check_file_info(file_path)
        string += "\n========\nDEBUG INFOS END\n"

    return string

def evaluate(parameters):
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    signal.signal(signal.SIGUSR2, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGQUIT, signal.SIG_IGN)

    max_val = 99999999999999999999999999999999999999999999999999999999999

    return_in_case_of_error = None

    if args.maximize:
        return_in_case_of_error = {"result": -max_val}
    else:
        return_in_case_of_error = {"result": max_val}

    try:
        print("parameters:", parameters)

        parameters_keys = list(parameters.keys())
        parameters_values = list(parameters.values())

        program_string_with_params = replace_parameters_in_string(parameters, joined_run_program)

        program_string_with_params = program_string_with_params.replace('\r', ' ').replace('\n', ' ')

        string = find_file_paths_and_print_infos(program_string_with_params)

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
except KeyboardInterrupt:
    sys.exit(0)
except signalUSR:
    print("\n:warning: USR1-Signal was sent. Cancelling loading ax.")
    sys.exit(0)
except signalINT:
    print("\n:warning: INT signal was sent. Cancelling loading ax.")
    sys.exit(0)

def disable_logging ():
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

def show_end_table_and_save_end_files ():
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    signal.signal(signal.SIGUSR2, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGQUIT, signal.SIG_IGN)

    warnings.filterwarnings("ignore", category=UserWarning, module="ax.modelbridge.cross_validation")
    warnings.filterwarnings("ignore", category=Warning, module="ax.modelbridge.cross_validation")
    warnings.filterwarnings("ignore", category=Warning, module="ax.modelbridge")
    warnings.filterwarnings("ignore", category=Warning, module="ax")

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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            best_parameters, (means, covariances) = ax_client.get_best_parameters()

            print_debug("[show_end_table_and_save_end_files] Got best params")
            best_result = means["result"]

            if str(best_result) == '1e+59':
                table_str = "Best result could not be determined"
                print_color("red", table_str)
            else:
                print_debug("[show_end_table_and_save_end_files] Creating table")
                table = Table(show_header=True, header_style="bold", title="Best parameter:")

                for key in best_parameters.keys():
                    table.add_column(key)

                print_debug("[show_end_table_and_save_end_files] Add last column to table")
                table.add_column("result (inexact)")

                print_debug("[show_end_table_and_save_end_files] Defining rows")
                row_without_result = [str(best_parameters[key]) for key in best_parameters.keys()];
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

def end_program ():
    print_debug("[end_program] end_program started")
    global end_program_ran

    if end_program_ran:
        print_debug("[end_program] end_program_ran was true. Returning.")
        return

    end_program_ran = True
    print_debug("[end_program] Setting end_program_ran = True")

    global ax_client
    global console
    global current_run_folder

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
        show_end_table_and_save_end_files()
        print_debug("[end_program] show_end_table_and_save_end_files called")
    except KeyboardInterrupt:
        print_color("red", "\n:warning: You pressed CTRL+C. Program execution halted.")
        print("\n:warning: KeyboardInterrupt signal was sent. Ending program will still run.")
        print_debug("[end_program] Calling show_end_table_and_save_end_files (in KeyboardInterrupt)")
        show_end_table_and_save_end_files()
        print_debug("[end_program] show_end_table_and_save_end_files called (in KeyboardInterrupt)")
    except TypeError:
        print_color("red", "\n:warning: The program has been halted without attaining any results.")
    except signalUSR:
        print("\n:warning: USR1-Signal was sent. Ending program will still run.")
        print_debug("[end_program] Calling show_end_table_and_save_end_files (in signalUSR)")
        show_end_table_and_save_end_files()
        print_debug("[end_program] show_end_table_and_save_end_files called (in signalUSR)")
    except signalINT:
        print("\n:warning: INT-Signal was sent. Ending program will still run.")
        print_debug("[end_program] Calling show_end_table_and_save_end_files (in signalINT)")
        show_end_table_and_save_end_files()
        print_debug("[end_program] show_end_table_and_save_end_files called (in signalINT)")

    pd_csv = f'{current_run_folder}/pd.csv'
    print_debug(f"[end_program] Trying to save file to {pd_csv}")

    save_pd_csv()

    for job, trial_index in jobs[:]:
        job.cancel()

    sys.exit(0)

def save_checkpoint ():
    global current_run_folder
    global ax_client

    checkpoint_filepath = f"{current_run_folder}/checkpoint.json"
    ax_client.save_to_json_file(filepath=checkpoint_filepath)

    print_debug("Checkpoint saved")

def save_pd_csv ():
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
    global args
    global current_run_folder

    min_or_max = "minimize"
    if args.maximize:
        min_or_max = "maximize"

    with open(f"{current_run_folder}/{min_or_max}", 'w') as f:
        print('The contents of this file do not matter. It is only relevant that it exists.', file=f)

    if args.parameter:
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
    else:
        print_color("red", f"No parameters defined")
        sys.exit(26)

def check_equation (variables, equation):
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

def main ():
    global args
    global file_number
    global folder_number
    global result_csv_file
    global current_run_folder
    global ax_client
    global jobs

    check_slurm_job_id()

    current_run_folder = f"{args.run_dir}/{args.experiment_name}/{folder_number}"
    while os.path.exists(f"{current_run_folder}"):
        current_run_folder = f"{args.run_dir}/{args.experiment_name}/{folder_number}"
        folder_number = folder_number + 1

    result_csv_file = create_folder_and_file(f"{current_run_folder}", "csv")

    with open(f"{current_run_folder}/env", 'a') as f:
        env = dict(os.environ)
        for key in env:
            print(str(key) + " = " + str(env[key]), file=f)

    with open(f"{current_run_folder}/run.sh", 'w') as f:
        print("bash run.sh '" + "' '".join(sys.argv[1:]) + "'", file=f)

    print(f"[yellow]CSV-File[/yellow]: [underline]{result_csv_file}[/underline]")
    print_color("green", program_name)

    experiment_parameters = None

    if args.parameter:
        experiment_parameters = parse_experiment_parameters(args.parameter)

        checkpoint_filepath = f"{current_run_folder}/checkpoint.json.parameters.json"

        with open(checkpoint_filepath, "w") as outfile:
            json.dump(experiment_parameters, outfile)

        print_overview_table(experiment_parameters)

    if not args.verbose:
        disable_logging()

    try:
        ax_client = AxClient(verbose_logging=args.verbose)

        minimize_or_maximize = not args.maximize

        experiment = None

        if args.load_checkpoint:
            ax_client = (AxClient.load_from_json_file(args.load_checkpoint))

            checkpoint_params_file = args.load_checkpoint + ".parameters.json"

            if not os.path.exists(checkpoint_params_file):
                print_color("red", f"{checkpoint_params_file} not found. Cannot continue without.")
                sys.exit(22)

            f = open(checkpoint_params_file)
            experiment_parameters = json.load(f)
            f.close()

            with open(f'{current_run_folder}/checkpoint_load_source', 'w') as f:
                print(f"Continuation from checkpoint {args.load_checkpoint}", file=f)
        else:
            experiment_args = {
                "name": args.experiment_name,
                "parameters": experiment_parameters,
                "objectives": {"result": ObjectiveProperties(minimize=minimize_or_maximize)},
                "choose_generation_strategy_kwargs": {
                    "num_trials": args.max_eval
                },
            }


            if args.experiment_constraints:
                constraints_string = " ".join(args.experiment_constraints[0])

                variables = [item['name'] for item in experiment_parameters]

                equation = check_equation(variables, constraints_string)

                if equation:
                    experiment_args["parameter_constraints"] = [constraints_string]
                else:
                    print_color("red", "Experiment constraints are invalid.")
                    sys.exit(28)

            try:
                experiment = ax_client.create_experiment(**experiment_args)
            except ValueError as error:
                print_color("red", f"An error has occured: {error}")
                sys.exit(29)

        log_folder = f"{current_run_folder}/%j"
        executor = submitit.AutoExecutor(folder=log_folder)

        # 'nodes': <class 'int'>, 'gpus_per_node': <class 'int'>, 'tasks_per_node': <class 'int'>

        executor.update_parameters(
            name=args.experiment_name,
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


        base_desc = f"Evaluating hyperparameter constellations, searching {searching_for} ({args.max_eval} in total)..."

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with tqdm(total=args.max_eval, disable=False, desc=base_desc) as progress_bar:
                start_str = f"[cyan]{base_desc}"

                progress_string = start_str

                progress_string = progress_string

                while submitted_jobs < args.max_eval or jobs:
                    # Schedule new jobs if there is availablity
                    try:
                        calculated_max_trials = min(args.num_parallel_jobs - len(jobs), args.max_eval - submitted_jobs)

                        print_debug(f"Trying to get the next {calculated_max_trials} trials.")

                        trial_index_to_param, _ = ax_client.get_next_trials(
                            max_trials=calculated_max_trials
                        )

                        print_debug(f"Got {len(trial_index_to_param.items())} new items.")

                        for trial_index, parameters in trial_index_to_param.items():
                            new_job = None
                            try:
                                print_debug(f"Trying to start new job.")
                                new_job = executor.submit(evaluate, parameters)
                                submitted_jobs += 1
                                jobs.append((new_job, trial_index))
                                time.sleep(1)
                            except submitit.core.utils.FailedJobError as error:
                                if "QOSMinGRES" in str(error) and args.gpus == 0:
                                    print_color("red", f"\n:warning: It seems like, on the chosen partition, you need at least one GPU. Use --gpus=1 (or more) as parameter.")
                                else:
                                    print_color("red", f"\n:warning: FAILED: {error}")
                                
                                try:
                                    new_job.cancel()

                                    jobs.remove((new_job, trial_index))

                                    progress_bar.update(1)

                                    save_checkpoint()
                                    save_pd_csv()
                                except Exception as e:
                                    print_color("red", f"\n:warning: Cancelling failed job FAILED: {e}")


                    except botorch.exceptions.errors.InputDataError as e:
                        print_color("red", f"Error: {e}")
                    except ax.exceptions.core.DataRequiredError:
                        print_color("red", f"Error: {e}")

                    for job, trial_index in jobs[:]:
                        # Poll if any jobs completed
                        # Local and debug jobs don't run until .result() is called.
                        if job.done() or type(job) in [LocalJob, DebugJob]:
                            try:
                                result = job.result()
                                print_debug("Got job result")
                                ax_client.complete_trial(trial_index=trial_index, raw_data=result)
                            except submitit.core.utils.UncompletedJobError as error:
                                print_color("red", str(error))

                                job.cancel()
                            except ax.exceptions.core.UserInputError as error:
                                if "None for metric" in str(error):
                                    print_color("red", f"\n:warning: It seems like the program that was about to be run didn't have 'RESULT: <NUMBER>' in it's output string.\nError: {error}")
                                else:
                                    print_color("red", f"\n:warning: {error}")

                                job.cancel()

                            jobs.remove((job, trial_index))

                            progress_bar.update(1)

                            save_checkpoint()
                            save_pd_csv()

                    time.sleep(0.1)
            end_program()
    except KeyboardInterrupt:
        print_color("red", "\n:warning: You pressed CTRL+C. Optimization stopped.")
    except signalUSR:
        print("\n:warning: USR1-Signal was sent. Cancelling optimization. Running end_program.")
        end_program()
        print("\n:warning: end_program ran.")
    except signalINT:
        print("\n:warning: INT-Signal was sent. Cancelling optimization Running end_program.")
        end_program()
        print("\n:warning: end_program ran.")

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
