ax_client = None
end_program_ran = False
program_name = "OmniOpt2"
current_run_folder = None
file_number = 0
folder_number = 0
args = None
result_csv_file = None

import sys

try:
    import os
    import socket
    import json
    import signal
    from tqdm import tqdm
except ModuleNotFoundError as e:
    print(f"Error loading module: {e}")
    sys.exit(24)

class userSignalOne (Exception):
    pass

class userSignalTwo (Exception):
    pass

class userSignalInt (Exception):
    pass

def receive_usr_signal_one (signum, stack):
    end_program()
    raise userSignalOne("USR1-signal received")

def receive_usr_signal_two (signum, stack):
    end_program()
    raise userSignalTwo("USR2-signal received")

def receive_usr_signal_int (signum, stack):
    end_program()
    raise userSignalInt("INT-signal received")

signal.signal(signal.SIGUSR1, receive_usr_signal_one)
signal.signal(signal.SIGUSR2, receive_usr_signal_two)
signal.signal(signal.SIGINT, receive_usr_signal_int)

import importlib.util 
spec = importlib.util.spec_from_file_location(
    name="helpers",
    location=".helpers.py",
)
my_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(my_module)

try:
    from rich.console import Console
    console = Console(force_terminal=True, force_interactive=True)
    with console.status("[bold green]Importing modules...") as status:
        #from rich.traceback import install
        #install(show_locals=True)

        from rich.table import Table
        from rich import print
        from rich.progress import Progress

        import time
        import csv
        import re
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
except userSignalOne:
    print("\n:warning: USR1 signal was sent. Cancelling.")
    sys.exit(0)
except userSignalTwo:
    print("\n:warning: USR2 signal was sent. Cancelling.")
    sys.exit(0)

def print_color (color, text):
    print(f"[{color}]{text}[/{color}]")

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

def dier (msg):
    pprint(msg)
    sys.exit(10)

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

def parse_experiment_parameters(args):
    params = []

    param_names = []

    i = 0

    while i < len(args):
        this_args = args[i]
        j = 0
        while j < len(this_args):
            name = this_args[j]

            invalid_names = ["start_time", "end_time", "run_time", "program_string", "result", "exit_code"]

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
                    print_color("red", f"\n:warning: --parameter for type range must have 5 parameters: <NAME> range <START> <END> (<TYPE (int or float)>)");
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

        return input_string
    except Exception as e:
        print_color("red", f"\n:warning: Error: {e}")
        return None

def execute_bash_code(code):
    try:
        result = subprocess.run(code, shell=True, check=True, text=True, capture_output=True)

        if result.returncode != 0:
            print(f"Exit-Code: {result.returncode}")

        return [result.stdout, result.stderr, result.returncode]

    except subprocess.CalledProcessError as e:
        print(f"Fehler beim Ausführen des Bash-Codes. Exit-Code: {e.returncode}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return [e.stdout, e.stderr, e.returncode]

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

def evaluate(parameters):
    print("parameters:", parameters)

    parameters_keys = list(parameters.keys())
    parameters_values = list(parameters.values())

    program_string_with_params = replace_parameters_in_string(parameters, args.run_program)

    print_color("green", program_string_with_params)

    start_time = int(time.time())

    stdout_stderr_exit_code = execute_bash_code(program_string_with_params)

    end_time = int(time.time())

    stdout = stdout_stderr_exit_code[0]
    stderr = stdout_stderr_exit_code[1]
    exit_code = stdout_stderr_exit_code[2]

    run_time = end_time - start_time

    print("stdout:")
    print(stdout)

    result = get_result(stdout)

    print(f"Result: {result}")

    headline = ["start_time", "end_time", "run_time", "program_string", *parameters_keys, "result", "exit_code", "hostname"];
    values = [start_time, end_time, run_time, program_string_with_params,  *parameters_values, result, exit_code, socket.gethostname()];

    headline = ['None' if element is None else element for element in headline]
    values = ['None' if element is None else element for element in values]

    add_to_csv(result_csv_file, headline, values)

    if type(result) == int:
        return {"result": int(result)}
    elif type(result) == float:
        return {"result": float(result)}
    else:
        max_val = 99999999999999999999999999999999999999999999999999999999999
        if args.maximize:
            return {"result": -max_val}
        else:
            return {"result": max_val}

try:
    with console.status("[bold green]Importing modules...") as status:
        import time
        try:
            import ax
            import botorch
            from ax.service.ax_client import AxClient, ObjectiveProperties
            from ax.modelbridge.dispatch_utils import choose_generation_strategy
            from ax.storage.json_store.save import save_experiment
            from ax.service.utils.report_utils import exp_to_df
        except ModuleNotFoundError as e:
            print_color("red", "\n:warning: ax could not be loaded. Did you create and load the virtual environment properly?")
            sys.exit(6)
        except KeyboardInterrupt:
            print_color("red", "\n:warning: You pressed CTRL+C. Program execution halted.")
            sys.exit(6)

        try:
            import submitit
            from submitit import AutoExecutor, LocalJob, DebugJob
        except userSignalOne:
            print("\n:warning: USR1 signal was sent. Cancelling.")
            sys.exit(0)
        except userSignalTwo:
            print("\n:warning: USR2 signal was sent. Cancelling.")
            sys.exit(0)
        except:
            print_color("red", "\n:warning: submitit could not be loaded. Did you create and load the virtual environment properly?")
            sys.exit(7)
except KeyboardInterrupt:
    sys.exit(0)
except userSignalOne:
    print("\n:warning: USR1 signal was sent. Cancelling.")
    sys.exit(0)
except userSignalTwo:
    print("\n:warning: USR2 signal was sent. Cancelling.")
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
    warnings.filterwarnings("ignore", category=UserWarning, module="ax.service.utils.report_utils")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.autograd")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.autograd.__init__")
    warnings.filterwarnings("ignore", category=UserWarning, module="botorch.optim.fit")
    warnings.filterwarnings("ignore", category=UserWarning, module="ax.core.parameter")

def end_program ():
    global end_program_ran

    if end_program_ran:
        return

    end_program_ran = True
    global ax_client
    global console
    global current_run_folder

    if current_run_folder is None:
        print("current_run_folder was empty. Not running end-algorithm.")
        return

    if ax_client is None:
        print("ax_client was empty. Not running end-algorithm.")
        return

    if console is None:
        print("console was empty. Not running end-algorithm.")
        return

    try:
        warnings.filterwarnings("ignore", category=UserWarning, module="ax.service.utils.report_utils")
        best_parameters, (means, covariances) = ax_client.get_best_parameters()

        best_result = means["result"]

        table = Table(show_header=True, header_style="bold", title="Best parameters:")

        # Dynamisch Spaltenüberschriften hinzufügen
        for key in best_parameters.keys():
            table.add_column(key)

        table.add_column("result (inexact)")

        # "best results" als Zeilenüberschrift hinzufügen
        row_without_result = [str(best_parameters[key]) for key in best_parameters.keys()];
        row = [*row_without_result, str(best_result)]

        table.add_row(*row)

        # Drucke die Tabelle
        console.print(table)

        with console.capture() as capture:
            console.print(table)
        table_str = capture.get()

        with open(f"{current_run_folder}/best_result.txt", "w") as text_file:
            text_file.write(table_str)
    except KeyboardInterrupt:
        print_color("red", "\n:warning: You pressed CTRL+C. Program execution halted.")
    except TypeError:
        print_color("red", "\n:warning: The program has been halted without attaining any results.")
    except userSignalOne:
        print("\n:warning: USR1 signal was sent. Cancelling.")
        sys.exit(0)
    except userSignalTwo:
        print("\n:warning: USR2 signal was sent. Cancelling.")
        sys.exit(0)

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
    except Exception as e:
        print_color("red", f"While saving all trials as a pandas-dataframe-csv, an error occured: {e}")
        sys.exit(17)

def main ():
    global args
    global file_number
    global folder_number
    global result_csv_file
    global current_run_folder
    global ax_client

    check_slurm_job_id()

    parser = argparse.ArgumentParser(
        prog=program_name,
        description='A hyperparameter optimizer for the HPC-system of the TU Dresden',
        epilog="Example:\n\npython3 run.py --num_parallel_jobs=1 --gpus=1 --max_eval=1 --parameter x range -10 10 float --parameter y range -10 10 int --run_program='bash test.sh $x $y' --maximize --timeout=10"
    )


    required = parser.add_argument_group('Required arguments', "These options have to be set")
    required_but_choice = parser.add_argument_group('Required arguments that allow a choice', "Of these arguments, one has to be set to continue.")
    optional = parser.add_argument_group('Optional', "These options are optional")
    debug = parser.add_argument_group('Debug', "These options are mainly useful for debugging")

    required.add_argument('--num_parallel_jobs', help='Number of parallel slurm jobs', type=int, required=True)
    required.add_argument('--max_eval', help='Maximum number of evaluations', type=int, required=True)
    required.add_argument('--timeout', help='Timeout for slurm jobs (i.e. for each single point to be optimized)', type=int, required=True)
    required.add_argument('--run_program', help='A program that should be run. Use, for example, $x for the parameter named x.', type=str, required=True)
    required.add_argument('--experiment_name', help='Name of the experiment. Not really used anywhere. Default: exp', type=str, required=True)

    required_but_choice.add_argument('--parameter', action='append', nargs='+', help="Experiment parameters in the formats (options in round brackets are optional): <NAME> range <LOWER BOUND> <UPPER BOUND> (<INT, FLOAT>) -- OR -- <NAME> fixed <VALUE> -- OR -- <NAME> choice <Comma-seperated list of values>", default=None)
    required_but_choice.add_argument('--load_checkpoint', help="Path of a checkpoint to be loaded", type=str, default=None)

    optional.add_argument('--cpus_per_task', help='CPUs per task', type=int, default=1)
    optional.add_argument('--mem_gb', help='Amount of RAM for each worker in GB (default: 1GB)', type=float, default=1)
    optional.add_argument('--gpus', help='Number of GPUs', type=int, default=0)
    optional.add_argument('--maximize', help='Maximize instead of minimize (which is default)', action='store_true', default=False)
    optional.add_argument('--experiment_constraints', help='Constraints for parameters. Example: x + y <= 2.0', type=str)
    optional.add_argument('--stderr_to_stdout', help='Redirect stderr to stdout for subjobs', action='store_true', default=False)
    optional.add_argument('--run_dir', help='Directory, in which runs should be saved. Default: runs', default="runs", type=str)

    debug.add_argument('--verbose', help='Verbose logging', action='store_true', default=False)

    args = parser.parse_args()

    if args.parameter is None and args.load_checkpoint is None:
        print_color("red", "Either --parameter or --load_checkpoint is required. Both were not found.")
        sys.exit(19)
    elif args.parameter is not None and args.load_checkpoint is not None:
        print_color("red", "You cannot use --parameter and --load_checkpoint. You have to decide for one.");
        sys.exit(20)
    elif args.load_checkpoint:
        if not os.path.exists(args.load_checkpoint):
            print_color("red", f"{args.load_checkpoint} could not be found!")
            sys.exit(21)


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
            if args.experiment_constraints:
                experiment = ax_client.create_experiment(
                    name=args.experiment_name,
                    parameters=experiment_parameters,
                    objectives={"result": ObjectiveProperties(minimize=minimize_or_maximize)},
                    parameter_constraints=[args.experiment_constraints]
                )
            else:
                experiment = ax_client.create_experiment(
                    name=args.experiment_name,
                    parameters=experiment_parameters,
                    objectives={"result": ObjectiveProperties(minimize=minimize_or_maximize)}
                )

        log_folder = f"{current_run_folder}/%j"
        executor = submitit.AutoExecutor(folder=log_folder)


        # 'name': <class 'str'>, 'nodes': <class 'int'>, 'gpus_per_node': <class 'int'>, 'tasks_per_node': <class 'int'>

        executor.update_parameters(
            name=args.experiment_name,
            timeout_min=args.timeout,
            slurm_gres=f"gpu:{args.gpus}",
            cpus_per_task=args.cpus_per_task,
            stderr_to_stdout=args.stderr_to_stdout,
            mem_gb=args.mem_gb,
        )

        jobs = []
        submitted_jobs = 0
        # Run until all the jobs have finished and our budget is used up.

        searching_for = "minimum"
        if args.maximize:
            searching_for = "maximum"

        with tqdm(total=args.max_eval, disable=False, desc=f"Evaluating hyperparameter constellations, searching {searching_for} ({args.max_eval} in total)...") as progress_bar:
            start_str = f"[cyan]Evaluating hyperparameter constellations, searching {searching_for} ({args.max_eval} in total)..."

            progress_string = start_str

            progress_string = progress_string

            while submitted_jobs < args.max_eval or jobs:
                for job, trial_index in jobs[:]:
                    # Poll if any jobs completed
                    # Local and debug jobs don't run until .result() is called.
                    if job.done() or type(job) in [LocalJob, DebugJob]:
                        try:
                            result = job.result()
                            ax_client.complete_trial(trial_index=trial_index, raw_data=result)
                            jobs.remove((job, trial_index))

                            #best_parameters, (means, covariances) = ax_client.get_best_parameters()
                            #best_result = means["result"]
                            #new_desc_string = f"best result: {best_result}"

                            progress_bar.update(1)

                            checkpoint_filepath = f"{current_run_folder}/checkpoint.json"
                            ax_client.save_to_json_file(filepath=checkpoint_filepath)

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
                            except Exception as e:
                                print_color("red", f"While saving all trials as a pandas-dataframe-csv, an error occured: {e}")
                        except submitit.core.utils.UncompletedJobError as error:
                                print_color("red", str(error))
                                sys.exit(27)
                        except ax.exceptions.core.UserInputError as error:
                            if "None for metric" in str(error):
                                print_color("red", f"\n:warning: It seems like the program that was about to be run didn't have 'RESULT: <NUMBER>' in it's output string.\nError: {error}")
                            else:
                                print_color("red", f"\n:warning: {error}")
                                sys.exit(25)
                
                # Schedule new jobs if there is availablity
                try:
                    trial_index_to_param, _ = ax_client.get_next_trials(
                        max_trials=min(args.num_parallel_jobs - len(jobs), args.max_eval - submitted_jobs)
                    )

                    for trial_index, parameters in trial_index_to_param.items():
                        try:
                            job = executor.submit(evaluate, parameters)
                            submitted_jobs += 1
                            jobs.append((job, trial_index))
                            time.sleep(1)
                        except submitit.core.utils.FailedJobError as error:
                            if "QOSMinGRES" in str(error) and args.gpus == 0:
                                print_color("red", f"\n:warning: It seems like, on the chosen partition, you need at least one GPU. Use --gpus=1 (or more) as parameter.")
                            else:
                                print_color("red", f"\n:warning: FAILED: {error}")

                            sys.exit(2)
                except botorch.exceptions.errors.InputDataError as e:
                    print_color("red", f"Error: {e}")
                
                # Sleep for a bit before checking the jobs again to avoid overloading the cluster. 
                # If you have a large number of jobs, consider adding a sleep statement in the job polling loop aswell.
                time.sleep(0.1)
    except KeyboardInterrupt:
        print_color("red", "\n:warning: You pressed CTRL+C. Optimization stopped.")
    except userSignalOne:
        print("\n:warning: USR1 signal was sent. Cancelling.")
    except userSignalTwo:
        print("\n:warning: USR2 signal was sent. Cancelling.")
    
    end_program()

if __name__ == "__main__":
    main()
