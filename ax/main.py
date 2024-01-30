program_name = "OmniAx"

try:
    from rich.console import Console
    console = Console()
    with console.status("[bold green]Importing modules...") as status:
        from rich.traceback import install
        #install(show_locals=True)

        from rich.table import Table
        from rich import print
        from rich.progress import Progress

        import time
        import csv
        import os
        import re
        import sys
        import argparse
        from rich.pretty import pprint
        import subprocess

        import logging
except KeyboardInterrupt:
    sys.exit(0)

def print_color (color, text):
    print(f"[{color}]{text}[/{color}]")

def dier (msg):
    pprint(msg)
    sys.exit(2)

file_number = 0

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

result_csv_file = create_folder_and_file(f"runs/{file_number}", "csv")

def parse_experiment_parameters(args):
    params = []

    i = 0

    while i < len(args):
        this_args = args[i]
        j = 0
        while j < len(this_args):
            name = this_args[j]

            param_type = this_args[j + 1]

            valid_types = ["range", "fixed", "choice", "string"]
            valid_types_string = ', '.join(valid_types)

            if param_type not in valid_types:
                print_color("red", f":warning: Invalid type {param_type}, valid types are: {valid_types_string}")
                sys.exit(3)

            if param_type == "range":
                if len(this_args) != 5 and len(this_args) != 4:
                    print_color("red", f":warning: --parameter for type range must have 5 parameters: <NAME> range <START> <END> (<TYPE (int or float)>)");
                    sys.exit(11)

                try:
                    lower_bound = float(this_args[j + 2])
                except:
                    print_color("red", f":warning: {this_args[j + 2]} does not seem to be a number")
                    sys.exit(4)

                try:
                    upper_bound = float(this_args[j + 3])
                except:
                    print_color("red", f":warning: {this_args[j + 3]} does not seem to be a number")
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

                #valid_value_types = ["int", "float", "bool", "str"]
                valid_value_types = ["int", "float"]

                if value_type not in valid_value_types:
                    ", ".join(valid_value_types)
                    print_color("red", f":warning: {value_type} is not a valid value type. Valid types for range are: {valid_value_types_string}")
                    sys.exit(10)

                param = {
                    "name": name,
                    "type": param_type,
                    "bounds": [lower_bound, upper_bound],
                    "value_type": value_type
                }

                params.append(param)

                j += skip
            else:
                print_color("red", f":warning: Parameter type {param_type} not yet implemented.");
                sys.exit(14)
                j += 4
        i += 1

    return params

print_color("green", program_name)

parser = argparse.ArgumentParser(
    prog=program_name,
    description='A hyperparameter optimizer for the HPC-system of the TU Dresden',
    epilog="Example:\n\npython3 run.py --num_parallel_jobs=1 --partition=alpha --gpus=1 --max_eval=1 --parameter x range -10 10 float --parameter y range -10 10 int --run_program='bash test.sh $x $y' --maximize"
)

parser.add_argument('--num_parallel_jobs', help='Number of parallel slurm jobs', type=int, required=True)
parser.add_argument('--max_eval', help='Maximum number of evaluations', type=int, required=True)
parser.add_argument('--cpus_per_task', help='CPUs per task', type=int, default=1)
parser.add_argument('--parameter', action='append', nargs='+', required=True, help='Experiment parameters in the format: name type lower_bound upper_bound')
parser.add_argument('--timeout_min', help='Timeout for slurm jobs', type=int, default=60)
parser.add_argument('--gpus', help='Number of GPUs', type=int, default=0)
parser.add_argument('--partition', help='Name of the partition it should run on', type=str, required=True)
parser.add_argument('--maximize', help='Maximize instead of minimize (which is default)', action='store_true', default=False)
parser.add_argument('--verbose', help='Verbose logging', action='store_true', default=False)
parser.add_argument('--experiment_constraints', help='Constraints for parameters. Example: x + y <= 2.0', type=str)
parser.add_argument('--experiment_name', help='Name of the experiment. Not really used anywhere. Default: exp', default="exp", type=str)
parser.add_argument('--run_program', help='A program that should be run. Use, for example, $x for the parameter named x.', type=str, required=True)

args = parser.parse_args()

experiment_parameters = parse_experiment_parameters(args.parameter)

rows = []
for param in experiment_parameters:
    rows.append([str(param["name"]), str(param["type"]), str(param["bounds"][0]), str(param["bounds"][1]), str(param["value_type"])])

table = Table(title="Experiment parameters:")
columns = ["Name", "Type", "Lower bound", "Upper bound", "Value-Type"]
for column in columns:
    table.add_column(column)
for row in rows:
    table.add_row(*row, style='bright_green')
console.print(table)


def replace_parameters_in_string(parameters, input_string):
    try:
        for param_item in parameters:
            input_string = input_string.replace(f"${param_item}", str(parameters[param_item]))

        return input_string
    except Exception as e:
        print_color("red", f":warning: Error: {e}")
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
    global experiment_parameters

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

    headline = ["start_time", "end_time", "run_time", "program_string", *parameters_keys, "result", "exit_code"];
    values = [start_time, end_time, run_time, program_string_with_params,  *parameters_values, result, exit_code];

    headline = ['None' if element is None else element for element in headline]
    values = ['None' if element is None else element for element in values]

    add_to_csv(result_csv_file, headline, values)

    if result:
        return {"result": float(result)}
    else:
        return {"result": None}

try:
    with console.status("[bold green]Importing modules...") as status:
        import time
        try:
            import ax
            from ax.service.ax_client import AxClient, ObjectiveProperties
            from ax.service.utils.report_utils import exp_to_df
        except:
            print_color("red", ":warning: ax could not be loaded. Did you create and load the virtual environment properly?")
            sys.exit(8)

        try:
            import submitit
            from submitit import AutoExecutor, LocalJob, DebugJob
        except:
            print_color("red", ":warning: submitit could not be loaded. Did you create and load the virtual environment properly?")
            sys.exit(9)

except KeyboardInterrupt:
    sys.exit(0)

try:
    ax_client = AxClient(verbose_logging=args.verbose)

    minimize = not args.maximize

    if args.experiment_constraints:
        ax_client.create_experiment(
            name=args.experiment_name,
            parameters=experiment_parameters,
            objectives={"result": ObjectiveProperties(minimize=minimize)},
            parameter_constraints=[args.experiment_constraints]
        )
    else:
        ax_client.create_experiment(
            name=args.experiment_name,
            parameters=experiment_parameters,
            objectives={"result": ObjectiveProperties(minimize=minimize)}
        )

    log_folder = f"runs/{file_number}/%j"
    executor = submitit.AutoExecutor(folder=log_folder)

    executor.update_parameters(
        timeout_min=args.timeout_min,
        slurm_partition=args.partition,
        slurm_gres=f"gpu:{args.gpus}",
        cpus_per_task=args.cpus_per_task
    )

    jobs = []
    submitted_jobs = 0
    # Run until all the jobs have finished and our budget is used up.
    with Progress() as progress:
        start_str = "Running jobs... "

        progress_bar = progress.add_task(f"[cyan]{start_str}", total=args.max_eval)

        while submitted_jobs < args.max_eval or jobs:
            for job, trial_index in jobs[:]:
                # Poll if any jobs completed
                # Local and debug jobs don't run until .result() is called.
                if job.done() or type(job) in [LocalJob, DebugJob]:
                    result = job.result()
                    try:
                        ax_client.complete_trial(trial_index=trial_index, raw_data=result)
                        jobs.remove((job, trial_index))

                        progress.update(progress_bar, advance=1)
                    except ax.exceptions.core.UserInputError as error:
                        if "None for metric" in str(error):
                            print_color("red", f":warning: It seems like the program that was about to be run didn't have 'RESULT: <NUMBER>' in it's output string.\nError: {error}")
                        else:
                            print_color("red", ":warning: ".error)
                            sys.exit(1)
            
            # Schedule new jobs if there is availablity
            trial_index_to_param, _ = ax_client.get_next_trials(
                max_trials=min(args.num_parallel_jobs - len(jobs), args.max_eval - submitted_jobs))
            for trial_index, parameters in trial_index_to_param.items():
                try:
                    job = executor.submit(evaluate, parameters)
                    submitted_jobs += 1
                    jobs.append((job, trial_index))
                    time.sleep(1)
                except submitit.core.utils.FailedJobError as error:
                    if "QOSMinGRES" in str(error) and args.gpus == 0:
                        print_color("red", f":warning: It seems like, on the chosen partition, you need at least one GPU. Use --gpus=1 (or more) as parameter.")
                    else:
                        print_color("red", f":warning: FAILED: {error}")

                    sys.exit(2)
            
            # Sleep for a bit before checking the jobs again to avoid overloading the cluster. 
            # If you have a large number of jobs, consider adding a sleep statement in the job polling loop aswell.
            time.sleep(0.1)
except KeyboardInterrupt:
    print_color("red", ":warning: You pressed CTRL+C. Program execution halted.")

try:
    best_parameters, (means, covariances) = ax_client.get_best_parameters()

    
    best_result = means["result"]

    table = Table(show_header=True, header_style="bold magenta", title="Best parameters")

    # Dynamisch Spaltenüberschriften hinzufügen
    for key in best_parameters.keys():
        table.add_column(key)

    table.add_column("result")

    # "best results" als Zeilenüberschrift hinzufügen
    row_without_result = [str(best_parameters[key]) for key in best_parameters.keys()];
    row = [*row_without_result, str(best_result)]

    table.add_row(*row)


    # Drucke die Tabelle
    console.print(table)

    #print_color("green", f'Best set of parameters: {best_parameters}')
    #print_color("green", f'Mean objective value: {best_result}')
except TypeError:
    print_color("red", ":warning: You pressed CTRL+C. Program execution halted.")
