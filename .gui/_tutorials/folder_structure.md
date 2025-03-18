# Folder structure of OmniOpt2 runs

<!-- What files are created for each run? -->

<div id="toc"></div>

## `runs`-folder

For every experiment you do, there will be a new folder created inside the `runs`-folder in your OmniOpt2-installation.

Each of these has a subfolder for each run that the experiment with that name was run. For example, if you run the experiment `my_experiment`
twice, the paths `runs/my_experiment/0` and `runs/my_experiment/1` exist.

### Single files

#### `best_result.txt`

This file contains an ANSI-table that shows you the best result and the parameters resulted in that result.

```
		      Best parameter:
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┓
┃ width_and_height ┃ validation_split ┃ learning_rate ┃ epochs ┃ result   ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━┩
│ 72               │ 0.184052         │ 0.001         │ 14     │ 1.612789 │
└──────────────────┴──────────────────┴───────────────┴────────┴──────────┘
```

#### `results.csv`

This file contains infos about every evaluation in this run, that is, it's number, the algorithm that created that point, its parameters, and it's result.

```
trial_index,arm_name,trial_status,generation_method,result,width_and_height,validation_split,learning_rate,epochs
0,0_0,COMPLETED,Sobol,1.690072,71,0.021625286340713503,0.20240612381696702,7
1,1_0,COMPLETED,Sobol,1.638602,65,0.02604435756802559,0.2448390863677487,6
2,2_0,COMPLETED,Sobol,1.718751,78,0.23111544810235501,0.38468948143068704,2
3,3_0,COMPLETED,Sobol,1.636012,93,0.0857066310942173,0.23433385196421297,15
4,4_0,COMPLETED,Sobol,1.624952,60,0.04056024849414826,0.11660899678524585,6
5,5_0,COMPLETED,Sobol,1.64169,76,0.1567445032298565,0.21590755908098072,10
6,6_0,COMPLETED,Sobol,1.639097,72,0.07228925675153733,0.1230122183514759,6
7,7_0,COMPLETED,Sobol,1.6279,74,0.04752136580646038,0.08336016669869424,3
8,8_0,COMPLETED,Sobol,1.618417,87,0.0058464851230382925,0.016544286970980468,7
9,9_0,COMPLETED,Sobol,1.627581,76,0.0673203308135271,0.08200951679609716,5
```

#### `get_next_trials.csv`

A CSV file that contains the current time, the number of jobs `ax_client.get_next_trials()` got and the number it requested to get.

```
2024-06-25 08:55:46,1,20
2024-06-25 08:56:41,2,20
2024-06-25 08:57:14,5,20
2024-06-25 08:57:33,7,20
2024-06-25 08:59:54,15,20
...
```

#### `worker_usage.csv`

This contains the unix-timestamp, the number of workers requested, the number of workers got and the percentage of numbers got in respective to the number requested.

```
1717234020.5216107,20,20,100
1717234020.701352,20,19,95
1717234022.9286764,20,18,90
1717234023.9625003,20,18,90
1717234024.123422,20,17,85
1717234029.7775924,20,15,75
1717234032.9025316,20,15,75
1717234034.1040723,20,14,70
1717234034.2609978,20,13,65
1717234036.3286648,20,13,65
1717234036.4710507,20,12,60
1717234039.5589206,20,12,60
1717234039.7017379,20,11,55
1717234043.816519,20,11,55
1717234043.9860802,20,10,50
1717234044.1203723,20,9,45
1717234046.1940708,20,9,45
```

#### GPU-usage-files (`gpu_usage_*.csv`)

GPU usage files. They are the output of `nvidia-smi` and are periodically taken, when you run on a system with SLURM that allows you to connect to
nodes that have running jobs on it with ssh.

Header line is omitted, but is: `timestamp, name, pci.bus_id, driver_version, pstate, pcie.link.gen.max, pcie.link.gen.current, temperature.gpu, utilization.gpu [%], utilization.memory [%], memory.total [MiB], memory.free [MiB], memory.used [MiB]`.

It may looks something like this:

```
2024/06/01 11:27:05.177, NVIDIA A100-SXM4-40GB, 00000000:3B:00.0, 545.23.08, P0, 4, 4, 44, 0 %, 0 %, 40960 MiB, 40333 MiB, 4 MiB
2024/06/01 11:27:05.188, NVIDIA A100-SXM4-40GB, 00000000:8B:00.0, 545.23.08, P0, 4, 4, 42, 0 %, 0 %, 40960 MiB, 40333 MiB, 4 MiB
2024/06/01 11:27:05.192, NVIDIA A100-SXM4-40GB, 00000000:0B:00.0, 545.23.08, P0, 4, 4, 43, 0 %, 0 %, 40960 MiB, 40333 MiB, 4 MiB
2024/06/01 11:27:15.309, NVIDIA A100-SXM4-40GB, 00000000:8B:00.0, 545.23.08, P0, 4, 4, 42, 3 %, 0 %, 40960 MiB, 1534 MiB, 38803 MiB
2024/06/01 11:27:15.311, NVIDIA A100-SXM4-40GB, 00000000:0B:00.0, 545.23.08, P0, 4, 4, 43, 3 %, 0 %, 40960 MiB, 1534 MiB, 38803 MiB
2024/06/01 11:27:15.311, NVIDIA A100-SXM4-40GB, 00000000:3B:00.0, 545.23.08, P0, 4, 4, 44, 3 %, 0 %, 40960 MiB, 1534 MiB, 38803 MiB
2024/06/01 11:27:25.361, NVIDIA A100-SXM4-40GB, 00000000:8B:00.0, 545.23.08, P0, 4, 4, 43, 3 %, 0 %, 40960 MiB, 666 MiB, 39671 MiB
2024/06/01 11:27:25.376, NVIDIA A100-SXM4-40GB, 00000000:3B:00.0, 545.23.08, P0, 4, 4, 44, 1 %, 0 %, 40960 MiB, 910 MiB, 39427 MiB
```

#### `job_infos.csv`

This is similar to the `results.csv`, but contains a little other info, i.e. the hostname the execution ran on and the full path that is run, also start- and endtime of execution and the exit code and signal that the job ended with.

```
start_time,end_time,run_time,program_string,width_and_height,validation_split,learning_rate,epochs,result,exit_code,signal,hostname
1719298546,1719298600,54,bash /home/s3811141/repos/OmniOpt/.tests/example_network/run.sh --learning_rate=0.20240612381696702 --epochs=7 --validation_split=0.021625286340713503 --width=71 --height=71 --dense=8 --dense_units=16 --conv=16 --conv_filters=16,71,0.021625286340713503,0.20240612381696702,7,1.690072,0,None,arbeitsrechner
1719298601,1719298633,32,bash /home/s3811141/repos/OmniOpt/.tests/example_network/run.sh --learning_rate=0.2448390863677487 --epochs=6 --validation_split=0.02604435756802559 --width=65 --height=65 --dense=8 --dense_units=16 --conv=16 --conv_filters=16,65,0.02604435756802559,0.2448390863677487,6,1.638602,0,None,arbeitsrechner
1719298635,1719298653,18,bash /home/s3811141/repos/OmniOpt/.tests/example_network/run.sh --learning_rate=0.38468948143068704 --epochs=2 --validation_split=0.23111544810235501 --width=78 --height=78 --dense=8 --dense_units=16 --conv=16 --conv_filters=16,78,0.23111544810235501,0.38468948143068704,2,1.718751,0,None,arbeitsrechner
1719298654,1719298793,139,bash /home/s3811141/repos/OmniOpt/.tests/example_network/run.sh --learning_rate=0.23433385196421297 --epochs=15 --validation_split=0.0857066310942173 --width=93 --height=93 --dense=8 --dense_units=16 --conv=16 --conv_filters=16,93,0.0857066310942173,0.23433385196421297,15,1.636012,0,None,arbeitsrechner
1719298794,1719298822,28,bash /home/s3811141/repos/OmniOpt/.tests/example_network/run.sh --learning_rate=0.11660899678524585 --epochs=6 --validation_split=0.04056024849414826 --width=60 --height=60 --dense=8 --dense_units=16 --conv=16 --conv_filters=16,60,0.04056024849414826,0.11660899678524585,6,1.624952,0,None,arbeitsrechner
1719298823,1719298881,58,bash /home/s3811141/repos/OmniOpt/.tests/example_network/run.sh --learning_rate=0.21590755908098072 --epochs=10 --validation_split=0.1567445032298565 --width=76 --height=76 --dense=8 --dense_units=16 --conv=16 --conv_filters=16,76,0.1567445032298565,0.21590755908098072,10,1.64169,0,None,arbeitsrechner
1719298882,1719298920,38,bash /home/s3811141/repos/OmniOpt/.tests/example_network/run.sh --learning_rate=0.1230122183514759 --epochs=6 --validation_split=0.07228925675153733 --width=72 --height=72 --dense=8 --dense_units=16 --conv=16 --conv_filters=16,72,0.07228925675153733,0.1230122183514759,6,1.639097,0,None,arbeitsrechner
1719298921,1719298947,26,bash /home/s3811141/repos/OmniOpt/.tests/example_network/run.sh --learning_rate=0.08336016669869424 --epochs=3 --validation_split=0.04752136580646038 --width=74 --height=74 --dense=8 --dense_units=16 --conv=16 --conv_filters=16,74,0.04752136580646038,0.08336016669869424,3,1.6279,0,None,arbeitsrechner
```

To add custom lines to this file, for example for analyzing certain parts of your program, you can print this to the standard output:

```python
print("OO-Info: outputname: 123")
```

This will add the column `OO_Info_outputname` to the `job_infos.csv`, and each line will have it's own output values then.

#### `oo_errors.txt`

This file, if it exists, contains a list of potential errors OmniOpt2 encountered during the run. If no errors were found, it may be empty or nonexistent.

#### `parameters.txt`

This file contains the parameter search space definition in a simple table. Example:

```
                            Experiment parameters:
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━┓
┃ Name             ┃ Type  ┃ Lower bound ┃ Upper bound ┃ Values ┃ Value-Type ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━┩
│ width_and_height │ range │ 60          │ 100         │        │ int        │
│ validation_split │ range │ 0           │ 0.4         │        │ float      │
│ learning_rate    │ range │ 0.001       │ 0.4         │        │ float      │
│ epochs           │ range │ 1           │ 15          │        │ int        │
└──────────────────┴───────┴─────────────┴─────────────┴────────┴────────────┘
```

### `state_files`

These files store some states used mainly to continue jobs. Not all of these files may be present.

- *checkpoint.json*: A JSON file containing all data needed to restore the experiment
- *checkpoint.json.parameters.json*: A list of parameters for this run
- *result_errors.log*: A list of errors for the run regarding the results
- *env*: A dump of the environment, OmniOpt2 works in (useful for debugging)
- *experiment_name*: The name of this experiment
- *time_overview.txt*: A table of the runtimes (min, max, avg, median)
- *failed_jobs*: The number of failed jobs
- *global_vars.json*: A variable that contains several global states that continued runs need to continue
- *experiment_overview.txt*: A table showing different parameters relating to the current experiment
- *gpus*: The number of GPUs this run has allocated per worker
- *joined_run_program*: The program string including parameters
- *get_next_nr_steps_tables.txt*: A debug table for get_next_nr_steps (only on SLURM-systems)
- *max_eval*: The max evals of this run
- *mem_gb *: The amount of Memory allocated per worker (in GB)
- *minimize *: If it exists, it means, the job was about to minimize
- *maximize *: If it exists, it means, the job was about to maximize
- *pd.json*: Contains data to restore the `ax_client`
- *phase_random_steps*: How many random steps have been generated
- *phase_systematic_steps*: How many non-random steps have been generated
- *run.sh*: A bash-file that allows you to re-run this program
- *submitted_jobs*: The number of submitted jobs
- *result_names_overview.txt*: The rich overview-table of all result names
- *time*: The time this job-sbatch has allocated
- *defective_nodes*: A list of nodes that were detected as defective, i.e. a GPU was allocated but none was given. Requires `--auto_exclude_defective_hosts` to be set.
- *succeeded_jobs*: Contains one line for each succeeded job.
- *args_overview.txt*: A table that shows all arg-parameters given to OO.
- *ui_url.txt*: Contains the URL that this run was started by
- *pareto_front_data.json*: Data about the pareto-front (exists only for multi-objective-optimizations)
- *follow*: Exists if the job was started with --follow. Only used for continued jobs.
- *generation_strategy.txt*: Infos about the generation strategy used internally.
- *main_process_gb*: Saves the amount of memory in the main process (if applicable). Only used for continued jobs.
- *original_ax_client_before_loading_tmp_one.json*: Required to re-load generation strategy in continued jobs
- *loaded_modules.json*: A list of all loaded modules with module information where available
- *cpu_ram_usage.csv*: Logs CPU and RAM usage
- *result_names.txt*: A list of RESULT names
- *pareto_front_table.txt*: If multiple RESULT names are specified, this is the table of the calculated pareto-front
- *run_uuid*: A file containing a [UUID](https://de.wikipedia.org/wiki/Universally_Unique_Identifier) for each run
- *uuid_of_continued_run*: A file containing a [UUID](https://de.wikipedia.org/wiki/Universally_Unique_Identifier) from the job it has been continued from
- *ax_client.experiment.json*: A JSON file containing the whole ax-client

### `eval_nodes_cpu_ram_logs.txt`

Run once a second on every host that runs the evaluate function to log CPU and memory usage.

### `single_runs`

This contains one folder for each subjob (i.e. single evaluation) that ran. Locally, it starts at a random number. On systems with SLURM, the folder names are the SLURM IDs.

Inside each folder there are 4 files:

- 2728975_0_log.err: This contains the stderr of the job.
- 2728975_0_log.out: This contains the stdout of the job. Usually, this is where you need to look if you want to search for why something has gone wrong.
- 2728975_0_result.pkl: Contains the result in a pickle-file
- 2728975_submitted.pkl: Contains job infos in a pickle file

(Replace '2728975' with the SLURM-Job-ID).
