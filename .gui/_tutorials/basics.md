# Basics and Docker

<!-- How to use OmniOpt locally, on HPC-Systems or in Docker -->

<div id="toc"></div>

## What is OmniOpt2 and what does it do?

OmniOpt2 is a highly parallelized hyperparameter optimizer based on Ax/Botorch. It explores various combinations of parameters within a given search space, runs a program with these parameters, and identifies promising areas for further evaluations.

## Key Features

- **Simple Hyperparameter Optimization**: OmniOpt2 allows for easy hyperparameter optimization within defined ranges.
- **Tool Agnostic**: It is completely agnostic regarding the code it runs. OmniOpt2 only requires command-line arguments with the hyperparameters and expects the program to output results, e.g., `print(f"RESULT: {loss}")`. [See here how to prepare your program for the use with OmniOpt2](tutorials.php?tutorial=run_sh)
- **Self-Installation**: OmniOpt2 installs itself into a virtual environment.
- **No Configuration Files**: All configuration is handled through the command line interface (CLI).

## Installation

OmniOpt2 is self-installing and does not require any additional manual setup. Simply run the `curl` command provided by the [GUI](index.php) and it will install itself into a virtual environment.

## Usage

```
./omniopt --partition=alpha --experiment_name=my_experiment --mem_gb=1 --time=60 --worker_timeout=30 --max_eval=500 --num_parallel_jobs=20 --gpus=0 --num_random_steps=20 --follow --show_sixel_graphics --run_program=YmFzaCAvcGF0aC90by9teV9leHBlcmltZW50L3J1bi5zaCAtLWVwb2Nocz0lKGVwb2NocykgLS1sZWFybmluZ19yYXRlPSUobGVhcm5pbmdfcmF0ZSkgLS1sYXllcnM9JShsYXllcnMp --cpus_per_task=1 --send_anonymized_usage_stats --model=BOTORCH_MODULAR --parameter learning_rate range 0 0.5 float --parameter epochs choice 1,10,20,30,100 --parameter layers fixed 10
```

This command includes all necessary options to run a hyperparameter optimization with OmniOpt2.

### Parameters

- `--partition=alpha`: Specifies the partition to use.
- `--experiment_name=my_experiment`: Sets the name of the experiment.
- `--mem_gb=1`: Allocates 1 GB of memory for the job.
- `--time=60`: Sets a timeout of 60 minutes for the job.
- `--worker_timeout=30`: Sets a timeout of 30 minutes for workers.
- `--max_eval=500`: Limits the maximum number of evaluations to 500.
- `--num_parallel_jobs=20`: Runs up to 20 parallel jobs.
- `--gpus=0`: Specifies the number of GPUs to use.
- `--num_random_steps=20`: Sets the number of random steps to 20.
- `--follow`: Follows the job's progress.
- `--show_sixel_graphics`: Displays sixel graphics.
- `--run_program=YmFzaCAvcGF0aC90by9teV9leHBlcmltZW50L3J1bi5zaCAtLWVwb2Nocz0lKGVwb2NocykgLS1sZWFybmluZ19yYXRlPSUobGVhcm5pbmdfcmF0ZSkgLS1sYXllcnM9JShsYXllcnMp`: Specifies the base64-encoded command to run the program. In this case, it resolves to `bash /path/to/my_experiment/run.sh --epochs=%(epochs) --learning_rate=%(learning_rate) --layers=%(layers)`.
- `--cpus_per_task=1`: Allocates 1 CPU per task.
- `--send_anonymized_usage_stats`: Sends anonymized usage statistics.
- `--model=BOTORCH_MODULAR`: Specifies the optimization model to use.
- `--parameter learning_rate range 0 0.5 float`: Defines the search space for the learning rate.
- `--parameter epochs choice 1,10,20,30,100`: Defines the choices for the epochs parameter.
- `--parameter layers fixed 10`: Sets the layers parameter to a fixed value of 10.

## `--run_program`

The `--run_program` parameter needs the program to be executed as a base64-string because parsing spaces and newline in bash, where it is partially evaluated, [is very difficult](https://en.wikipedia.org/wiki/Delimiter#Delimiter_collision). However, it is possible to use a human-readable string, though it has to be converted to base64 by your shell:

```
--run_program=$(echo -n "bash /path/to/my_experiment/run.sh --epochs=%(epochs) --learning_rate=%(learning_rate) --layers=%(layers)" | base64 -w 0)
```

## Integration

OmniOpt2 is compatible with any program that can run on Linux, regardless of the programming language (e.g., C, Python). The program must accept parameters via command line and output a result string (e.g., `print(f"RESULT: {loss}")`).

## Scalability

OmniOpt2, when SLURM is installed, automatically starts sub-jobs on different nodes to maximize resource utilization. Use the flag `--num_parallel_jobs n` with n being the number of workers you want to start jobs in parallel. When no SLURM is installed on your system, OmniOpt2 will run the jobs sequentially.

## Error Handling

OmniOpt2 provides helpful error messages and suggestions for common issues that may arise during usage.

## Use Cases

OmniOpt2 is particularly useful in fields such as AI research and simulations, where hyperparameter optimization can significantly impact performance and results.

## Run locally or in Docker

You can also run OmniOpt2 locally or inside docker.

### Run locally

To run OmniOpt2 locally, simply fill the GUI, copy the curl-command, and run it locally. OmniOpt2 will be installed into a virtualenv once in the beginning, which may take up to 20 minutes. From then on, it will not install itself again, so you only need to wait once.

### Run in docker

To build a docker container, simply run `./omniopt_docker` in the main folder. Docker is not supported on the HPC system, though. If you have Debian or systems based on it, it will automatically install docker if it's not installed. For other systems, you need to install docker yourself.

The `./omniopt_docker` command will build the container. You can also run several commands directly from the `./omniopt_docker` command like this:

```
./omniopt_docker omniopt --tests
```

For example, this will install docker (on Debian), build the container, and run OmniOpt2 with the `--tests` parameter.

The current folder where you run the following command from is mounted inside docker as `/var/opt/omniopt/docker_user_dir`.

Keep your program there and use this as a base path for your run in the GUI. In the GUI, in the additional-parameters-table, you can choose "Run-Mode" → Docker to automatically start jobs generated by the GUI in docker.

## Contact

Idea: peter.winkler1 at tu-dresden.de, Technical Support: norman.koch at tu-dresden.de.

