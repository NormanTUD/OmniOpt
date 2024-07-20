# OmniOpt2
Basically the same as OmniOpt, but based on ax/botorch instead of hyperopt

# Main program

```command
./omniopt --partition=alpha --experiment_name=example --mem_gb=1 --time=60 --worker_timeout=60 --max_eval=500 --num_parallel_jobs=500 --gpus=1 --follow --run_program=ZWNobyAiUkVTVUxUOiAlKHBhcmFtKSI= --parameter param range 0 1000 float
```

This will automatically install all dependencies. Internally, it calls a pythonscript. 

# Show results

```command
./evaluate-run
```

# Plot results

```command
./plot --run_dir runs/example/0
# Or, with --min and --max:
./plot --run_dir runs/example/0 --min 0 --max 100
```

# Run tests

Runs the main test suite. Runs an optimization, continues it, tries to continue one that doesn't exit, and runs a job with many different faulty jobs that fail in all sorts of ways (to test how OmniOpt2 reacts to it).

```command
./tests/main_tests
```

# Install from repo

`pip3 install -e git+https://github.com/NormanTUD/OmniOpt2.git#egg=OmniOpt2`

# Exit-Codes

| Exit Code | Error group description                                                      |
|-----------|------------------------------------------------------------------------------|
| 2         | Loading of environment failed                                                |
| 3         | Invalid exit code detected                                                   |
| 10        | Usually only returned by dier (for debugging).                               |
| 11        | Required program not found (check logs)                                      |
| 12        | Error with pip, check logs                                                   |
| 13        | Run folder already exists                                                    |
| 15        | Unimplemented error.                                                         |
| 18        | test_wronggoing_stuff program not found (only --tests).                      |
| 19        | Something was wrong with your parameters. See output for details.            |
| 31        | Basic modules could not be loaded or you cancelled loading them.             |
| 44        | Continuation of previous job failed.                                         |
| 47        | Missing checkpoint or defective file or state files (check output).          |
| 49        | Something went wrong while creating the experiment.                          |
| 89        | Search space exhausted or search cancelled.                                  |
| 99        | It seems like the run folder was deleted during the run.                     |
| 100       | --mem_gb or --gpus, which must be int, has received a value that is not int. |
| 103       | --time is not in minutes or HH:MM:SS format                                  |
| 104       | One of the parameters --mem_gb, --time, or --experiment_name is missing.     |
| 105       | Continued job error: previous job has missing state files.                   |
| 181       | Error parsing --parameter. Check output for more details.                    |
| 192       | Unknown data type (--tests).                                                 |
| 193       | Error in printing logs. You may be on a read only file system.               |
| 199       | This happens on unstable file systems when trying to write a file.           |
| 203       | Unsupported --model.                                                         |
| 233       | No random steps set.                                                         |
| 242       | Error in Models like THOMPSON or EMPIRICAL_BAYES_THOMPSON. Not sure why.     |
| 243       | Job was not found in squeue anymore, it may got cancelled before it ran.     |
