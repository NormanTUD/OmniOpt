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

| Exit Code | Error group description                                                     |
|-----------|-----------------------------------------------------------------------------|
| 100       | --mem_gb or --gpus, which must be int, has received a value that is not int |
| 103       | --time is not in minutes or HH:MM:SS format                                 |
| 104       | One of the parameters --mem_gb, --time, or --experiment_name is missing     |
| 105       | Continued job error: previous job has missing state files                   |
| 243       | Job was not found in squeue anymore, it may got cancelled before it ran     |
