# OmniOpt2
Basically the same as OmniOpt, but based on ax/botorch instead of hyperopt

# Main program

```command
./omniopt --partition=alpha --experiment_name=example --mem_gb=1 --time=60 --worker_timeout=60 --max_eval=500 --num_parallel_jobs=500 --gpus=1 --follow --run_program=ZWNobyAiUkVTVUxUOiAlKHBhcmFtKSI= --parameter param range 0 1000 float
```

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
