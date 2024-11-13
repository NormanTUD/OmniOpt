# OmniOpt - Hyperparameter Optimizer for SLURM-based Systems

![Current build status](https://github.com/NormanTUD/OmniOpt/actions/workflows/main.yml/badge.svg?event=push)
![Latest Release](https://img.shields.io/github/v/release/NormanTUD/OmniOpt)
![Open Issues](https://img.shields.io/github/issues/NormanTUD/OmniOpt)

![Open Pull Requests](https://img.shields.io/github/issues-pr/NormanTUD/OmniOpt)
![License](https://img.shields.io/badge/license-GNU-blue.svg)
![Bug Issues](https://img.shields.io/github/issues/NormanTUD/OmniOpt/bug)

![GitHub Repo stars](https://img.shields.io/github/stars/NormanTUD/OmniOpt)
![Pull Requests](https://img.shields.io/github/issues-pr/NormanTUD/OmniOpt)
![Stars](https://img.shields.io/github/stars/NormanTUD/OmniOpt)

![Forks](https://img.shields.io/github/forks/NormanTUD/OmniOpt)
![Contributors](https://img.shields.io/github/contributors/NormanTUD/OmniOpt)
![Last Commit](https://img.shields.io/github/last-commit/NormanTUD/OmniOpt)

[![Coverage Status](https://coveralls.io/repos/github/NormanTUD/OmniOpt/badge.svg?branch=main)](https://coveralls.io/github/NormanTUD/OmniOpt?branch=main)

OmniOpt is a tool designed to assist researchers, engineers, and data
scientists with hyperparameter optimization on SLURM-based clusters. It
simplifies large-scale optimization tasks with built-in fault tolerance and
flexibility. A graphical user interface (GUI) is available for command
creation, accessible at [OmniOpt GUI](https://imageseg.scads.de/omniax/).
For tutorials on configuration, exit codes, and debugging, visit
[OmniOpt Tutorials](https://imageseg.scads.de/omniax/tutorials.php).

## Main program

```command
./omniopt --partition=alpha --experiment_name=example --mem_gb=1 --time=60 \
    --worker_timeout=60 --max_eval=500 --num_parallel_jobs=500 --gpus=1 \
    --follow --run_program=ZWNobyAiUkVTVUxUOiAlKHBhcmFtKSI= \
    --parameter param range 0 1000 float
```

This command initiates OmniOpt and installs dependencies if not already
installed. The parameter `--run_program` uses a Base64-encoded string to
specify commands.

## Plot Results

Generates visualizations, such as scatter and hex scatter plots.
`--min` and `--max` adjust the plotted result value range.

Or, with `--min` and `--max`:

```command
./plot --run_dir runs/example/0
./plot --run_dir runs/example/0 --min 0 --max 100
```

## Run Tests (Developer Use Only)

The test suite simulates various scenarios, including handling faulty
jobs and ensuring program resilience.

```command
./tests/main_tests
```

## Install from pypi

```command
pip3 install omniopt2
```

## Install from repo

```command
pip3 install -e git+https://github.com/NormanTUD/OmniOpt2.git#egg=OmniOpt2
```

Alternatively, it can be executed directly, as OmniOpt will install its
dependencies automatically if required.

## Error Codes

For common issues and exit codes, see the
[exit codes tutorial](https://imageseg.scads.de/omniax/tutorials.php?tutorial=exit_codes_and_bash_scripting).

## Contributions

I'd be glad to see your contributions!
