# OmniOpt2 - Hyperparameter Optimizer for SLURM-based Systems

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

OmniOpt2 is a tool designed to assist researchers, engineers, and data
scientists with hyperparameter optimization on SLURM-based clusters, even
though it works without it as well. It simplifies large-scale optimization
tasks with built-in fault tolerance and flexibility. A graphical user interface
(GUI) is available for command creation, accessible at 
[OmniOpt2 GUI](https://imageseg.scads.de/omniax/). For tutorials on 
configuration, exit codes, and debugging, visit
[OmniOpt2 Tutorials](https://imageseg.scads.de/omniax/tutorials).

## Main program

```command
omniopt --partition=alpha --experiment_name=example --mem_gb=1 --time=60 \
    --worker_timeout=60 --max_eval=500 --num_parallel_jobs=500 --gpus=1 \
    --follow --run_program=ZWNobyAiUkVTVUxUOiAlKHBhcmFtKSI= \
    --parameter param range 0 1000 float
```

This command initiates OmniOpt2 and installs dependencies if not already
installed. The parameter `--run_program` uses a
[Base64](https://de.wikipedia.org/wiki/Base64)-encoded string to
specify commands.

## Plot Results

Generates visualizations, such as scatter and hex scatter plots.
`--min` and `--max` adjust the plotted result value range.

Or, with `--min` and `--max`:

```command
omniopt_plot --run_dir runs/example/0
omniopt_plot --run_dir runs/example/0 --min 0 --max 100
```

## Run Tests (Developer Use Only)

The test suite simulates various scenarios, including handling faulty
jobs and ensuring program resilience.

```command
./tests/main
```

See
[the automated tests tutorial page](https://imageseg.scads.de/omniax/tutorials?tutorial=tests)
for more details.

## Install from pypi

This may not use the latest version.

```command
pip3 install omniopt2
```

## Install from repo

```command
pip3 install -e git+https://github.com/NormanTUD/OmniOpt2.git#egg=OmniOpt2
```

Alternatively, it can be executed directly, as OmniOpt2 will install its
dependencies automatically if required.

## Error Codes

For common issues and exit codes, see the
[exit codes tutorial-page](https://imageseg.scads.de/omniax/tutorials?tutorial=exit_codes_and_bash_scripting).

## Autocompletions

Autocomplete files for zsh and bash are in `.shells`. Run

```bash
bash .shells/install
```

to install them.

## Contributions

I'd be glad to see your contributions!

## Old OmniOpt

The old OmniOpt version, based on HyperOpt, is not supported anymore. It is still available, though, at [https://github.com/NormanTUD/LegacyOmniOpt](https://github.com/NormanTUD/LegacyOmniOpt).
