# OmniOpt2

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

A hyperparameter optimizer for SLURM-based systems. GUI for creating commands is available at https://imageseg.scads.de/omniax/ , where also additional help can be found.

## Main program

```command
./omniopt --partition=alpha --experiment_name=example --mem_gb=1 --time=60 \
    --worker_timeout=60 --max_eval=500 --num_parallel_jobs=500 --gpus=1 \
    --follow --run_program=ZWNobyAiUkVTVUxUOiAlKHBhcmFtKSI= \
    --parameter param range 0 1000 float
```

This will automatically install all dependencies. Internally, it calls a
python-script.

## Show results

```command
./omniopt_evaluate
```

## Plot results

```command
./plot --run_dir runs/example/0
```

Or, with --min and --max:

```command
./plot --run_dir runs/example/0 --min 0 --max 100
```

## Run tests

Runs the main test suite. Runs an optimization, continues it, tries to
continue one that doesn't exit, and runs a job with many different faulty jobs
that fail in all sorts of ways (to test how OmniOpt2 reacts to it).

```command
./tests/main_tests
```

## Install from repo

```command
pip3 install -e git+https://github.com/NormanTUD/OmniOpt2.git#egg=OmniOpt2
```

## Install from pypi

```command
pip3 install omniopt2
```
