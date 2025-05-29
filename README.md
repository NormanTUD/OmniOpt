# OmniOpt2 - Hyperparameter Optimizer for SLURM-based Systems

![Current build status](https://github.com/NormanTUD/OmniOpt/actions/workflows/main.yml/badge.svg?event=push)

OmniOpt2 is a tool designed to assist researchers, engineers, and data
scientists with hyperparameter optimization on SLURM-based clusters, even
though it works without it as well. It simplifies large-scale optimization
tasks with built-in fault tolerance and flexibility. A graphical user interface
(GUI) is available for command creation, accessible at 
[OmniOpt2 GUI](https://imageseg.scads.de/omniax/gui). For tutorials on 
configuration, exit codes, and debugging, visit
[OmniOpt2 Tutorials](https://imageseg.scads.de/omniax/tutorials).

## Main program

```command
omniopt --partition=alpha --experiment_name=example --mem_gb=1 --time=60 \
    --worker_timeout=60 --max_eval=500 --num_parallel_jobs=500 --gpus=1 \
    --follow --run_program=$(echo 'echo "RESULT: %(param)"' | base64 -w0) \
    --parameter param range 0 1000 float
```

This command initiates OmniOpt2 and installs dependencies if not already
installed. The parameter `--run_program` uses a
[Base64](https://de.wikipedia.org/wiki/Base64)-encoded string to
specify commands. It is recommended to use the
[GUI](https://imageseg.scads.de/omniax/gui), though.

## Plot Results

Generates visualizations, such as scatter and hex scatter plots.
`--min` and `--max` adjust the plotted result value range.

Or, with `--min` and `--max`:

```command
omniopt_plot --run_dir runs/example/0
omniopt_plot --run_dir runs/example/0 --min 0 --max 100
```

## Using live-share

Use `--live_share` (also enablable via GUI) to automatically share the job. You will get a URL
where your job data is hosted publically for 30 days, meaning everyone can access your results,
and you can see all kinds of visualizations and export them.

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

This may not use the bleeding-edge version, but if you get the version from here it means, the test suite has completety tested it properly.

```command
pip3 install omniopt2
```

## Install from repo (bleeding edge, may contain untested changes)

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

## Issues

If you experience any problems, please write issues at my [Github Issues page](https://github.com/NormanTUD/OmniOpt/issues).

## Old OmniOpt

The old OmniOpt version, based on HyperOpt, is not supported anymore. It is still available, though, at [https://github.com/NormanTUD/LegacyOmniOpt](https://github.com/NormanTUD/LegacyOmniOpt).
