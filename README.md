# OmniOpt2

OmniOpt2 is a highly parallel hyperparameter optimizer for Linux, SLURM
clusters, and local workstations. It explores a parameter search space, runs
your program with the proposed parameters, reads a result from its output, and
uses Ax/BoTorch, Optuna, or other generators to decide what to try next.

OmniOpt2 is program-agnostic: your program can be Python, C, R, a shell script,
a simulation, or anything else that runs on Linux. It only needs to

1. accept parameters on the command line, and
2. print a result line such as `RESULT: 0.123`.

- GUI: <https://imageseg.scads.de/omniax/gui>
- Tutorials: <https://imageseg.scads.de/omniax/tutorials>
- TU Dresden HPC: <https://compendium.hpc.tu-dresden.de/software/hyperparameter_optimization/>
- Issues: <https://github.com/NormanTUD/OmniOpt/issues>

## Contents

- [Highlights](#highlights)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [HPC, SLURM, Lmod, and Virtual Environments](#hpc-slurm-lmod-and-virtual-environments)
- [Optuna](#optuna)
- [Configuration Files and Continuation](#configuration-files-and-continuation)
- [Results, Plots, Evaluation, and Sharing](#results-plots-evaluation-and-sharing)
- [Exit Codes and Diagnostics](#exit-codes-and-diagnostics)
- [Tests and Development](#tests-and-development)
- [License and Contact](#license-and-contact)

## Highlights

- **GUI and CLI**: build a complete `omniopt` command in the browser or use the
  CLI directly.
- **SLURM and local execution**: workers run as SLURM jobs when `sbatch` is
  available, otherwise locally through submitit's local executor.
- **HPC-ready environment handling**: automatic Lmod/Load-Modules detection,
  optional per-evaluation `--modules` and `--venv`, and automatic
  re-exec/bootstrap with the right Python.
- **Many generators**: BoTorch GP (`BOTORCH_MODULAR`), SAASBO, mixed-variable
  BO, Sobol, factorial/grid search, random search, random forest, external
  generators, TPE, and Optuna samplers including multi-objective ones.
- **Multi-objective optimization**: multiple `--result_names`, Pareto fronts,
  optional OCC (optimization with combined criteria), and constraints.
- **Formula mode**: optimize a mathematical expression directly with
  `--formula` (LaTeX or Python-style infix) instead of a run script.
- **Robust runs**: checkpoints, continuation, state files, orchestrator hooks,
  automatic exclusion of defective GPU nodes, and actionable diagnostics for
  failed evaluations.
- **Results and sharing**: CSV results, interactive plots, terminal graphics,
  OmniOpt2-Share, and optional live sharing.

## Installation

OmniOpt2 is self-installing. The first run creates a virtual environment under
`~/.omniax_venvs/<python-version>/<architecture>` and installs its dependencies
there. Later runs reuse it. Use `--root_venv_dir=/path` to choose a different
base directory.

### Run directly from the repository

```bash
git clone https://github.com/NormanTUD/OmniOpt.git
cd OmniOpt
./omniopt --help
./omniopt --install    # bootstrap dependencies and exit
```

### Install with pip

The PyPI version is the released version:

```bash
pip3 install omniopt2
omniopt --help
```

The bleeding-edge version from Git:

```bash
pip3 install "git+https://github.com/NormanTUD/OmniOpt.git"
```

### Docker

```bash
./omniopt_docker
./omniopt_docker omniopt --tests
```

The current directory is mounted inside the container. In the GUI, the run mode
can be switched to Docker. Docker is intended for local use and is not
supported on the HPC systems.

### Requirements

- Linux
- `python3`
- Optional: SLURM (`sbatch`, `squeue`, ...)
- Optional: an Lmod/Environment Modules installation (`ml`, `module`)
- Optional: Docker
- System packages listed in `apt-dependencies.txt`

## Quick Start

### GUI

Open the [OmniOpt2 GUI](https://imageseg.scads.de/omniax/gui), fill in the
experiment, copy the generated command, and run it on the login node or
locally. The GUI also documents every option and can produce a reproducible
configuration.

### CLI

```bash
./omniopt \
    --experiment_name=example \
    --mem_gb=1 \
    --time=60 \
    --worker_timeout=60 \
    --max_eval=100 \
    --num_parallel_jobs=4 \
    --num_random_steps=10 \
    --gpus=0 \
    --model=BOTORCH_MODULAR \
    --result_names RESULT=min \
    --parameter x range 0 10 float \
    --run_program="$(printf '%s' 'bash ./run.sh --x=%(x)' | base64 -w0)" \
    --follow \
    --force_local_execution
```

Remove `--force_local_execution` and add `--partition=<name>` to run on SLURM.
`--run_program` is Base64-encoded so spaces, quotes, and newlines survive shell
parsing. The GUI generates this automatically.

A minimal `run.sh`:

```bash
#!/bin/bash
# $@ contains --x=<value>
python3 train.py "$@"
```

The program must print a result that OmniOpt2 can parse:

```text
RESULT: 0.12345
```

Use `--result_names LOSS=min ACCURACY=max` for multiple objectives and
`--occ` for optimization with combined criteria. Without an explicit
`min`/`max`, minimizing is assumed. See the
[run.sh tutorial](https://imageseg.scads.de/omniax/tutorials?tutorial=run_sh)
and the [results.csv documentation](https://imageseg.scads.de/omniax/tutorials?tutorial=results_csv).

### Parameters and search space

```bash
--parameter learning_rate range 0.0001 0.5 float
--parameter epochs choice 1,10,20,50,100
--parameter layers fixed 10
--parameter dropout range 0 0.8 float true
```

Range parameters support explicit log scaling. Fixed parameters are passed to
the program but do not consume search dimensions. Use
`--experiment_constraints` for linear constraints such as
`a + b <= 10`. See the
[constraints tutorial](https://imageseg.scads.de/omniax/tutorials?tutorial=constraints).

### Models

The current model list is available with `./omniopt --help` and includes:

- `BOTORCH_MODULAR` (default BoTorch GP)
- `SAASBO`, `BO_MIXED`
- `SOBOL`, `FACTORIAL`, `UNIFORM`, `PSEUDORANDOM`
- `RANDOMFOREST`
- `TPE`
- `EXTERNAL_GENERATOR`
- `OPTUNA_TPE`, `OPTUNA_CMAES`, `OPTUNA_GP`, `OPTUNA_RANDOM`,
  `OPTUNA_GRID`, `OPTUNA_QMC`, `OPTUNA_BruteForce`, `OPTUNA_NSGAII`,
  `OPTUNA_NSGAIII`, `OPTUNA_MOTPE`

See the [surrogate models tutorial](https://imageseg.scads.de/omniax/tutorials?tutorial=models)
for guidance on choosing a model. Custom Ax generation strategies can be passed
with `--generation_strategy` (see the
[custom generation strategy tutorial](https://imageseg.scads.de/omniax/tutorials?tutorial=custom_generation_strategy)),
and external optimizers can be connected with `--model=EXTERNAL_GENERATOR`
and `--external_generator=<program>`.

### Formula mode

Instead of a run program, you can optimize an equation:

```bash
./omniopt \
    --experiment_name=formula_example \
    --formula 'x**2 + y**2' \
    --formula_mode=infix \
    --parameter x range -10 10 float \
    --parameter y range -10 10 float \
    --max_eval=50 --num_random_steps=10 --mem_gb=1 --time=30 \
    --result_names RESULT=min --force_local_execution
```

LaTeX and Python-style infix are supported. See the
[formula tutorial](https://imageseg.scads.de/omniax/tutorials?tutorial=formulas).

## HPC, SLURM, Lmod, and Virtual Environments

When `sbatch` is available, OmniOpt2 submits the main process as a SLURM job
unless `--force_local_execution` is set. Workers are scheduled through
submitit. Common SLURM options include:

```bash
--partition=alpha --account=<account> --reservation=<name> \
--gpus=1 --cpus_per_task=4 --nodes_per_job=1 \
--time=120 --worker_timeout=30 --num_parallel_jobs=20 \
--dependency=afterok:<jobid> --follow --auto_exclude_defective_hosts
```

`--dependency=omniopt_singleton` serializes OmniOpt2 jobs with the same name.
`--wait_until_ended` waits for the submitted main job and prints its log.

### Lmod modules per evaluation

Use `--modules` to load Lmod modules in every evaluation. This replaces a
`run.sh` that only does `module load ...; python3 ...`.

```bash
--modules='ml release/24.10 GCCcore/13.3.0 Python/3.12.3'
```

All of these are accepted and mean the same thing:

```text
release/24.10 GCCcore/13.3.0 Python/3.12.3
module release/24.10 GCCcore/13.3.0 Python/3.12.3
ml release/24.10 GCCcore/13.3.0 Python/3.12.3
module load release/24.10 GCCcore/13.3.0 Python/3.12.3
```

Multiple lines are allowed. On systems without Lmod, OmniOpt2 prints a warning
and continues without loading the modules. Modules are checked before the
main SLURM job is submitted, so a typo is reported immediately on the login
node. A failing module load is reported as a red, actionable error and OmniOpt2
exits with code `35`. See the
[TU Dresden module documentation](https://compendium.hpc.tu-dresden.de/software/modules/)
for release and toolchain examples.

### Virtualenv per evaluation

Use `--venv` to activate a virtualenv inside every evaluation:

```bash
--venv=/path/to/venv
```

The order is always:

```text
modules -> virtualenv -> run program
```

Use `--workdir=/path` if relative paths in your run program should resolve from
a specific directory instead of the one where OmniOpt2 was started. Use
`--run_program_once` for setup steps that should run once before the search.

`--venv` may point to the virtualenv directory or directly to its
`bin/activate` script. The main OmniOpt2 process is not affected. A missing or
broken virtualenv is reported inside the evaluation with exit code `36` and an
actionable message.

Both `--modules` and `--venv` are stored in the run's state files and restored
when continuing a run. See the
[virtualenv tutorial](https://imageseg.scads.de/omniax/tutorials?tutorial=virtualenv).

### Automatic Lmod handling

OmniOpt2 also detects Lmod automatically to build its own virtual environment
with the cluster's Python/toolchain. Set `LOAD_MODULES=0` to disable the
automatic cluster toolchain snippet. Set `OMNIOPT_INSTALL_DEBUG=1` for a
detailed installer trace (detected Lmod commands, Python candidates, venv
rebuilds, pip attempts).

## Optuna

`omniopt_optuna` exposes Optuna samplers through OmniOpt2:

```bash
# Single-objective TPE
omniopt_optuna run --max-eval=20 --parameter 'x range -10 10 int' \
    --run-program 'echo "RESULT: %(x)"'

# Multi-objective NSGA-II
omniopt_optuna run --model=OPTUNA_NSGAII --max-eval=30 \
    --result-names RESULT1=min RESULT2=max \
    --parameter 'x range 0 5 float' \
    --run-program 'echo "RESULT1: %(x)"; echo "RESULT2: %(x)"'
```

Persistent studies and remote control:

```bash
omniopt_optuna study create --workdir runs/my_study --sampler=tpe
omniopt_optuna study add --workdir runs/my_study --params-file p.json \
    --values '{"RESULT": 3.5}'
omniopt_optuna study suggest --workdir runs/my_study --parameters-file spec.json
```

See the [Optuna tutorial](https://imageseg.scads.de/omniax/tutorials?tutorial=optuna).

## Configuration Files and Continuation

OmniOpt2 is CLI-first, but it can also read YAML, JSON, or TOML:

```bash
./omniopt --config_yaml config.yaml
./omniopt --config_json config.json
./omniopt --config_toml config.toml
```

Use `--dump_config` to write the effective configuration.

Continue a previous run with:

```bash
./omniopt --continue runs/my_experiment/0
```

State files and `run.sh` are written into the run directory. See the
[continue tutorial](https://imageseg.scads.de/omniax/tutorials?tutorial=continue_job)
and the [folder structure](https://imageseg.scads.de/omniax/tutorials?tutorial=folder_structure).

For custom recovery behavior, `--orchestrator_file` can react to errors in the
worker output (restart, exclude a node, etc.). See the
[orchestrator tutorial](https://imageseg.scads.de/omniax/tutorials?tutorial=orchestrator).

## Results, Plots, Evaluation, and Sharing

Each run stores its results and state under `runs/<experiment>/<number>/` by
default (use `--run_dir` to change the base directory). The main result file is
`results.csv`.

```bash
# Plot a finished run
omniopt_plot --run_dir runs/example/0
omniopt_plot --run_dir runs/example/0 --plot_type=scatter
omniopt_plot --run_dir runs/example/0 --plot_type=scatter_hex --min 0 --max 100

# Interactive terminal evaluation
omniopt_evaluate --projectdir=runs

# Upload a run (link valid for 30 days)
omniopt_share runs/example/0
```

Plot types include `cpu_ram_usage`, `general`, `gpu_usage`, `kde`, `scatter`,
`scatter_generation_method`, `scatter_hex`, `time_and_exit_code`,
`trial_index_result`, and `worker`.

Use `--live_share` to upload results while the run is active. Both live sharing
and `--send_anonymized_usage_stats` are opt-in. See the
[OmniOpt2-Share tutorial](https://imageseg.scads.de/omniax/tutorials?tutorial=oo_share).

## Exit Codes and Diagnostics

OmniOpt2 uses distinct exit codes so scripts can react precisely. The complete
list is in the
[exit codes tutorial](https://imageseg.scads.de/omniax/tutorials?tutorial=exit_codes_and_bash_scripting).
Recent additions include:

- `4`: bootstrap/import failure while loading base modules
- `35`: the Lmod modules requested with `--modules` could not be loaded
- `36`: the virtualenv requested with `--venv` could not be activated
- `87`: search space exhausted
- `89`: all jobs failed
- `248`: the external generator failed repeatedly (its input folder is kept)

Failed evaluations include a red summary panel with actionable hints for common
problems: missing results, Python traceback output, missing files, wrong working
directory, missing Python packages, and malformed result values. For a quick
sanity check without a full optimization run, use:

```bash
./omniopt --dryrun ...
```

Use `--verbose`, `--debug`, or `--prettyprint` for more diagnostics. See the
[debugging tutorial](https://imageseg.scads.de/omniax/tutorials?tutorial=debug)
and the [dry-run tutorial](https://imageseg.scads.de/omniax/tutorials?tutorial=dryrun).

## Tests and Development

The test suite lives in `.tests`:

```bash
./omniopt --tests                   # quick built-in self-test
python3 .tests/main                 # run everything
python3 .tests/main --quick         # quick subset
python3 .tests/main --list          # list matching tests
python3 .tests/main --only=suite:basic
python3 .tests/main --no_plots --no_linter
python3 .tests/smoke_tests          # pre-commit smoke tests
python3 .tests/test_lmod_modules_and_venv  # real Lmod/venv integration checks
```

The Lmod/venv test runs the real `./omniopt` binary for the no-Lmod, fake-Lmod,
failing-module, and real-Lmod scenarios and prints the full command and output.
The smoke tests also cover the GUI JavaScript, PHP, Python unit tests, linting,
typing, and the installer.

Repository entry points:

| Path | Purpose |
| --- | --- |
| `omniopt` | Main optimizer and bootstrap |
| `omniopt_plot` | Plot results |
| `omniopt_evaluate` | Interactive result browser |
| `omniopt_share` | Upload/share results |
| `omniopt_optuna` | Optuna runner and study control |
| `omniopt_docker` | Docker build/run helper |
| `.ax.py` | Ax/BoTorch backend abstraction |
| `.executor_backend.py` | submitit/SLURM/local execution backend |
| `.gui/` | PHP/JavaScript GUI and documentation |
| `.tests/` | Test framework and test suite |

## Shell Completion

Completion definitions for bash and zsh are in `.shells`:

```bash
bash .shells/install
```

See [.shells/README.md](.shells/README.md) for details.

## License and Contact

OmniOpt2 is licensed under the GNU General Public License v3. See
[LICENSE](LICENSE).

- Technical support: norman.koch@tu-dresden.de
- Ideas and original concept: peter.winkler1@tu-dresden.de
- Issues: <https://github.com/NormanTUD/OmniOpt/issues>

## Old OmniOpt

The old HyperOpt-based OmniOpt version is no longer supported. It remains
available at <https://github.com/NormanTUD/LegacyOmniOpt>.
