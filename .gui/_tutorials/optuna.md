# <img class='emoji_nav' src='emojis/spy.svg' /> Optuna Integration

<!-- How to run Optuna through OmniOpt -->

<!-- Category: Models -->

<div id="toc"></div>

OmniOpt ships every Optuna sampler as a *special case* of its `ExternalProgramGenerationNode`. The integration is intentionally thin: OmniOpt already knows how to read OmniOpt-style parameters, constraints, and trials; the Optuna runner reads the same `input.json` and writes the same `results.json`. That means every Optuna feature (single-objective, multi-objective, pruning, persistent studies, remote control) is reachable through OmniOpt's existing CLI surface.

## Quick start

The simplest way to drive Optuna through OmniOpt:

```bash
omniopt_optuna --max-eval=20 \
    --parameter "x range -10 10 int" \
    --run-program 'echo "RESULT: $x"'
```

This is equivalent to:

```bash
omniopt --model=OPTUNA_TPE --max_eval=20 --num_random_steps=0 \
    --run_mode=local \
    --run_program='echo "RESULT: $x"' \
    --parameter "x range -10 10 int"
```

The `omniopt_optuna` entry point is a thin wrapper that picks a sensible default for every `--optuna_*` flag, so it is the recommended interface unless you need fine-grained control.

## Picking a sampler

Every Optuna sampler has its own `--model` value:

| Sampler | `--model` | Use case |
| --- | --- | --- |
| TPE | `OPTUNA_TPE` | Default. Works well on noisy / mixed search spaces. |
| CMA-ES | `OPTUNA_CMAES` | Continuous, low-to-medium dimensional spaces. |
| GP | `OPTUNA_GP` | Expensive evaluations; uses Gaussian Processes (experimental). |
| Random | `OPTUNA_RANDOM` | Cheap baseline / sanity check. |
| Grid | `OPTUNA_GRID` | Fully discrete search spaces. |
| QMC | `OPTUNA_QMC` | Quasi-Monte-Carlo exploration (experimental). |
| Brute force | `OPTUNA_BruteForce` | Exhaustive enumeration (experimental). |
| NSGA-II | `OPTUNA_NSGAII` | Multi-objective optimization. |
| NSGA-III | `OPTUNA_NSGAIII` | Many-objective optimization (≥3 objectives). |
| Multi-Objective TPE | `OPTUNA_MOTPE` | Multi-objective with TPE. |

Pick one explicitly or pass `--optuna_sampler=NAME` to override what the model implies.

```bash
omniopt_optuna --model=OPTUNA_CMAES --max-eval=50 \
    --parameter "lr range 1e-5 1e-1 float" \
    --run-program 'python train.py --lr=%(lr)s'
```

## Multi-objective optimization

For multi-objective runs, declare multiple objectives with `--result-names`:

```bash
omniopt_optuna --model=OPTUNA_NSGAII --max-eval=30 \
    --result-names RESULT1=min RESULT2=max \
    --parameter "x range 0 5 float" \
    --run-program 'echo "RESULT1: $((x*x))" && echo "RESULT2: $((25-x*x))"'
```

NSGA-II, NSGA-III and MOTPE require at least 2 objectives; the single-objective samplers reject configurations with more than one.

## Pruning

Optuna can stop unpromising trials early through a *pruner*. The default is `NopPruner` (no pruning). Other choices are `median`, `hyperband`, `threshold`, `successive_halving`:

```bash
omniopt_optuna --model=OPTUNA_TPE --optuna_pruner=median \
    --optuna_n_startup_trials=5 ...
```

## Persistent studies (Optuna's storage)

By default every Optuna study lives in `runs/<experiment>/0/optuna_study.db` (a SQLite file). To share a study across OmniOpt invocations, pass `--optuna_storage=URL`:

```bash
omniopt_optuna --model=OPTUNA_TPE \
    --optuna_storage=postgresql://user:pass@host/db \
    --optuna_study_name=shared_study
```

`--optuna_no_load_if_exists` makes OmniOpt always start fresh instead of continuing the named study.

## Remote control (no Python required)

`omniopt_optuna study …` exposes the Optuna study as a persistent object on disk so any process can drive it without ever importing Optuna. This is useful for distributed setups where OmniOpt runs on one machine and external workers submit results from another.

```bash
# Create the persistent study
omniopt_optuna study create --workdir runs/my_study \
    --sampler=tpe --seed=42 --objectives=RESULT

# Record trials from anywhere (a worker, a shell script, another language)
omniopt_optuna study add --workdir runs/my_study \
    --params-file p.json --values '{"RESULT": 3.5}'

# Ask Optuna for the next point
omniopt_optuna study suggest --workdir runs/my_study \
    --parameters-file spec.json

# Inspect
omniopt_optuna study best   --workdir runs/my_study
omniopt_optuna study trials --workdir runs/my_study

# Tear down
omniopt_optuna study delete --workdir runs/my_study --study-name my_study
```

The `study …` subcommands delegate to `.optuna_runner.py study …`, which is also callable directly:

```bash
python3 .optuna_runner.py study best --workdir runs/my_study
```

## All flags

| Flag | Default | What it does |
| --- | --- | --- |
| `--optuna_sampler` | (set by `--model`) | Override the default sampler. |
| `--optuna_pruner` | `none` | Choose a pruner. |
| `--optuna_n_startup_trials` | `10` | Random trials before the sampler model kicks in. |
| `--optuna_multivariate` | off | Pass `multivariate=True` to TPESampler. |
| `--optuna_group` | off | Pass `group=True` to TPESampler. |
| `--optuna_constraints` | off | Pass an empty `constraints_func` to Optuna's samplers. |
| `--optuna_n_ei_candidates` | `0` | `n_ei_candidates` for TPESampler (0 → Optuna default). |
| `--optuna_storage` | (sqlite in workdir) | Optuna storage URL. |
| `--optuna_study_name` | `omniopt_study` | Optuna study name. |
| `--optuna_no_load_if_exists` | off | Start fresh instead of continuing the study. |
| `--optuna_extra_iters` | `1` | How many extra trials Optuna runs per OmniOpt suggest call. |

The same flags are also available as environment variables (prefix `OMNIOPT_OPTUNA_*`) so the runner can be steered without touching the CLI:

| Env var | CLI flag |
| --- | --- |
| `OMNIOPT_OPTUNA_SAMPLER` | `--optuna_sampler` |
| `OMNIOPT_OPTUNA_PRUNER` | `--optuna_pruner` |
| `OMNIOPT_OPTUNA_N_STARTUP_TRIALS` | `--optuna_n_startup_trials` |
| `OMNIOPT_OPTUNA_STORAGE` | `--optuna_storage` |
| `OMNIOPT_OPTUNA_STUDY_NAME` | `--optuna_study_name` |
| `OMNIOPT_OPTUNA_EXTRA_ITERS` | `--optuna_extra_iters` |
| `OMNIOPT_OPTUNA_LOG_LEVEL` | (no CLI flag) |

CLI flags win over environment variables.

## Why every Optuna feature is a special case of OmniOpt

The runner reads OmniOpt's `input.json` (parameters, constraints, seed, trials, objectives) and writes OmniOpt's `results.json` (`{parameters: {...}}`). Both are the same files `.tpe.py` already handles, so OmniOpt's existing `ExternalProgramGenerationNode` treats the Optuna runner as a black box that fills in `results.json` exactly the same way.

That gives you, *for free*:

- **Single CLI surface.** `--parameter`, `--result-names`, `--run_program`, `--live_share`, `--generation_strategy`, `--continue`, the GUI, `omniopt_plot`, `omniopt_share` — every OmniOpt feature works with `--model=OPTUNA_*` and the new `omniopt_optuna` entry point.
- **Persistence.** OmniOpt's run folder, `results.csv`, `state_files/`, `live_share` uploads, and the `--continue` workflow all work unchanged.
- **Plotting.** Run `omniopt_plot --run_dir runs/<experiment>/0` exactly as for any other OmniOpt model.
- **Multi-objective support.** Multiple objectives declared via `--result-names` flow through Optuna's multi-objective samplers (NSGA-II / NSGA-III / MOTPE) and produce the same Pareto-front plots.
