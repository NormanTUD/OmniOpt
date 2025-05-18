# <span class="invert_in_dark_mode">♻️</span> Using external generators

<!-- How to use your own programs to generate new points and include them easily in OmniOpt2 -->

<!-- Category: Models -->

<div id="toc"></div>

## Basic idea

It is possible to use external generators in OmniOpt2, which means that you use external programs to generate new points that should be examined. That means you can use any algorithm you like in any programming language you
want, as long as you follow the standards required by OmniOpt2.

The external generator works by putting a JSON file that contains all previously generated data, the seed, the constraints, all parameters and their types in a JSON file.

You can specify your program with the `--external_generator` parameter, though it must be in base64. To take effect, the `--model` must be set to `EXTERNAL_GENERATOR`. See the last parameters here:

```bash
./omniopt \
    --partition=alpha \
    --experiment_name=EXTERNAL_GENERATOR_test \
    --mem_gb=1 \
    --time=60 \
    --worker_timeout=60 \
    --max_eval=2 \
    --num_parallel_jobs=5 \
    --gpus=1 \
    --num_random_steps=1 \
    --follow \
    --live_share \
    --send_anonymized_usage_stats \
    --result_names RESULT=max \
    --run_program=ZWNobyAiUkVTVUxUOiAlKHgpJSh5KSIgJiYgZWNobyAiUkVTVUxUMjogJXoi \
    --cpus_per_task=1 \
    --nodes_per_job=1 \
    --generate_all_jobs_at_once \
    --revert_to_random_when_seemingly_exhausted \
    --run_mode=local \
    --decimalrounding=4 \
    --occ_type=euclid \
    --main_process_gb=8 \
    --max_nr_of_zero_results=1 \
    --pareto_front_confidence=1 \
    --slurm_signal_delay_s=0 \
    --n_estimators_randomforest=100 \
    --parameter x range 123 100000000 int false \
    --parameter y range 1234 4321 \
    --parameter z range 111 222 int \
    --experiment_constraint "x >= y" \
    --seed 1234 \
    --model=EXTERNAL_GENERATOR \
    --external_generator $(echo "python3 $(pwd)/.tests/example_external.py" | base64 -w0)
```

This then gets called with a temporary directory as first parameter, in which a JSON file called `input.json` like this resides:

```json
{
    "parameters": {
        "x": {
            "parameter_type": "RANGE",
            "type": "INT",
            "range": [
                123,
                100000000
            ]
        },
        "y": {
            "parameter_type": "RANGE",
            "type": "FLOAT",
            "range": [
                1234.0,
                4321.0
            ]
        },
        "z": {
            "parameter_type": "RANGE",
            "type": "INT",
            "range": [
                111,
                222
            ]
        }
    },
    "constraints": [
        "y <= x"
    ],
    "seed": 1234,
    "trials": [
        [
            {
                "x": 46164761,
                "y": 2179.7038996219635,
                "z": 221
            }
        ],
        [
            {
                "RESULT": 461647612179.7039
            }
        ]
    ],
    "objectives": {
        "RESULT": "max"
    }
}
```

Your program must take this JSON file and create new hyperparameters, and put them in the same folder as `results.json`. The parameters, constraints and so on are, of course, dependent on the way you run OmniOpt2 and
it's parameters.

The `results.json` file your program must write in the folder given as parameter may look like this:

```json
{
    "parameters": {
        "x": 1234,
        "y": "5431",
        "z": "111"
    }
}
```

This file is then read, parsed and used to run a new hyperparameter set. `x`, `y` and `z` are the hyperparameter names; of course, those are also dependent on your OmniOpt2 run.

For each new hyperparameter (after the SOBOL-phase), the program will be invoked newly.

## Another example run code and `input.json`-file

```bash
./omniopt \
	--partition=alpha \
	--experiment_name=EXTERNAL_GENERATOR_test \
	--mem_gb=1 \
	--time=60 \
	--worker_timeout=60 \
	--max_eval=2 \
	--num_parallel_jobs=5 \
	--gpus=1 \
	--num_random_steps=1 \
	--follow \
	--live_share \
	--send_anonymized_usage_stats \
	--result_names RESULT=max \
	--run_program=ZWNobyAiUkVTVUxUOiAlKHgpJSh5KSIgJiYgZWNobyAiUkVTVUxUMjogJXoi \
	--cpus_per_task=1 \
	--nodes_per_job=1 \
	--generate_all_jobs_at_once \
	--revert_to_random_when_seemingly_exhausted \
	--run_mode=local \
	--decimalrounding=4 \
	--occ_type=euclid \
	--main_process_gb=8 \
	--max_nr_of_zero_results=1 \
	--pareto_front_confidence=1 \
	--slurm_signal_delay_s=0 \
	--n_estimators_randomforest=100 \
	--parameter x range 123 100000000 int false \
	--parameter y choice 5431,1234 \
	--parameter z fixed 111 \
	--model=EXTERNAL_GENERATOR \
    --external_generator $(echo "python3 $(pwd)/.tests/example_external.py" | base64 -w0)
```

```json
{
    "parameters": {
        "x": {
            "parameter_type": "RANGE",
            "type": "INT",
            "range": [
                123,
                100000000
            ]
        },
        "y": {
            "parameter_type": "CHOICE",
            "type": "STRING",
            "values": [
                "5431",
                "1234"
            ]
        },
        "z": {
            "parameter_type": "FIXED",
            "type": "STRING",
            "value": "111"
        }
    },
    "constraints": [],
    "seed": null,
    "trials": [
        [
            {
                "x": 55988092,
                "y": 1234,
                "z": 111
            }
        ],
        [
            {
                "RESULT": 559880921234.0
            }
        ]
    ]
}
```

## Example program

This is an example python-program that generated random points that lie within the ranges and parameter boundaries of your experiment:

```python[../.random_generator.py]
```

## Caveats

External Generator does not work with [custom generation strategies](tutorials?tutorial=custom_generation_strategy).
