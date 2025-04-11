# Using external generators

<!-- What are external generators and how to use them -->

<div id="toc"></div>

## Basic idea

It is possible to use external generators in OmniOpt2, which means that you use external programs to generate new points that should be examined. That means you can use any algorithm you like in any programming language you
want, as long as you follow the standards required by OmniOpt2. 

The external generator works by putting a JSON file that contains all previously generated data, the seed, the constraints, all parameters and their types in a JSON file. 

You can specify your program with the `--external_generator` parameter, though it must be in base64. To take effect, the `--model` must be set to `EXTERNAL_GENERATOR`. See the last parameter here:

```bash
./omniopt \
	--partition=alpha \
	--experiment_name=small_test_experiment \
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
	--model=EXTERNAL_GENERATOR \
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
    --external_generator $(echo "python3 $(pwd)/.tests/example_external.py" | base64 -w0) 
```

This then gets called with a temporary directory as first parameter, in which a JSON file called `input.json` like this resides:

```json
{
    'parameters': {
        'x': {
            'parameter_type': 'RANGE',
            'type': 'INT',
            'range': [123, 100000000]
        },
        'y': {'
            parameter_type': 'CHOICE',
            'type': 'STRING',
            'values': ['5431', '1234']
        },
        'z': {
            'parameter_type': 'FIXED',
            'type': 'STRING',
            'value': '111'
        }
    },
    'constraints': None,
    'seed': None,
    'trials': [
        [
            {'x': 2848395, 'y': 1234, 'z': 111}
        ],
        [
            {'RESULT': 28483951234.0}
        ]
    ]
}
```

Your program must take this JSON file and create new hyperameters, and put them in the same folder as `results.json`. It may look like this:

```json
{
    "parameters": {
        "x": 1234,
        "y": "5431",
        "z": "111"
    }
}
```

This file is then read, parsed and used to run a new hyperparameter set.

For each new hyperparameter (after the SOBOL-phase), the program will be invoked newly.

## Example program

This is an example python-program that generated random points that lie within the ranges and parameter boundaries of your experiment:

```python
import sys
import os
import json
import random

def generate_random_value(parameter):
    try:
        if parameter['parameter_type'] == 'RANGE':
            range_min, range_max = parameter['range']
            if parameter['type'] == 'INT':
                return random.randint(range_min, range_max)

            if parameter['type'] == 'FLOAT':
                return random.uniform(range_min, range_max)
        elif parameter['parameter_type'] == 'CHOICE':
            values = parameter['values']
            if parameter['type'] == 'INT':
                return random.choice(values)

            if parameter['type'] == 'STRING':
                return random.choice(values)

            return random.choice(values)
        elif parameter['parameter_type'] == 'FIXED':
            return parameter['value']
    except KeyError as e:
        print(f"KeyError: Missing {e} in parameter")
        sys.exit(4)

    return None

def generate_random_point(parameters):
    point = {}
    for param_data in parameters.items():
        p = param_data[1]

        if p and not isinstance(p, list):
            for param_name in list(p.keys()):
                point[param_name] = generate_random_value(param_data[1][param_name])
    return point

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <path>")
        sys.exit(1)

    path = sys.argv[1]

    if not os.path.isdir(path):
        print(f"Error: The path '{path}' is not a valid folder.")
        sys.exit(2)

    json_file_path = os.path.join(path, 'input.json')
    results_file_path = os.path.join(path, 'results.json')

    try:
        with open(json_file_path, mode='r', encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {json_file_path} not found.")
        sys.exit(3)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON in {json_file_path}.")
        sys.exit(4)

    random_point = generate_random_point(data)

    with open(results_file_path, mode='w', encoding="utf-8") as f:
        json.dump({"parameters": random_point}, f, indent=4)

if __name__ == "__main__":
    main()
``` 

## Caveat

External Generator does not work with custom generation strategies.
