# <tt>config.toml</tt>, <tt>config.yaml</tt>, <tt>config.json</tt>

<!-- How to load parameters from OmniOpt2-config-files -->

<div id="toc"></div>

## What are TOML, JSON and YAML files? What are they good for?

[TOML](https://en.wikipedia.org/wiki/TOML), [JSON](https://en.wikipedia.org/wiki/JSON) and [YAML](https://en.wikipedia.org/wiki/YAML) are file interchange formats that allow you to save data structures on your disk. OmniOpt2 allows you to load parameters via <tt>--config_toml path/to/your/configuration.toml</tt>, <tt>--config_json path/to/your/configuration.json</tt> or <tt>--config_yaml path/to/your/configuration.yaml</tt>.

Basically, any parameter that can be given to the CLI can also be given over one of those two options. You can also merge CLI parameters with config-files, while CLI parameters take precedence.

Only one of these can be used. You cannot, for example, use <tt>--config_toml</tt> and <tt>--config_json</tt> together.

## Example files

### TOML

```
live_share = true
partition = "alpha"
experiment_name = "__main__tests__"
mem_gb = 4
time = 60
worker_timeout = 5
max_eval = 2
num_parallel_jobs = 20
gpus = 0
run_program = "./.tests/optimization_example --int_param='%(int_param)' --float_param='%(float_param)' --choice_param='%(choice_param)' --int_param_two='%(int_param_two)'"

[[parameter]]
param = ["int_param", "range", "-100", "10", "int"]

[[parameter]]
param = ["float_param", "range", "-100", "10", "float"]

[[parameter]]
param = ["choice_param", "choice", "1,2,4,8,16,hallo"]

[[parameter]]
param = ["int_param_two", "range", "-100", "10", "int"]

num_random_steps = 1
model = "BOTORCH_MODULAR"
auto_exclude_defective_hosts = true
```

### YAML

```
live_share: true
partition: alpha
experiment_name: __main__tests__
mem_gb: 4
time: 60
worker_timeout: 5
max_eval: 2
num_parallel_jobs: 20
gpus: 0
run_program: >-
  ./.tests/optimization_example --int_param='%(int_param)'
  --float_param='%(float_param)' --choice_param='%(choice_param)'
  --int_param_two='%(int_param_two)'
parameter:
  - param:
      - int_param
      - range
      - '-100'
      - '10'
      - int
  - param:
      - float_param
      - range
      - '-100'
      - '10'
      - float
  - param:
      - choice_param
      - choice
      - '1,2,4,8,16,hallo'
  - param:
      - int_param_two
      - range
      - '-100'
      - '10'
      - int
    num_random_steps: 1
    model: BOTORCH_MODULAR
    auto_exclude_defective_hosts: true
```

### JSON

```
{
  "experiment_name": "__main__tests__",
  "gpus": 0,
  "live_share": true,
  "max_eval": 2,
  "mem_gb": 4,
  "num_parallel_jobs": 20,
  "partition": "alpha",
  "run_program": "./.tests/optimization_example --int_param='%(int_param)' --float_param='%(float_param)' --choice_param='%(choice_param)' --int_param_two='%(int_param_two)'",
  "time": 60,
  "worker_timeout": 5,
  "parameter": [
    {
      "param": [
        "int_param",
        "range",
        "-100",
        "10",
        "int"
      ]
    },
    {
      "param": [
        "float_param",
        "range",
        "-100",
        "10",
        "float"
      ]
    },
    {
      "param": [
        "choice_param",
        "choice",
        "1,2,4,8,16,hallo"
      ]
    },
    {
      "auto_exclude_defective_hosts": true,
      "model": "BOTORCH_MODULAR",
      "num_random_steps": 1,
      "param": [
        "int_param_two",
        "range",
        "-100",
        "10",
        "int"
      ]
    }
  ]
}
```
