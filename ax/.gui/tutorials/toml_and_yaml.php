<h1><tt>config.toml</tt>, <tt>config.yaml</tt></h1>
    
<div id="toc"></div>

<h2 id="what_is_omniopt">What are TOML and YAML files? What are they good for?</h2>

<p><a href="https://en.wikipedia.org/wiki/TOML" target="_blank">TOML</a> and <a href="https://en.wikipedia.org/wiki/YAML" target="_blank">YAML</a> are file interchange formats that allow you to save data structures on your disk. OmniOpt2 allows you to load parameters via <tt>--config_toml path/to/your/configuration.toml</tt> or <tt>--config_yaml path/to/your/configuration.yaml</tt>.</p>

<p>Basically, any parameter that can be given to the CLI can also be given over one of those two options. You can also merge CLI parameters with config-files, while CLI parameters take precedence.</p>

<h2 id="example_files">Example files</h2>
<h3 id="toml">TOML</h3>

<pre><code class="language-bash">
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
hide_ascii_plots = true
</code></pre>
<h3 id="yaml">YAML</h3>
<pre><code class="language-bash">
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
    hide_ascii_plots: true
</code></pre>

<h2 id="caveats">Caveats</h2>

<p>Currently, SLURM-related parameters that are given to the main script are not parsed properly and need to be given manually via the CLI.</p>
