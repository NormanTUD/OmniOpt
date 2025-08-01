<h1>🧾 Exit-Codes and Bash-scripting</h1>

<!-- How to script OmniOpt2 -->

<!-- Category: Advanced Usage -->

<div id="toc"></div>

<h2 id="what_are_exit_codes">What are exit-codes?</h2>

<p>Each program on Linux, after it runs, returns a value to the operating system to tell if it has succeeded and if not, what error may have occurred.</p>

<p>0 means 'everything was fine', every other value (possible is 1-255) mean 'something went wrong' and you can assign errors or groups of errors
to one exit code. This is what OmniOpt2 extensively does, to make scripting it easier.</p>

<h2 id="exit_code_groups">Exit code groups in OmniOpt2</h2>

<p>Depending on the error, if any, occurred, OmniOpt2 ends with the following exit codes:</p>

<?php
    $GLOBALS["HIDE_SUBZERO"] = true;
    require "exit_code_table.php";
?>

<h2 id="how_to_script_omniopt">How to script OmniOpt2 with exit codes</h2>

<p>This example runs OmniOpt2 and, depending on the exit-code, does something else.</p>

<pre class="invert_in_dark_mode"><code class="language-bash">#!/bin/bash
omniopt \
	--partition=alpha \
	--experiment_name=my_experiment \
	--mem_gb=1 \
	--time=60 \
	--worker_timeout=30 \
	--max_eval=500 \
	--num_parallel_jobs=20 \
	--gpus=0 \
	--num_random_steps=20 \
	--follow \
	--run_program=$(echo -n "bash /path/to/my_experiment/run.sh --epochs=%(epochs) --learning_rate=%(learning_rate) --layers=%(layers)" | base64 -w 0) \
	--cpus_per_task=1 \
	--send_anonymized_usage_stats \
	--model=BOTORCH_MODULAR \
	--parameter learning_rate range 0 0.5 float \
	--parameter epochs choice 1,10,20,30,100 \
	--parameter layers fixed 10

exit_code=$? # Special bash variable

if [[ $exit_code -eq 0 ]]; then
    omniopt --continue runs/my_experiment/0 # Run again with the same parameters, but load previous data
elif [[ $exit_code -eq 87 ]]; then # 87 = Search space exhausted
    echo "The search space was exhausted. Trying further will not find new points."
    # OmniOpt2 call for expanded search space here
fi
</code></pre>

<h2 id="fire_and_forget">Start multiple jobs with a bash script, or how to fire-and-forget</h2>

<p>When you want to start multiple jobs, it's recommended to omit the <tt>--follow</tt>-parameter. This way, jobs will get started, but you dont need to wait for them to be finished.</p>
