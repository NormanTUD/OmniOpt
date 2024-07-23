<?php
	include("../_header_base.php");
?>
<h1>Basics and Docker</h1>
    
<div id="toc"></div>

<h2 id="what_is_omniopt">What is OmniOpt2 and what does it do?</h2>

<p>OmniOpt2 is a highly parallelized hyperparameter optimizer based on Ax/Botorch. It explores various combinations of parameters within a given search space, 
runs a program with these parameters, and identifies promising areas for further evaluations.</p>

<h2 id="key_features">Key Features</h2>

<ul>
	<li>Simple Hyperparameter Optimization: OmniOpt2 allows for easy hyperparameter optimization within defined ranges.</li>
	<li>Tool Agnostic: It is completely agnostic regarding the code it runs. OmniOpt2 only requires command-line arguments with the hyperparameters and expects the program to output results, e.g., <code class='language-python'>print(f"RESULT: {loss}")</code>. <a href="tutorials.php?tutorial=run_sh">See here how to prepare your program for the use with OmniOpt2</a></li>
	<li>Self-Installation: OmniOpt2 installs itself into a virtual environment.</li>
	<li>No Configuration Files: All configuration is handled through the command line interface (CLI).</li>
</ul>

<h2 id="installation">Installation</h2>

<p>OmniOpt2 is self-installing and does not require any additional manual setup. Simply run the <tt>curl</tt> the <a href="index.php">GUI</a> provides command and it will install itself into a virtual environment.</p>

<h2 id="usage">Usage</h2>

<pre><code class="language-bash">./omniopt --partition=alpha --experiment_name=my_experiment --mem_gb=1 --time=60 --worker_timeout=30 --max_eval=500 --num_parallel_jobs=20 --gpus=0 --num_random_steps=20 --follow --show_sixel_graphics --run_program=YmFzaCAvcGF0aC90by9teV9leHBlcmltZW50L3J1bi5zaCAtLWVwb2Nocz0lKGVwb2NocykgLS1sZWFybmluZ19yYXRlPSUobGVhcm5pbmdfcmF0ZSkgLS1sYXllcnM9JShsYXllcnMp --cpus_per_task=1 --send_anonymized_usage_stats --model=BOTORCH_MODULAR --parameter learning_rate range 0 0.5 float --parameter epochs choice 1,10,20,30,100 --parameter layers fixed 10</code></pre>

<p>This command includes all necessary options to run a hyperparameter optimization with OmniOpt2.</p>


<h3 id="parameters">Parameters</h3>

<ul>
	<li><code class='language-bash'>--partition=alpha</code>: Specifies the partition to use.</li>
	<li><code class='language-bash'>--experiment_name=my_experiment</code>: Sets the name of the experiment.</li>
	<li><code class='language-bash'>--mem_gb=1</code>: Allocates 1 GB of memory for the job.</li>
	<li><code class='language-bash'>--time=60</code>: Sets a timeout of 60 minutes for the job.</li>
	<li><code class='language-bash'>--worker_timeout=30</code>: Sets a timeout of 30 minutes for workers.</li>
	<li><code class='language-bash'>--max_eval=500</code>: Limits the maximum number of evaluations to 500.</li>
	<li><code class='language-bash'>--num_parallel_jobs=20</code>: Runs up to 20 parallel jobs.</li>
	<li><code class='language-bash'>--gpus=0</code>: Specifies the number of GPUs to use.</li>
	<li><code class='language-bash'>--num_random_steps=20</code>: Sets the number of random steps to 20.</li>
	<li><code class='language-bash'>--follow</code>: Follows the job's progress.</li>
	<li><code class='language-bash'>--show_sixel_graphics</code>: Displays sixel graphics.</li>
	<li><code class='language-bash'>--run_program=YmFzaCAvcGF0aC90by9teV9leHBlcmltZW50L3J1bi5zaCAtLWVwb2Nocz0lKGVwb2NocykgLS1sZWFybmluZ19yYXRlPSUobGVhcm5pbmdfcmF0ZSkgLS1sYXllcnM9JShsYXllcnMp</code>: Specifies the <a target='_blank' href="https://en.wikipedia.org/wiki/Base64">base64</a>-encoded command to run the program. In this case this resolves to <code class="language-bash">bash /path/to/my_experiment/run.sh --epochs=%(epochs) --learning_rate=%(learning_rate) --layers=%(layers)</code>.</li>
	<li><code class='language-bash'>--cpus_per_task=1</code>: Allocates 1 CPU per task.</li>
	<li><code class='language-bash'>--send_anonymized_usage_stats</code>: Sends anonymized usage statistics.</li>
	<li><code class='language-bash'>--model=BOTORCH_MODULAR</code>: Specifies the optimization model to use.</li>
	<li><code class='language-bash'>--parameter learning_rate range 0 0.5 float</code>: Defines the search space for the learning rate.</li>
	<li><code class='language-bash'>--parameter epochs choice 1,10,20,30,100</code>: Defines the choices for the epochs parameter.</li>
	<li><code class='language-bash'>--parameter layers fixed 10</code>: Sets the layers parameter to a fixed value of 10</li>
</ul>

<h2 id="run_program"><tt>--run_program</tt></h2>
<p>The <code class="language-bash">--run_program</code>-parameter needs the program to be executed as a base64-string, because parsing spaces and newline in bash, where it is party evaluated, <a target='_blank' href="https://en.wikipedia.org/wiki/Delimiter#Delimiter_collision">is very difficult</a>. It is possible to use a human readable string, though it has to be converted to base64 by your shell:<p>

<code class="language-bash">--run_program=$(echo -n "bash /path/to/my_experiment/run.sh --epochs=%(epochs) --learning_rate=%(learning_rate) --layers=%(layers)" | base64 -w 0)</code>

<h2 id="integration">Integration</h2>

<p>OmniOpt2 is compatible with any program that can run on Linux, regardless of the programming language (e.g., C, Python). The program must accept parameters via command line and output a result string (e.g., <code class="language-python">print(f"RESULT: {loss}")</code>).</p>

<h2 id="scalability">Scalability</h2>

<p>OmniOpt2, when SLURM is installed, automatically starts sub-jobs on different nodes to maximize resource utilization. Use the flag <tt>--num_parallel_jobs n</tt> with n being the number of workers you want to start jobs in parallel. When no SLURM is installed on your system, OmniOpt2 will run the jobs sequentially.</p>


<h2 class="error_handling">Error Handling</h2>

<p>OmniOpt2 provides helpful error messages and suggestions for common issues that may arise during usage.

<h2 id="use_cases">Use Cases</h2>

<p>OmniOpt2 is particularly useful in fields such as AI research and simulations, where hyperparameter optimization can significantly impact performance and results.</p>

<h2 id="run_local_or_docker">Run locally or in Docker</h2>
<p>You can also run OmniOpt2 locally or inside docker.</p>

<h3 id="run_locally">Run locally</h3>

<p>To run OmniOpt2 locally, simply fill the GUI, copy the curl-command and run it locally. OmniOpt2 will be installed into a virtualenv once in the beginning, which may take up to 20 minutes. From then on, it will not install itself again, so you only need to wait once.</p>

<h3 id="run_docker">Run in docker</h3>
<p>To build a docker container, simply run <code class="language-bash">./omniopt_docker</code> in the main folder. Docker is not supported on the HPC System though. If you have Debian or systems based on it, it will automatically install docker if it's not installed. For other systems, you need to install docker yourself.</p>

<p>The <code class="language-bash">./omniopt_docker</code>-command will build the container. You can also run several commands directly from the <code class="language-bash">./omniopt_docker</code>-command like this:</p>

<p><code class="language-bash">./omniopt_docker omniopt --tests</code> for example, will install docker (on Debian), build the container and run OmniOpt2 with the <code class="language-bash">--tests</code>-parameter.</p>

<p>The current folder where you run the <pre><code class="language-bash">./omniopt_docker</code></pre> from is mounted inside docker as <pre><code class="language-bash">/var/opt/omniopt/docker_user_dir</code></pre></p>

<p> Keep your program there and use this as a base path for your run in the GUI. In the GUI, in the additional-parameters-table, you can chose "Run-Mode" &rarr; Docker to automatically start Jobs generated by the GUI in docker.</p>

<h2 id="contact">Contact</h2>
<p>Idea: peter.winkler1 at tu-dresden.de, Technical Support: norman.koch at tu-dresden.de.</p>
