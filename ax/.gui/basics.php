<?php
	include("_header_base.php");
?>
	<link href="tutorial.css" rel="stylesheet" />
	<link href="jquery-ui.css" rel="stylesheet">
	<link href="prism.css" rel="stylesheet" />

	<h1>Basics</h1>
    
	<div id="toc"></div>

	<h2 id="what_is_omniopt">What is OmniOpt2 and what does it do?</h2>

	<p>OmniOpt2 is a highly parallelized hyperparameter optimizer based on Ax/Botorch. It explores various combinations of parameters within a given search space, 
	runs a program with these parameters, and identifies promising areas for further evaluations.</p>

	<h2 id="key_features">Key Features</h2>

	<ul>
		<li>Simple Hyperparameter Optimization: OmniOpt2 allows for easy hyperparameter optimization within defined ranges.</li>
		<li>Tool Agnostic: It is completely agnostic regarding the code it runs. OmniOpt2 only requires command-line arguments with the hyperparameters and expects the program to output results, e.g., <code class='language-bash'>print(f"RESULT: {loss}")</code>. <a href="run_sh.php">See here how to prepare your program for the use with OmniOpt2</a></li>
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
		<li><code class='language-bash'>--run_program=YmFzaCAvcGF0aC90by9teV9leHBlcmltZW50L3J1bi5zaCAtLWVwb2Nocz0lKGVwb2NocykgLS1sZWFybmluZ19yYXRlPSUobGVhcm5pbmdfcmF0ZSkgLS1sYXllcnM9JShsYXllcnMp</code>: Specifies the base64-encoded command to run the program. In this case this resolves to <code class="language-bash">bash /path/to/my_experiment/run.sh --epochs=%(epochs) --learning_rate=%(learning_rate) --layers=%(layers)</code>.</li>
		<li><code class='language-bash'>--cpus_per_task=1</code>: Allocates 1 CPU per task.</li>
		<li><code class='language-bash'>--send_anonymized_usage_stats</code>: Sends anonymized usage statistics.</li>
		<li><code class='language-bash'>--model=BOTORCH_MODULAR</code>: Specifies the optimization model to use.</li>
		<li><code class='language-bash'>--parameter learning_rate range 0 0.5 float</code>: Defines the search space for the learning rate.</li>
		<li><code class='language-bash'>--parameter epochs choice 1,10,20,30,100</code>: Defines the choices for the epochs parameter.</li>
		<li><code class='language-bash'>--parameter layers fixed 10</code>: Sets the layers parameter to a fixed value of 10</li>
	</ul>

	<h2 id="integration">Integration</h2>

	<p>OmniOpt2 is compatible with any program that can run on Linux, regardless of the programming language (e.g., C, Python). The program must accept parameters via command line and output a result string (e.g., <code class="language-bash">print(f"RESULT: {loss}")</code>).</p>

	<h2 id="scalability">Scalability</h2>

	<p>OmniOpt2 can scale across multiple HPC clusters if SLURM is available. It starts sub-jobs on different clusters to maximize resource utilization.</p>


	<h2 class="error_handling">Error Handling</h2>

	<p>OmniOpt2 provides helpful error messages and suggestions for common issues that may arise during usage.

	<h2 id="use_cases">Use Cases</h2>

	<p>OmniOpt2 is particularly useful in fields such as AI research and simulations, where hyperparameter optimization can significantly impact performance and results.</p>

	<p>For support, error reporting, and contributions, contact: peter.winkler1@tu-dresden.de, for technical stuff norman.koch@tu-dresden.de.</p>

	<h2 id="">Run locally or in Docker</h2>
	TODO

	<script src="prism.js"></script>
	<script src="footer.js"></script>
</body>
</html>
