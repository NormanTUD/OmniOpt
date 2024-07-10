<?php
	include("../_header_base.php");
?>
<h1>Plot your results</h1>

<div id="toc"></div>

There are many plots available and multiple options to show them. Here's a brief overview.

<h2 id="plot-over-x11">Plot over X11</h2>
<h3 id="plot-overview">Plot from overview</h3>

To plot over X11, make sure you are connected with <tt>ssh -X user@login2.barnard.hpc.tu-dresden.de</tt> (of course, use the HPC system you wish instead of barnard, if applicable, and change it to your user).

Then, <tt>cd</tt> into your OmniOpt2 directory. Assuming you have already ran an OmniOpt-run and the results are in <tt>runs/my_experiment/0</tt>, run this:

<pre><code class="language-bash">./omniopt_plot --run_dir runs/my_experiment/0</code></pre>

You will be presented by a menu like this:<br>

<img src="imgs/plot_overview.png" /><br>

Use your arrow keys to navigate to the plot type you like, and then press enter.

<h3 id="plot-overview">Plot directly</h3>
If you know what plot you want, you can directly plot it by using:
<pre><code class="language-bash">./omniopt_plot --run_dir runs/my_experiment/0 --plot_type=scatter # change plot_type accordingly</code></pre>

<h3 id="plot_to_file">Plot to file</h3>
All, except the 3d scatter, support to export your plot to a file.
<pre><code class="language-bash">./omniopt_plot --run_dir runs/my_experiment/0 --plot_type=scatter --save_to_file filename.svg # change plot_type and file name accordingly. Allowed are svg and png.</code></pre>

<h2 id="plot-types">Plot types</h2>
<p>There are many different plot types, some of which can only be shown on jobs that ran on Taurus, or jobs with more than a specific number of results or parameters. If you run the <tt>omniopt_plot</tt>-script, it will automatically show you plots that are readily available.</p>

<h3 id="trial_index_result">Plot trial index/result</h3>
<pre><code class="language-bash">./omniopt_plot --run_dir runs/my_experiment/0 --plot_type=trial_index_result</code></pre>
<img src="imgs/trial_index_result.png" /><br>
<p>The trial-index is a continous number that, for each run that is completed, is increased. Using it as <i>x</i>-axis allows you to trace how the results developed over time. Usually, the result should go down (at minimization runs) over time, though it may spike out a bit.</p>

<h4 id="trial_index_result_options"><tt>--plot_type=trial_index_result</tt> Options</h4>
<pre><?php include("plot_helps/trial_index_result.txt"); ?></pre>

<h3 id="time_and_exit_code">Plot time and exit code infos</h3>
<pre><code class="language-bash">./omniopt_plot --run_dir runs/my_experiment/0 --plot_type=time_and_exit_code</code></pre>
<img src="imgs/time_and_exit_code.png" /><br>

<p>This graph has 4 subgraphs that show different information regarding the job runtime, it's results and it's exit codes.</p>


<ul>
	<li><i>Distribution of Run Time</i>: This shouws you how many jobs had which runtime. The <i>y</i>-Axis shows you the number of jobs in one specific time-bin, while the <i>x</i>-axis shows you the number of seconds that the jobs in those bins ran.</li>
	<li><i>Result over Time</i>: This shows you a distribution of results and when they were started and the results attained, so you can find out how long jobs took and how well their results were. </li>
	<li><i>Run Time Distribution by Exit Code</i>: Every job as an exit code and a run time, and this shows you a violin plot of the runtimes and exit-code distribution of a job. It may be helpful when larger jobs fail to find out how long they need until they fail.</li>
	<li><i>Run Time by Hostname</i>: Shows a boxplot of runtime by each hostname where it ran on. Useful to detect nodes that may execute code slower than other codes or to find out which nodes larger models were scheduled to.</li>
</ul>

<h4 id="time_and_exit_code_options"><tt>--plot_type=time_and_exit_code</tt> Options</h4>
<pre><?php include("plot_helps/time_and_exit_code.txt"); ?></pre>

<h3 id="scatter">Scatter</h3>
<pre><code class="language-bash">./omniopt_plot --run_dir runs/my_experiment/0 --plot_type=scatter</code></pre>
<img src="imgs/scatter.png" /><br>
<p>The scatter plot shows you all 2d combinations of the hyperparameter space and, for each evaluation, a dot is printed. The color of the dot depends on the result value of this specific run. The lower, the greener, and the higher, the more red they are. Thus, you can see how many results were attained and how they were, and where they have been searched.</p>

<h4 id="scatter_options"><tt>--plot_type=scatter</tt> Options</h4>
<pre><?php include("plot_helps/scatter.txt"); ?></pre>

<h3 id="hex_scatter">Hex-Scatter</h3>
<pre><code class="language-bash">./omniopt_plot --run_dir runs/my_experiment/0 --plot_type=scatter_hex</code></pre>
<img src="imgs/scatter_hex.png" /><br>

<p>Similiar to scatter plot, but here many runs are grouped into hexagonal subspaces of the parameter combinations, and the groups are coloured by their average result, and as such you can see an approximation of the function space. This allows you to quickly grasp 'good' areas of your hyperparameter space.</p>

<h4 id="scatter_hex_options"><tt>--plot_type=scatter_hex</tt> Options</h4>
<pre><?php include("plot_helps/scatter_hex.txt"); ?></pre>

<h3 id="scatter_generation_method">Scatter-Generation-Method</h3>
<pre><code class="language-bash">./omniopt_plot --run_dir runs/my_experiment/0 --plot_type=scatter_generation_method</code></pre>
<img src="imgs/scatter_generation_method.png" /><br>

<p>This is similiar to the scatter plot, but also shows you which generation method (i.e. SOBOL, BoTorch, ...) is responsible for creating that point, and how the generation methods are scattered over each axis of the hyperparameter optimization problem. Thus, you can see how many runs have been tried and where exactly.</p>

<h4 id="scatter_generation_method_options"><tt>--plot_type=scatter_generation_method</tt> Options</h4>
<pre><?php include("plot_helps/scatter_generation_method.txt"); ?></pre>

<h3 id="kde">KDE</h3>
<pre><code class="language-bash">./omniopt_plot --run_dir runs/my_experiment/0 --plot_type=kde</code></pre>
<img src="imgs/kde.png" /><br>

<p>Kernel-Density-Estimation-Plots, short <i>KDE</i>-Plots, group different runs into so-called bins by their result range and parameter range.</p>

<p>Each grouped result gets a color, green means lower, red means higher, and is plotted as overlaying bar charts.</p>

<p>These graphs thus show you, which parameter range yields which results, and how many of them have been tried, and how 'good' they were, i.e. closer to the minimum (green).</p>

<h4 id="kde_options"><tt>--plot_type=kde</tt> Options</h4>
<pre><?php include("plot_helps/kde.txt"); ?></pre>

<h3 id="get_next_trials">get_next_trials got/requested</h3>
<pre><code class="language-bash">./omniopt_plot --run_dir runs/my_experiment/0 --plot_type=get_next_trials</code></pre>
<img src="imgs/get_next_trials.png" /><br>
<p>Each time the <tt>ax_client.get_next_trials()</tt>-function is called, it is logged how many new evaluations should be retrieved, and how many actually are retrieved. This graph is probably not useful for anyone except for the developer of OmniOpt for debugging, but still, I included it here.</p>

<h4 id="get_next_trials_options"><tt>--plot_type=get_next_trials</tt> Options</h4>
<pre><?php include("plot_helps/get_next_trials.txt"); ?></pre>

<h3 id="general">General job infos</h3>
<pre><code class="language-bash">./omniopt_plot --run_dir runs/my_experiment/0 --plot_type=general</code></pre>

<img src="imgs/general.png" /><br>
<p>The <tt>general</tt>-plot shows you general info about your job. It consists of four subgraphs:</p>

<ul>
	<li><i>Results by Generation Method</i>: This shows the different generation methods, SOBOL meaning random step, and BoTorch being the model that is executed after the first random steps. The <i>y</i>-value is the Result. Most values are inside the blue box, little dots outside are considered outliars. Usually, you can see that the nonrandom model has far better results than the first random evaluations.</li>
	<li><i>Distribution of job status</i>: How many jobs were run and in which status they were. Different status include:</li>
	<ul>
		<li><i>COMPLETED</i>: That means the job has completed and has a result</li>
		<li><i>ABANDONED</i>: That means the job has been started, but, for example, due to timeout errors, the job was not able to finish with results</li>
		<li><i>MANUAL</i>: That means the job has been imported from a previous run</li>
		<li><i>FAILED</i>: That means the job has started but it failed and gained no result</li>
	</ul>
	<li><i>Correlation Matrix</i>: Shows you how each of the parameters correlates with each other and the final result. The higher the values, the more likely there's a correlation</li>
	<li><i>Distribution of Results by Generation Method</i>: This puts different results into so-called bins, i.e. groups of results in a certain range, and plots colored bar charts that tell you where how many results have been found by which method.</li>
</ul>

<h4 id="general_options"><tt>--plot_type=general</tt> Options</h4>
<pre><?php include("plot_helps/general.txt"); ?></pre>

<h3 id="3d">3d</h3>
<pre><code class="language-bash">./omniopt_plot --run_dir runs/my_experiment/0 --plot_type=3d</code></pre>
<img src="imgs/3d.png" /><br>

<p>Very similiar to the 2d-scatter plot, but in 3d.</p>

<h4 id="3d_options"><tt>--plot_type=3d</tt> Options</h4>
<pre><?php include("plot_helps/3d.txt"); ?></pre>

<h3 id="gpu_usage">GPU usage</h3>
<pre><code class="language-bash">./omniopt_plot --run_dir runs/my_experiment/0 --plot_type=gpu_usage</code></pre>
<img src="imgs/gpu_usage.png" /><br>
<p>Shows the workload of different GPUs on all nodes that jobs of an evaluation has run on over time.</p>

<h4 id="gpu_usage_options"><tt>--plot_type=gpu_usage</tt> Options</h4>
<pre><?php include("plot_helps/gpu_usage.txt"); ?></pre>

<h3 id="worker">Worker usage</h3>
<pre><code class="language-bash">./omniopt_plot --run_dir runs/my_experiment/0 --plot_type=worker</code></pre>
<img src="imgs/worker_usage.png" /><br>
<h4 id="worker_options"><tt>--plot_type=worker</tt> Options</h4>
<pre><?php include("plot_helps/worker.txt"); ?></pre>

Shows the amount of requested workers, and the amount of real workers over time.
