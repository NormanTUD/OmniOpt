<h1>Plot your results</h1>

<!-- What kinds of plots are available and how to use them -->

<div id="toc"></div>

There are many plots available and multiple options to show them. Here's a brief overview.

<h2 id="plot-over-x11">Plot over X11</h2>
<h3 id="plot-overview">Plot from overview</h3>

To plot over X11, make sure you are connected with <samp>ssh -X user@login2.barnard.hpc.tu-dresden.de</samp> (of course, use the HPC system you wish instead of barnard, if applicable, and change it to your user).

Then, <samp>cd</samp> into your OmniOpt2 directory. Assuming you have already ran an OmniOpt2-run and the results are in <samp>runs/my_experiment/0</samp> (adapt this to your experiment folder!), run this:

<pre class="invert_in_dark_mode"><code class="language-bash">./omniopt_plot runs/my_experiment/0</code></pre>

You will be presented by a menu like this:<br>

<img alt="Plot-Overview" src="imgs/plot_overview.png" /><br>

Use your arrow keys to navigate to the plot type you like, and then press enter.

<h3 id="plot-directly">Plot directly</h3>
If you know what plot you want, you can directly plot it by using:
<pre class="invert_in_dark_mode"><code class="language-bash">./omniopt_plot runs/my_experiment/0 scatter # change plot_type accordingly</code></pre>

<h3 id="plot_to_file">Plot to file</h3>
All plot scripts support to export your plot to a file.
<pre class="invert_in_dark_mode"><code class="language-bash">./omniopt_plot runs/my_experiment/0 scatter --save_to_file filename.svg # change plot_type and file name accordingly. Allowed are svg and png.</code></pre>

<h2 id="plot-types">Plot types</h2>
<p>There are many different plot types, some of which can only be shown on jobs that ran on Taurus, or jobs with more than a specific number of results or parameters. If you run the <samp>omniopt_plot</samp>-script, it will automatically show you plots that are readily available.</p>

<h3 id="trial_index_result">Plot trial index/result</h3>
<pre class="invert_in_dark_mode"><code class="language-bash">./omniopt_plot runs/my_experiment/0 trial_index_result</code></pre>
<img alt="Trial-Index-Result" src="imgs/trial_index_result.png" /><br>
<p>The trial-index is a continuous number that, for each run that is completed, is increased. Using it as <i>x</i>-axis allows you to trace how the results developed over time. Usually, the result should go down (at minimization runs) over time, though it may spike out a bit.</p>

<h4 id="trial_index_result_options"><samp>trial_index_result</samp> Options</h4>
<pre><?php require "plot_helps/trial_index_result.txt"; ?></pre>

<h3 id="time_and_exit_code">Plot time and exit code infos</h3>
<pre class="invert_in_dark_mode"><code class="language-bash">./omniopt_plot runs/my_experiment/0 time_and_exit_code</code></pre>
<img alt="Time-and-exit-Code" src="imgs/time_and_exit_code.png" /><br>

<p>This graph has 4 subgraphs that show different information regarding the job runtime, it's results and it's exit codes.</p>


<ul>
    <li><i>Distribution of Run Time</i>: This shows you how many jobs had which runtime. The <i>y</i>-Axis shows you the number of jobs in one specific time-bin, while the <i>x</i>-axis shows you the number of seconds that the jobs in those bins ran.</li>
    <li><i>Result over Time</i>: This shows you a distribution of results and when they were started and the results attained, so you can find out how long jobs took and how well their results were. </li>
    <li><i>Run Time Distribution by Exit Code</i>: Every job as an exit code and a run time, and this shows you a violin plot of the runtimes and exit-code distribution of a job. It may be helpful when larger jobs fail to find out how long they need until they fail.</li>
    <li><i>Run Time by Hostname</i>: Shows a boxplot of runtime by each hostname where it ran on. Useful to detect nodes that may execute code slower than other codes or to find out which nodes larger models were scheduled to.</li>
</ul>

<h4 id="time_and_exit_code_options"><samp>time_and_exit_code</samp> Options</h4>
<pre><?php require "plot_helps/time_and_exit_code.txt"; ?></pre>

<h3 id="scatter">Scatter</h3>
<pre class="invert_in_dark_mode"><code class="language-bash">./omniopt_plot runs/my_experiment/0 scatter</code></pre>
<img alt="Scatter" src="imgs/scatter.png" /><br>
<p>The scatter plot shows you all 2d combinations of the hyperparameter space and, for each evaluation, a dot is printed. The color of the dot depends on the result value of this specific run. The lower, the greener, and the higher, the more red they are. Thus, you can see how many results were attained and how they were, and where they have been searched.</p>

<h4 id="scatter_options"><samp>scatter</samp> Options</h4>
<pre><?php require "plot_helps/scatter.txt"; ?></pre>

<h3 id="hex_scatter">Hex-Scatter</h3>
<pre class="invert_in_dark_mode"><code class="language-bash">./omniopt_plot runs/my_experiment/0 scatter_hex</code></pre>
<img alt="Scatter-Hex" src="imgs/scatter_hex.png" /><br>

<p>Similar to scatter plot, but here many runs are grouped into hexagonal subspaces of the parameter combinations, and the groups are coloured by their average result, and as such you can see an approximation of the function space. This allows you to quickly grasp 'good' areas of your hyperparameter space.</p>

<h4 id="scatter_hex_options"><samp>scatter_hex</samp> Options</h4>
<pre><?php require "plot_helps/scatter_hex.txt"; ?></pre>

<h3 id="scatter_generation_method">Scatter-Generation-Method</h3>
<pre class="invert_in_dark_mode"><code class="language-bash">./omniopt_plot runs/my_experiment/0 scatter_generation_method</code></pre>
<img alt="Scatter-Generation-Method" src="imgs/scatter_generation_method.png" /><br>

<p>This is similar to the scatter plot, but also shows you which generation method (i.e. SOBOL, BoTorch, ...) is responsible for creating that point, and how the generation methods are scattered over each axis of the hyperparameter optimization problem. Thus, you can see how many runs have been tried and where exactly.</p>

<h4 id="scatter_generation_method_options"><samp>scatter_generation_method</samp> Options</h4>
<pre><?php require "plot_helps/scatter_generation_method.txt"; ?></pre>

<h3 id="kde">KDE</h3>
<pre class="invert_in_dark_mode"><code class="language-bash">./omniopt_plot runs/my_experiment/0 kde</code></pre>
<img alt="KDE (Kernel Density Estimation)" src="imgs/kde.png" /><br>

<p>Kernel-Density-Estimation-Plots, short <i>KDE</i>-Plots, group different runs into so-called bins by their result range and parameter range.</p>

<p>Each grouped result gets a color, green means lower, red means higher, and is plotted as overlaying bar charts.</p>

<p>These graphs thus show you, which parameter range yields which results, and how many of them have been tried, and how 'good' they were, i.e. closer to the minimum (green).</p>

<h4 id="kde_options"><samp>kde</samp> Options</h4>
<pre><?php require "plot_helps/kde.txt"; ?></pre>

<h3 id="get_next_trials">get_next_trials got/requested</h3>
<pre class="invert_in_dark_mode"><code class="language-bash">./omniopt_plot runs/my_experiment/0 get_next_trials</code></pre>
<img alt="Get next trials" src="imgs/get_next_trials.png" /><br>
<p>Each time the <samp>ax_client.get_next_trials()</samp>-function is called, it is logged how many new evaluations should be retrieved, and how many actually are retrieved. This graph is probably not useful for anyone except for the developer of OmniOpt2 for debugging, but still, I included it here.</p>

<h4 id="get_next_trials_options"><samp>get_next_trials</samp> Options</h4>
<pre><?php require "plot_helps/get_next_trials.txt"; ?></pre>

<h3 id="general">General job infos</h3>
<pre class="invert_in_dark_mode"><code class="language-bash">./omniopt_plot runs/my_experiment/0 general</code></pre>

<img alt="General" src="imgs/general.png" /><br>
<p>The <samp>general</samp>-plot shows you general info about your job. It consists of four subgraphs:</p>

<ul>
    <li><i>Results by Generation Method</i>: This shows the different generation methods, SOBOL meaning random step, and BoTorch being the model that is executed after the first random steps. The <i>y</i>-value is the Result. Most values are inside the blue box, little dots outside are considered outliers. Usually, you can see that the nonrandom model has far better results than the first random evaluations.</li>
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

<h4 id="general_options"><samp>general</samp> Options</h4>
<pre><?php require "plot_helps/general.txt"; ?></pre>

<h3 id="cpu_ram_usage">CPU and RAM Usage</h3>
<pre class="invert_in_dark_mode"><code class="language-bash">./omniopt_plot runs/my_experiment/0 cpu_ram_usage</code></pre>
<img alt="CPU-Ram-Usage" src="imgs/cpu_ram_usage.png" /><br>

<p>Very similar to the 2d-scatter plot, but in 3d.</p>

<h4 id="cpu_ram_usage_options"><samp>cpu_ram_usage</samp> Options</h4>
<pre><?php require "plot_helps/cpu_ram_usage.txt"; ?></pre>

<h3 id="gpu_usage">GPU usage</h3>
<pre class="invert_in_dark_mode"><code class="language-bash">./omniopt_plot runs/my_experiment/0 gpu_usage</code></pre>
<img alt="GPU-Usage" src="imgs/gpu_usage.png" /><br>
<p>Shows the workload of different GPUs on all nodes that jobs of an evaluation has run on over time.</p>

<h4 id="gpu_usage_options"><samp>gpu_usage</samp> Options</h4>
<pre><?php require "plot_helps/gpu_usage.txt"; ?></pre>

<h3 id="worker">Worker usage</h3>
<pre class="invert_in_dark_mode"><code class="language-bash">./omniopt_plot runs/my_experiment/0 worker</code></pre>
<img alt="Worker" src="imgs/worker_usage.png" /><br>
<h4 id="worker_options"><samp>worker</samp> Options</h4>
<pre><?php require "plot_helps/worker.txt"; ?></pre>

Shows the amount of requested workers, and the amount of real workers over time.
