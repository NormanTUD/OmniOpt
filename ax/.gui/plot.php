<?php
	include("_header_base.php");
?>
	<link href="jquery-ui.css" rel="stylesheet">
	<style>
		body {
			font-family: Verdana, sans-serif;
		}
		.toc {
			margin-bottom: 20px;
		}
		.toc ul {
			list-style-type: none;
			padding: 0;
		}
		.toc li {
			margin-bottom: 5px;
		}
		.toc a {
			text-decoration: none;
			color: #007bff;
		}
			.toc a:hover {
			text-decoration: underline;
		}
	</style>
	<link href="prism.css" rel="stylesheet" />

	<h1>Plot your results</h1>
    
	<div class="toc">
		<h2>Table of Contents</h2>
		<ul>
			<li><a href="#plot-over-x11">Plot over X11</a></li>
			<ul>
				<li><a href="#plot-overview">Plot from overview</a></li>
				<li><a href="#plot_to_file">Plot to file</a></li>
			</ul>
			<li><a href="#plot-types">Plot over X11</a></li>
			<ul>
				<li><a href="#trial_index_result">Plot trial index/result</a></li>
				<li><a href="#time_and_exit_code">Plot time and exit code infos</a></li>
				<li><a href="#scatter">Scatter</a></li>
				<li><a href="#hex_scatter">Hex-Scatter</a></li>
				<li><a href="#scatter_generation_method">Scatter-Generation-Method</a></li>
				<li><a href="#kde">KDE</a></li>
				<li><a href="#get_next_trials">get_next_trials got/requested</a></li>
				<li><a href="#general">General job infos</a></li>
			</ul>
		</ul>
	</div>

	There are many plots available and multiple options to show them. Here's a brief overview.

	<h2 id="plot-over-x11">Plot over X11</h2>
	<h3 id="plot-overview">Plot from overview</h3>

	To plot over X11, make sure you are connected with <tt>ssh -X user@login2.barnard.hpc.tu-dresden.de</tt> (of course, use the HPC system you wish instead of barnard, if applicable, and change it to your user).

	Then, <tt>cd</tt> into your OmniOpt2 directory. Assuming you have already ran an OmniOpt-run and the results are in <tt>runs/my_experiment/0</tt>, run this:

	<pre><code class="language-bash">./omniopt_plot --run_dir runs/my_experiment/0</code></pre>

	You will be presented by a menu like this:<br>

	<img src="plot_overview.png" /><br>

	Use your arrow keys to navigate to the plot type you like, and then press enter.

	<h3 id="plot-overview">Plot directly</h3>
	If you know what plot you want, you can directly plot it by using:
	<pre><code class="language-bash">./omniopt_plot --run_dir runs/my_experiment/0 --plot_type=scatter # change plot_type accordingly</code></pre>

	<h3 id="plot_to_file">Plot to file</h3>
	All, except the 3d scatter, support to export your plot to a file.
	<pre><code class="language-bash">./omniopt_plot --run_dir runs/my_experiment/0 --plot_type=scatter --save_to_file filename.svg # change plot_type and file name accordingly. Allowed are svg and png.</code></pre>

	<h2 id="plot-types">Plot types</h2>
	TODO
	<h3 id="trial_index_result">Plot trial index/result</h3>
	<img src="results_to_trial_index.png" /><br>
	TODO
	<h3 id="time_and_exit_code">Plot time and exit code infos</h3>
	<img src="exit_code_and_times.png" /><br>
	TODO
	<h3 id="scatter">Scatter</h3>
	<img src="scatter.png" /><br>
	TODO
	<h3 id="hex_scatter">Hex-Scatter</h3>
	<img src="scatter_hex.png" /><br>
	TODO
	<h3 id="scatter_generation_method">Scatter-Generation-Method</h3>
	TODO
	<h3 id="kde">KDE</h3>
	TODO
	<h3 id="get_next_trials">get_next_trials got/requested</h3>
	TODO
	<h3 id="general">General job infos</h3>
	TODO

	<script src="prism.js"></script>
	<script>
		Prism.highlightAll();
	</script>
</body>
</html>

