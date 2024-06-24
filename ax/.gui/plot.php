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
			</ul>
		</ul>
	</div>

	There are many plots available and multiple options to show them. Here's a brief overview.

	<h2 id="plot-over-x11">Plot over X11</h2>
	<h3 id="plot-overview">Plot from overview</h3>

	To plot over X11, make sure you are connected with <tt>ssh -X user@login2.barnard.hpc.tu-dresden.de</tt> (of course, use the HPC system you wish instead of barnard, if applicable, and change it to your user).

	Then, <tt>cd</tt> into your OmniOpt2 directory. Assuming you have already ran an OmniOpt-run and the results are in <tt>runs/my_experiment/0</tt>, run this:

	<pre><code class="language-bash">#!/bin/bash -l
./omniopt_plot --run_dir runs/my_experiment/0
</code></pre>

	You will be presented by a menu like this:<br>

	<img src="plot_overview.png" /><br>

	Use your arrow keys to navigate to the plot type you like, and then press enter.

	<script src="prism.js"></script>
	<script>
		Prism.highlightAll();
	</script>
</body>
</html>

