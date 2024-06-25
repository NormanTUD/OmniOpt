<?php
	include("_header_base.php");
?>
	<link href="tutorial.css" rel="stylesheet" />
	<link href="jquery-ui.css" rel="stylesheet">
	<link href="prism.css" rel="stylesheet" />

	<h1>Folder structure of OmniOpt runs</h1>
    
	<div id="toc"></div>

	<h2 id="runs_folder"><tt>runs</tt>-folder</h2>

	<p>For every experiment you do, there will be a new folder created inside the <tt>runs</tt>-folder in your OmniOpt2-installation.</p>

	<p>Each of these has a subfolder for each run that the experiment with that name was run. For example, if you run the experiment <tt>my_experiment</tt>
	twice, the paths <tt>runs/my_experiment/0</tt> and <tt>runs/my_experiment/1</tt> exist.

	<h3 id="runs_folder">Single files</h3>
	<pre><code class="language-bash">best_result.txt  get_next_trials.csv  gpu_usage__i8033.csv  gpu_usage__i8037.csv  job_infos.csv  oo_errors.txt  parameters.txt  results.csv  single_runs  state_files  worker_usage.csv</code></pre>
	<script src="prism.js"></script>
	<script src="footer.js"></script>
</body>
</html>

