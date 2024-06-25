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
	<pre><code class="language-bash">ls
best_result.txt  get_next_trials.csv  gpu_usage__i8033.csv  gpu_usage__i8037.csv  job_infos.csv  oo_errors.txt  parameters.txt  results.csv  single_runs  state_files  worker_usage.csv</code></pre>

	<h4 id="best_result"><tt>best_result.txt</tt></h4>

	<p>This file contains an ANSI-table that shows you the best result and the parameters resulted in that result.</p>

	<pre>
			      Best parameter:                              
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┓
┃ width_and_height ┃ validation_split ┃ learning_rate ┃ epochs ┃ result   ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━┩
│ 72               │ 0.184052         │ 0.001         │ 14     │ 1.612789 │
└──────────────────┴──────────────────┴───────────────┴────────┴──────────┘
</pre>
	
	<h4 id="get_next_trials"><tt>get_next_trials.csv</tt></h4>

	<p>A CSV file that contains the current time, the number of jobs <tt>ax_client.get_next_trials()</tt> got and the number it requested to get.</p>

	<pre>2024-06-25 08:55:46,1,20
2024-06-25 08:56:41,2,20
2024-06-25 08:57:14,5,20
2024-06-25 08:57:33,7,20
2024-06-25 08:59:54,15,20
...</pre>

	<script src="prism.js"></script>
	<script src="footer.js"></script>
</body>
</html>
