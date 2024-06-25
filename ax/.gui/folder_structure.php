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
single_runs  state_files  worker_usage.csv</code></pre>

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
	
	<h4 id="results_csv"><tt>results.csv</tt></h4>

	<p>This file contains infos about every evaluation in this run, that is, it's number, the algorithm that craeted that point, its parameters, and it's result.</p>

	<pre>trial_index,arm_name,trial_status,generation_method,result,width_and_height,validation_split,learning_rate,epochs
0,0_0,COMPLETED,Sobol,1.690072,71,0.021625286340713503,0.20240612381696702,7
1,1_0,COMPLETED,Sobol,1.638602,65,0.02604435756802559,0.2448390863677487,6
2,2_0,COMPLETED,Sobol,1.718751,78,0.23111544810235501,0.38468948143068704,2
3,3_0,COMPLETED,Sobol,1.636012,93,0.0857066310942173,0.23433385196421297,15
4,4_0,COMPLETED,Sobol,1.624952,60,0.04056024849414826,0.11660899678524585,6
5,5_0,COMPLETED,Sobol,1.64169,76,0.1567445032298565,0.21590755908098072,10
6,6_0,COMPLETED,Sobol,1.639097,72,0.07228925675153733,0.1230122183514759,6
7,7_0,COMPLETED,Sobol,1.6279,74,0.04752136580646038,0.08336016669869424,3
8,8_0,COMPLETED,Sobol,1.618417,87,0.0058464851230382925,0.016544286970980468,7
9,9_0,COMPLETED,Sobol,1.627581,76,0.0673203308135271,0.08200951679609716,5</pre>

	<h4 id="get_next_trials"><tt>get_next_trials.csv</tt></h4>

	<p>A CSV file that contains the current time, the number of jobs <tt>ax_client.get_next_trials()</tt> got and the number it requested to get.</p>

	<pre>2024-06-25 08:55:46,1,20
2024-06-25 08:56:41,2,20
2024-06-25 08:57:14,5,20
2024-06-25 08:57:33,7,20
2024-06-25 08:59:54,15,20
...</pre>

	<h4 id="gpu_usage">GPU-usage-files (<tt>gpu_usage_*.csv</tt>)</h4>

	<p>GPU usage files. They are the output of <tt>nvidia-smi</tt> and are periodically taken, when you run on a system with SLURM that allows you to connect to
	nodes that have running jobs on it with ssh.</tt>

	<p>Header line is omitted, but is: <tt>timestamp, name, pci.bus_id, driver_version, pstate, pcie.link.gen.max, pcie.link.gen.current, temperature.gpu, utilization.gpu [%], utilization.memory [%], memory.total [MiB], memory.free [MiB], memory.used [MiB]</tt>.</p>

	<p>It may looks something like this:</p>

	<pre>2024/06/01 11:27:05.177, NVIDIA A100-SXM4-40GB, 00000000:3B:00.0, 545.23.08, P0, 4, 4, 44, 0 %, 0 %, 40960 MiB, 40333 MiB, 4 MiB
2024/06/01 11:27:05.188, NVIDIA A100-SXM4-40GB, 00000000:8B:00.0, 545.23.08, P0, 4, 4, 42, 0 %, 0 %, 40960 MiB, 40333 MiB, 4 MiB
2024/06/01 11:27:05.192, NVIDIA A100-SXM4-40GB, 00000000:0B:00.0, 545.23.08, P0, 4, 4, 43, 0 %, 0 %, 40960 MiB, 40333 MiB, 4 MiB
2024/06/01 11:27:15.309, NVIDIA A100-SXM4-40GB, 00000000:8B:00.0, 545.23.08, P0, 4, 4, 42, 3 %, 0 %, 40960 MiB, 1534 MiB, 38803 MiB
2024/06/01 11:27:15.311, NVIDIA A100-SXM4-40GB, 00000000:0B:00.0, 545.23.08, P0, 4, 4, 43, 3 %, 0 %, 40960 MiB, 1534 MiB, 38803 MiB
2024/06/01 11:27:15.311, NVIDIA A100-SXM4-40GB, 00000000:3B:00.0, 545.23.08, P0, 4, 4, 44, 3 %, 0 %, 40960 MiB, 1534 MiB, 38803 MiB
2024/06/01 11:27:25.361, NVIDIA A100-SXM4-40GB, 00000000:8B:00.0, 545.23.08, P0, 4, 4, 43, 3 %, 0 %, 40960 MiB, 666 MiB, 39671 MiB
2024/06/01 11:27:25.376, NVIDIA A100-SXM4-40GB, 00000000:3B:00.0, 545.23.08, P0, 4, 4, 44, 1 %, 0 %, 40960 MiB, 910 MiB, 39427 MiB</pre>

	<h4 id="job_infos"><tt>job_infos.txt</tt></h4>

	<p>This is similiar to the <tt>results.csv</tt>, but contains a little other info, i.e. the hostname the execution ran on and the full path that is run, also start- and endtime of execution and the exit code and signal that the job ended with.</p>

	<pre>start_time,end_time,run_time,program_string,width_and_height,validation_split,learning_rate,epochs,result,exit_code,signal,hostname
1719298546,1719298600,54,bash /home/s3811141/repos/OmniOpt/ax/.tests/example_network/run.sh --learning_rate=0.20240612381696702 --epochs=7 --validation_split=0.021625286340713503 --width=71 --height=71 --dense=8 --dense_units=16 --conv=16 --conv_filters=16,71,0.021625286340713503,0.20240612381696702,7,1.690072,0,None,arbeitsrechner
1719298601,1719298633,32,bash /home/s3811141/repos/OmniOpt/ax/.tests/example_network/run.sh --learning_rate=0.2448390863677487 --epochs=6 --validation_split=0.02604435756802559 --width=65 --height=65 --dense=8 --dense_units=16 --conv=16 --conv_filters=16,65,0.02604435756802559,0.2448390863677487,6,1.638602,0,None,arbeitsrechner
1719298635,1719298653,18,bash /home/s3811141/repos/OmniOpt/ax/.tests/example_network/run.sh --learning_rate=0.38468948143068704 --epochs=2 --validation_split=0.23111544810235501 --width=78 --height=78 --dense=8 --dense_units=16 --conv=16 --conv_filters=16,78,0.23111544810235501,0.38468948143068704,2,1.718751,0,None,arbeitsrechner
1719298654,1719298793,139,bash /home/s3811141/repos/OmniOpt/ax/.tests/example_network/run.sh --learning_rate=0.23433385196421297 --epochs=15 --validation_split=0.0857066310942173 --width=93 --height=93 --dense=8 --dense_units=16 --conv=16 --conv_filters=16,93,0.0857066310942173,0.23433385196421297,15,1.636012,0,None,arbeitsrechner
1719298794,1719298822,28,bash /home/s3811141/repos/OmniOpt/ax/.tests/example_network/run.sh --learning_rate=0.11660899678524585 --epochs=6 --validation_split=0.04056024849414826 --width=60 --height=60 --dense=8 --dense_units=16 --conv=16 --conv_filters=16,60,0.04056024849414826,0.11660899678524585,6,1.624952,0,None,arbeitsrechner
1719298823,1719298881,58,bash /home/s3811141/repos/OmniOpt/ax/.tests/example_network/run.sh --learning_rate=0.21590755908098072 --epochs=10 --validation_split=0.1567445032298565 --width=76 --height=76 --dense=8 --dense_units=16 --conv=16 --conv_filters=16,76,0.1567445032298565,0.21590755908098072,10,1.64169,0,None,arbeitsrechner
1719298882,1719298920,38,bash /home/s3811141/repos/OmniOpt/ax/.tests/example_network/run.sh --learning_rate=0.1230122183514759 --epochs=6 --validation_split=0.07228925675153733 --width=72 --height=72 --dense=8 --dense_units=16 --conv=16 --conv_filters=16,72,0.07228925675153733,0.1230122183514759,6,1.639097,0,None,arbeitsrechner
1719298921,1719298947,26,bash /home/s3811141/repos/OmniOpt/ax/.tests/example_network/run.sh --learning_rate=0.08336016669869424 --epochs=3 --validation_split=0.04752136580646038 --width=74 --height=74 --dense=8 --dense_units=16 --conv=16 --conv_filters=16,74,0.04752136580646038,0.08336016669869424,3,1.6279,0,None,arbeitsrechner</pre>

	<h4 id="oo_errors"><tt>oo_errors.txt</tt></h4>

	<p>This file, if it exists, contains a list of potential errors OmniOpt2 encountered during the run. If no errors were found, it may be empty or non-existant.</p>

	<h4 id="parameters"><tt>parameters.txt</tt></h4>

	<p>This file contains the parameter search space definition in a simple table. Example:

	<pre>                            Experiment parameters:                            
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━┓
┃ Name             ┃ Type  ┃ Lower bound ┃ Upper bound ┃ Values ┃ Value-Type ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━┩
│ width_and_height │ range │ 60          │ 100         │        │ int        │
│ validation_split │ range │ 0           │ 0.4         │        │ float      │
│ learning_rate    │ range │ 0.001       │ 0.4         │        │ float      │
│ epochs           │ range │ 1           │ 15          │        │ int        │
└──────────────────┴───────┴─────────────┴─────────────┴────────┴────────────┘
</pre>

	<script src="prism.js"></script>
	<script src="footer.js"></script>
</body>
</html>
