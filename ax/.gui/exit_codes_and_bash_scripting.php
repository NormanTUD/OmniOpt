<?php
	include("_header_base.php");
?>
	<link href="tutorial.css" rel="stylesheet" />
	<link href="jquery-ui.css" rel="stylesheet">
	<link href="prism.css" rel="stylesheet" />

	<h1>Basics</h1>
    
	<div id="toc"></div>

	<h2 id="what_are_exit_codes">What are exit-codes?</h2>

	<p>Each program on Linux, after it runs, returns a value to the operating system to tell if it has succeeded and if not, what error may have occured.</p>

	<p>0 means 'everything was fine', every other value (possible is 1-255) mean 'something went wrong' and you can assign errors or groups of errors
	to one exit code. This is what OmniOpt2 extensively does, to make scripting it easier.</p>

	<h2 id="exit_code_groups">Exit code groups in OmniOpt</h2>

	<p>Depending on the error, if any, occured, OmniOpt2 ends with the following exit codes:</p>

	<h2>Exit Code Information</h2>
		<table>
		<tr>
			<th>Exit Code</th>
			<th>Error Group Description</th>
		</tr>
		<?php
			$exit_code_info = [
				"-1" => "No proper Exit code found",
				"2" => "Loading of Environment failed",
				"3" => "Invalid exit code detected",
				0 => "Seems to have worked properly",
				10 => "Usually only returned by dier (for debugging).",
				11 => "Required program not found (check logs)",
				12 => "Error with pip, check logs.",
				15 => "Unimplemented error.",
				18 => "test_wronggoing_stuff program not found (only --tests).",
				19 => "Something was wrong with your parameters. See output for details.",
				31 => "Basic modules could not be loaded or you cancelled loading them.",
				44 => "Continuation of previous job failed.",
				47 => "Missing checkpoint or defective file or state files (check output).",
				49 => "Something went wrong while creating the experiment.",
				87 => "Search space exhausted or search cancelled.",
				99 => "It seems like the run folder was deleted during the run.",
				100 => "--mem_gb or --gpus, which must be int, has received a value that is not int.",
				103 => "--time is not in minutes or HH:MM format.",
				104 => "One of the parameters --mem_gb, --time, or --experiment_name is missing.",
				105 => "Continued job error: previous job has missing state files.",
				181 => "Error parsing --parameter. Check output for more details.",
				192 => "Unknown data type (--tests).",
				199 => "This happens on unstable file systems when trying to write a file.",
				203 => "Unsupported --model.",
				233 => "No random steps set.",
				142 => "Error in Models like THOMPSON or EMPIRICAL_BAYES_THOMPSON. Not sure why.",
				243 => "Job was not found in squeue anymore, it may got cancelled before it ran."
			];

			foreach ($exit_code_info as $code => $description) {
			    echo "<tr>";
			    echo "<td>$code</td>";
			    echo "<td>$description</td>";
			    echo "</tr>";
			}
		?>
		</table>

	<h2 id="how_to_script_omniopt">How to script OmniOpt2 with exit codes</h2>

	<p>This example runs OmniOpt and, depending on the exit-code, does something else.</p>

	<pre><code class="language-bash">
#!/bin/bash
./omniopt --partition=alpha --experiment_name=my_experiment --mem_gb=1 --time=60 --worker_timeout=30 --max_eval=500 --num_parallel_jobs=20 --gpus=0 --num_random_steps=20 --follow --show_sixel_graphics ----run_program=$(echo -n "bash /path/to/my_experiment/run.sh --epochs=%(epochs) --learning_rate=%(learning_rate) --layers=%(layers)" | base64 -w 0) --cpus_per_task=1 --send_anonymized_usage_stats --model=BOTORCH_MODULAR --parameter learning_rate range 0 0.5 float --parameter epochs choice 1,10,20,30,100 --parameter layers fixed 10
exit_code=$? # Special bash variable

if [[ $exit_code -eq 0 ]]; then
	./omniopt --continue runs/my_experiment/0 # Run again with the same parameters, but load previous data
elif [[ $exit_code -eq 87 ]]; then # 87 = Search space exhausted
	echo "The search space was exhausted. Trying further will not find new points."
	# OmniOpt call for expanded search space here
fi
</code></pre>


	<script src="prism.js"></script>
	<script src="footer.js"></script>
</body>
</html>
