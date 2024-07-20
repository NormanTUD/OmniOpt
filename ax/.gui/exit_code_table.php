	<h2>Exit Code Information</h2>
		<table>
		<tr>
			<th>Exit Code</th>
			<th>Error Group Description</th>
		</tr>
		<?php
			$exit_code_info = [];

			array_unshift($exit_code_info, [
				0 => "Seems to have worked properly",
				"2" => "Loading of Environment failed",
				"3" => "Invalid exit code detected",
				10 => "Usually only returned by dier (for debugging).",
				11 => "Required program not found (check logs)",
				12 => "Error with pip, check logs.",
				13 => "Run folder already exists",
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
				142 => "Error in Models like THOMPSON or EMPIRICAL_BAYES_THOMPSON. Not sure why.",
				181 => "Error parsing --parameter. Check output for more details.",
				192 => "Unknown data type (--tests).",
				193 => "Error in printing logs. You may be on a read only file system.",
				199 => "This happens on unstable file systems when trying to write a file.",
				203 => "Unsupported --model.",
				233 => "No random steps set.",
				243 => "Job was not found in squeue anymore, it may got cancelled before it ran."
			]);

			if(!array_key_exists("HIDE_SUBZERO", $GLOBALS)) {
				array_unshift($exit_code_info, [
					"-1" => "No proper Exit code found",
				]);
			}

			foreach ($exit_code_info as $code_block_id => $code_block_content) {
				foreach ($code_block_content as $code => $description) {
					echo "<tr>";
					echo "<td>$code</td>";
					echo "<td>$description</td>";
					echo "</tr>";
				}
			}
		?>
		</table>
