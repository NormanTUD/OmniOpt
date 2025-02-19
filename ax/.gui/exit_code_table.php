<?php
		$exit_code_info = [
			0 => "Seems to have worked properly",
			1 => "There was an error regarding the python-syntax. Is your python-version too old? If you are on a slurm-system, this may mean that the job cannot be started. Check stdout.",
			2 => "Wrong CLI arguments",
			3 => "Invalid exit code detected",
			4 => "Failed loading modules",
			5 => "Errors regarding toml or yaml config files",
			6 => "Error creating .logs dir",
			7 => "Probably versioning error. Try removing virtualenv and try again",
			8 => "Probably something went wrong trying to plot sixel graphics",
			9 => "Probably something went wrong trying to use or define the ax_client or executor",
			10 => "Usually only returned by dier (for debugging)",
			11 => "Required program not found (check logs)",
			12 => "Error with pip, check logs",
			13 => "Run folder already exists",
			14 => "Error installing OmniOpt2 via install_omniax.sh",
			15 => "Unimplemented error",
			16 => "Wrongly called .py files: Probably you tried to call them directly instead of over the bash file",
			18 => "test_wronggoing_stuff program not found (only --tests)",
			19 => "Something was wrong with your parameters. See output for details",
			20 => "Something went wrong installing the required modules",
			21 => "requirements.txt or test_requirements.txt not found",
			22 => "Python header-files not found",
			23 => "Loading of Environment failed",
			31 => "Basic modules could not be loaded or you cancelled loading them",
			32 => "Error regarding the --signed_weighted_euclidean_weights parameter. Check output for details",
			44 => "Continuation of previous job failed",
			45 => "Could not create log dir",
			47 => "Missing checkpoint or defective file or state files (check output)",
			49 => "Something went wrong while creating the experiment",
			50 => "Something went wrong with the --result_names option (check output)",
			87 => "Search space exhausted or search was cancelled",
			88 => "Search was done according to ax",
			89 => "All jobs failed",
			90 => "Error creating the experiment_args",
			91 => "Error creating the experiment_args",
			92 => "Error loading torch: This can happen when your disk is full",
			99 => "It seems like the run folder was deleted during the run",
			100 => "--mem_gb or --gpus, which must be int, has received a value that is not int",
			101 => "Error using ax_client: it was not defined where it should have been",
			103 => "--time is not in minutes or HH:MM format",
			104 => "One of the parameters --mem_gb, --time, --run_program or --experiment_name is missing",
			105 => "Continued job error: previous job has missing state files",
			106 => "--num_parallel_jobs must be equal to or larger than 1",
			123 => "Something is wrong the the --generation_strategy",
			130 => "Interrupt-Signal detected",
			133 => "Error loading --config_toml, --config_json or --config_yaml",
			137 => "OOM-Killer on Slurm-Systems",
			138 => "USR-Signal detected",
			142 => "Error in Models like THOMPSON or EMPIRICAL_BAYES_THOMPSON. Not sure why",
			143 => "Slurm-Job was cancelled",
			146 => "CONT-Signal detected",
			181 => "Error parsing --parameter. Check output for more details",
			191 => "Could not create workdir ",
			192 => "Unknown data type (--tests)",
			193 => "Error in printing logs. You may be on a read only file system or your hard disk is full",
			199 => "This happens on unstable file systems when trying to write a file",
			203 => "Unsupported --model",
			206 => "Invalid orchestrator file",
			210 => "Unknown orchestrator mode",
			211 => "Git checkout failed (--checkout_to_latest_tested_version)",
			233 => "No random steps set",
			242 => "Error at fetching new trials",
			243 => "Job was not found in squeue anymore, it may got cancelled before it ran",
			244 => "get_executor() failed. See logs for more details.",
			245 => "python3 is not installed",
			246 => "A path that should have been a file is actually a folder. Check output for more details.",
			255 => "sbatch error"
		];

		if (isset($_GET["exit_code"])) {
			$_ec = $_GET["exit_code"];

			if (preg_match("/^\d*$/", $_ec)) {
				if(0 <= $_ec && $_ec <= 255) {
					if(isset($exit_code_info[$_ec])) {
						print "Exit-Code $_ec, this means: ".$exit_code_info[$_ec];
					} else {
						print "Unknown exit code: $_ec.";
					}				
				} else {
					print "Exit-codes must be between 0 and 255, given one was $_ec";
				}
			} else {
				print "Invalid exit-code given";
			}
		} else {
			print "<h2>Exit Code Information</h2>";
			print "	<table>";
			print "		<tr class='invert_in_dark_mode'>";
			print "			<th>Exit Code</th>";
			print "			<th>Error Group Description</th>";
			print "		</tr>";

			foreach ($exit_code_info as $code => $description) {
				echo "<tr>";
				echo "<td>$code</td>";
				echo "<td>$description</td>";
				echo "</tr>";
			}

			print "</table>\n";
		}
?>
