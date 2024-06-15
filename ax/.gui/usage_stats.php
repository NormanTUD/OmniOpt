<?php
	error_reporting(E_ALL);
	set_error_handler(function ($severity, $message, $file, $line) {
		throw new \ErrorException($message, $severity, $severity, $file, $line);
	});

	ini_set('display_errors', 1);


	function dier($msg) {
		print("<pre>".print_r($msg, true)."</pre>");
		exit(1);
	}
?>
<!DOCTYPE html>
<html lang="de">
	<head>
		<meta charset="UTF-8">
		<meta name="viewport" content="width=device-width, initial-scale=1.0">
		<title>Usage Statistics</title>
		<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
		<style>
			table {
				width: 100%;
				border-collapse: collapse;
			}
			th, td {
				border: 1px solid #ddd;
				padding: 8px;
			}
			th {
				padding-top: 12px;
				padding-bottom: 12px;
				text-align: left;
				background-color: #4CAF50;
				color: white;
			}
			tr:nth-child(even) {
				background-color: #f2f2f2;
			}
		</style>
	</head>
<body>
<?php
	function log_error($error_message) {
		error_log($error_message);
		echo "<p>Error: $error_message</p>";
	}

	function validate_parameters($params) {
		assert(is_array($params), "Parameters should be an array");

		$required_params = ['anon_user', 'has_sbatch', 'run_uuid', 'git_hash', 'exit_code', 'runtime'];
		$patterns = [
			'anon_user' => '/^[a-f0-9]{32}$/',
			'has_sbatch' => '/^[01]$/',
			'run_uuid' => '/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/',
			'git_hash' => '/^[0-9a-f]{40}$/',
			'exit_code' => '/^\d{1,3}$/',
			'runtime' => '/^\d+(\.\d+)?$/'  // Positive number (integer or decimal)
		];

		foreach ($required_params as $param) {
			if (!isset($params[$param])) {
				return false;
			}
			if (!preg_match($patterns[$param], $params[$param])) {
				log_error("Invalid format for parameter: $param");
				return false;
			}
		}

		$exit_code = intval($params['exit_code']);
			if ($exit_code < 0 || $exit_code > 255) {
			log_error("Invalid exit_code value: $exit_code");
			return false;
		}

		$runtime = floatval($params['runtime']);
		if ($runtime < 0) {
			log_error("Invalid runtime value: $runtime");
			return false;
		}

		return true;
	}

	function append_to_csv($params, $filepath) {
		assert(is_array($params), "Parameters should be an array");
		assert(is_string($filepath), "Filepath should be a string");

		$headers = ['anon_user', 'has_sbatch', 'run_uuid', 'git_hash', 'exit_code', 'runtime', 'time'];
		$file_exists = file_exists($filepath);
		$params["time"] = time();

		try {
			$file = fopen($filepath, 'a');
			if (!$file_exists) {
				fputcsv($file, $headers);
			}
			fputcsv($file, $params);
			fclose($file);
		} catch (Exception $e) {
			log_error("Failed to write to CSV: " . $e->getMessage(). ". Make sure <tt>$filepath</tt> is owned by the www-data group and do <tt>chmod g+w $filepath</tt>");
			exit(1);
		}
	}

	function validate_csv($filepath) {
		if (!file_exists($filepath) || !is_readable($filepath)) {
			log_error("CSV file does not exist or is not readable.");
			return false;
		}

		try {
			$file = fopen($filepath, 'r');
			$content = fread($file, filesize($filepath));
			fclose($file);
		} catch (Exception $e) {
			log_error("Failed to read CSV file: " . $e->getMessage());
			return false;
		}

		return true;
	}

	function filter_data($data) {
		$developer_ids = [];
		$test_ids = [];
		$regular_data = [];

		foreach ($data as $row) {
			if ($row[0] == 'affeaffeaffeaffeaffeaffeaffeaffe') {
				$developer_ids[] = $row;
			} elseif ($row[0] == 'affed00faffed00faffed00faffed00f') {
				$test_ids[] = $row;
			} else {
				$regular_data[] = $row;
			}
		}

		return [$developer_ids, $test_ids, $regular_data];
	}

	function display_plots($data, $title, $element_id) {
		$anon_users = array_column($data, 0);
		$has_sbatch = array_column($data, 1);
		$exit_codes = array_map('intval', array_column($data, 4));
		$runtimes = array_map('floatval', array_column($data, 5));

		$unique_sbatch = array_unique($has_sbatch);
		$show_sbatch_plot = count($unique_sbatch) > 1 ? '1' : 0;

		echo "<div id='$element_id-exit-codes' style='height: 400px;'></div>";
		echo "<div id='$element_id-runs' style='height: 400px;'></div>";
		echo "<div id='$element_id-runtimes' style='height: 400px;'></div>";
		echo "<div id='$element_id-runtime-vs-exit-code' style='height: 400px;'></div>";

		if ($show_sbatch_plot) {
			echo "<div id='$element_id-sbatch' style='height: 400px;'></div>";
		}

		echo "<script>
			var anon_users_$element_id = " . json_encode($anon_users) . ";
			var has_sbatch_$element_id = " . json_encode($has_sbatch) . ";
			var exit_codes_$element_id = " . json_encode($exit_codes) . ";
			var runtimes_$element_id = " . json_encode($runtimes) . ";

			var exitCodePlot = {
				x: exit_codes_$element_id,
				type: 'histogram',
				marker: {
					color: exit_codes_$element_id.map(function(code) { return code == 0 ? 'blue' : 'red'; })
				},
				name: 'Exit Codes'
			};

			var userPlot = {
				x: anon_users_$element_id,
				type: 'histogram',
				name: 'Runs per User'
			};

			var runtimePlot = {
				x: runtimes_$element_id,
				type: 'histogram',
				name: 'Runtimes'
			};

			var runtimeVsExitCodePlot = {
				x: exit_codes_$element_id,
				y: runtimes_$element_id,
				mode: 'markers',
				type: 'scatter',
				marker: {
					color: exit_codes_$element_id.map(function(code) { return code == 0 ? 'blue' : 'red'; })
				},
				name: 'Runtime vs Exit Code'
			};

			Plotly.newPlot('$element_id-exit-codes', [exitCodePlot], {title: '$title - Exit Codes'});
			Plotly.newPlot('$element_id-runs', [userPlot], {title: '$title - Runs per User'});
			Plotly.newPlot('$element_id-runtimes', [runtimePlot], {title: '$title - Runtimes'});
			Plotly.newPlot('$element_id-runtime-vs-exit-code', [runtimeVsExitCodePlot], {title: '$title - Runtime vs Exit Code'});

			if ($show_sbatch_plot) {
					var sbatchPlot = {
						x: has_sbatch_$element_id,
						type: 'bar',
						name: 'SBatch Usage'
					};
					Plotly.newPlot('$element_id-sbatch', [sbatchPlot], {title: '$title - SBatch Usage'});
				}
		</script>";
	}

	function generate_html_table($data, $headers) {
		echo "<table>";
		echo "<tr>";
		foreach ($headers as $header) {
			echo "<th>$header</th>";
		}
		echo "</tr>";

		foreach ($data as $row) {
			echo "<tr>";
				foreach ($row as $cell) {
					echo "<td>$cell</td>";
				}
				echo "</tr>";
			}

		echo "</table>";
	}

	$params = $_GET;
	$stats_dir = 'stats';
	$csv_path = $stats_dir . '/usage_statistics.csv';

	if (validate_parameters($params)) {
		if (!file_exists($stats_dir)) {
			mkdir($stats_dir, 0777, true);
		}

		if (is_writable($stats_dir)) {
			append_to_csv($params, $csv_path);
			echo "<p>Data successfully written to CSV.</p>";
			exit(0);
		} else {
			log_error("Stats directory is not writable.");
		}
	}

	if (validate_csv($csv_path)) {
		$data = array_map('str_getcsv', file($csv_path));
		$headers = array_shift($data);

		[$developer_ids, $test_ids, $regular_data] = filter_data($data);

		echo "<h2>Regular Users</h2>";
		generate_html_table($regular_data, $headers);
		display_plots($regular_data, "Regular Users Statistics", "regular_plots");

		echo "<h2>Developer Machines</h2>";
		generate_html_table($developer_ids, $headers);
		display_plots($developer_ids, "Developer Machines Statistics", "developer_plots");

		echo "<h2>Automated Tests</h2>";
		generate_html_table($test_ids, $headers);
		display_plots($test_ids, "Automated Tests Statistics", "test_plots");
	} else {
		log_error("No valid data available to display.");
	}
?>
	<h2>Exit Code Information</h2>
		<table>
		<tr>
			<th>Exit Code</th>
			<th>Error Group Description</th>
		</tr>
		<?php
			$exit_code_info = [
				"-1" => "No proper Exit code found",
				0 => "Seems to have worked properly",
				10 => "Usually only returned by dier (for debugging).",
				15 => "Unimplemented error.",
				18 => "test_wronggoing_stuff program not found (only --tests).",
				19 => "Something was wrong with your parameters. See output for details.",
				31 => "Basic modules could not be loaded or you cancelled loading them.",
				44 => "Continuation of previous job failed.",
				47 => "Missing checkpoint or defective file or state files (check output).",
				49 => "Something went wrong while creating the experiment.",
				99 => "It seems like the run folder was deleted during the run.",
				100 => "--mem_gb or --gpus, which must be int, has received a value that is not int.",
				103 => "--time is not in minutes or HH:MM format.",
				104 => "One of the parameters --mem_gb, --time, or --experiment_name is missing.",
				105 => "Continued job error: previous job has missing state files.",
				142 => "Error in Models like THOMPSON or EMPIRICAL_BAYES_THOMPSON. Not sure why.",
				181 => "Error parsing --parameter. Check output for more details.",
				192 => "Unknown data type (--tests).",
				199 => "This happens on unstable file systems when trying to write a file.",
				203 => "Unsupported --model.",
				233 => "No random steps set.",
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
	</body>
</html>
