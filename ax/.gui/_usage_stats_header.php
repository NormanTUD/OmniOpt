<?php
	error_reporting(E_ALL);
	set_error_handler(
		function ($severity, $message, $file, $line) {
			throw new \ErrorException($message, $severity, $severity, $file, $line);
		}
	);

	ini_set('display_errors', 1);

	function dier($msg) {
		print("<pre>" . print_r($msg, true) . "</pre>");
		exit(1);
	}

	function log_error($error_message) {
		error_log($error_message);
		echo "<p>Error: $error_message</p>";
	}

	require "_header_base.php";
?>
	<script src="plotly-3.0.1.min.js"></script>
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
	<link rel="stylesheet" href="jquery-ui.css">
	<script src="jquery-ui.js"></script>
	<script>
		$(function() {
			$("#tabs").tabs();
		});
	</script>
<?php
	function print_js_code_for_plot ($element_id, $data) {
		$anon_users = array_column($data, 0);
		$has_sbatch = array_column($data, 1);
		$exit_codes = array_map('intval', array_column($data, 4));
		$runtimes = array_map('floatval', array_column($data, 5));

		$show_sbatch_plot = count(array_unique($has_sbatch)) > 1 ? '1' : 0;

		if ($show_sbatch_plot) {
			echo "<div id='$element_id-sbatch' style='height: 400px;'></div>";
		}

		echo "
			<script>
				$(document).ready(function() {
					showSpinnerOverlay(\"Loading data for $element_id\");
					$.ajax({
						url: 'get_usage_stat.php',
						type: 'GET',
						data: { element_id: '$element_id' },
						dataType: 'json',
						error: function (d) {
							err('Error loading data');
							removeSpinnerOverlay();
						},
						success: function(data) {
							var anon_users_$element_id = data.anon_users;
							var has_sbatch_$element_id = data.has_sbatch;
							var exit_codes_$element_id = data.exit_codes;
							var runtimes_$element_id = data.runtimes;

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
								name: 'Runtime vs Exit Code',
								marker: {
									color: exit_codes_$element_id.map(function(code) { return code == 0 ? 'blue' : 'red'; })
								}
							};

							var exitCodeCounts = {};
							exit_codes_$element_id.forEach(function(code) {
								if (!exitCodeCounts[code]) {
									exitCodeCounts[code] = 0;
								}
								exitCodeCounts[code]++;
							});

							var exitCodePie = {
								values: Object.values(exitCodeCounts),
								labels: Object.keys(exitCodeCounts),
								type: 'pie',
								name: 'Exit Code Distribution'
							};

							var avgRuntimes = {};
							var runtimeSum = {};
							var runtimeCount = {};

							for (var i = 0; i < exit_codes_$element_id.length; i++) {
								var code = exit_codes_" . $element_id . "[i];
								var runtime = runtimes_" . $element_id . "[i];
								if (!(code in runtimeSum)) {
									runtimeSum[code] = 0;
									runtimeCount[code] = 0;
								}
								runtimeSum[code] += runtime;
								runtimeCount[code] += 1;
							}

							for (var code in runtimeSum) {
								avgRuntimes[code] = runtimeSum[code] / runtimeCount[code];
							}

							var avgRuntimeBar = {
							x: Object.keys(avgRuntimes),
								y: Object.values(avgRuntimes),
								type: 'bar',
								name: 'Average Runtime per Exit Code'
							};

							var runtimeBox = {
								y: runtimes_$element_id,
								type: 'box',
								name: 'Runtime Distribution'
							};

							// Top N users by number of jobs
							var userJobCounts = {};
							anon_users_$element_id.forEach(function(user) {
								if (!userJobCounts[user]) {
									userJobCounts[user] = 0;
								}
								userJobCounts[user]++;
							});

							var topNUsers = Object.entries(userJobCounts).sort((a, b) => b[1] - a[1]).slice(0, 10);
							var topUserBar = {
								x: topNUsers.map(item => item[0]),
								y: topNUsers.map(item => item[1]),
								type: 'bar',
								name: 'Top Users by Number of Jobs'
							};

							Plotly.newPlot('$element_id-exit-codes', [exitCodePlot], {title: 'Exit Codes', paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', width: 800});
							Plotly.newPlot('$element_id-runs', [userPlot], {title: 'Runs per User', paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', width: 800});
							Plotly.newPlot('$element_id-runtimes', [runtimePlot], {title: 'Runtimes', paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', width: 800});
							Plotly.newPlot('$element_id-runtime-vs-exit-code', [runtimeVsExitCodePlot], {title: 'Runtime vs Exit Code', paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', width: 800});
							Plotly.newPlot('$element_id-exit-code-pie', [exitCodePie], {title: 'Exit Code Distribution', paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0), width: 800'});
							Plotly.newPlot('$element_id-avg-runtime-bar', [avgRuntimeBar], {title: 'Average Runtime per Exit Code', paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', width: 800});
							Plotly.newPlot('$element_id-runtime-box', [runtimeBox], {title: 'Runtime Distribution', paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', width: 800});
							Plotly.newPlot('$element_id-top-users', [topUserBar], {title: 'Top Users by Number of Jobs', paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', width: 800});

							if ($show_sbatch_plot) {
								var sbatchPlot = {
									x: has_sbatch_$element_id,
									type: 'histogram',
									name: 'Runs with and without sbatch'
								};
								Plotly.newPlot('$element_id-sbatch', [sbatchPlot], {title: 'Runs with and without sbatch', paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)'});
							}
							removeSpinnerOverlay();
						}
					});
				});
			</script>
		";
	}

	function display_statistics($stats) {
		echo "<div class='statistics'>";
		echo "<h3>Statistics</h3>";
		echo "<p>Total jobs: {$stats['total_jobs']}</p>";
		echo "<p>Failed jobs: {$stats['failed_jobs']} (" . number_format($stats['failure_rate'], 2) . "%)</p>";
		echo "<p>Successful jobs: {$stats['successful_jobs']}</p>";

		if (isset($stats["average_runtime"])) {
			$runtime_labels = [
				"average_runtime" => "Average runtime",
				"median_runtime" => "Median runtime",
				"max_runtime" => "Max runtime",
				"min_runtime" => "Min runtime",
				"avg_success_runtime" => "Average success runtime",
				"median_success_runtime" => "Median success runtime",
				"avg_failed_runtime" => "Average failed runtime",
				"median_failed_runtime" => "Median failed runtime"
			];

			foreach ($runtime_labels as $key => $label) {
				echo "<p>$label: " . gmdate("H:i:s", intval($stats[$key])) . "</p>";
			}
		}

		echo "</div>";
	}

	function calculate_statistics_from_db($db_path, $element_id) {
		try {
			$db = new SQLite3($db_path);
			$query = "SELECT exit_code, runtime FROM usage_statistics";

			if($element_id == "test") {
				$query .= " where anon_user = 'affed00faffed00faffed00faffed00f'";
			} else if ($element_id == "developer") {
				$query .= " where anon_user = 'affeaffeaffeaffeaffeaffeaffeaffe'";
			} else {
				$query .= " where anon_user != 'affeaffeaffeaffeaffeaffeaffeaffe' and anon_user != 'affed00faffed00faffed00faffed00f'";
			}

			$result = $db->query($query);

			$total_jobs = 0;
			$failed_jobs = 0;
			$runtimes = [];
			$successful_runtimes = [];
			$failed_runtimes = [];

			while ($row = $result->fetchArray(SQLITE3_NUM)) {
				$exit_code = intval($row[0]);
				$runtime = floatval($row[1]);

				$runtimes[] = $runtime;
				$total_jobs++;

				if ($exit_code !== 0) {
					$failed_jobs++;
					$failed_runtimes[] = $runtime;
				} else {
					$successful_runtimes[] = $runtime;
				}
			}

			$db->close();

			if ($total_jobs === 0) {
				return [
					'total_jobs' => 0,
					'failed_jobs' => 0,
					'successful_jobs' => 0,
					'failure_rate' => 0
				];
			}

			$successful_jobs = $total_jobs - $failed_jobs;
			$failure_rate = ($failed_jobs / $total_jobs) * 100;
			$total_runtime = array_sum($runtimes);
			$average_runtime = $total_runtime / $total_jobs;
			$median_runtime = calculate_median($runtimes);

			return [
				'total_jobs' => $total_jobs,
				'failed_jobs' => $failed_jobs,
				'successful_jobs' => $successful_jobs,
				'failure_rate' => $failure_rate,
				'average_runtime' => $average_runtime,
				'median_runtime' => $median_runtime,
				'max_runtime' => max($runtimes),
				'min_runtime' => min($runtimes),
				'avg_success_runtime' => count($successful_runtimes) ? array_sum($successful_runtimes) / count($successful_runtimes) : 0,
				'median_success_runtime' => calculate_median($successful_runtimes),
				'avg_failed_runtime' => count($failed_runtimes) ? array_sum($failed_runtimes) / count($failed_runtimes) : 0,
				'median_failed_runtime' => calculate_median($failed_runtimes)
			];
		} catch (Exception $e) {
			die("Failed to fetch data: " . $e->getMessage());
		}
	}

	function display_plots($data, $element_id, $db_path) {
		$statistics = calculate_statistics_from_db($db_path, $element_id);

		display_statistics($statistics);

		$plots = array(
			"Exit-Code distribution" => 'exit-codes',
			"Runs" => 'runs',
			"Runtimes" => 'runtimes',
			"Runtime vs. exit code" => 'runtime-vs-exit-code',
			"Exit-Code piechart" => 'exit-code-pie',
			"Average runtime" => 'avg-runtime-bar',
			"Runtime box plot" => 'runtime-box',
			"Top users" => 'top-users'
		);

		foreach ($plots as $plot_header => $plot) {
			echo "<h2>$plot_header</h2>\n";
			echo "<div class='usage_plot' id='$element_id-$plot' style='height: 400px;'></div>";
		}

		print_js_code_for_plot($element_id, $data);
	}

	function validate_parameters($params, $filepath) {
		assert(is_array($params), "Parameters should be an array");
		assert(is_string($filepath), "Filepath should be a string");

		$required_params = ['anon_user', 'has_sbatch', 'run_uuid', 'git_hash', 'exit_code', 'runtime'];
		$patterns = [
			'anon_user' => '/^[a-f0-9]{32}$/',
			'has_sbatch' => '/^[01]$/',
			'run_uuid' => '/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/',
			'git_hash' => '/^[0-9a-f]{40}$/',
			'exit_code' => '/^-?\d{1,3}$/',
			'runtime' => '/^\d+(\.\d+)?$/'
		];

		foreach ($required_params as $param) {
			if (!isset($params[$param])) {
				dier("$param is not set");
			}
			if (!preg_match($patterns[$param], $params[$param])) {
				dier("Invalid format for parameter: $param");
			}
		}

		$exit_code = intval($params['exit_code']);
		if ($exit_code < -1 || $exit_code > 255) {
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

	function validate_csv($filepath) {
		if (!file_exists($filepath) || !is_readable($filepath)) {
			log_error("CSV file does not exist or is not readable.");
			return false;
		}

		try {
			$file = fopen($filepath, 'r');
			fread($file, filesize($filepath));
			fclose($file);
		} catch (Exception $e) {
			log_error("Failed to read CSV file: " . $e->getMessage());
			return false;
		}

		return true;
	}

	function calculate_median($values) {
		$count = count($values);
		if ($count === 0) return 0;

		sort($values);
		$middle = floor($count / 2);

		return ($count % 2 === 0)
			? ($values[$middle - 1] + $values[$middle]) / 2
			: $values[$middle];
	}
?>
