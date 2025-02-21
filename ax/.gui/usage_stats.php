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

	require "_usage_stats_header.php";

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

	function append_to_csv($params, $filepath) {
		if (validate_parameters($params)) {
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
				log_error("Failed to write to CSV: " . $e->getMessage() . ". Make sure <tt>$filepath</tt> is owned by the www-data group and do <tt>chmod g+w $filepath</tt>");
				exit(1);
			}
		}
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

	function calculate_statistics($data) {
		$total_jobs = count($data);
		$failed_jobs = count(
			array_filter(
				$data,
				function ($row) {
					return intval($row[4]) != 0;
				}
		)
		);
		$successful_jobs = $total_jobs - $failed_jobs;
		$failure_rate = $total_jobs > 0 ? ($failed_jobs / $total_jobs) * 100 : 0;

		$runtimes = array_map('floatval', array_column($data, 5));
		$total_runtime = array_sum($runtimes);
		$average_runtime = $total_jobs > 0 ? $total_runtime / $total_jobs : 0;

		sort($runtimes);
		$median_runtime = $total_jobs > 0 ? (count($runtimes) % 2 == 0 ? ($runtimes[count($runtimes) / 2 - 1] + $runtimes[count($runtimes) / 2]) / 2 : $runtimes[floor(count($runtimes) / 2)]) : 0;

		$successful_runtimes = array_filter(
			$data,
			function ($row) {
				return intval($row[4]) == 0;
			}
		);
		$successful_runtimes = array_map('floatval', array_column($successful_runtimes, 5));
		$avg_success_runtime = !empty($successful_runtimes) ? array_sum($successful_runtimes) / count($successful_runtimes) : 0;
		sort($successful_runtimes);
		$median_success_runtime = !empty($successful_runtimes) ? (count($successful_runtimes) % 2 == 0 ? ($successful_runtimes[count($successful_runtimes) / 2 - 1] + $successful_runtimes[count($successful_runtimes) / 2]) / 2 : $successful_runtimes[floor(count($successful_runtimes) / 2)]) : 0;

		$failed_runtimes = array_filter(
			$data,
			function ($row) {
				return intval($row[4]) != 0;
			}
		);
		$failed_runtimes = array_map('floatval', array_column($failed_runtimes, 5));
		$avg_failed_runtime = !empty($failed_runtimes) ? array_sum($failed_runtimes) / count($failed_runtimes) : 0;
		sort($failed_runtimes);
		$median_failed_runtime = !empty($failed_runtimes) ? (count($failed_runtimes) % 2 == 0 ? ($failed_runtimes[count($failed_runtimes) / 2 - 1] + $failed_runtimes[count($failed_runtimes) / 2]) / 2 : $failed_runtimes[floor(count($failed_runtimes) / 2)]) : 0;

		if (count($runtimes)) {
			return [
				'total_jobs' => $total_jobs,
				'failed_jobs' => $failed_jobs,
				'successful_jobs' => $successful_jobs,
				'failure_rate' => $failure_rate,
				'average_runtime' => $average_runtime,
				'median_runtime' => $median_runtime,
				'max_runtime' => max($runtimes),
				'min_runtime' => min($runtimes),
				'avg_success_runtime' => $avg_success_runtime,
				'median_success_runtime' => $median_success_runtime,
				'avg_failed_runtime' => $avg_failed_runtime,
				'median_failed_runtime' => $median_failed_runtime
			];
		} else {
			return [
				'total_jobs' => $total_jobs,
				'failed_jobs' => $failed_jobs,
				'successful_jobs' => $successful_jobs,
				'failure_rate' => $failure_rate
			];
		}
	}

	// Main code execution

	$data_filepath = 'stats/usage_statistics.csv';

	if (isset($_SERVER["REQUEST_METHOD"]) && isset($_GET["anon_user"])) {
		append_to_csv($_GET, $data_filepath);
	}

	if (validate_csv($data_filepath)) {
		$data = array_map('str_getcsv', file($data_filepath));
		array_shift($data); // Remove header row

		list($developer_ids, $test_ids, $regular_data) = filter_data($data);
?>
		<br>
		<div id="tabs">
<?php
			$links = [
			    'regular_data' => 'Regular Users',
			    'test_ids' => 'Tests',
			    'developer_ids' => 'Developer'
			];
?>
			<ul>
<?php
				foreach ($links as $key => $label) {
					if (count(${$key})) {
						echo '<li class="invert_in_dark_mode"><a href="#' . $key . '">' . $label . '</a></li>';
					}
				}
?>
				<li class="invert_in_dark_mode"><a href="#exit_codes">Exit-Codes</a></li>
			</ul>
<?php
		$sections = [
			'regular_data' => 'Regular Users',
			'test_ids' => 'Test Users',
			'developer_ids' => 'Developer Users'
		];

		foreach ($sections as $key => $title) {
			if (count(${$key})) {
				echo '<div id="' . $key . '">';
				echo "<h2>$title</h2>";
				display_plots(${$key}, explode('_', $key)[0]);
				echo '</div>';
			}
		}
?>
		<div id="exit_codes">
<?php
			include "exit_code_table.php";
?>
		</div>
<?php
	} else {
		echo "No valid CSV file found";
	}
	include("footer.php");
?>
