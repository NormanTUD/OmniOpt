<?php
	require "_usage_stats_header.php";

	function append_to_csv($params, $filepath) {
		if (validate_parameters($params, $filepath)) {
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
		} else {
			log_error("Parameters contain wrong values. Cannot save.");
			exit(1);
		}
	}

	function group_data($data) {
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
		$failed_jobs = count(array_filter($data, fn($row) => intval($row[4]) !== 0));
		$successful_jobs = $total_jobs - $failed_jobs;
		$failure_rate = $total_jobs > 0 ? ($failed_jobs / $total_jobs) * 100 : 0;

		$runtimes = array_map('floatval', array_column($data, 5));

		if ($total_jobs === 0) {
			return [
				'total_jobs' => 0,
				'failed_jobs' => 0,
				'successful_jobs' => 0,
				'failure_rate' => 0
			];
		}

		$total_runtime = array_sum($runtimes);
		$average_runtime = $total_runtime / $total_jobs;
		$median_runtime = calculate_median($runtimes);

		$successful_runtimes = array_map('floatval', array_column(array_filter($data, fn($row) => intval($row[4]) === 0), 5));
		$failed_runtimes = array_map('floatval', array_column(array_filter($data, fn($row) => intval($row[4]) !== 0), 5));

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
	}

	// Main code execution

	$data_filepath = 'stats/usage_statistics.csv';

	if (isset($_SERVER["REQUEST_METHOD"]) && isset($_GET["anon_user"])) {
		append_to_csv($_GET, $data_filepath);
	}

	if (validate_csv($data_filepath)) {
		$data = array_map('str_getcsv', file($data_filepath));
		array_shift($data); // Remove header row

		list($developer_ids, $test_ids, $regular_data) = group_data($data);
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
