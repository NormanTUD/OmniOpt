<?php
	require "_header_base.php";
?>
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
	<link rel="stylesheet" href="jquery-ui.css">
	<script src="jquery-ui.js"></script>
	<script>
		$(function() {
			$("#tabs").tabs();
		});
	</script>
<?php
	function print_js_code_for_plot ($element_id, $anon_users, $has_sbatch, $exit_codes, $runtimes, $show_sbatch_plot) {
		echo "
			<script>
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

				Plotly.newPlot('$element_id-exit-codes', [exitCodePlot], {title: 'Exit Codes', paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)'});
				Plotly.newPlot('$element_id-runs', [userPlot], {title: 'Runs per User', paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)'});
				Plotly.newPlot('$element_id-runtimes', [runtimePlot], {title: 'Runtimes', paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)'});
				Plotly.newPlot('$element_id-runtime-vs-exit-code', [runtimeVsExitCodePlot], {title: 'Runtime vs Exit Code', paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)'});
				Plotly.newPlot('$element_id-exit-code-pie', [exitCodePie], {title: 'Exit Code Distribution', paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)'});
				Plotly.newPlot('$element_id-avg-runtime-bar', [avgRuntimeBar], {title: 'Average Runtime per Exit Code', paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)'});
				Plotly.newPlot('$element_id-runtime-box', [runtimeBox], {title: 'Runtime Distribution', paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)'});
				Plotly.newPlot('$element_id-top-users', [topUserBar], {title: 'Top Users by Number of Jobs', paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)'});

				if ($show_sbatch_plot) {
					var sbatchPlot = {
						x: has_sbatch_$element_id,
						type: 'histogram',
						name: 'Runs with and without sbatch'
					};
					Plotly.newPlot('$element_id-sbatch', [sbatchPlot], {title: 'Runs with and without sbatch', paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)'});
				}
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

	function display_plots($data, $element_id) {
		$statistics = calculate_statistics($data);
		display_statistics($statistics);

		$anon_users = array_column($data, 0);
		$has_sbatch = array_column($data, 1);
		$exit_codes = array_map('intval', array_column($data, 4));
		$runtimes = array_map('floatval', array_column($data, 5));

		$unique_sbatch = array_unique($has_sbatch);
		$show_sbatch_plot = count($unique_sbatch) > 1 ? '1' : 0;

		$plots = [
			'exit-codes',
			'runs',
			'runtimes',
			'runtime-vs-exit-code',
			'exit-code-pie',
			'avg-runtime-bar',
			'runtime-box',
			'top-users'
		];

		foreach ($plots as $plot) {
			echo "<div class='usage_plot' id='$element_id-$plot' style='height: 400px;'></div>";
		}

		if ($show_sbatch_plot) {
			echo "<div id='$element_id-sbatch' style='height: 400px;'></div>";
		}

		print_js_code_for_plot($element_id, $anon_users, $has_sbatch, $exit_codes, $runtimes, $show_sbatch_plot);
	}
?>
