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
?>
