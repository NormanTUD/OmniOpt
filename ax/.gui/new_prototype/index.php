<?php
	$tabs = [
		'Overview' => [
			'id' => 'tab_overview',
			'content' => '<pre>Overview</pre>',
		],
		'Experiment Overview' => [
			'id' => 'tab_experiment_overview',
			'content' => '<pre>Experiment overview</pre>',
		],
		'Progressbar-Logs' => [
			'id' => 'tab_progressbar_logs',
			'content' => '<pre>Progressbar-Logs</pre>',
		],
		'Job-Infos' => [
			'id' => 'tab_job_infos',
			'content' => '<pre>Job-Infos</pre>',
		],
		'Worker-Usage' => [
			'id' => 'tab_worker_usage',
			'content' => '<pre>Worker-Usage</pre>',
		],
		'Main-Log' => [
			'id' => 'tab_main_log',
			'content' => '<pre>Main Log</pre>',
		],
		'Worker-CPU-RAM-Graphs' => [
			'id' => 'tab_worker_cpu_ram_graphs',
			'content' => '<pre>Worker CPU RAM Graphs</pre>',
		],
		'Debug-Log' => [
			'id' => 'tab_debug_log',
			'content' => '<pre>Debug Log</pre>',
		],
		'Args Overview' => [
			'id' => 'tab_args_overview',
			'content' => '<pre>Args-Overview</pre>',
		],
		'CPU/Ram Usage' => [
			'id' => 'tab_cpu_ram_usage',
			'content' => '<pre>CPU-Ram-Usage</pre>',
		],
		'Trial-Index-to-Param' => [
			'id' => 'tab_trial_index_to_param',
			'content' => '<pre>Trial index to param</pre>',
		],
		'Next-Trials' => [
			'id' => 'tab_next_trials',
			'content' => '<p>Next Trials</p>',
		],
		'2D-Scatter' => [
			'id' => 'tab_scatter_2d',
			'content' => '<div id="scatter2d"></div>',
		],
		'3D-Scatter' => [
			'id' => 'tab_scatter_3d',
			'content' => '<div id="scatter3d"></div>',
		],
		'Parallel Plot' => [
			'id' => 'tab_parallel',
			'content' => '<div id="parallel"></div>',
		],
		'Results' => [
			'id' => 'tab_results',
			'content' => '<div id="table"></div>',
		],
		'Single Logs' => [
			'id' => 'tab_logs',
			'content' => generate_log_tabs(50),  // Beispiel: 50 Logs dynamisch generiert
		],
	];

	function generate_log_tabs($nr_files) {
		$output = '<section class="tabs" style="width: 100%"><menu role="tablist" aria-label="Single-Runs">';
		for ($i = 0; $i < $nr_files; $i++) {
			$output .= '<button role="tab" ' . ($i == 0 ? 'aria-selected="true"' : '') . ' aria-controls="single_run_' . $i . '">Single-Run-' . $i . '</button>';
		}
		$output .= '</menu>';
		for ($i = 0; $i < $nr_files; $i++) {
			$output .= '<article role="tabpanel" id="single_run_' . $i . '"><pre>C:\WINDOWS\SYSTEM32> Single-Run ' . $i . '</pre></article>';
		}
		$output .= '</section>';
		return $output;
	}
?>
<!DOCTYPE html>
<html lang="en">
	<head>
		<meta charset="UTF-8">
		<meta name="viewport" content="width=device-width, initial-scale=1.0">
		<title>OmniOpt2-Share</title>
		<script src="../plotly-latest.min.js"></script>
		<script src="gridjs.umd.js"></script>
		<link href="mermaid.min.css" rel="stylesheet" />
		<link href="tabler.min.css" rel="stylesheet">
		<?php include("css.php"); ?>
	</head>
	<body>
		<div class="page window" style='font-family: sans-serif'>
			<div class="title-bar">
				<div class="title-bar-text">OmniOpt2-Share</div>
			</div>
			<div id="spinner" class="spinner"></div>

			<div id="main_window" style="display: none" class="container py-4 has-space">
				<section class="tabs" style="width: 100%">
					<menu role="tablist" aria-label="OmniOpt2-Run">
<?php
						$first_tab = true;
						foreach ($tabs as $tab_name => $tab_data) {
							echo '<button role="tab" aria-controls="' . $tab_data['id'] . '" ' . ($first_tab ? 'aria-selected="true"' : '') . '>' . $tab_name . '</button>';
							$first_tab = false;
						}
?>
					</menu>

<?php
					foreach ($tabs as $tab_name => $tab_data) {
						echo '<article role="tabpanel" id="' . $tab_data['id'] . '" ' . ($tab_name === 'General Info' ? '' : 'hidden') . '>';
						echo $tab_data['content'];
						echo '</article>';
					}
?>
				</section>
			</div>
		</div>

		<script src="functions.js"></script>
		<script src="main.js"></script>
	</body>
</html>
