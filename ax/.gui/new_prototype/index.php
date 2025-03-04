<?php
	function get_get($name, $default = null) {
		if(isset($_GET[$name])) {
			return $_GET[$name];
		}

		return $default;
	}

	function generateFolderButtons($folderPath, $new_param_name) {
		if (is_dir($folderPath)) {
			$dir = opendir($folderPath);

			// Aktuelle URL abrufen   
			$currentUrl = $_SERVER['REQUEST_URI'];

			// Ordner in einem Array speichern
			$folders = [];
			while (($folder = readdir($dir)) !== false) {
				if ($folder != "." && $folder != ".." && is_dir($folderPath . '/' . $folder)) {
					$folders[] = $folder;
				}
			}

			// Schließen des Verzeichnisses
			closedir($dir);

			// Sortieren der Ordner
			usort($folders, function($a, $b) {
				// Überprüfen, ob beide Ordner numerisch sind
				if (is_numeric($a) && is_numeric($b)) {
					return (int)$a - (int)$b;  // Numerisch aufsteigend
				}
				return strcmp($a, $b);  // Alphabetisch aufsteigend
			});

			// Erstellen der Buttons
			foreach ($folders as $folder) {
				// URL mit dem neuen Parameter an die aktuelle URL anhängen
				$url = $currentUrl . (strpos($currentUrl, '?') === false ? '?' : '&') . $new_param_name . '=' . urlencode($folder);

				// Button als Link mit der erzeugten URL
				echo '<a href="' . htmlspecialchars($url) . '" style="margin: 10px;">';
				echo '<button type="button">' . htmlspecialchars($folder) . '</button>';
				echo '</a><br><br>';
			}
		} else {
			echo "The specified folder does not exist.";
		}
	}

	function is_valid_user_or_experiment_name ($name) {
		if(preg_match("/^[a-zA-Z0-9_]+$/", $name)) {
			return true;
		}

		return false;
	}

	function string_is_numeric ($name) {
		if(preg_match("/^\d+$/", $name)) {
			return true;
		}

		return false;
	}

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

	$share_folder = "shares";

	$errors = [];
	if(!is_dir($share_folder)) {
		$errors[] = "Folder <tt>$share_folder</tt> not found.";
	}


	$user_id = get_get("user_id");
	$experiment_name = get_get("experiment_name");
	$run_nr = get_get("run_nr", -1);

	if($user_id) {
		if(!is_valid_user_or_experiment_name($user_id)) {
			$errors[] = "<tt>".htmlentities($user_id)."</tt> is not a valid username";
		}
	}

	if($experiment_name) {
		if(!is_valid_user_or_experiment_name($experiment_name)) {
			$errors[] = "<tt>".htmlentities($experiment_name)."</tt> is not a valid experiment name";
		}
	}

	if($run_nr != -1) {
		if(!string_is_numeric($run_nr)) {
			$errors[] = "<tt>".htmlentities($run_nr)."</tt> is not a valid run nr";
		}
	}

	if($run_nr == -1) {
		$run_nr = null;
	}
?>
<!DOCTYPE html>
<html lang="en">
	<head>
		<meta charset="UTF-8">
		<meta name="viewport" content="width=device-width, initial-scale=1.0">
		<title>OmniOpt2-Share</title>
		<script src="../plotly-latest.min.js"></script>
		<script src="../jquery-3.7.1.js"></script>
		<script src="gridjs.umd.js"></script>
		<link href="mermaid.min.css" rel="stylesheet" />
		<link href="tabler.min.css" rel="stylesheet">
		<?php include("css.php"); ?>
		<script src="functions.js"></script>
		<script src="main.js"></script>
	</head>
	<body>
		<div class="page window" style='font-family: sans-serif'>
			<div class="title-bar">
				<div class="title-bar-text">OmniOpt2-Share</div>
				<div class="title-bar-controls">
					<button onclick='close_main_window()' aria-label="Close"></button>
				</div>
			</div>
			<div id="spinner" class="spinner"></div>

			<div id="main_window" style="display: none" class="container py-4 has-space">
<?php
				if(count($errors)) {
					if (count($errors) > 1) {
						print "<h2>Errors:</h2>\n";
						print "<ul>\n";
						foreach ($errors as $error) {
							print "<li>$error</li>";
						}
						print "</ul>\n";
					} else {
						print "<h2>Error:</h2>\n";
						print $errors[0];
					}

?>
					<script>
						show_main_window();
					</script>
<?php
				} else {
					if($user_id && $experiment_name && !is_null($run_nr)) {
?>
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
<?php
					} else {
						if(!$user_id && !$experiment_name && !$run_nr) {
							generateFolderButtons($share_folder, "user_id");
						} else if($user_id && !$experiment_name && !$run_nr) {
							generateFolderButtons("$share_folder/$user_id", "experiment_name");
						} else if($user_id && $experiment_name && !$run_nr) {
							generateFolderButtons("$share_folder/$user_id/$experiment_name", "run_nr");
						} else {
							print "DONT KNOW!!! >>$run_nr<<";
						}
					}
				}
?>
			</div>
		</div>
	</body>
</html>
