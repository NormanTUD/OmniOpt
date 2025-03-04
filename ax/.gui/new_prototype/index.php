<?php
	$SPECIAL_COL_NAMES = [
		"trial_index",
		"arm_name",
		"trial_status",
		"generation_method",
		"generation_node"
	];

	function read_file_as_array($filePath) {
		if (!is_readable($filePath)) {
			trigger_error("File cannot be read: $filePath", E_USER_WARNING);
			return [];
		}

		$lines = file($filePath, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES);

		if ($lines === false) {
			trigger_error("Error while reading this file: $filePath", E_USER_WARNING);
			return [];
		}

		return $lines;
	}

	function add_simple_pre_tab_from_file ($tabs, $filename, $name, $id) {
		if(is_file($filename)) {
			$tabs[$name] = [
				'id' => $id,
				'content' => '<pre>'.htmlentities(remove_ansi_colors(file_get_contents($filename))).'</pre>',
			];
		}

		return $tabs;
	}

	function get_log_files($run_dir) {
		$log_files = [];

		if (!is_dir($run_dir)) {
			error_log("Fehler: Verzeichnis existiert nicht - $run_dir");
			return $log_files;
		}

		$files = scandir($run_dir);
		if ($files === false) {
			error_log("Fehler: Konnte Verzeichnis nicht lesen - $run_dir");
			return $log_files;
		}

		foreach ($files as $file) {
			if (preg_match('/^(\d+)_0_log\.out$/', $file, $matches)) {
				$nr = $matches[1];
				$log_files[$nr] = $file;
			}
		}

		return $log_files;
	}

	if (!function_exists("dier")) {
		function dier($data, $enable_html = 0, $exception = 0) {
			#$source_data = debug_backtrace()[0];
			#@$source = 'Aufgerufen von <b>' . debug_backtrace()[1]['file'] . '</b>::<i>' . debug_backtrace()[1]['function'] . '</i>, line ' . htmlentities($source_data['line']) . "<br>\n";
			#$print = $source;
			$print = "";

			$print .= "<pre>\n";
			ob_start();
			print_r($data);
			$buffer = ob_get_clean();
			if ($enable_html) {
				$print .= $buffer;
			} else {
				$print .= htmlentities($buffer);
			}
			$print .= "</pre>\n";

			$print .= "Backtrace:\n";
			$print .= "<pre>\n";
			foreach (debug_backtrace() as $trace) {
				$print .= htmlentities(sprintf("\n%s:%s %s", $trace['file'], $trace['line'], $trace['function']));
			}
			$print .= "</pre>\n";

			if (!$exception) {
				print $print;
				exit();
			} else {
				throw new Exception($print);
			}
		}
	}

	function getCsvDataAsArray($filePath, $delimiter = ",") {
		if (!file_exists($filePath) || !is_readable($filePath)) {
			error_log("CSV file not found or not readable: " . $filePath);
			return [];
		}

		$data = [];

		if (($handle = fopen($filePath, "r")) !== false) {
			while (($row = fgetcsv($handle, 0, $delimiter)) !== false) {
				$data[] = $row;
			}
			fclose($handle);
		} else {
			error_log("Failed to open CSV file: " . $filePath);
		}

		return $data;
	}

	function get_get($name, $default = null) {
		if(isset($_GET[$name])) {
			return $_GET[$name];
		}

		return $default;
	}

	function remove_ansi_colors($contents) {
		$contents = preg_replace('#\\x1b[[][^A-Za-z]*[A-Za-z]#', '', $contents);
		return $contents;
	}

	function generateFolderButtons($folderPath, $new_param_name) {
		if(!isset($_SERVER["REQUEST_URI"])) {
			return; // Don't run this in CLI
		}
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
		]
	];

	$tabs = [];

	function file_contains_results($filename, $names) {
		if (!file_exists($filename) || !is_readable($filename)) {
			return false;
		}

		$file_content = file_get_contents($filename);

		foreach ($names as $name) {
			$pattern = '/^\s*' . preg_quote($name, '/') . ':\s*[-+]?\d+(?:\.\d+)?\s*$/m';

			if (!preg_match($pattern, $file_content)) {
				return false;
			}
		}

		return true;
	}

	function generate_log_tabs($run_dir, $log_files, $result_names) {
		$red_cross = "&#10060;";
		$green_checkmark = "&#9989;";

		$output = '<section class="tabs" style="width: 100%"><menu role="tablist" aria-label="Single-Runs">';

		// Tab-Buttons erstellen
		$i = 0;
		foreach ($log_files as $nr => $file) {
			$checkmark = file_contains_results("$run_dir/$file", $result_names) ? $green_checkmark : $red_cross;
			$output .= '<button role="tab" ' . ($i == 0 ? 'aria-selected="true"' : '') . ' aria-controls="single_run_' . $i . '">' . $nr . $checkmark . '</button>';
			$i++;
		}

		$output .= '</menu>';

		// Tab-Inhalte erstellen
		$i = 0;
		foreach ($log_files as $nr => $file) {
			$file_path = $run_dir . '/' . $file; // Hier den vollständigen Pfad zur Datei anpassen
			$content = file_get_contents($file_path); // Inhalt der Datei holen
			$output .= '<article role="tabpanel" id="single_run_' . $i . '"><pre>' . htmlspecialchars($content) . '</pre></article>';
			$i++;
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

	if($user_id) {
		$run_dir = "$share_folder/$user_id";

		if(!is_dir($run_dir)) {
			$errors[] = "<tt>".htmlentities($run_dir)."</tt> cannot be found!";
		}
	}

	if($run_nr == -1) {
		$run_nr = null;
	} else {
		if(!count($errors)) {
			$run_dir = "$share_folder/$user_id/$experiment_name/$run_nr";

			if(!is_dir($run_dir)) {
				$errors[] = "<tt>".htmlentities($run_dir)."</tt> cannot be found!";
			}
		}
	}

	$run_dir = "";

	$global_json_data = [];

	if(!count($errors)) {
		$run_dir = "$share_folder/$user_id/$experiment_name/$run_nr";

		$result_names_file = "$run_dir/result_names.txt";
		$result_min_max_file = "$run_dir/result_min_max.txt";

		$result_names = ["RESULT"];
		$result_min_max = ["min"];

		if(is_file($result_names_file)) {
			$result_names = read_file_as_array($result_names_file);
		}
		if(is_file($result_min_max_file)) {
			$result_min_max = read_file_as_array($result_min_max_file);
		}

		$global_json_data["result_names"] = $result_names;
		$global_json_data["result_min_max"] = $result_min_max;

		$best_results_txt = "$run_dir/best_result.txt";
		$results_csv = "$run_dir/results.csv";

		$overview_html = "";
		$results_html = "";

		if($results_csv) {
			$csv_contents = getCsvDataAsArray($results_csv);   
			$headers = $csv_contents[0]; // Erste Zeile als Header speichern
			$csv_contents_no_header = $csv_contents;
			array_shift($csv_contents_no_header); // Entferne die Kopfzeile

			$results_csv_json = $csv_contents_no_header;
			$results_headers_json = $headers;

			$global_json_data["results_csv_json"] = $results_csv_json;
			$global_json_data["results_headers_json"] = $results_headers_json;

			$results_html .= "<button onclick='copy_to_clipboard_base64(\"".base64_encode(htmlentities(file_get_contents($results_csv)))."\")'>Copy raw data to clipboard</button>\n";
			$results_html .= "<div id='results_csv_table'></div>\n";
			$results_html .= "<script>\n\tcreateTable(results_csv_json, results_headers_json, 'results_csv_table')</script>\n";
			$results_html .= "<button onclick='copy_to_clipboard_base64(\"".base64_encode(htmlentities(file_get_contents($results_csv)))."\")'>Copy raw data to clipboard</button>\n";
		}

		if(is_file($best_results_txt)) {
			$overview_html .= "<h3>Best results:</h3>\n<pre>\n".htmlentities(remove_ansi_colors(file_get_contents($best_results_txt)))."</pre>";
		}

		if($overview_html != "") {
			$tabs['Overview'] = [
				'id' => 'tab_overview',
				'content' => $overview_html
			];
		}

		$tabs = add_simple_pre_tab_from_file($tabs, "$run_dir/args_overview.txt", "Args Overview", "tab_args_overview");

		$out_files = get_log_files($run_dir);

		if(count($out_files)) {
			$tabs['Single Logs'] = [
				'id' => 'tab_logs',
				'content' => generate_log_tabs($run_dir, $out_files, $result_names)
			];
		}

		if ($results_html != "") {
			$tabs['Results'] = [
				'id' => 'tab_results',
				'content' => $results_html,
			];
		}
	}

	if(!count($tabs) && $run_dir != "" && count($errors)) {
		$errors[] = "Cannot plot any data in <tt>".htmlentities($run_dir)."</tt>";
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
<?php
		if(count($global_json_data)) {
			print "<script>\n";
			print "<!-- global_json_data -->\n";
			foreach ($global_json_data as $json_name => $json_data) {
				print "\tvar $json_name = ".json_encode($json_data).";\n";
			}
			print "</script>\n";
		}
?>
		<div class="page window" style='font-family: sans-serif'>
			<div class="title-bar">
				<div class="title-bar-text">OmniOpt2-Share</div>
<?php
				if(get_get("user_id") || get_get("experiment_name") || get_get("run_nr")) {
?>
					<div class="title-bar-controls">
						<button onclick='close_main_window()' aria-label="Close"></button>
					</div>
<?php
				}
?>
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
								echo '<article role="tabpanel" id="' . $tab_data['id'] . '" ' . ($tab_name === 'General Info' ? '' : 'hidden') . ">\n";
								echo $tab_data['content'];
								echo "</article>\n";
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
