<?php
	$GLOBALS["json_data"] = [];
	$GLOBALS["functions_after_tab_creation"] = [];

	$SPECIAL_COL_NAMES = [
		"trial_index",
		"arm_name",
		"trial_status",
		"generation_method",
		"generation_node"
	];

	function filter_empty_columns($csv_data) {
		$filtered_data = [];

		foreach ($csv_data as $row) {
			$filtered_row = [];

			foreach ($row as $column) {
				if ($column !== "") {
					$filtered_row[] = $column;
				}
			}

			if (!empty($filtered_row)) {
				$filtered_data[] = $filtered_row;
			}
		}

		return $filtered_data;
	}

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

	function getStatusForResultsCsv($csvFilePath) {
		if (!file_exists($csvFilePath) || !is_readable($csvFilePath)) {
			return json_encode(["error" => "File not found or not readable"], JSON_PRETTY_PRINT);
		}

		$statuses = [
			"failed" => 0,
			"succeeded" => 0,
			"running" => 0,
			"total" => 0
		];

		$k = 0;

		if (($handle = fopen($csvFilePath, "r")) !== false) {
			while (($data = fgetcsv($handle, 0, ",", "\"", "\\")) !== false) {
				if (count($data) < 3) continue;

				$status = strtolower(trim($data[2]));

				if ($k > 0) {
					$statuses["total"]++;

					if ($status === "completed") {
						$statuses["succeeded"]++;
					} elseif ($status === "failed") {
						$statuses["failed"]++;
					} elseif ($status === "running") {
						$statuses["running"]++;
					}
				}

				$k++;
			}
			fclose($handle);
		}

		return $statuses;
	}

	function copy_id_to_clipboard_string($id) {
		return "<br><button onclick='copy_to_clipboard_from_id(\"".$id."\")'>Copy raw data to clipboard</button><br><br>\n";
	}

	function add_worker_cpu_ram_from_file($tabs, $filename, $name, $id) {
		if(is_file($filename)) {
			$html = '<div id="cpuRamWorkerChartContainer"></div><br>';
			$html .= copy_id_to_clipboard_string("worker_cpu_ram_pre");
			$html .= '<pre id="worker_cpu_ram_pre">'.htmlentities(file_get_contents($filename)).'</pre>';
			$html .= copy_id_to_clipboard_string("worker_cpu_ram_pre");

			$tabs[$name] = [
				'id' => $id,
				'content' => $html,
			];

			$GLOBALS["functions_after_tab_creation"][] = "plot_worker_cpu_ram();";
		}

		return $tabs;
	}

	function add_log_from_file($tabs, $filename, $name, $id) {
		if(is_file($filename)) {
			// Attempt to read the file content
			$fileContent = file_get_contents($filename);

			// Check if reading the file was successful
			if ($fileContent === false) {
				$output = "Error loading the file!";
				exit;
			}

			// Split the file content into individual lines
			$lines = explode("\n", $fileContent);

			// Start the HTML table
			$output = "<table border='1'>";
			$output .= "<thead><tr><th>Time</th><th>Function Stack</th><th>Message</th></tr></thead>";
			$output .= "<tbody>";

			// Process each line
			foreach ($lines as $line) {
				// Skip empty lines
				if (trim($line) === "") {
					continue;
				}

				// Try to parse the line as JSON
				$jsonData = json_decode($line, true);

				// Check if JSON parsing was successful
				if ($jsonData === null) {
					$output .= "<tr><td colspan='3'>Error parsing the JSON data: $line</td></tr>";
					continue;
				}

				// Extract Time, Function Stack, and Message
				$time = isset($jsonData['time']) ? $jsonData['time'] : 'Not available';
				$msg = isset($jsonData['msg']) ? $jsonData['msg'] : 'Not available';

				// Format the Function Stack
				$functionStack = '';
				if (isset($jsonData['function_stack'])) {
					foreach ($jsonData['function_stack'] as $functionData) {
						$function = isset($functionData['function']) ? $functionData['function'] : 'Unknown';
						if($function != "_get_debug_json") {
							$lineNumber = isset($functionData['line_number']) ? $functionData['line_number'] : 'Unknown';
							$functionStack .= "$function (Line $lineNumber)<br>";
						}
					}
				}

				// Add a row to the table
				$output .= "<tr>";
				$output .= "<td style='border: 1px solid black;'>$time</td>";
				$output .= "<td style='border: 1px solid black;'>$functionStack</td>";
				$output .= "<td style='border: 1px solid black;'>$msg</td>";
				$output .= "</tr>";
			}

			// End the HTML table
			$output .= "</tbody></table>";

			$tabs[$name] = [
				'id' => $id,
				'content' => $output
			];

			$GLOBALS["functions_after_tab_creation"][] = "plotCPUAndRAMUsage();";
		}

		return $tabs;
	}

	function add_cpu_ram_usage_main_worker_from_file($tabs, $filename, $name, $id) {
		if(is_file($filename)) {
			$html = "<div id='mainWorkerCPURAM'></div>";
			$html .= copy_id_to_clipboard_string("pre_$id");
			$html .= '<pre id="pre_' . $id . '">'.htmlentities(remove_ansi_colors(file_get_contents($filename))).'</pre>';
			$html .= copy_id_to_clipboard_string("pre_$id");

			$csv_contents = getCsvDataAsArray($filename);   
			$headers = $csv_contents[0];
			$csv_contents_no_header = $csv_contents;
			array_shift($csv_contents_no_header);

			$csv_json = $csv_contents_no_header;
			$headers_json = $headers;

			$GLOBALS["json_data"]["${id}_csv_json"] = $csv_json;
			$GLOBALS["json_data"]["${id}_headers_json"] = $headers_json;

			$tabs[$name] = [
				'id' => $id,
				'content' => $html
			];

			$GLOBALS["functions_after_tab_creation"][] = "plotCPUAndRAMUsage();";
		}

		return $tabs;
	}

	function add_scatter_3d_plots($tabs, $filename, $name, $id) {
		if(is_file($filename)) {
			$html = "<div id='plotScatter3d'></div>";

			$tabs[$name] = [
				'id' => $id,
				'content' => $html
			];

			$GLOBALS["functions_after_tab_creation"][] = "plotScatter3d();";
		}

		return $tabs;
	}

	function add_scatter_2d_plots($tabs, $filename, $name, $id) {
		if(is_file($filename)) {
			$html = "<div id='plotScatter2d'></div>";

			$tabs[$name] = [
				'id' => $id,
				'content' => $html
			];

			$GLOBALS["functions_after_tab_creation"][] = "plotScatter2d();";
		}

		return $tabs;
	}

	function add_worker_usage_plot_from_file($tabs, $filename, $name, $id) {
		if(is_file($filename)) {
			$html = "<div id='workerUsagePlot'></div>";
			$html .= copy_id_to_clipboard_string("pre_$id");
			$html .= '<pre id="pre_'.$id.'">'.htmlentities(remove_ansi_colors(file_get_contents($filename))).'</pre>';
			$html .= copy_id_to_clipboard_string("pre_$id");

			$csv_contents = getCsvDataAsArray($filename);   

			$GLOBALS["json_data"]["${id}_csv_json"] = $csv_contents;

			$tabs[$name] = [
				'id' => $id,
				'content' => $html
			];

			$GLOBALS["functions_after_tab_creation"][] = "plotWorkerUsage(tab_worker_usage_csv_json);";
		}

		return $tabs;
	}

	function add_simple_pre_tab_from_file ($tabs, $filename, $name, $id, $remove_ansi_colors = false) {
		if(is_file($filename)) {
			$contents = file_get_contents($filename);
			if(!$remove_ansi_colors) {
				$contents = remove_ansi_colors($contents);
			} else {
				$contents = ansi_to_html(htmlspecialchars($contents));
			}
			$html = copy_id_to_clipboard_string("simple_pre_tab_$id");
			if(!$remove_ansi_colors) {
				$html .= '<pre id="simple_pre_tab_' . $i . '">'.htmlentities($contents).'</pre>';
			} else {
				$html .= '<pre id="simple_pre_tab_' . $i . '">'.$contents.'</pre>';
			}
			$html = copy_id_to_clipboard_string("simple_pre_tab_$id");

			$tabs[$name] = [
				'id' => $id,
				'content' => $html
			];
		}

		return $tabs;
	}

	function add_parallel_plot_tab ($tabs) {
		$html = '<div id="parallel-plot" style="min-width: 1600px; width: 1600px; height: 800px;"></div>';

		$tabs['Parallel Plot'] = [
			'id' => 'tab_parallel',
			'content' => $html
		];

		return $tabs;
	}

	function add_simple_csv_tab_from_file ($tabs, $filename, $name, $id) {
		if(is_file($filename)) {
			$csv_contents = getCsvDataAsArray($filename);   
			$headers = $csv_contents[0];
			$csv_contents_no_header = $csv_contents;
			array_shift($csv_contents_no_header);

			$csv_json = $csv_contents_no_header;
			$headers_json = $headers;

			$GLOBALS["json_data"]["${id}_headers_json"] = $headers_json;
			$GLOBALS["json_data"]["${id}_csv_json"] = $csv_json;
			$GLOBALS["json_data"]["${id}_csv_json_non_empty"] = filter_empty_columns($csv_json);

			$results_html = "<div id='${id}_csv_table'></div>\n";
			$results_html .= copy_id_to_clipboard_string("${id}_csv_table_pre");
			$results_html = "<pre id='${id}_csv_table_pre'>".htmlentities(file_get_contents($filename))."</pre>\n";
			$results_html .= copy_id_to_clipboard_string("${id}_csv_table_pre");
			$results_html .= "<script>\n\tcreateTable(${id}_csv_json, ${id}_headers_json, '${id}_csv_table')</script>\n";

			$tabs[$name] = [
				'id' => $id,
				'content' => $results_html,
			];
		}

		return $tabs;
	}

	function get_log_files($run_dir) {
		$log_files = [];

		if (!is_dir($run_dir)) {
			error_log("Error: Directory does not exist - $run_dir");
			return $log_files;
		}

		$files = scandir($run_dir);
		if ($files === false) {
			error_log("Error: Could not read Directory- $run_dir");
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
				foreach ($row as &$value) {
					if (is_numeric($value)) {
						if (strpos($value, '.') !== false || stripos($value, 'e') !== false) {
							$value = (float)$value;
						} else {
							$value = (int)$value;
						}
					}
					// Sonst bleibt der Wert wie er ist
				}
				unset($value); // Referenz lösen
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
			return;
		}
		if (is_dir($folderPath)) {
			$dir = opendir($folderPath);

			$currentUrl = $_SERVER['REQUEST_URI'];

			$folders = [];
			while (($folder = readdir($dir)) !== false) {
				if ($folder != "." && $folder != ".." && is_dir($folderPath . '/' . $folder)) {
					$folders[] = $folder;
				}
			}

			closedir($dir);

			usort($folders, function($a, $b) {
				if (is_numeric($a) && is_numeric($b)) {
					return (int)$a - (int)$b;
				}
				return strcmp($a, $b);
			});

			foreach ($folders as $folder) {
				$url = $currentUrl . (strpos($currentUrl, '?') === false ? '?' : '&') . $new_param_name . '=' . urlencode($folder);

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

	$tabs = [];

	function ansi_to_html($string) {
		$ansi_colors = [
			'30' => 'black', '31' => 'red', '32' => 'green', '33' => 'yellow',
			'34' => 'blue', '35' => 'magenta', '36' => 'cyan', '37' => 'white',
			'90' => 'brightblack', '91' => 'brightred', '92' => 'brightgreen',
			'93' => 'brightyellow', '94' => 'brightblue', '95' => 'brightmagenta',
			'96' => 'brightcyan', '97' => 'brightwhite'
		];

		$pattern = '/\x1b\[(\d+)(;\d+)*m/';
		return preg_replace_callback($pattern, function($matches) use ($ansi_colors) {
			$codes = explode(';', $matches[1]);
			$style = '';

			foreach ($codes as $code) {
				if (isset($ansi_colors[$code])) {
					$style = 'color:' . $ansi_colors[$code] . ';';
					break;
				}
			}

			return $style ? '<span style="' . $style . '">' : '';
		}, $string);
	}

	function file_contains_results($filename, $names) {
		if (!file_exists($filename) || !is_readable($filename)) {
			return false;
		}

		$file_content = file_get_contents($filename);

		foreach ($names as $name) {
			$pattern = '/(^|\s)\s*' . preg_quote($name, '/') . ':\s*[-+]?\d+(?:\.\d+)?\s*$/mi';

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

		$i = 0;
		foreach ($log_files as $nr => $file) {
			$checkmark = file_contains_results("$run_dir/$file", $result_names) ? $green_checkmark : $red_cross;
			$output .= '<button role="tab" ' . ($i == 0 ? 'aria-selected="true"' : '') . ' aria-controls="single_run_' . $i . '">' . $nr . $checkmark . '</button>';
			$i++;
		}

		$output .= '</menu>';

		$i = 0;
		foreach ($log_files as $nr => $file) {
			$file_path = $run_dir . '/' . $file;
			$content = file_get_contents($file_path);
			$output .= '<article role="tabpanel" id="single_run_' . $i . '">';
			$output .= copy_id_to_clipboard_string("single_run_$i");
			$output .= '<pre>' . ansi_to_html(htmlspecialchars($content)) . '</pre>';
			$output .= copy_id_to_clipboard_string("single_run_$i");
			$output .= '</article>';
			$i++;
		}

		$output .= '</section>';
		return $output;
	}

	$share_folder = "../shares";

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

	$run_dir = "$share_folder/$user_id/$experiment_name/$run_nr";

	if(!count($errors) && $user_id && $experiment_name && $run_nr != -1 && $run_nr !== null && is_dir($run_dir)) {
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

		$GLOBALS["json_data"]["result_names"] = $result_names;
		$GLOBALS["json_data"]["result_min_max"] = $result_min_max;

		$best_results_txt = "$run_dir/best_result.txt";
		$overview_html = "";

		if(is_file("$run_dir/ui_url.txt")) {
			$filePath = "$run_dir/ui_url.txt";
			$firstLine = fgets(fopen($filePath, 'r'));

			if (filter_var($firstLine, FILTER_VALIDATE_URL) && (strpos($firstLine, 'http://') === 0 || strpos($firstLine, 'https://') === 0)) {
				$overview_html .= "<button onclick=\"window.open('".htmlspecialchars($firstLine)."', '_blank')\">GUI page with all the settings of this job</button><br><br>";
			}
		}

		if(is_file($best_results_txt)) {
			$overview_html .= "<pre>\n".htmlentities(remove_ansi_colors(file_get_contents($best_results_txt)))."</pre>";
		}

		if(is_file("$run_dir/parameters.txt")) {
			$overview_html .= "<pre>\n".htmlentities(remove_ansi_colors(file_get_contents("$run_dir/parameters.txt")))."</pre>";
		}

		$status_data = null;

		if(is_file("$run_dir/results.csv")) {
			$status_data = getStatusForResultsCsv("$run_dir/results.csv");

			$overview_table = '<table border="1">';
			$overview_table .= '<tbody>';
			$overview_table .= '<tr>';

			foreach ($status_data as $key => $value) {
				$capitalizedKey = ucfirst($key);
				$overview_table .= '<th style="border: 1px solid black">' . $capitalizedKey . '</th>';
			}
			$overview_table .= '</tr>';

			$overview_table .= '<tr>';

			foreach ($status_data as $value) {
				$overview_table .= '<td style="border: 1px solid black">' . $value . '</td>';
			}
			$overview_table .= '</tr>';

			$overview_table .= '</tbody>';
			$overview_table .= '</table>';

			$overview_html .= "<br>$overview_table";
		}

		if(count($result_names)) {
			$result_names_table = '<br><table border="1">';
			$result_names_table .= '<tr><th style="border: 1px solid black">name</th><th style="border: 1px solid black">min/max</th></tr>';
			for ($i = 0; $i < count($result_names); $i++) {
				$min_or_max = "min";

				if(isset($result_min_max[$i])) {
					$min_or_max = $result_min_max[$i];
				}

				$result_names_table .= '<tr>';
				$result_names_table .= '<td style="border: 1px solid black">' . htmlspecialchars($result_names[$i]) . '</td>';
				$result_names_table .= '<td style="border: 1px solid black">' . htmlspecialchars($min_or_max) . '</td>';
				$result_names_table .= '</tr>';
			}
			$result_names_table .= '</table><br>';

			$overview_html .= $result_names_table;
		}

		if($overview_html != "") {
			$tabs['Overview'] = [
				'id' => 'tab_overview',
				'content' => $overview_html
			];
		}

		if(file_exists("$run_dir/pareto_front_data.json") && file_exists("$run_dir/pareto_front_table.txt")) {
			$pareto_front_html = "";

			$pareto_front_txt_file = "$run_dir/pareto_front_table.txt";

			$pareto_front_txt = remove_ansi_colors(file_get_contents($pareto_front_txt_file));

			if($pareto_front_txt) {
				$pareto_front_html .= "<pre>$pareto_front_txt</pre><br>\n";
			}

			$GLOBALS["json_data"]["pareto_front_data"] = json_decode(file_get_contents("$run_dir/pareto_front_data.json"));

			$GLOBALS["functions_after_tab_creation"][] = "load_pareto_graph();";

			if($pareto_front_html) {
				$pareto_front_html = "<div id='pareto_front_graphs_container'></div>\n$pareto_front_html";

				$tabs['Pareto-Fronts'] = [
					'id' => 'tab_pareto_fronts',
					'content' => $pareto_front_html
				];
			}
		}

		if($status_data && isset($status_data["succeeded"]) && $status_data["succeeded"] > 0) {
			$tabs = add_parallel_plot_tab($tabs);
			$GLOBALS["functions_after_tab_creation"][] = "createParallelPlot(tab_results_csv_json, tab_results_headers_json, result_names, special_col_names);";
		}

		$tabs = add_simple_csv_tab_from_file($tabs, "$run_dir/results.csv", "Results", "tab_results");
		$tabs = add_simple_csv_tab_from_file($tabs, "$run_dir/job_infos.csv", "Job-Infos", "tab_job_infos");
		$tabs = add_simple_csv_tab_from_file($tabs, "$run_dir/get_next_trials.csv", "Get-Next-Trials", "tab_get_next_trials");
		$tabs = add_simple_pre_tab_from_file($tabs, "$run_dir/outfile", "Main-Log", "tab_main_log", true);
		$tabs = add_simple_pre_tab_from_file($tabs, "$run_dir/trial_index_to_params", "Trial-Index-to-Param", "tab_trial_index_to_param");
		$tabs = add_simple_pre_tab_from_file($tabs, "$run_dir/experiment_overview.txt", "Experiment Overview", "tab_experiment_overview");
		$tabs = add_simple_pre_tab_from_file($tabs, "$run_dir/progressbar", "Progressbar log", "tab_progressbar_log");
		$tabs = add_simple_pre_tab_from_file($tabs, "$run_dir/args_overview.txt", "Args Overview", "tab_args_overview");
		$tabs = add_worker_usage_plot_from_file($tabs, "$run_dir/worker_usage.csv", "Worker-Usage", "tab_worker_usage");
		$tabs = add_cpu_ram_usage_main_worker_from_file($tabs, "$run_dir/cpu_ram_usage.csv", "CPU/RAM-Usage (main worker)", "tab_main_worker_cpu_ram");
		$tabs = add_log_from_file($tabs, "$run_dir/log", "Debug-Logs", "tab_debug_logs");
		$tabs = add_worker_cpu_ram_from_file($tabs, "$run_dir/eval_nodes_cpu_ram_logs.txt", "Worker-CPU-RAM-Graphs", "tab_worker_cpu_ram_graphs");

		$out_files = get_log_files($run_dir);

		if (count($result_names) == 1) {
			$tabs = add_scatter_2d_plots($tabs, "$run_dir/results.csv", "Scatter-2D", "tab_scatter_2d");

			$difference = array_diff($GLOBALS["json_data"]["tab_results_headers_json"], $SPECIAL_COL_NAMES);

			if(count($difference) >= 3) {
				$tabs = add_scatter_3d_plots($tabs, "$run_dir/results.csv", "Scatter-3D", "tab_scatter_3d");
			}
		}

		if(count($out_files)) {
			$tabs['Single Logs'] = [
				'id' => 'tab_logs',
				'content' => generate_log_tabs($run_dir, $out_files, $result_names)
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
		<script src="new_share_functions.js"></script>
		<script src="main.js"></script>
	</head>
	<body>
		<script>
			var special_col_names = <?php print json_encode($SPECIAL_COL_NAMES); ?>;
<?php
		if(count($GLOBALS["json_data"])) {
			print "<!-- GLOBALS[json_data] -->\n";
			foreach ($GLOBALS["json_data"] as $json_name => $json_data) {
				print "\tvar $json_name = ".json_encode($json_data).";\n";
			}
		}
?>
		</script>
		<div class="page window" style='font-family: sans-serif'>
			<div class="title-bar" style="height: fit-content;">
				<div class="title-bar-text">OmniOpt2-Share
<?php
				if(get_get("user_id") || get_get("experiment_name") || get_get("run_nr")) {
					function is_valid_user_id($value) {
						return preg_match('/^[a-zA-Z0-9_]+$/', $value);
					}

					function is_valid_experiment_name($value) {
						return preg_match('/^[a-zA-Z0-9_]+$/', $value);
					}

					function is_valid_run_nr($value) {
						return preg_match('/^\d+$/', $value);
					}

					$user_id_link = get_get("user_id");
					$experiment_name_link = get_get("experiment_name");
					$run_nr_link = get_get("run_nr");

					if (!is_valid_user_id($user_id_link)) {
						$user_id_link = '';
					}

					if (!is_valid_experiment_name($experiment_name_link)) {
						$experiment_name_link = '';
					}

					if (!is_valid_run_nr($run_nr_link)) {
						$run_nr_link = '';
					}

					$base_url = "?";

					$links = [];

					if (!empty($user_id_link)) {
						$links[] = '<a class="top_link" href="' . $base_url . 'user_id=' . urlencode($user_id_link) . '">' . $user_id_link . '</a>';
					}

					if (!empty($experiment_name_link)) {
						$links[] = '<a class="top_link" href="' . $base_url . 'user_id=' . urlencode($user_id_link) . '&experiment_name=' . urlencode($experiment_name_link) . '">' . $experiment_name_link . '</a>';
					}

					if ($run_nr_link != "") {
						$links[] = '<a class="top_link" href="' . $base_url . 'user_id=' . urlencode($user_id_link) . '&experiment_name=' . urlencode($experiment_name_link) . '&run_nr=' . urlencode($run_nr_link) . '">' . $run_nr_link . '</a>';
					}

					if(count($links)) {
						$home = $_SERVER["PHP_SELF"];
						$home = preg_replace("/.*\//", "", $home);
						$home = preg_replace("/\.php$/", "", $home);

						array_unshift($links, "<a class='top_link' href='$home'>Home</a>");
					}

					$path_with_links = implode(" / ", $links);

					if(count($links)) {
						echo " ($path_with_links)";
					}
				}
?>
				</div>
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

			<!--
			<div id="left_tree_view">
				<ul class="tree-view">
					<li>OmniOpt2-Share</li>
					<li>
							<ul>
								<li>
										<summary>test_job</summary>
										<ul>
											<li>0</li>
											<li>1</li>
										</ul>
								</li>
							</ul>
					</li>
				</ul>
			</div>
			-->

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
		<script>
<?php
			if(count($GLOBALS["functions_after_tab_creation"])) {
				foreach ($GLOBALS["functions_after_tab_creation"] as $fn) {
					print "\t\t\t$fn\n";
				}
			}
?>
			show_main_window();
		</script>
	</body>
</html>
