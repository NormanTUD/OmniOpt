<?php
	error_reporting(E_ALL);
	set_error_handler(
		function ($severity, $message, $file, $line) {
			throw new \ErrorException($message, $severity, $severity, $file, $line);
		}
	);

	ini_set('display_errors', 1);

	$GLOBALS["sharesPath"] = "shares/";

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

	function respond_with_error($error_message) {
		header('Content-Type: application/json');

		print json_encode(array("error" => $error_message));
		exit(1);
	}

	function validate_param($param_name, $pattern, $error_message) {
		$value = get_get($param_name);
		if (!preg_match($pattern, $value)) {
			throw new Exception($error_message);
		}
		return $value;
	}

	function validate_directory($dir_path) {
		if (!is_dir($dir_path)) {
			throw new Exception("$dir_path not found");
		}
	}

	function respond_with_json($data) {
		header('Content-Type: application/json');

		print json_encode(array(
			"data" => $data,
			"hash" => hash("md5", json_encode($data))
		));
		exit(0);
	}


	function build_run_folder_path($user_id, $experiment_name, $run_nr) {
		return "$user_id/$experiment_name/$run_nr/";
	}

	function get_get($name, $default = null) {
		if(isset($_GET[$name])) {
			return $_GET[$name];
		}

		return $default;
	}

	function ansi_to_html($string) {
		$ansi_colors = [
			'30' => 'black', '31' => 'red', '32' => 'green', '33' => 'yellow',
			'34' => 'blue', '35' => 'magenta', '36' => 'cyan', '37' => 'white',
			'90' => 'brightblack', '91' => 'brightred', '92' => 'brightgreen',
			'93' => 'brightyellow', '94' => 'brightblue', '95' => 'brightmagenta',
			'96' => 'brightcyan', '97' => 'brightwhite'
		];

		$pattern = '/\x1b\[(\d+)(;\d+)*m/';

		$current_style = '';

		return preg_replace_callback($pattern, function($matches) use ($ansi_colors, &$current_style) {
			$codes = explode(';', $matches[1]);
			$style = '';

			foreach ($codes as $code) {
				if (isset($ansi_colors[$code])) {
					$style = 'color:' . $ansi_colors[$code] . ';';
					break;
				}
			}

			$output = '';
			if ($style !== $current_style) {
				if ($current_style) {
					$output .= '</span>';
				}
				if ($style) {
					$output .= '<span style="' . $style . '">';
				}
				$current_style = $style;
			}

			return $output;
		}, $string) . ($current_style ? '</span>' : '');
	}

	function remove_sixel ($output) {
		$output = preg_replace("/(P[0-9;]+q.*?)(?=(P|$|\s))/", "<br>", $output);

		return $output;
	}

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
			$html = "<button onclick='plot_worker_cpu_ram()' id='plot_worker_cpu_ram_button'>Plot this data (may be slow)</button>\n";
			$html .= '<div id="cpuRamWorkerChartContainer"></div><br>';
			$html .= copy_id_to_clipboard_string("worker_cpu_ram_pre");
			$html .= '<pre id="worker_cpu_ram_pre">'.htmlentities(file_get_contents($filename)).'</pre>';
			$html .= copy_id_to_clipboard_string("worker_cpu_ram_pre");

			$tabs[$name] = [
				'id' => $id,
				'content' => $html
			];
		}

		return $tabs;
	}

	function add_debug_log_from_file($tabs, $filename, $name, $id) {
		if(is_file($filename)) {
			$output = "<div id='debug_log_spinner' class='spinner'></div>";
			$output .= "<div id='here_debuglogs_go'></div>";

			$tabs[$name] = [
				'id' => $id,
				'content' => $output,
				'onclick' => "load_debug_log()"
			];
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
				'content' => $html,
				"onclick" => "plotCPUAndRAMUsage();"
			];
		}

		return $tabs;
	}

	function add_scatter_3d_plots($tabs, $filename, $name, $id) {
		if(is_file($filename)) {
			$html = "<div id='plotScatter3d'></div>";

			$tabs[$name] = [
				'id' => $id,
				'content' => $html,
				"onclick" => "plotScatter3d()"
			];
		}

		return $tabs;
	}

	function add_scatter_2d_plots($tabs, $filename, $name, $id) {
		if(is_file($filename)) {
			$html = "<div id='plotScatter2d'></div>";

			$tabs[$name] = [
				'id' => $id,
				'content' => $html,
				"onclick" => "plotScatter2d();"
			];
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
				'content' => $html,
				"onclick" => "plotWorkerUsage();"
			];
		}

		return $tabs;
	}

	function add_simple_pre_tab_from_file ($tabs, $filename, $name, $id, $remove_ansi_colors = false) {
		if(is_file($filename) && filesize($filename) > 0) {
			$contents = file_get_contents($filename);
			if(!$remove_ansi_colors) {
				$contents = remove_ansi_colors($contents);
			} else {
				$contents = removeAnsiEscapeSequences(ansi_to_html(htmlspecialchars($contents)));
			}

			$html = copy_id_to_clipboard_string("simple_pre_tab_$id");
			if(!$remove_ansi_colors) {
				$contents = htmlentities($contents);
			} else {
				$contents = remove_sixel($contents);
			}

			$html .= "<pre id='simple_pre_tab_$id'>$contents</pre>";

			$html .= copy_id_to_clipboard_string("simple_pre_tab_$id");

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
			'content' => $html,
			"onclick" => "createParallelPlot(tab_results_csv_json, tab_results_headers_json, result_names, special_col_names);"
		];

		return $tabs;
	}

	function add_simple_csv_tab_from_file ($tabs, $filename, $name, $id, $header_line = null) {
		if(is_file($filename) && filesize($filename)) {
			$csv_contents = getCsvDataAsArray($filename, ",", $header_line);
			$headers = $csv_contents[0];
			$csv_contents_no_header = $csv_contents;
			array_shift($csv_contents_no_header);

			$csv_json = $csv_contents_no_header;
			$headers_json = $headers;

			$GLOBALS["json_data"]["${id}_headers_json"] = $headers_json;
			$GLOBALS["json_data"]["${id}_csv_json"] = $csv_json;

			$content = htmlentities(file_get_contents($filename));

			if($content && $header_line) {
				$content = implode(",", $header_line)."\n$content";
			}

			$results_html = "<div id='${id}_csv_table'></div>\n";
			$results_html .= copy_id_to_clipboard_string("${id}_csv_table_pre");
			$results_html .= "<pre id='${id}_csv_table_pre'>".$content."</pre>\n";
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

	function getCsvDataAsArray($filePath, $delimiter = ",", $header_line = null) {
		if (!file_exists($filePath) || !is_readable($filePath)) {
			error_log("CSV file not found or not readable: " . $filePath);
			return [];
		}

		$data = [];

		if($header_line != null) {
			$data[] = $header_line;
		}

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
				}
				unset($value);
				$data[] = $row;
			}
			fclose($handle);
		} else {
			error_log("Failed to open CSV file: " . $filePath);
		}

		return $data;
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

				echo '<a class="share_folder_buttons" href="' . htmlspecialchars($url) . '">';
				echo '<button type="button">' . htmlspecialchars($folder) . '</button>';
				echo '</a><br>';
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
		$red_cross = "<span class='invert_in_dark_mode'>&#10060;</span>";
		$green_checkmark = "<span class='invert_in_dark_mode'>&#9989;</span>";

		$output = '<section class="tabs" style="width: 100%"><menu role="tablist" aria-label="Single-Runs">';

		$i = 0;
		foreach ($log_files as $nr => $file) {
			$checkmark = file_contains_results("$run_dir/$file", $result_names) ? $green_checkmark : $red_cross;


			$runtime_string = get_runtime_from_outfile(file_get_contents("$run_dir/$file"));

			if($runtime_string == "0s" || !$runtime_string) {
				$runtime_string = "";
			} else {
				$runtime_string = " ($runtime_string) ";
			}

			$tabname = "$nr$runtime_string$checkmark";

			$output .= '<button onclick="load_log_file('.$i.', \''.$file.'\')" role="tab" '.(
				$i == 0 ? 'aria-selected="true"' : ''
			).' aria-controls="single_run_'.$i.'">'.$tabname.'</button>';
			$i++;
		}

		$output .= '</menu>';

		$i = 0;
		foreach ($log_files as $nr => $file) {
			$file_path = $run_dir . '/' . $file;
			$output .= '<article role="tabpanel" id="single_run_' . $i . '">';
			if($i != 0) {
				$output .= "<div id='spinner_log_$i' class='spinner'></div>";
			}

			$output .= copy_id_to_clipboard_string("single_run_${i}_pre");
			if ($i == 0) {
				$content = file_get_contents($file_path);
				$output .= '<pre id="single_run_'.$i.'_pre" data-loaded="true">' . highlightDebugInfo(ansi_to_html(htmlspecialchars($content))) . '</pre>';
			} else {
				$output .= '<pre id="single_run_'.$i.'_pre"></pre>';
			}
			$output .= copy_id_to_clipboard_string("single_run_${i}_pre");
			$output .= '</article>';
			$i++;
		}

		$output .= '</section>';
		return $output;
	}

	function is_valid_user_id($value) {
		if($value === null) {
			return false;
		}
		return preg_match('/^[a-zA-Z0-9_]+$/', $value);
	}

	function is_valid_experiment_name($value) {
		if($value === null) {
			return false;
		}
		return preg_match('/^[a-zA-Z0-9_]+$/', $value);
	}

	function is_valid_run_nr($value) {
		if($value === null) {
			return false;
		}
		return preg_match('/^\d+$/', $value);
	}

	function removeAnsiEscapeSequences($string) {
		$string = preg_replace('/.\[?(1A|2(5[hl]|K))/', '', $string);
		$string = preg_replace('/\[/', '', $string);
		return $string;
	}

	function highlightDebugInfo($log) {
		$log = preg_replace('/(E[0-9]{4}.*?)(?=\n|$)/', '<span style="color:red;">$1</span>', $log);

		$log = preg_replace('/(WARNING:.*?)(?=\n|$)/', '<span style="color:orange;">$1</span>', $log);

		$log = preg_replace('/(INFO.*?)(?=\n|$)/', '<span style="color:green;">$1</span>', $log);

		
		$log = preg_replace_callback('/(DEBUG INFOS START.*?DEBUG INFOS END)/s', function($matches) {
			$debugInfo = $matches[0];

			$debugInfo = preg_replace('/(Program-Code:.*?)(?=\n|$)/', '<span style="color:green;">$1</span>', $debugInfo);

			$debugInfo = preg_replace('/(File:.*?)(?=\n|$)/', '<span style="color:blue;">$1</span>', $debugInfo);

			$debugInfo = preg_replace('/(UID:.*?)(?=\n|$)/', '<span style="color:gray;">$1</span>', $debugInfo);
			$debugInfo = preg_replace('/(GID:.*?)(?=\n|$)/', '<span style="color:gray;">$1</span>', $debugInfo);

			$debugInfo = preg_replace('/(Status-Change-Time:.*?)(?=\n|$)/', '<span style="color:blue;">$1</span>', $debugInfo);
			$debugInfo = preg_replace('/(Last access:.*?)(?=\n|$)/', '<span style="color:blue;">$1</span>', $debugInfo);
			$debugInfo = preg_replace('/(Last modification:.*?)(?=\n|$)/', '<span style="color:blue;">$1</span>', $debugInfo);

			$debugInfo = preg_replace('/(Size:.*?)(?=\n|$)/', '<span style="color:purple;">$1</span>', $debugInfo);
			$debugInfo = preg_replace('/(Permissions:.*?)(?=\n|$)/', '<span style="color:purple;">$1</span>', $debugInfo);

			$debugInfo = preg_replace('/(Owner:.*?)(?=\n|$)/', '<span style="color:green;">$1</span>', $debugInfo);

			$debugInfo = preg_replace('/(Hostname:.*?)(?=\n|$)/', '<span style="color:orange;">$1</span>', $debugInfo);

			return '<div style="background-color:#f0f0f0;padding:10px;border:1px solid #ddd;">' . $debugInfo . '</div>';
		}, $log);

		$log = preg_replace('/(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})/', '<span style="color:blue;">$1</span>', $log);

		return $log;
	}

	function get_runtime_from_outfile ($string) {
		if(!$string) {
			return null;
		}

		$pattern = '/submitit INFO \((\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\)/';

		preg_match_all($pattern, $string, $matches);

		$dates = $matches[1];

		$unixTimes = [];
		foreach ($dates as $date) {
			$formattedDate = str_replace(',', '.', $date);
			$unixTimes[] = strtotime($formattedDate);
		}

		$minTime = min($unixTimes);
		$maxTime = max($unixTimes);
		$timeDiff = $maxTime - $minTime;

		$hours = floor($timeDiff / 3600);
		$minutes = floor(($timeDiff % 3600) / 60);
		$seconds = $timeDiff % 60;

		$result = [];

		if ($hours > 0) {
			$result[] = "{$hours}h";
		}
		if ($minutes > 0) {
			$result[] = "{$minutes}m";
		}
		if ($seconds > 0 || empty($result)) {
			$result[] = "{$seconds}s";
		}

		return implode(":", $result);
	}
?>
