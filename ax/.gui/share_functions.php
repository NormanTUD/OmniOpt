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
		return "<button onclick='copy_to_clipboard_from_id(\"".$id."\")'>Copy raw data to clipboard</button>\n";
	}

	function add_worker_cpu_ram_from_file($tabs, $warnings, $filename, $name, $id) {
		if(is_file($filename) && filesize($filename)) {
			$worker_info = file_get_contents($filename);
			$min_max_table = extract_min_max_ram_cpu_from_worker_info($worker_info);

			$html = $min_max_table;
			$html .= "<button onclick='plot_worker_cpu_ram()' id='plot_worker_cpu_ram_button'>Plot this data (may be slow)</button>\n";
			$html .= '<div class="invert_in_dark_mode" id="cpuRamWorkerChartContainer"></div><br>';
			$html .= copy_id_to_clipboard_string("worker_cpu_ram_pre");
			$html .= '<pre id="worker_cpu_ram_pre">'.htmlentities($worker_info).'</pre>';
			$html .= copy_id_to_clipboard_string("worker_cpu_ram_pre");

			$tabs[$name] = [
				'id' => $id,
				'content' => $html
			];
		} else {
			if(!is_file($filename)) {
				$warnings[] = "$filename does not exist";
			} else if(!filesize($filename)) {
				$warnings[] = "$filename is empty";
			}
		}

		return [$tabs, $warnings];
	}

	function add_debug_log_from_file($tabs, $warnings, $filename, $name, $id) {
		if(is_file($filename) && filesize($filename)) {
			$output = "<div id='debug_log_spinner' class='spinner'></div>";
			$output .= "<div id='here_debuglogs_go'></div>";

			$tabs[$name] = [
				'id' => $id,
				'content' => $output,
				'onclick' => "load_debug_log()"
			];
		} else {
			if(!is_file($filename)) {
				$warnings[] = "$filename does not exist";
			} else if(!filesize($filename)) {
				$warnings[] = "$filename is empty";
			}
		}

		return [$tabs, $warnings];
	}

	function add_cpu_ram_usage_main_worker_from_file($tabs, $warnings, $filename, $name, $id) {
		if(is_file($filename) && filesize($filename)) {
			$html = "<div class='invert_in_dark_mode' id='mainWorkerCPURAM'></div>";
			$html .= copy_id_to_clipboard_string("pre_$id");
			$html .= '<pre id="pre_' . $id . '">'.htmlentities(remove_ansi_colors(file_get_contents($filename))).'</pre>';
			$html .= copy_id_to_clipboard_string("pre_$id");

			$csv_contents = getCsvDataAsArray($filename);
			$headers = $csv_contents[0];
			$csv_contents_no_header = $csv_contents;
			array_shift($csv_contents_no_header);

			$csv_json = $csv_contents_no_header;
			$headers_json = $headers;

			$GLOBALS["json_data"]["{$id}_csv_json"] = $csv_json;
			$GLOBALS["json_data"]["{$id}_headers_json"] = $headers_json;

			$tabs[$name] = [
				'id' => $id,
				'content' => $html,
				"onclick" => "plotCPUAndRAMUsage();"
			];
		} else {
			if(!is_file($filename)) {
				$warnings[] = "$filename does not exist";
			} else if(!filesize($filename)) {
				$warnings[] = "$filename is empty";
			}
		}

		return [$tabs, $warnings];
	}

	function add_scatter_3d_plots($tabs, $filename, $name, $id) {
		if(is_file($filename)) {
			$html = "<div class='invert_in_dark_mode' id='plotScatter3d'></div>";

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
			$html = "<div class='invert_in_dark_mode' id='plotScatter2d'></div>";

			$tabs[$name] = [
				'id' => $id,
				'content' => $html,
				"onclick" => "plotScatter2d();"
			];
		}

		return $tabs;
	}

	function add_worker_usage_plot_from_file($tabs, $warnings, $filename, $name, $id) {
		if(is_file($filename) && filesize($filename)) {
			$html = "<div class='invert_in_dark_mode' id='workerUsagePlot'></div>";
			$html .= copy_id_to_clipboard_string("pre_$id");
			$html .= '<pre id="pre_'.$id.'">'.htmlentities(remove_ansi_colors(file_get_contents($filename))).'</pre>';
			$html .= copy_id_to_clipboard_string("pre_$id");

			$csv_contents = getCsvDataAsArray($filename);

			$GLOBALS["json_data"]["{$id}_csv_json"] = $csv_contents;

			$tabs[$name] = [
				'id' => $id,
				'content' => $html,
				"onclick" => "plotWorkerUsage();"
			];
		} else {
			if(!is_file($filename)) {
				$warnings[] = "$filename does not exist";
			} else if(!filesize($filename)) {
				$warnings[] = "$filename is empty";
			}
		}

		return [$tabs, $warnings];
	}

	function add_simple_table_from_ascii_table_file($tabs, $warnings, $filename, $name, $id, $remove_ansi_colors = false) {
		if(is_file($filename) && filesize($filename) > 0) {
			$contents = file_get_contents($filename);
			if(!$remove_ansi_colors) {
				$contents = remove_ansi_colors($contents);
			} else {
				$contents = removeAnsiEscapeSequences(ansi_to_html(htmlspecialchars($contents)));
			}

			if(!$remove_ansi_colors) {
				$contents = htmlentities($contents);
			} else {
				$contents = remove_sixel($contents);
			}

			$html_table = asciiTableToHtml($contents);
			$html = $html_table;


			$tabs[$name] = [
				'id' => $id,
				'content' => $html
			];
		} else {
			if(!is_file($filename)) {
				$warnings[] = "$filename does not exist";
			} else if(!filesize($filename)) {
				$warnings[] = "$filename is empty";
			}
		}

		return [$tabs, $warnings];
	}

	function add_simple_pre_tab_from_file ($tabs, $warnings, $filename, $name, $id, $remove_ansi_colors = false) {
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
		} else {
			if(!is_file($filename)) {
				$warnings[] = "$filename does not exist";
			} else if(!filesize($filename)) {
				$warnings[] = "$filename is empty";
			}
		}

		return [$tabs, $warnings];
	}

	function add_exit_codes_pie_plot($tabs) {
		$html = '<div class="invert_in_dark_mode" id="plotExitCodesPieChart"></div>';

		$tabs['Exit-Codes'] = [
			'id' => 'tab_exit_codes_plot',
			'content' => $html,
			"onclick" => "plotExitCodesPieChart();"
		];

		return $tabs;
	}


	function add_violin_plot ($tabs) {
		$html = '<div class="invert_in_dark_mode" id="plotViolin"></div>';

		$tabs['Violin'] = [
			'id' => 'tab_violin',
			'content' => $html,
			"onclick" => "plotViolin();"
		];

		return $tabs;
	}

	function add_result_evolution_tab ($tabs) {
		$html = '<div class="invert_in_dark_mode" id="plotResultEvolution"></div>';

		$tabs['Evolution'] = [
			'id' => 'tab_hyperparam_evolution',
			'content' => $html,
			"onclick" => "plotResultEvolution();"
		];

		return $tabs;
	}

	function add_plot_result_pairs ($tabs) {
		$html = '<div class="invert_in_dark_mode" id="plotResultPairs"></div>';

		$tabs['Result-Pairs'] = [
			'id' => 'tab_result_pairs',
			'content' => $html,
			"onclick" => "plotResultPairs();"
		];

		return $tabs;
	}

	function add_histogram_plot ($tabs) {
		$html = '<div class="invert_in_dark_mode" id="plotHistogram"></div>';

		$tabs['Histogram'] = [
			'id' => 'tab_histogram',
			'content' => $html,
			"onclick" => "plotHistogram();"
		];

		return $tabs;
	}

	function add_heatmap_plot_tab ($tabs) {
		$explanation = "
    <h1>Correlation Heatmap Explanation</h1>

    <p>
        This is a heatmap that visualizes the correlation between numerical columns in a dataset. The values represented in the heatmap show the strength and direction of relationships between different variables.
    </p>

    <h2>How It Works</h2>
    <p>
        The heatmap uses a matrix to represent correlations between each pair of numerical columns. The calculation behind this is based on the concept of \"correlation,\" which measures how strongly two variables are related. A correlation can be positive, negative, or zero:
    </p>
    <ul>
        <li><strong>Positive correlation</strong>: Both variables increase or decrease together (e.g., if the temperature rises, ice cream sales increase).</li>
        <li><strong>Negative correlation</strong>: As one variable increases, the other decreases (e.g., as the price of a product rises, the demand for it decreases).</li>
        <li><strong>Zero correlation</strong>: There is no relationship between the two variables (e.g., height and shoe size might show zero correlation in some contexts).</li>
    </ul>

    <h2>Color Scale: Yellow to Purple (Viridis)</h2>
    <p>
        The heatmap uses a color scale called \"Viridis,\" which ranges from yellow to purple. Here's what the colors represent:
    </p>
    <ul>
        <li><strong>Yellow (brightest)</strong>: A strong positive correlation (close to +1). This indicates that as one variable increases, the other increases in a very predictable manner.</li>
        <li><strong>Green</strong>: A moderate positive correlation. Variables are still positively related, but the relationship is not as strong.</li>
        <li><strong>Blue</strong>: A weak or near-zero correlation. There is a small or no discernible relationship between the variables.</li>
        <li><strong>Purple (darkest)</strong>: A strong negative correlation (close to -1). This indicates that as one variable increases, the other decreases in a very predictable manner.</li>
    </ul>

    <h2>What the Heatmap Shows</h2>
    <p>
        In the heatmap, each cell represents the correlation between two numerical columns. The color of the cell is determined by the correlation coefficient: from yellow for strong positive correlations, through green and blue for weaker correlations, to purple for strong negative correlations.
    </p>
";
		$html = '<div class="invert_in_dark_mode" id="plotHeatmap"></div><br>'.$explanation;

		$tabs['Heatmap'] = [
			'id' => 'tab_heatmap',
			'content' => $html,
			"onclick" => "plotHeatmap();"
		];

		return $tabs;
	}

	function add_box_plot_tab ($tabs) {
		$html = '<div class="invert_in_dark_mode" id="plotBoxplot"></div>';

		$tabs['Boxplots'] = [
			'id' => 'tab_boxplots',
			'content' => $html,
			"onclick" => "plotBoxplot();"
		];

		return $tabs;
	}

	function add_parallel_plot_tab ($tabs) {
		$html = '<div class="invert_in_dark_mode" id="parallel-plot"></div>';

		$tabs['Parallel Plot'] = [
			'id' => 'tab_parallel',
			'content' => $html,
			"onclick" => "createParallelPlot(tab_results_csv_json, tab_results_headers_json, result_names, special_col_names);"
		];

		return $tabs;
	}

	function add_simple_csv_tab_from_file ($tabs, $warnings, $filename, $name, $id, $header_line = null) {
		if(is_file($filename) && filesize($filename)) {
			$csv_contents = getCsvDataAsArray($filename, ",", $header_line);
			$headers = $csv_contents[0];
			$csv_contents_no_header = $csv_contents;
			array_shift($csv_contents_no_header);

			$csv_json = $csv_contents_no_header;
			$headers_json = $headers;

			$GLOBALS["json_data"]["{$id}_headers_json"] = $headers_json;
			$GLOBALS["json_data"]["{$id}_csv_json"] = $csv_json;

			$content = htmlentities(file_get_contents($filename));

			if($content && $header_line) {
				$content = implode(",", $header_line)."\n$content";
			}

			$results_html = "<div id='{$id}_csv_table'></div>\n";
			$results_html .= copy_id_to_clipboard_string("{$id}_csv_table_pre");
			$results_html .= "<pre id='{$id}_csv_table_pre'>".$content."</pre>\n";
			$results_html .= copy_id_to_clipboard_string("{$id}_csv_table_pre");
			$results_html .= "<script>\n\tcreateTable({$id}_csv_json, {$id}_headers_json, '{$id}_csv_table')</script>\n";

			$tabs[$name] = [
				'id' => $id,
				'content' => $results_html,
			];
		} else {
			if(!is_file($filename)) {
				$warnings[] = "$filename does not exist";
			} else if(!filesize($filename)) {
				$warnings[] = "$filename is empty";
			}
		}

		return [$tabs, $warnings];
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

			if(count($folders)) {
				foreach ($folders as $folder) {
					$url = $currentUrl . (strpos($currentUrl, '?') === false ? '?' : '&') . $new_param_name . '=' . urlencode($folder);

					echo '<a class="share_folder_buttons" href="' . htmlspecialchars($url) . '">';
					echo '<button type="button">' . htmlspecialchars($folder) . '</button>';
					echo '</a><br>';
				}
			} else {
				print "<h2>Sorry, no jobs have been uploaded yet.</h2>";
			}
		} else {
			echo "The specified folder does not exist.";
		}
	}

	function is_valid_user_or_experiment_name ($name) {
		if(preg_match("/^[a-zA-Z0-9_-]+$/", $name)) {
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

			$output .= copy_id_to_clipboard_string("single_run_{$i}_pre");
			if ($i == 0) {
				$content = file_get_contents($file_path);
				$output .= '<pre id="single_run_'.$i.'_pre" data-loaded="true">' . highlightDebugInfo(ansi_to_html(htmlspecialchars($content))) . '</pre>';
			} else {
				$output .= '<pre id="single_run_'.$i.'_pre"></pre>';
			}
			$output .= copy_id_to_clipboard_string("single_run_{$i}_pre");
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
		return preg_match('/^[a-zA-Z-0-9_]+$/', $value);
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

	function extract_min_max_ram_cpu_from_worker_info($data) {
		preg_match_all('/CPU: ([\d\.]+)%, RAM: ([\d\.]+) MB/', $data, $matches);

		$cpu_values = $matches[1];
		$ram_values = $matches[2];

		if (empty($cpu_values) || empty($ram_values)) {
			echo "";
			exit;
		}

		// Hilfsfunktionen für Durchschnitt und Median
		function calculate_average($values) {
			return array_sum($values) / count($values);
		}

		function calculate_median($values) {
			sort($values);
			$count = count($values);
			$middle = floor($count / 2);
			if ($count % 2) {
				return $values[$middle];
			} else {
				return ($values[$middle - 1] + $values[$middle]) / 2;
			}
		}

		// Werte berechnen
		$min_cpu = min($cpu_values);
		$max_cpu = max($cpu_values);
		$avg_cpu = calculate_average($cpu_values);
		$median_cpu = calculate_median($cpu_values);

		$min_ram = min($ram_values);
		$max_ram = max($ram_values);
		$avg_ram = calculate_average($ram_values);
		$median_ram = calculate_median($ram_values);

		$html = '<table border="1">';
		$html .= '<tr><th>Min RAM (MB)</th><th>Max RAM (MB)</th><th>Avg RAM (MB)</th><th>Median RAM (MB)</th>';
		$html .= '<th>Min CPU (%)</th><th>Max CPU (%)</th><th>Avg CPU (%)</th><th>Median CPU (%)</th></tr>';
		$html .= '<tr>';
		$html .= '<td>' . htmlspecialchars($min_ram) . '</td>';
		$html .= '<td>' . htmlspecialchars($max_ram) . '</td>';
		$html .= '<td>' . htmlspecialchars(round($avg_ram, 2)) . '</td>';
		$html .= '<td>' . htmlspecialchars($median_ram) . '</td>';
		$html .= '<td>' . htmlspecialchars($min_cpu) . '</td>';
		$html .= '<td>' . htmlspecialchars($max_cpu) . '</td>';
		$html .= '<td>' . htmlspecialchars(round($avg_cpu, 2)) . '</td>';
		$html .= '<td>' . htmlspecialchars($median_cpu) . '</td>';
		$html .= '</tr>';
		$html .= '</table>';

		return $html;
	}

	function checkFolderPermissions($directory, $expectedUser, $expectedGroup, $alternativeUser, $alternativeGroup, $expectedPermissions) {
		if (getenv('CI') !== false) {
			return false;
		}

		if (!is_dir($directory)) {
			echo "<i>Error: '$directory' is not a valid directory</i>\n";
			return true;
		}

		// Get stat information
		$stat = stat($directory);
		if ($stat === false) {
			echo "<i>Error: Unable to retrieve information for '$directory'</i><br>\n";
			return;
		}

		// Get current ownership and permissions
		$currentUser = posix_getpwuid($stat['uid'])['name'] ?? 'unknown';
		$currentGroup = posix_getgrgid($stat['gid'])['name'] ?? 'unknown';
		$currentPermissions = substr(sprintf('%o', $stat['mode']), -4);

		$issues = false;

		// Check user
		if ($currentUser !== $expectedUser) {
			if ($currentUser !== $alternativeUser) {
				$issues = true;
				echo "<i>Ownership issue: Current user is '$currentUser'. Expected user is '$expectedUser'</i><br>\n";
				echo "<samp>chown $expectedUser $directory</samp>\n<br>";
			}
		}

		// Check group
		if ($currentGroup !== $expectedGroup) {
			if ($currentGroup !== $alternativeGroup) {
				$issues = true;
				echo "<i>Ownership issue: Current group is '$currentGroup'. Expected group is '$expectedGroup'</i><br>\n";
				echo "<samp>chown :$expectedGroup $directory</samp><br>\n";
			}
		}

		// Check permissions
		if (intval($currentPermissions, 8) !== $expectedPermissions) {
			$issues = true;
			echo "<i>Permissions issue: Current permissions are '$currentPermissions'. Expected permissions are '" . sprintf('%o', $expectedPermissions) . "'</i><br>\n";
			echo "<samp>chmod " . sprintf('%o', $expectedPermissions) . " $directory\n</samp><br>";
		}

		return $issues;
	}




	function warn($message) {
		echo "Warning: " . $message . "\n";
	}

	function findMatchingUUIDRunFolder(string $targetUUID, $sharesPath): ?string {
		$glob_str = "$sharesPath/*/*/*/run_uuid";
		$files = glob($glob_str);

		foreach ($files as $file) {
			$fileContent = preg_replace('/\s+/', '', file_get_contents($file));

			if ($fileContent === $targetUUID) {
				return dirname($file);
			}
		}

		return null;
	}

	function deleteFolder($folder) {
		$files = array_diff(scandir($folder), array('.', '..'));

		foreach ($files as $file) {
			(is_dir("$folder/$file")) ? deleteFolder("$folder/$file") : unlink("$folder/$file");
		}

		return rmdir($folder);
	}

	function createNewFolder($path, $user_id, $experiment_name) {
		$i = 0;

		$newFolder = $path . "/$user_id/$experiment_name/$i";

		do {
			$newFolder = $path . "/$user_id/$experiment_name/$i";
			$i++;
		} while (file_exists($newFolder));

		try {
			mkdir($newFolder, 0777, true);
		} catch (Exception $e) {
			print("Error trying to create directory $newFolder. Error:\n\n$e\n\n");
			exit(1);
		}
		return $newFolder;
	}

	function searchForHashFile($directory, $new_upload_md5, $userFolder) {
		$files = glob($directory);

		foreach ($files as $file) {
			try {
				$file_content = file_get_contents($file);

				if ($file_content === $new_upload_md5) {
					return [true, dirname($file)];
				}
			} catch (AssertionError $e) {
				print($e->getMessage());
			}
		}

		try {
			$destinationPath = "$userFolder/hash.md5";
			assert(is_writable(dirname($destinationPath)), "Directory is not writable: " . dirname($destinationPath));

			$write_success = file_put_contents($destinationPath, $new_upload_md5);
			assert($write_success !== false, "Failed to write to file: $destinationPath");
		} catch (\Throwable $e) {
			print($e->getMessage());
		}

		return [false, null];
	}

	function extractPathComponents($found_hash_file_dir, $sharesPath) {
		$pattern = "#^$sharesPath/([^/]+)/([^/]+)/(\d+)$#";

		if (preg_match($pattern, $found_hash_file_dir, $matches)) {
			assert(isset($matches[1]), "Failed to extract user from path: $found_hash_file_dir");
			assert(isset($matches[2]), "Failed to extract experiment name from path: $found_hash_file_dir");
			assert(isset($matches[3]), "Failed to extract run ID from path: $found_hash_file_dir");

			$user = $matches[1];
			$experiment_name = $matches[2];
			$run_dir = $matches[3];

			return [$user, $experiment_name, $run_dir];
		} else {
			warn("The provided path does not match the expected pattern: $found_hash_file_dir");
			return [null, null, null];
		}
	}

	function get_user_folder($sharesPath, $_uuid_folder, $user_id, $experiment_name, $run_nr="") {
		$probe_dir = "$sharesPath/$user_id/$experiment_name/$run_nr";

		if($run_nr != "" && $run_nr >= 0 && is_dir($probe_dir)) {
			return $probe_dir;
		}

		if(getenv("disable_folder_creation")) {
			return;
		}

		if (!$_uuid_folder) {
			$userFolder = createNewFolder($sharesPath, $user_id, $experiment_name);
		} else {
			$userFolder = $_uuid_folder;
		}

		return $userFolder;
	}

	function is_valid_zip_file($path) {
		if (!file_exists($path) || !is_readable($path)) {
			return false;
		}

		$handle = fopen($path, 'rb');
		if (!$handle) {
			return false;
		}

		$signature = fread($handle, 4);
		fclose($handle);

		// ZIP-Files begin with "PK\x03\x04"
		return $signature === "PK\x03\x04";
	}

	function move_files($offered_files, $added_files, $userFolder, $msgUpdate, $msg) {
		$empty_files = [];

		foreach ($offered_files as $offered_file) {
			$file = $offered_file["file"];
			$filename = $offered_file["filename"];

			if ($file) {
				if(file_exists($file)) {
					$content = file_get_contents($file);
					$content_encoding = mb_detect_encoding($content);
					if ($content_encoding == "ASCII" || $content_encoding == "UTF-8" || is_valid_zip_file($file)) {
						if (filesize($file)) {
							try {
								move_uploaded_file($file, "$userFolder/$filename");
								$added_files++;
							} catch (Exception $e) {
								echo "An exception occured trying to move $file to $userFolder/$filename\n";
							}
						} else {
							$empty_files[] = $filename;
						}
					} else {
						dier("$filename: \$content was not ASCII, but $content_encoding");
					}
				} else {
					print("$file does not exist");
				}
			}
		}

		if ($added_files) {
			if (isset($_GET["update"])) {
				eval('echo "$msgUpdate";');
			} else {
				eval('echo "$msg";');
			}
			exit(0);
		} else {
			if (count($empty_files)) {
				$empty_files_string = implode(", ", $empty_files);
				echo "Error sharing the job. The following files were empty: $empty_files_string. \n";
			} else {
				echo "Error sharing the job. No Files were found. \n";
			}
			exit(1);
		}
	}

	function remove_extra_slashes_from_url($string) {
		$pattern = '/(?<!:)(\/{2,})/';

		$cleaned_string = preg_replace($pattern, '/', $string);

		return $cleaned_string;
	}

	function move_files_if_not_already_there($new_upload_md5_string, $update_uuid, $BASEURL, $user_id, $experiment_name, $run_id, $offered_files, $userFolder, $uuid_folder, $sharesPath) {
		$added_files = 0;
		$project_md5 = hash('md5', $new_upload_md5_string);

		$found_hash_file_data = searchForHashFile("$sharesPath/*/*/*/hash.md5", $project_md5, $userFolder);

		$found_hash_file = $found_hash_file_data[0];
		$found_hash_file_dir = $found_hash_file_data[1];

		if ($found_hash_file && is_null($update_uuid)) {
			list($user, $experiment_name, $run_id) = extractPathComponents($found_hash_file_dir, $sharesPath);
			$old_url = remove_extra_slashes_from_url("$BASEURL/share?user_id=$user_id&experiment_name=$experiment_name&run_nr=$run_id");
			echo "This project already seems to have been uploaded. See $old_url\n";
			exit(0);
		} else {
			if (!$uuid_folder || !is_dir($uuid_folder)) {
				$url = remove_extra_slashes_from_url("$BASEURL/share?user_id=$user_id&experiment_name=$experiment_name&run_nr=$run_id");

				move_files(
					$offered_files,
					$added_files,
					$userFolder,
					"See $url for live-results.\n",
					"Run was successfully shared. See $url\nYou can share the link. It is valid for 30 days.\n"
				);
			} else {
				$url = remove_extra_slashes_from_url("$BASEURL/share?user_id=$user_id&experiment_name=$experiment_name&run_nr=$run_id");

				move_files(
					$offered_files,
					$added_files,
					$uuid_folder,
					"See $url for live-results.\n",
					"See $url for live-results.\n"
				);
			}
		}
	}

	function get_offered_files($acceptable_files, $acceptable_file_names, $i) {
		foreach ($acceptable_files as $acceptable_file) {
			$offered_files[$acceptable_file] = array(
				"file" => $_FILES[$acceptable_file]['tmp_name'] ?? null,
				"filename" => $acceptable_file_names[$i]
			);
			$i++;
		}

		return [$offered_files, $i];
	}

	function rrmdir($dir) {
		if (is_dir($dir)) {
			$objects = scandir($dir);

			foreach ($objects as $object) {
				if ($object != '.' && $object != '..') {
					$object_path = $dir.'/'.$object;
					if (filetype($object_path) == 'dir') {
						rrmdir($object_path);
					} else {
						if (file_exists($object_path)) {
							unlink($object_path);
						}
					}
				}
			}

			reset($objects);

			if(is_dir($dir)) {
				rmdir($dir);
			}
		}
	}

	function deleteEmptyDirectories(string $directory, bool $is_recursive_call): bool {                                                                                                                                            
		if (!is_dir($directory)) {       
			return false;                              
		}                                                       

		$files = array_diff(scandir($directory), ['.', '..']);

		foreach ($files as $file) {                            
			$path = $directory . DIRECTORY_SEPARATOR . $file;
			if (is_dir($path)) {             
				deleteEmptyDirectories($path, true);
			}                                
		}                                           

		$filesAfterCheck = array_diff(scandir($directory), ['.', '..']);

		// Überprüfung, ob das Verzeichnis leer ist und älter als ein Tag
		if ($is_recursive_call && empty($filesAfterCheck) && filemtime($directory) < time() - 86400) {
			rmdir($directory);                             
			return true;                 
		}                                                                                                                               
		return false;                                                                                                     
	}

	function _delete_old_shares($dir) {
		$oldDirectories = [];
		$currentTime = time();

		// Helper function to check if a directory is empty
		function is_dir_empty($dir) {
			return (is_readable($dir) && count(scandir($dir)) == 2); // Only '.' and '..' are present in an empty directory
		}

		foreach (glob("$dir/*/*/*", GLOB_ONLYDIR) as $subdir) {
			$pathParts = explode('/', $subdir);
			$secondDir = $pathParts[1] ?? '';

			// Skip Elias's project directory
			if ($secondDir != "s4122485") {
				$threshold = ($secondDir === 'runner') ? 86400 : (30 * 24 * 3600);

				if(is_dir($subdir)) {
					$dir_date = filemtime($subdir);

					// Check if the directory is older than the threshold and is either empty or meets the original condition
					if (is_dir($subdir) && ($dir_date < ($currentTime - $threshold))) {
						$oldDirectories[] = $subdir;
						rrmdir($subdir);
					}

					if (is_dir($subdir) && is_dir_empty($subdir)) {
						$oldDirectories[] = $subdir;
						rrmdir($subdir);
					}
				}
			}
		}

		return $oldDirectories;
	}

	function delete_old_shares () {
		$directoryToCheck = 'shares';
		deleteEmptyDirectories($directoryToCheck, false);
		$oldDirs = _delete_old_shares($directoryToCheck);
		deleteEmptyDirectories($directoryToCheck, false);

		return $oldDirs;
	}

	function asciiTableToHtml($asciiTable) {
		$lines = explode("\n", trim($asciiTable));

		while (!empty($lines) && trim($lines[0]) === '') {
			array_shift($lines);
		}

		$headerText = null;
		if (!empty($lines) && !preg_match('/^[\s]*[┏━┡┩└─]+/u', $lines[0])) {
			$headerText = array_shift($lines);
		}

		$lines = array_filter($lines, function ($line) {
			return !preg_match('/^[\s]*[┏━┡┩└─]+/u', $line);
		});

		if (empty($lines)) return '<p>Fehler: Keine gültige Tabelle erkannt.</p>';

		$headerLine = array_shift($lines);
		$headerCells = preg_split('/\s*[┃│]\s*/u', trim($headerLine, "┃│"));

		$html = $headerText ? "<h2>$headerText</h2>" : '';
		$html .= '<table border="1" cellspacing="0" cellpadding="5"><thead><tr>';
		foreach ($headerCells as $cell) {
			$html .= '<th>' . $cell . '</th>';
		}
		$html .= '</tr></thead><tbody>';

		foreach ($lines as $line) {
			$cells = preg_split('/\s*[┃│]\s*/u', trim($line, "┃│"));
			if (count($cells) === count($headerCells)) {
				$html .= '<tr>';
				foreach ($cells as $cell) {
					$html .= '<td>' . $cell . '</td>';
				}
				$html .= '</tr>';
			}
		}

		$html .= '</tbody></table>';
		return $html;
	}
?>
