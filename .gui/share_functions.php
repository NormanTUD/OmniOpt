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

	function copy_id_to_clipboard_string($id, $filename) {
		$filename = basename($filename);

		$str = "<button onclick='copy_to_clipboard_from_id(\"".$id."\")'><span class='invert_in_dark_mode'>&#128203;</span> Copy raw data to clipboard</button>\n";
		$str .= "<button onclick='download_as_file(\"".$id."\", \"".htmlentities($filename)."\")'>&DoubleDownArrow; Download &raquo;".htmlentities($filename)."&laquo; as file</button>\n";

		return $str;
	}

	function add_worker_cpu_ram_from_file($tabs, $warnings, $filename, $name, $id) {
		if(is_file($filename) && filesize($filename)) {
			$worker_info = file_get_contents($filename);
			$min_max_table = extract_min_max_ram_cpu_from_worker_info($worker_info);

			if($min_max_table) {
				$warnings[] = htmlentities($filename)." does not contain valid worker info";
				return [$tabs, $warnings];
			}

			$html = $min_max_table;
			$html .= "<button onclick='plot_worker_cpu_ram()' id='plot_worker_cpu_ram_button'>Plot this data (may be slow)</button>\n";
			$html .= '<div class="invert_in_dark_mode" id="cpuRamWorkerChartContainer"></div><br>';
			$html .= copy_id_to_clipboard_string("worker_cpu_ram_pre", $filename);
			$html .= '<pre id="worker_cpu_ram_pre">'.htmlentities($worker_info).'</pre>';
			$html .= copy_id_to_clipboard_string("worker_cpu_ram_pre", $filename);

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
			$html .= copy_id_to_clipboard_string("pre_$id", $filename);
			$html .= '<pre id="pre_' . $id . '">'.htmlentities(remove_ansi_colors(file_get_contents($filename))).'</pre>';
			$html .= copy_id_to_clipboard_string("pre_$id", $filename);

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
			$html .= copy_id_to_clipboard_string("pre_$id", $filename);
			$html .= '<pre id="pre_'.$id.'">'.htmlentities(remove_ansi_colors(file_get_contents($filename))).'</pre>';
			$html .= copy_id_to_clipboard_string("pre_$id", $filename);

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

			$html = copy_id_to_clipboard_string("simple_pre_tab_$id", $filename);
			if(!$remove_ansi_colors) {
				$contents = htmlentities($contents);
			} else {
				$contents = remove_sixel($contents);
			}

			$html .= "<pre id='simple_pre_tab_$id'>$contents</pre>";

			$html .= copy_id_to_clipboard_string("simple_pre_tab_$id", $filename);

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

	function add_result_evolution_tab ($tabs, $warnings, $result_names) {
		if(count($result_names)) {
			if (isset($GLOBALS["json_data"]["tab_job_infos_headers_json"])) {
				$html = '<div class="invert_in_dark_mode" id="plotResultEvolution"></div>';

				$tabs['Evolution'] = [
					'id' => 'tab_hyperparam_evolution',
					'content' => $html,
					"onclick" => "plotResultEvolution();"
				];
			} else {
				$warnings[] = "tab_job_infos_headers_json not found in global json_data";
			}
		} else {
			$warnings[] = "Not adding evolution tab because no result names could be found";
		}

		return [$tabs, $warnings];
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

	function add_job_status_distribution($tabs) {
		$html = '<div class="invert_in_dark_mode" id="plotJobStatusDistribution"></div>';

		$tabs['Job Status Distribution'] = [
			'id' => 'tab_plot_job_status_distribution',
			'content' => $html,
			"onclick" => "plotJobStatusDistribution();"
		];

		return $tabs;
	}

	function add_results_distribution_by_generation_method ($tabs) {
		$html = '<div class="invert_in_dark_mode" id="plotResultsDistributionByGenerationMethod"></div>';

		$tabs['Results by Generation Method'] = [
			'id' => 'tab_plot_results_distribution_by_generation_method',
			'content' => $html,
			"onclick" => "plotResultsDistributionByGenerationMethod();"
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

	function add_gpu_plots ($tabs) {
		$html = '<div class="invert_in_dark_mode" id="gpu-plot"></div>';

		$tabs['GPU Usage'] = [
			'id' => 'tab_gpu_usage',
			'content' => $html,
			"onclick" => "plotGPUUsage();"
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

	function has_real_char($filename) {
		return file_exists($filename) && preg_match('/\S/', file_get_contents($filename));
	}

	function add_simple_csv_tab_from_file ($tabs, $warnings, $filename, $name, $id, $header_line = null) {
		if(is_file($filename) && filesize($filename) && has_real_char($filename)) {
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
			$results_html .= copy_id_to_clipboard_string("{$id}_csv_table_pre", $filename);
			$results_html .= "<pre id='{$id}_csv_table_pre'>".$content."</pre>\n";
			$results_html .= copy_id_to_clipboard_string("{$id}_csv_table_pre", $filename);
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

		$enclosure = "\"";
		$escape = "\\";

		if (($handle = fopen($filePath, "r")) !== false) {
			while (($row = fgetcsv($handle, 0, $delimiter, $enclosure, $escape)) !== false) {
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
		$contents = preg_replace('#\[(?:0|91)m#', '', $contents);
		return $contents;
	}

	function generateFolderButtons($folderPath, $new_param_name) {
		if (!isset($_SERVER["REQUEST_URI"])) {
			return;
		}

		$sort = isset($_GET['sort']) ? $_GET['sort'] : 'time_desc';

		echo getSortOptions();

		if (is_dir($folderPath)) {
			$dir = opendir($folderPath);
			$currentUrl = $_SERVER['REQUEST_URI'];
			$folders = [];

			while (($folder = readdir($dir)) !== false) {
				if ($folder != "." && $folder != ".." && is_dir($folderPath . '/' . $folder) && preg_match("/^[a-zA-Z0-9-_]+$/", $folder)) {
					$folders[] = $folder;
				}
			}
			closedir($dir);

			function getLatestModificationTime($folderPath) {
				$latestTime = 0;
				$dir = opendir($folderPath);

				while (($file = readdir($dir)) !== false) {
					$filePath = $folderPath . '/' . $file;
					if ($file != "." && $file != "..") {
						if (is_dir($filePath)) {
							$latestTime = max($latestTime, getLatestModificationTime($filePath));
						} else {
							$latestTime = max($latestTime, filemtime($filePath));
						}
					}
				}
				closedir($dir);
				return $latestTime;
			}

			switch ($sort) {
			case 'time_asc':
				usort($folders, function($a, $b) use ($folderPath) {
					$timeA = getLatestModificationTime($folderPath . '/' . $a);
					$timeB = getLatestModificationTime($folderPath . '/' . $b);
					return $timeA - $timeB;
				});
				break;
			case 'time_desc':
				usort($folders, function($a, $b) use ($folderPath) {
					$timeA = getLatestModificationTime($folderPath . '/' . $a);
					$timeB = getLatestModificationTime($folderPath . '/' . $b);
					return $timeB - $timeA;
				});
				break;
			case 'nr_asc':
				sort($folders);
				break;
			case 'nr_desc':
				rsort($folders);
				break;
			}

			if (count($folders)) {
				foreach ($folders as $folder) {
					$folderPathWithFile = $folderPath . '/' . $folder;
					$url = $currentUrl . (strpos($currentUrl, '?') === false ? '?' : '&') . $new_param_name . '=' . urlencode($folder);
					if ($sort != 'nr_asc') {
						$url .= '&sort=' . urlencode($sort);
					}

					$timestamp = getLatestModificationTime($folderPathWithFile);
					$lastModified = date("F d Y H:i:s", $timestamp);
					$timeSince = timeSince($timestamp);

					if(hasNonEmptyFolder($folderPathWithFile)) {
						echo '<a class="share_folder_buttons" href="' . htmlspecialchars($url) . '">';
						echo '<button type="button">' . htmlspecialchars($folder) . ' (' . $lastModified . ' | ' . $timeSince . ')</button>';
						echo '</a><br>';
					}
				}
			} else {
				print "<h2>Sorry, no jobs have been uploaded yet.</h2>";
			}
		} else {
			echo "The specified folder does not exist.";
		}
	}

	function timeSince($timestamp) {
		$diff = time() - $timestamp;

		$units = [
			31536000 => 'year',
			2592000  => 'month',
			86400    => 'day',
			3600     => 'hour',
			60       => 'minute',
			1        => 'second'
		];

		foreach ($units as $unitSeconds => $unitName) {
			if ($diff >= $unitSeconds) {
				$count = floor($diff / $unitSeconds);
				return "$count $unitName" . ($count > 1 ? 's' : '') . " ago";
			}
		}

		return "just now";
	}

	function getSortOptions() {
		$sort = isset($_GET['sort']) ? $_GET['sort'] : 'time_desc';

		$currentUrl = $_SERVER['REQUEST_URI'];
		$urlParts = parse_url($currentUrl);
		parse_str($urlParts['query'] ?? '', $queryParams);
		unset($queryParams['sort']);

		return '
			<form id="sortForm" method="get">
				<select name="sort" onchange="updateUrl()">
					<option value="time_asc"' . ($sort == 'time_asc' ? ' selected' : '') . '>Time (ascending)</option>
					<option value="time_desc"' . ($sort == 'time_desc' ? ' selected' : '') . '>Time (descending)</option>
					<option value="nr_asc"' . ($sort == 'nr_asc' ? ' selected' : '') . '>Name (ascending)</option>
					<option value="nr_desc"' . ($sort == 'nr_desc' ? ' selected' : '') . '>Name (descending)</option>
				</select>
			</form>
			<script>
				function updateUrl() {
					const currentUrl = window.location.href;
					const url = new URL(currentUrl);

					const sortValue = document.querySelector("select[name=\'sort\']").value;

					url.searchParams.set("sort", sortValue);

					window.location.href = url.toString();
				}
			</script>
		';
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
			$pattern = '/(^|\s)\s*' . preg_quote($name, '/') . ':\s*[-+]?\d+(?:\.\d+)?\s*/mi';

			if (!preg_match($pattern, $file_content)) {
				return false;
			}
		}

		return true;
	}

	function endsWithSubmititInfo($file) {
		if (file_exists($file)) {
			$string = file_get_contents($file);
			$ret = preg_match('/submitit INFO \(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}\) - Exiting after successful completion$/', $string) === 1;

			return $ret;
		}

		return true;
	}

	function generate_log_tabs($run_dir, $log_files, $result_names) {
		$red_cross = "<span class='invert_in_dark_mode'>&#10060;</span>";
		$green_checkmark = "<span class='invert_in_dark_mode'>&#9989;</span>";
		$gear = "<span class='invert_in_dark_mode'>&#9881;</span>";

		$output = '<section class="tabs" style="width: 100%"><menu role="tablist" aria-label="Single-Runs">';

		$i = 0;
		foreach ($log_files as $nr => $file) {
			$checkmark = $red_cross;
			if (file_contains_results("$run_dir/$file", $result_names)) {
				$checkmark = $green_checkmark;
			} else {
				if(endsWithSubmititInfo("$run_dir/$file")) {
					$checkmark = $red_cross;
				} else {
					$checkmark = $gear;
				}
			}

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

			$output .= copy_id_to_clipboard_string("single_run_{$i}_pre", $file_path);
			if ($i == 0) {
				$content = file_get_contents($file_path);
				$output .= '<pre id="single_run_'.$i.'_pre" data-loaded="true">' . highlightDebugInfo(ansi_to_html(htmlspecialchars($content))) . '</pre>';
			} else {
				$output .= '<pre id="single_run_'.$i.'_pre"></pre>';
			}
			$output .= copy_id_to_clipboard_string("single_run_{$i}_pre", $file_path);
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
		return !!preg_match('/^[a-zA-Z0-9_]+$/', $value);
	}

	function is_valid_experiment_name($value) {
		if($value === null) {
			return false;
		}
		return !!preg_match('/^[a-zA-Z-0-9_]+$/', $value);
	}

	function is_valid_run_nr($value) {
		if($value === null) {
			return false;
		}
		return !!preg_match('/^\d+$/', $value);
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

		if(!count($unixTimes)) {
			return "";
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
			return "";
		}

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

		$stat = stat($directory);
		if ($stat === false) {
			echo "<i>Error: Unable to retrieve information for '$directory'</i><br>\n";
			return;
		}

		$currentUser = posix_getpwuid($stat['uid'])['name'] ?? 'unknown';
		$currentGroup = posix_getgrgid($stat['gid'])['name'] ?? 'unknown';
		$currentPermissions = substr(sprintf('%o', $stat['mode']), -4);

		$issues = false;

		if ($currentUser !== $expectedUser) {
			if ($currentUser !== $alternativeUser) {
				$issues = true;
				echo "<i>Ownership issue: Current user is '$currentUser'. Expected user is '$expectedUser'</i><br>\n";
				echo "<samp>chown $expectedUser $directory</samp>\n<br>";
			}
		}

		if ($currentGroup !== $expectedGroup) {
			if ($currentGroup !== $alternativeGroup) {
				$issues = true;
				echo "<i>Ownership issue: Current group is '$currentGroup'. Expected group is '$expectedGroup'</i><br>\n";
				echo "<samp>chown :$expectedGroup $directory</samp><br>\n";
			}
		}

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
								#dier(file_get_contents($offered_files["6749a9f3-c2de-4d60-8c31-d8458877e291_log"]["file"]));
								move_uploaded_file($file, "$userFolder/$filename");
								$added_files++;
							} catch (Exception $e) {
								echo "\nAn exception occured trying to move $file to $userFolder/$filename: $e\n";
							}
						} else {
							$empty_files[] = $filename;
						}
					} else {
						dier("$filename: \$content was not ASCII, but $content_encoding");
					}
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

		if ($is_recursive_call && empty($filesAfterCheck) && filemtime($directory) < time() - 86400) {
			rmdir($directory);
			return true;
		}
		return false;
	}

	function _delete_old_shares($dir) {
		$oldDirectories = [];
		$currentTime = time();

		function is_dir_empty($dir) {
			return (is_readable($dir) && count(scandir($dir)) == 2);
		}

		foreach (glob("$dir/*/*/*", GLOB_ONLYDIR) as $subdir) {
			$pathParts = explode('/', $subdir);
			$username_dir = $pathParts[1] ?? '';

			if ($username_dir != "s4122485" && $username_dir != "pwinkler") {
				$threshold = ($username_dir === 'runner' || $username_dir === "defaultuser" || $username_dir === "admin") ? 3600 : (30 * 24 * 3600);

				if(is_dir($subdir)) {
					$dir_date = filemtime($subdir);

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

		if (empty($lines)) return '<p>Error: No valid table found.</p>';

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

	function analyze_column_types($csv_data, $column_indices) {
		$column_analysis = [];

		foreach ($column_indices as $index => $column_name) {
			$has_string = false;
			$has_numeric = false;

			foreach ($csv_data as $row) {
				if (!isset($row[$index])) {
					continue;
				}
				if (is_numeric($row[$index])) {
					$has_numeric = true;
				} else {
					$has_string = true;
				}

				if ($has_numeric && $has_string) {
					break;
				}
			}

			$column_analysis[$column_name] = [
				'numeric' => $has_numeric,
				'string' => $has_string
			];
		}

		return $column_analysis;
	}

	function count_column_types($column_analysis) {
		$nr_numerical = 0;
		$nr_string = 0;

		foreach ($column_analysis as $column => $types) {
			if (!empty($types['numeric']) && empty($types['string'])) {
				$nr_numerical++;
			} elseif (!empty($types['string'])) {
				$nr_string++;
			}
		}

		return [$nr_numerical, $nr_string];
	}

	if (!function_exists('str_starts_with')) {
		function str_starts_with(string $haystack, string $needle): bool {
			return substr($haystack, 0, strlen($needle)) === $needle;
		}
	}

	function add_pareto_from_from_file($tabs, $warnings, $run_dir) {
		$pareto_front_txt_file = "$run_dir/pareto_front_table.txt";
		$pareto_front_json_file = "$run_dir/pareto_front_data.json";

		if(file_exists($pareto_front_json_file) && file_exists($pareto_front_txt_file) && filesize($pareto_front_json_file) && filesize($pareto_front_txt_file)) {
			$pareto_front_html = "";

			$pareto_front_text = remove_ansi_colors(htmlentities(file_get_contents($pareto_front_txt_file)));

			if($pareto_front_text) {
				$pareto_front_html .= "<pre>$pareto_front_text</pre>";
			}

			$GLOBALS["json_data"]["pareto_front_data"] = json_decode(file_get_contents($pareto_front_json_file));

			if($pareto_front_html) {
				$pareto_front_html = "<div id='pareto_front_graphs_container'></div>\n$pareto_front_html";

				$tabs['Pareto-Fronts'] = [
					'id' => 'tab_pareto_fronts',
					'content' => $pareto_front_html,
					'onclick' => "load_pareto_graph();"
				];
			}
		} else {
			if(!file_exists($pareto_front_json_file)) {
				$warnings[] = "$pareto_front_json_file not found";
			} else if(!filesize($pareto_front_json_file)) {
				$warnings[] = "$pareto_front_json_file is empty";
			}

			if(!file_exists("$pareto_front_txt_file")) {
				$warnings[] = "$pareto_front_txt_file not found";
			} else if(!filesize("$pareto_front_txt_file")) {
				$warnings[] = "$pareto_front_txt_file is empty";
			}
		}

		return [$tabs, $warnings];
	}

	function get_outfiles_tab_from_run_dir ($run_dir, $tabs, $warnings, $result_names) {
		$out_files = get_log_files($run_dir);

		if(count($out_files)) {
			$tabs['Single Logs'] = [
				'id' => 'tab_logs',
				'content' => generate_log_tabs($run_dir, $out_files, $result_names)
			];
		} else {
			$warnings[] = "No out-files found";
		}

		return [$tabs, $warnings];
	}

	function get_result_names_and_min_max ($run_dir, $warnings) {
		$result_names_file = "$run_dir/result_names.txt";
		$result_min_max_file = "$run_dir/result_min_max";

		if (!file_exists($result_min_max_file)) {
			$result_min_max_file = "$run_dir/result_min_max.txt";
		}

		$result_names = [];
		$result_min_max = [];

		if(is_file($result_names_file)) {
			$result_names = read_file_as_array($result_names_file);
		} else {
			$warnings[] = "$result_names_file not found";
		}

		if(is_file($result_min_max_file)) {
			$result_min_max = read_file_as_array($result_min_max_file);
		} else {
			$warnings[] = "$result_min_max_file not found";
		}

		return [$result_names, $result_min_max, $warnings];
	}

	function add_ui_url_from_file_to_overview($run_dir, $overview_html, $warnings) {
		$ui_url_txt = "$run_dir/ui_url.txt";
		if(is_file($ui_url_txt)) {
			$firstLine = fgets(fopen($ui_url_txt, 'r'));

			if (filter_var($firstLine, FILTER_VALIDATE_URL) && (strpos($firstLine, 'http://') === 0 || strpos($firstLine, 'https://') === 0)) {
				$overview_html .= "<button onclick=\"window.open('".htmlspecialchars($firstLine)."', '_blank')\">GUI page with all the settings of this job</button><br><br>";
			}
		} else {
			$warnings[] = "$ui_url_txt not found";
		}

		return [$overview_html, $warnings];
	}

	function add_experiment_overview_to_overview ($run_dir, $overview_html, $warnings) {
		$experiment_overview = "$run_dir/experiment_overview.txt";
		if(file_exists($experiment_overview) && filesize($experiment_overview)) {
			$experiment_overview_table = asciiTableToHtml(remove_ansi_colors(htmlentities(file_get_contents($experiment_overview))));
			if($experiment_overview_table) {
				$experiment_overview .= $experiment_overview_table;

				$overview_html .= $experiment_overview_table;
			} else {
				$warnings[] = "Could not create \$experiment_overview_table";
			}
		} else {
			if(!file_exists($experiment_overview)) {
				$warnings[] = "$experiment_overview not found";
			} else if(!filesize($experiment_overview)) {
				$warnings[] = "$experiment_overview is empty";
			}
		}

		return [$overview_html, $warnings];
	}

	function add_best_results_to_overview ($run_dir, $overview_html, $warnings) {
		$best_results_txt = "$run_dir/best_result.txt";
		if(is_file($best_results_txt)) {
			$overview_html .= asciiTableToHtml(remove_ansi_colors(htmlentities(file_get_contents($best_results_txt))));
		} else {
			if(!is_file($best_results_txt)) {
				$warnings[] = "$best_results_txt not found";
			} else if (!filesize($best_results_txt)) {
				$warnings[] = "$best_results_txt is empty";
			}
		}

		return [$overview_html, $warnings];
	}

	function add_parameters_to_overview ($run_dir, $overview_html, $warnings) {
		$parameters_txt_file = "$run_dir/parameters.txt";
		if(is_file($parameters_txt_file) && filesize($parameters_txt_file)) {
			$overview_html .= asciiTableToHtml(remove_ansi_colors(htmlentities(file_get_contents("$run_dir/parameters.txt"))));
		} else {
			if(!is_file($parameters_txt_file)) {
				$warnings[] = "$run_dir/parameters.txt not found";
			} else if (!filesize($parameters_txt_file)) {
				$warnings[] = "$run_dir/parameters.txt is empty";
			}
		}

		return [$overview_html, $warnings];
	}

	function add_progressbar_to_overview($run_dir, $overview_html, $warnings) {
		$progressbar_file = "$run_dir/progressbar";
		if(file_exists($progressbar_file) && filesize($progressbar_file)) {
			$lastLine = trim(array_slice(file($progressbar_file), -1)[0]);

			$overview_html .= "<h2>Last progressbar status:</h2>\n";
			$overview_html .= "<tt>".htmlentities($lastLine)."</tt>";
		} else {
			if(!is_file($progressbar_file)) {
				$warnings[] = "$progressbar_file not found";
			} else if(!filesize($progressbar_file)) {
				$warnings[] = "$progressbar_file is empty";
			}
		}

		return [$overview_html, $warnings];
	}

	function add_result_names_table_to_overview ($result_names, $result_min_max, $overview_html, $warnings) {
		if(count($result_names)) {
			$result_names_table = '<h2>Result names and types:</h2>'."\n";
			$result_names_table .= '<br><table border="1">'."\n";
			$result_names_table .= '<tr><th style="border: 1px solid black">name</th><th style="border: 1px solid black">min/max</th></tr>'."\n";
			for ($i = 0; $i < count($result_names); $i++) {
				$min_or_max = "min";

				if(isset($result_min_max[$i])) {
					$min_or_max = $result_min_max[$i];
				}

				$result_names_table .= '<tr>'."\n";
				$result_names_table .= '<td style="border: 1px solid black">' . htmlspecialchars($result_names[$i]) . '</td>'."\n";
				$result_names_table .= '<td style="border: 1px solid black">' . htmlspecialchars($min_or_max) . '</td>'."\n";
				$result_names_table .= '</tr>'."\n";
			}
			$result_names_table .= '</table><br>'."\n";

			$overview_html .= $result_names_table;
		} else {
			$warnings[] = "No result-names could be found";
		}

		return [$overview_html, $warnings];
	}

	function add_git_version_to_overview ($run_dir, $overview_html, $warnings) {
		$git_version_file = "$run_dir/git_version";
		if(file_exists($git_version_file) && filesize($git_version_file)) {
			$lastLine = htmlentities(file_get_contents($git_version_file));

			$overview_html .= "<h2>Git-Version:</h2>\n";
			$overview_html .= "<tt>".htmlentities($lastLine)."</tt>";
		} else {
			if(!is_file($git_version_file)) {
				$warnings[] = "$git_version_file not found";
			} else if (!filesize($git_version_file)) {
				$warnings[] = "$git_version_file empty";
			}
		}

		return [$overview_html, $warnings];
	}

	function add_overview_table_to_overview_and_get_status_data ($run_dir, $status_data, $overview_html, $warnings) {
		$results_csv_file = "$run_dir/results.csv";

		if(is_file($results_csv_file) && filesize($results_csv_file)) {
			$status_data = getStatusForResultsCsv($results_csv_file);

			if($status_data["total"]) {
				$overview_table = '<h2>Number of evaluations:</h2>'."\n";
				$overview_table .= '<table border="1">'."\n";
				$overview_table .= '<tbody>'."\n";
				$overview_table .= '<tr>'."\n";

				foreach ($status_data as $key => $value) {
					$capitalizedKey = ucfirst($key);
					$overview_table .= '<th style="border: 1px solid black">' . $capitalizedKey . '</th>'."\n";
				}
				$overview_table .= '</tr>'."\n";

				$overview_table .= '<tr>'."\n";

				foreach ($status_data as $value) {
					$overview_table .= '<td style="border: 1px solid black">' . $value . '</td>'."\n";
				}
				$overview_table .= '</tr>'."\n";

				$overview_table .= '</tbody>'."\n";
				$overview_table .= '</table>'."\n";

				$overview_html .= "<br>$overview_table";
			} else {
				$warnings[] = "No evaluations detected";
			}
		} else {
			if(!is_file($results_csv_file)) {
				$warnings[] = "$results_csv_file not found";
			} else if (!filesize($results_csv_file)) {
				$warnings[] = "$results_csv_file is empty";
			}
		}

		return [$overview_html, $warnings, $status_data];
	}

	function add_overview_tab($tabs, $warnings, $run_dir, $status_data, $result_names, $result_min_max) {
		$overview_html = "";

		[$overview_html, $warnings] = add_ui_url_from_file_to_overview($run_dir, $overview_html, $warnings);
		[$overview_html, $warnings] = add_experiment_overview_to_overview($run_dir, $overview_html, $warnings);
		[$overview_html, $warnings] = add_best_results_to_overview($run_dir, $overview_html, $warnings);
		[$overview_html, $warnings] = add_parameters_to_overview($run_dir, $overview_html, $warnings);
		[$overview_html, $warnings, $status_data] = add_overview_table_to_overview_and_get_status_data($run_dir, $status_data, $overview_html, $warnings);
		[$overview_html, $warnings] = add_result_names_table_to_overview($result_names, $result_min_max, $overview_html, $warnings);
		[$overview_html, $warnings] = add_progressbar_to_overview($run_dir, $overview_html, $warnings);
		[$overview_html, $warnings] = add_git_version_to_overview($run_dir, $overview_html, $warnings);

		if($overview_html != "") {
			$tabs['Overview'] = [
				'id' => 'tab_overview',
				'content' => $overview_html
			];
		} else {
			$warnings[] = "\$overview_html was empty";
		}

		return [$tabs, $warnings, $status_data];
	}

	function find_gpu_usage_files($run_dir) {
		if (!is_dir($run_dir)) {
			error_log("Error: Directory '$run_dir' does not exist or is not accessible.");
			return [];
		}

		$pattern = $run_dir . DIRECTORY_SEPARATOR . 'gpu_usage__*.csv';
		$files = glob($pattern);

		if ($files === false) {
			return [];
		}

		return $files;
	}

	function parse_gpu_usage_files($files) {
		$gpu_usage_data = [];

		$headers = [
			"timestamp", "name", "pci.bus_id", "driver_version", "pstate", 
			"pcie.link.gen.max", "pcie.link.gen.current", "temperature.gpu", 
			"utilization.gpu", "utilization.memory", "memory.total", 
			"memory.free", "memory.used"
		];

		foreach ($files as $file) {
			$basename = basename($file);
			if (preg_match('/gpu_usage__i(\d+)\.csv/', $basename, $matches)) {
				$index = $matches[1];
				$gpu_usage_data[$index] = [];

				$handle = fopen($file, "r");
				if ($handle !== false) {
					while (($line = fgets($handle)) !== false) {
						$data = str_getcsv($line, ",", "\"", "\\");

						if (count($data) !== count($headers)) {
							error_log("Warning: Skipping malformed line in '$file'.");
							continue;
						}

						$data = array_map(function ($item) {
							$item = trim(str_replace(["MiB", "%"], "", $item));
							return trim($item);
						}, $data);

						$entry = array_combine($headers, $data);
						if ($entry === false) {
							error_log("Error: array_combine failed for '$file'.");
							continue;
						}

						$entry['timestamp_unix'] = strtotime($entry['timestamp']);

						if(count($entry)) {
							$gpu_usage_data[$index][] = $entry;
						}
					}
					fclose($handle);
				} else {
					error_log("Error: Could not open file '$file'.");
				}
			}
		}

		return $gpu_usage_data;
	}

	function hasNonEmptyFolder($dir) {
		if (!is_dir($dir)) {
			return false;
		}

		$files = new RecursiveIteratorIterator(new RecursiveDirectoryIterator($dir), RecursiveIteratorIterator::LEAVES_ONLY);

		foreach ($files as $file) {
			if (!$file->isDir()) {
				return true;
			}
		}

		return false;
	}
?>
