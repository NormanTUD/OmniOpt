<?php
	include_once("share_functions.php");

	try {
		$run_nr = validate_param("run_nr", "/^\d+$/", "Invalid run_nr");
		$user_id = validate_param("user_id", "/^[a-zA-Z0-9_]+$/", "Invalid user_id");
		$experiment_name = validate_param("experiment_name", "/^[a-zA-Z0-9_-]+$/", "Invalid experiment_name");
		$run_folder_without_shares = build_run_folder_path($user_id, $experiment_name, $run_nr);
		$run_folder = $GLOBALS["sharesPath"]."/$run_folder_without_shares";

		validate_directory($run_folder);

		$path = "$run_folder/log";;

		if(file_exists($path)) {
			$out_file = get_debug_log_html_table($path);
			respond_with_json($out_file);
		} else {
			respond_with_error("Invalid path $path found");
		}
	} catch (Exception $e) {
		respond_with_error($e->getMessage());
	}

	function get_debug_log_html_table ($filename) {
		$fileContent = file_get_contents($filename);

		if ($fileContent === false) {
			$output = "Error loading the file!";
			exit;
		}

		$lines = explode("\n", $fileContent);

		$output = "<table border='1'>";
		$output .= "<thead><tr><th>Time</th><th>Function Stack</th><th>Message</th></tr></thead>";
		$output .= "<tbody>";

		foreach ($lines as $line) {
			if (trim($line) === "") {
				continue;
			}

			$jsonData = json_decode($line, true);

			if ($jsonData === null) {
				$output .= "<tr><td colspan='3'>Error parsing the JSON data: $line</td></tr>";
				continue;
			}

			$time = isset($jsonData['time']) ? $jsonData['time'] : 'Not available';
			$msg = isset($jsonData['msg']) ? $jsonData['msg'] : 'Not available';

			$functionStack = '';
			if (isset($jsonData['function_stack'])) {
				foreach ($jsonData['function_stack'] as $functionData) {
					$function = isset($functionData['function']) ? $functionData['function'] : 'Unknown';
					if($function != "_get_debug_json") {
						$lineNumber = isset($functionData['line_number']) ? $functionData['line_number'] : 'Unknown';
						$functionStack .= "<tt>$function</tt> (Line $lineNumber)<br>";
					}
				}
			}

			$output .= "<tr>\n";
			$output .= "<td style='border: 1px solid black;'>$time</td>\n";
			$output .= "<td style='border: 1px solid black;'>$functionStack</td>\n";
			$output .= "<td style='border: 1px solid black;'><pre class='invert_in_dark_mode debug_log_pre'>$msg</pre></td>\n";
			$output .= "</tr>\n";
		}

		$output .= "</tbody></table>";

		return $output;
	}
?>
