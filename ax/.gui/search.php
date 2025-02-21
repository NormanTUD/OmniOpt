<?php
	$GLOBALS["max_results"] = 500;
	$GLOBALS["cnt"] = 0;

	function assertCondition($condition, $errorText) {
		if (!$condition) {
			throw new Exception($errorText);
		}
	}

	function parsePath($path) {
		try {
			assertCondition(strpos($path, "shares/") === 0, "Path must begin with 'shares/'");

			$trimmedPath = substr($path, strlen("shares/"));

			$pathComponents = explode("/", $trimmedPath);

			assertCondition(count($pathComponents) === 3, "Path must contain exactly 3 components after 'shares/'");

			$user = $pathComponents[0];
			$directory = $pathComponents[1];
			$file = $pathComponents[2];

			return [
				'user' => $user,
				'directory' => $directory,
				'file' => $file
			];
		} catch (Exception $e) {
			echo("Error: " . $e->getMessage());
		}
	}

	function scan_share_directories($output, $root_dir, $regex_pattern) {
		if (!is_dir($root_dir)) {
			return $output;
		}

		$user_dirs = scandir($root_dir);

		foreach ($user_dirs as $user_dir) {
			if ($user_dir === '.' || $user_dir === '..' || $GLOBALS["cnt"] > $GLOBALS["max_results"]) {
				continue;
			}

			$user_path = $root_dir . '/' . $user_dir;

			if (!is_dir($user_path)) {
				continue;
			}

			$experiment_dirs = scandir($user_path);

			foreach ($experiment_dirs as $experiment_dir) {
				if ($experiment_dir === '.' || $experiment_dir === '..' || $GLOBALS["cnt"] > $GLOBALS["max_results"]) {
					continue;
				}

				$experiment_path = $user_path . '/' . $experiment_dir;

				if (!is_dir($experiment_path)) {
					continue;
				}

				$run_path = $experiment_path . '/';

				if (!is_dir($run_path)) {
					continue;
				}

				if (preg_match($regex_pattern, $run_path)) {
					$parsedPath = parsePath($run_path);
					$url = "share.php?user_id=" . $parsedPath['user'] . "&experiment_name=" . $parsedPath['directory'] . "&run_nr=" . $parsedPath['file'];
					$entry = [
						'link' => $url,
						'content' => "OmniOpt2-Share: $run_path"
					];
					$output[] = $entry;

					$GLOBALS["cnt"]++;
				}
			}
		}

		return $output;
	}

	function read_file_content($file_path) {
		try {
			if (!file_exists($file_path)) {
				throw new Exception("File not found: $file_path");
			}
			$content = file($file_path, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES);
			if ($content === false) {
				throw new Exception("Error while reading file: $file_path");
			}
			return $content;
		} catch (Exception $e) {
			log_error($e->getMessage());
			return false;
		}
	}

	function extract_html_from_php($filename) {
		if (!file_exists($filename)) {
			return 'Error: File does not exist.';
		}

		$command = escapeshellcmd("php $filename");

		ob_start();

		try {
			passthru($command, $return_var);

			$output = ob_get_clean();

			if ($return_var !== 0) {
				throw new Exception('Fehler beim Ausführen des PHP-Skripts.');
			}

			$html_content = preg_replace("/<head>.*<\/head>/is", "", $output);

			return $html_content;
		} catch (Throwable $e) {
			ob_end_clean();
			return 'Fehler: ' . $e->getMessage();
		}
	}



	function strip_html_tags($html_content) {
		$res = strip_tags($html_content);
		return $res;
	}

	function search_text_with_context($text_lines, $regex) {
		$results = [];
		foreach ($text_lines as $line_number => $line) {
			$clean_line = strip_html_tags($line);
			if (preg_match($regex, $clean_line)) {
				$context = find_nearest_heading($text_lines, $line_number);
				$results[] = [
					'line' => trim($clean_line),
					'context' => $context
				];
			}
		}
		return $results;
	}

	function find_nearest_heading($text_lines, $current_line) {
		for ($i = $current_line; $i >= 0; $i--) {
			if (preg_match('/<(h[1-6])\s+[^>]*id=["\']([^"\']+)["\']/', $text_lines[$i], $matches)) {
				return [
					'tag' => $matches[1],
					'id' => $matches[2]
				];
			}
		}
		return null;
	}

	function log_error($message) {
		error_log($message);
		header('Content-Type: application/json');
		echo json_encode(["error" => $message]);
		exit;
	}

	$php_files = []; // Liste der zu durchsuchenden Dateien

	require "searchable_php_files.php";

	foreach ($files as $fn => $n) {
		if (is_array($n)) {
			foreach ($n["entries"] as $sub_fn => $sub_n) {
				$php_files[] = "tutorials/$sub_fn.php";
			}
		} else {
			$php_files[] = "$fn.php";
		}
	}

	if (isset($_GET['regex']) || getenv("regex")) {
		$regex = isset($_GET['regex']) ? $_GET['regex'] : getenv("regex");

		// Hinzufügen von "/" Begrenzer, wenn nicht vorhanden
		if (substr($regex, 0, 1) !== '/') {
			$regex = '/' . $regex;
		}
		if (substr($regex, -1) !== '/') {
			$regex = $regex . '/i';
		}
		if (@preg_match($regex, '') === false) {
			log_error("Ungültiger regulärer Ausdruck: $regex");
		}
	} else {
		header('Content-Type: application/json');
		print(json_encode(array("error" => "No 'regex' parameter given for search")));
		exit(0);
	}

	$output = [];

	foreach ($php_files as $file_path) {
		if ($file_path != "share.php" && $file_path != "usage_stats.php") {
			$file_content = read_file_content($file_path);
			if ($file_content !== false) {
				$html_content = extract_html_from_php($file_path);
				$text_lines = explode("\n", $html_content); // Hier HTML-Inhalt in Zeilen aufteilen

				$search_results = search_text_with_context($text_lines, $regex);
				if (!empty($search_results)) {
					foreach ($search_results as $result) {
						if ($result["line"]) {
							$entry = [
								'content' => $result['line']
							];
							if ($result['context']) {
								$tutorial_file = preg_replace("/(tutorial=)*/", "", preg_replace("/\.php$/", "", preg_replace("/tutorials\//", "tutorial=", $file_path)));
								$entry['link'] = "tutorials.php?tutorial=" . $tutorial_file . '#' . $result['context']['id'];
								$output[] = $entry;
								$GLOBALS["cnt"] += 1;
							}
						}
					}
				}
			}
		}
	}

	$output = scan_share_directories($output, "shares", $regex);

	header('Content-Type: application/json');
	echo json_encode($output);
