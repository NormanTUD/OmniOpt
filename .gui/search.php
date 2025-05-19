<?php
	$GLOBALS["max_results"] = 500;
	$GLOBALS["cnt"] = 0;

	require_once "searchable_php_files.php";

	function assertCondition($condition, $errorText) {
		if (!$condition) {
			throw new Exception($errorText);
		}
	}

	function log_error($message) {
		error_log($message);
		header('Content-Type: application/json');
		echo json_encode(["error" => $message]);
		exit;
	}

	function parsePath($path) {
		try {
			assertCondition(strpos($path, "shares/") === 0, "Path must begin with 'shares/'");

			$parts = explode("/", substr($path, strlen("shares/")));

			assertCondition(count($parts) === 3, "Path must contain exactly 3 components after 'shares/'");

			return [
				'user' => $parts[0],
				'directory' => $parts[1],
				'file' => $parts[2]
			];
		} catch (Exception $e) {
			echo("Error: " . $e->getMessage());
		}
		return null;
	}

	function read_file_content($file_path) {
		try {
			if (!file_exists($file_path)) {
				throw new Exception("File not found: $file_path");
			}

			$content = file($file_path, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES);
			if ($content === false) {
				throw new Exception("Error reading file: $file_path");
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
				throw new Exception('Error executing PHP script.');
			}

			return preg_replace("/<head>.*<\/head>/is", "", $output);
		} catch (Throwable $e) {
			ob_end_clean();
			return 'Error: ' . $e->getMessage();
		}
	}

	function strip_html_tags($html) {
		return strip_tags($html);
	}

	function find_nearest_heading($lines, $index) {
		for ($i = $index; $i >= 0; $i--) {
			if (preg_match('/<(h[1-6])\s+[^>]*id=["\']([^"\']+)["\']/', $lines[$i], $matches)) {
				return [
					'tag' => $matches[1],
					'id' => $matches[2]
				];
			}
		}
		return null;
	}

	function search_text_with_context($lines, $regex) {
		$results = [];

		foreach ($lines as $line_number => $line) {
			$stripped = strip_html_tags($line);

			if (preg_match($regex, $stripped)) {
				$context = find_nearest_heading($lines, $line_number);
				$results[] = [
					'line' => trim($stripped),
					'context' => $context
				];
			}
		}

		return $results;
	}

	function scan_share_directories($output, $root_dir, $regex_pattern) {
		if (!is_dir($root_dir)) {
			return $output;
		}

		$user_dirs = scandir($root_dir);

		foreach ($user_dirs as $user_dir) {
			if ($user_dir === '.' || $user_dir === '..') {
				continue;
			}

			$user_path = "$root_dir/$user_dir";
			if (!is_dir($user_path)) {
				continue;
			}

			$experiment_dirs = scandir($user_path);
			foreach ($experiment_dirs as $experiment_dir) {
				if ($experiment_dir === '.' || $experiment_dir === '..') {
					continue;
				}

				if ($GLOBALS["cnt"] >= $GLOBALS["max_results"]) {
					return $output;
				}

				$run_path = "$user_path/$experiment_dir/";

				if (is_dir($run_path) && preg_match($regex_pattern, $run_path)) {
					$parsed = parsePath($run_path);
					if ($parsed === null) {
						continue;
					}

					$url = "share?user_id={$parsed['user']}&experiment_name={$parsed['directory']}";
					if (!empty($parsed['file'])) {
						$url .= "&run_nr={$parsed['file']}";
					}

					$output[] = [
						'link' => $url,
						'content' => "OmniOpt2-Share: $run_path"
					];

					$GLOBALS["cnt"]++;
				}
			}
		}

		return $output;
	}

	function validate_regex($regex) {
		if (substr($regex, 0, 1) !== '/') {
			$regex = '/' . $regex;
		}
		if (substr($regex, -1) !== '/') {
			$regex .= '/i';
		}
		if (@preg_match($regex, '') === false) {
			log_error("Invalid regex pattern: $regex");
		}

		return $regex;
	}

	if (!isset($_GET['regex']) && !getenv("regex")) {
		header('Content-Type: application/json');
		echo json_encode(["error" => "No 'regex' parameter given for search"]);
		exit;
	}

	$regex_raw = isset($_GET['regex']) ? $_GET['regex'] : getenv("regex");
	$regex = validate_regex($regex_raw);

	$php_files = [];

	foreach ($files as $fn => $n) {
		if (is_array($n)) {
			foreach ($n["entries"] as $sub_fn => $sub_n) {
				$file_base = "_tutorials/$sub_fn";
				if (file_exists("$file_base.php")) {
					$php_files[] = "$file_base.php";
				} else if (file_exists("$file_base.md")) {
					$php_files[] = "$file_base.md";
				}
			}
		} else {
			$php_files[] = "$fn.php";
		}
	}

	$output = [];

	foreach ($php_files as $file_path) {
		if (in_array(basename($file_path), ["share.php", "usage_stats.php"])) {
			continue;
		}

		$file_content = read_file_content($file_path);
		if ($file_content === false) {
			continue;
		}

		$html_content = preg_match("/\.md$/", $file_path)
			? convertMarkdownToHtml(file_get_contents($file_path))
			: extract_html_from_php($file_path);

		$lines = explode("\n", $html_content);
		$results = search_text_with_context($lines, $regex);

		foreach ($results as $result) {
			if (!empty($result['line'])) {
				$tutorial = preg_replace("/(_tutorial=)*/", "", preg_replace("/\.(md|php)$/", "", preg_replace("/tutorials\//", "tutorial=", $file_path)));

				$entry = [
					'content' => $result['line'],
					'link' => "tutorials?tutorial=$tutorial"
				];

				if (!empty($result['context']['id'])) {
					$entry['link'] .= '#' . $result['context']['id'];
				}

				$output[] = $entry;
				$GLOBALS["cnt"]++;

				if ($GLOBALS["cnt"] >= $GLOBALS["max_results"]) {
					break 2;
				}
			}
		}
	}

	$output = scan_share_directories($output, "shares", $regex);

	header('Content-Type: application/json');
	echo json_encode($output);
?>
