<?php
	$GLOBALS["max_results"] = 500;
	$GLOBALS["cnt"] = 0;
	$GLOBALS["index_tutorials"] = 1;

	require_once "searchable_php_files.php";
	require_once "_functions.php";

	function assert_condition($condition, $errorText) {
		if (!$condition) throw new Exception($errorText);
	}

	function log_error_and_exit($message) {
		error_log($message);
		header('Content-Type: application/json');
		echo json_encode(["error" => $message]);
		exit;
	}

	function parse_path($path) {
		try {
			assert_condition(strpos($path, "shares/") === 0, "Path must begin with 'shares/'");

			$parts = explode("/", substr($path, strlen("shares/")));
			assert_condition(count($parts) === 3, "Path must contain exactly 3 components after 'shares/'");

			return [
				'user' => $parts[0],
				'directory' => $parts[1],
				'file' => $parts[2]
			];
		} catch (Exception $e) {
			log_error_and_exit("parse_path error: " . $e->getMessage());
		}
	}

	function read_file_content($file_path) {
		if (!file_exists($file_path)) log_error_and_exit("File not found: $file_path");

		$content = file($file_path, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES);
		if ($content === false) log_error_and_exit("Error reading file: $file_path");

		return $content;
	}

	function extract_html_from_php($filename) {
		if (!file_exists($filename)) return 'Error: File does not exist.';

		ob_start();
		try {
			include $filename; 
			$output = ob_get_clean();
			return preg_replace("/<head>.*<\/head>/is", "", $output);
		} catch (Throwable $e) {
			ob_end_clean();
			return "Error: " . $e->getMessage();
		}
	}


	function strip_html_tags_safe($html) {
		return strip_tags($html);
	}

	function find_nearest_heading($lines, $index) {
		for ($i = $index; $i >= 0; $i--) {
			if (preg_match('/<(h[1-6])\s+[^>]*id=["\']([^"\']+)["\']/', $lines[$i], $matches)) {
				return ['tag' => $matches[1], 'id' => $matches[2]];
			}
		}
		return null;
	}

	function search_text_with_context($lines, $regex) {
		$results = [];
		foreach ($lines as $line_number => $line) {
			$stripped = html_entity_decode(strip_html_tags_safe($line), ENT_QUOTES | ENT_HTML5, 'UTF-8');
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

	function create_share_url($parsed) {
		$url = "share?user_id={$parsed['user']}&experiment_name={$parsed['directory']}";
		if (!empty($parsed['file'])) $url .= "&run_nr={$parsed['file']}";
		return $url;
	}

	function generate_breadcrumb_buttons($path) {
		$html_parts = [];
		$segments = explode('/', trim($path, "/"));

		if (!empty($segments) && $segments[0] === 'shares') array_shift($segments);

		$html_parts[] = '<div class="search_share_path" class="invert_in_dark_mode title-bar-text">';
		$param_parts = [];
		$href = 'share?';

		if (isset($segments[0])) {
			$user_id = htmlspecialchars($segments[0]);
			$param_parts[] = "user_id=" . rawurlencode($user_id);
			$html_parts[] = '<button onclick="window.location.href=\''.$href.'&'.implode('&', $param_parts).'\'">'.$user_id.'</button>';
		}

		if (isset($segments[1])) {
			$experiment = htmlspecialchars($segments[1]);
			$param_parts[] = "experiment_name=" . rawurlencode($experiment);
			$html_parts[] = ' / <button onclick="window.location.href=\''.$href.'&'.implode('&',$param_parts).'\'">'.$experiment.'</button>';
		}

		if (isset($segments[2])) {
			$run = htmlspecialchars($segments[2]);
			$param_parts[] = "run_nr=" . rawurlencode($run);
			$html_parts[] = ' / <button onclick="window.location.href=\''.$href.'&'.implode('&',$param_parts).'\'">'.$run.'</button>';
		}

		$html_parts[] = '</div>';
		return implode('', $html_parts);
	}

	function scan_share_directories(&$categorized, $root_dir, $regex_pattern) {
		if (!is_dir($root_dir)) return;

		foreach (scandir($root_dir) as $user_dir) {
			if (in_array($user_dir, ['.','..'])) continue;

			$user_path = "$root_dir/$user_dir";
			if (!is_dir($user_path)) continue;

			foreach (scandir($user_path) as $exp_dir) {
				if (in_array($exp_dir, ['.','..'])) continue;
				if ($GLOBALS["cnt"] >= $GLOBALS["max_results"]) return;

				$run_path = "$user_path/$exp_dir/";
				if (is_dir($run_path) && (preg_match($regex_pattern, $exp_dir) || preg_match($regex_pattern, $user_dir))) {
					$parsed = parse_path($run_path);
					$categorized["Shares"][] = [
						'link' => create_share_url($parsed),
						'content' => generate_breadcrumb_buttons($run_path)
					];
					$GLOBALS["cnt"]++;
				}
			}
		}
	}

	function validate_regex($regex) {
		if (substr($regex, 0, 1) !== '/') $regex = '/' . $regex;
		if (substr($regex, -1) !== '/') $regex .= '/i';
		if (@preg_match($regex, '') === false) log_error_and_exit("Invalid regex pattern: $regex");
		return $regex;
	}

	function build_php_file_list($files) {
		$list = [];
		foreach ($files as $fn => $n) {
			if (is_array($n)) {
				foreach ($n["entries"] as $sub_fn => $sub_n) {
					$base = "_tutorials/$sub_fn";
					if (file_exists("$base.php")) $list[] = "$base.php";
					elseif (file_exists("$base.md")) $list[] = "$base.md";
				}
			} else {
				$list[] = "$fn.php";
			}
		}
		return $list;
	}

	function process_php_files($php_files, $regex, &$categorized) {
		foreach ($php_files as $file_path) {
			if (in_array(basename($file_path), ["share.php", "usage_stats.php", "gui.php"])) continue;

			$content = read_file_content($file_path);
			$html_content = preg_match("/\.md$/", $file_path)
				? convert_markdown_to_html(implode("\n", $content))
				: extract_html_from_php($file_path);

			$results = search_text_with_context(explode("\n", $html_content), $regex);
			foreach ($results as $result) {
				if (!empty($result['line'])) {
					$tutorial = preg_replace("/(_tutorial=)*/","",
						preg_replace("/\.(md|php)$/","",
						preg_replace("/tutorials\//","tutorial=", $file_path)));

					$headline = get_first_heading_content($file_path);
					$entry = [
						'content' => $result['line'],
						'headline' => $headline,
						'link' => "tutorials?tutorial=$tutorial"
					];

					if (!preg_match("/conceptdrift/", $tutorial)) {
						if (!empty($result['context']['id']))
							$entry['link'] .= '#' . $result['context']['id'];
						$categorized["Tutorials"][] = $entry;
					}

					$GLOBALS["cnt"]++;
					if ($GLOBALS["cnt"] >= $GLOBALS["max_results"]) return;
				}
			}
		}
	}

	function remove_empty_subarrays(array $array): array {
		$out = [];
		foreach ($array as $k => $v) {
			if (is_array($v) && count($v) > 0) $out[$k] = $v;
		}
		return $out;
	}

	function fetch_search_pattern() {
		$regex_raw = $_GET['regex'] ?? getenv("regex");
		if (!$regex_raw) log_error_and_exit("No 'regex' parameter given for search");
		return validate_regex($regex_raw);
	}

	function init_categories() {
		return ["Tutorials" => [], "Shares" => []];
	}

	function execute_search($regex, &$categorized, $files) {
		$php_files = build_php_file_list($files);
		process_php_files($php_files, $regex, $categorized);
		scan_share_directories($categorized, "shares", $regex);
		$categorized = remove_empty_subarrays($categorized);
		return $categorized;
	}

	function output_json($data) {
		header('Content-Type: application/json');
		echo json_encode($data);
	}

	function main() {
		global $files;
		$regex = fetch_search_pattern();
		$categorized = init_categories();
		$result = execute_search($regex, $categorized, $files);
		output_json($result);
	}

	main();
?>
