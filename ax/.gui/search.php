<?php
	function assertCondition($condition, $errorText) {
		if (!$condition) {
			throw new Exception($errorText);
		}
	}

	function parsePath($path) {
		try {
			// Prüfen, ob der Pfad mit "shares/" beginnt
			assertCondition(strpos($path, "shares/") === 0, "Der Pfad muss mit 'shares/' beginnen");

			// Entfernen des "shares/"-Teils
			$trimmedPath = substr($path, strlen("shares/"));

			// Aufteilen des Pfades in seine Bestandteile
			$pathComponents = explode("/", $trimmedPath);

			// Prüfen, ob die Anzahl der Komponenten korrekt ist
			assertCondition(count($pathComponents) === 3, "Der Pfad muss genau drei Komponenten nach 'shares/' enthalten");

			// Extrahieren der einzelnen Komponenten
			$user = $pathComponents[0];
			$directory = $pathComponents[1];
			$file = $pathComponents[2];

			// Ausgabe der extrahierten Komponenten

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
		// Check if the root directory exists
		if (!is_dir($root_dir)) {
			throw new Exception("The root directory does not exist: " . $root_dir);
		}

		// Get the list of directories in the root directory
		$user_dirs = scandir($root_dir);

		foreach ($user_dirs as $user_dir) {
			if ($user_dir === '.' || $user_dir === '..') {
				continue;
			}

			$user_path = $root_dir . '/' . $user_dir;

			if (!is_dir($user_path)) {
				continue;
			}

			// Get the list of experiments for the user
			$experiment_dirs = scandir($user_path);

			foreach ($experiment_dirs as $experiment_dir) {
				if ($experiment_dir === '.' || $experiment_dir === '..') {
					continue;
				}

				$experiment_path = $user_path . '/' . $experiment_dir;

				if (!is_dir($experiment_path)) {
					continue;
				}

				// Get the list of run numbers for the experiment
				$run_dirs = scandir($experiment_path);

				foreach ($run_dirs as $run_dir) {
					if ($run_dir === '.' || $run_dir === '..') {
						continue;
					}

					$run_path = $experiment_path . '/' . $run_dir;

					if (!is_dir($run_path)) {
						continue;
					}

					// Check if the run directory name matches the regex pattern
					if (preg_match($regex_pattern, $run_path, $matches)) {
						$parsedPath = parsePath($run_path);
						$url = "share.php?user=".$parsedPath['user']."&experiment=".$parsedPath['directory']."&run_nr=".$parsedPath['file'];
						$entry = [
							'link' => $url,
							'content' => "OmniOpt-Share: $run_path"
						];
						$output[] = $entry;
					}
				}
			}
		}

		return $output;
	}

	// Funktion zum Lesen des Inhalts einer Datei
	function read_file_content($file_path) {
		try {
			if (!file_exists($file_path)) {
				throw new Exception("Datei nicht gefunden: $file_path");
			}
			$content = file($file_path, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES);
			if ($content === false) {
				throw new Exception("Fehler beim Lesen der Datei: $file_path");
			}
			return $content;
		} catch (Exception $e) {
			log_error($e->getMessage());
			return false;
		}
	}

	// Funktion zum Extrahieren von HTML-Code aus PHP-Datei
	function extract_html_from_php($file_content) {
		ob_start();
		eval('?>' . implode("\n", $file_content));
		$html_content = ob_get_clean();
		$html_content = preg_replace("/<head>.*<\/head>/is", "", $html_content);
		return $html_content;
	}

	// Funktion zum Entfernen von HTML-Tags
	function strip_html_tags($html_content) {
		$res = strip_tags($html_content);
		return $res;
	}

	// Funktion zum Durchsuchen des Textes und Finden der Positionen
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

	// Funktion zum Finden der nächsten vor der Zeile liegenden <h1>, <h2>, ... mit ID
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

	// Funktion zum Loggen von Fehlern
	function log_error($message) {
		error_log($message);
		header('Content-Type: application/json');
		echo json_encode(["error" => $message]);
		exit;
	}

	// Hauptprogramm
	$php_files = []; // Liste der zu durchsuchenden Dateien

	include("searchable_php_files.php");

	foreach ($files as $fn => $n) {
		if (is_array($n)) {
			foreach ($n["entries"] as $sub_fn => $sub_n) {
				$php_files[] = "tutorials/$sub_fn.php";
			}
		} else {
			$php_files[] = "$fn.php";
		}
	}

	// Überprüfen und Validieren des regulären Ausdrucks
	if (isset($_GET['regex'])) {
		$regex = $_GET['regex'];
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
		if($file_path != "share.php" && $file_path != "usage_stats.php") {
			$file_content = read_file_content($file_path);
			if ($file_content !== false) {
				$html_content = extract_html_from_php($file_content);
				$text_lines = explode("\n", $html_content); // Hier HTML-Inhalt in Zeilen aufteilen

				$search_results = search_text_with_context($text_lines, $regex);
				if (!empty($search_results)) {
					foreach ($search_results as $result) {
						if($result["line"]) {
							$entry = [
								'content' => $result['line']
							];
							if ($result['context']) {
								$tutorial_file = preg_replace("/(tutorial=)*/", "", preg_replace("/\.php$/", "", preg_replace("/tutorials\//", "tutorial=", $file_path)));
								$entry['link'] = "tutorials.php?tutorial=" . $tutorial_file . '#' . $result['context']['id'];
								$output[] = $entry;
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
?>
