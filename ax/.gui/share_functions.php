<?php
	function countNonMatchingHeaders($csvFilePath) {
		// Die Header, die wir ignorieren möchten
		$ignoredHeaders = [
			'trial_index', 'arm_name', 'trial_status',
			'generation_method', 'result'
		];

		// Prüfen, ob die Datei existiert und lesbar ist
		if (!file_exists($csvFilePath) || !is_readable($csvFilePath)) {
			trigger_error('Die Datei existiert nicht oder ist nicht lesbar.', E_USER_WARNING);
			return false;
		}

		// Datei öffnen
		if (($handle = fopen($csvFilePath, 'r')) !== false) {
			// Erste Zeile (Header) einlesen
			$headers = fgetcsv($handle);

			// Datei schließen
			fclose($handle);

			// Falls die Datei leer ist oder keine Header hat
			if ($headers === false) {
				trigger_error('Die CSV-Datei enthält keine Header-Zeile.', E_USER_WARNING);
				return false;
			}

			// Anzahl der Header, die nicht in der Liste der ignorierten Header enthalten sind
			$nonMatchingHeaderCount = 0;

			foreach ($headers as $header) {
				if (!in_array(trim($header), $ignoredHeaders)) {
					$nonMatchingHeaderCount++;
				}
			}

			return $nonMatchingHeaderCount;
		} else {
			// Falls die Datei nicht geöffnet werden konnte
			trigger_error('Fehler beim Öffnen der Datei.', E_USER_WARNING);
			return false;
		}
	}

	function copy_button($name_to_search_for) {
		return "<button class='copy_to_clipboard_button invert_in_dark_mode' onclick='find_closest_element_behind_and_copy_content_to_clipboard(this, \"$name_to_search_for\")'>📋 Copy raw data to clipboard</button>";
	}

	function removeHTags($string) {
		// Regex to match all h1 to h6 tags
		$pattern = '/<\/?h[1-6][^>]*>/i';

		// Replace the tags with an empty string
		$cleanedString = preg_replace($pattern, '', $string);

		// Check for errors during the replacement process
		if ($cleanedString === null) {
			// In case of an error, log the issue and return the original string
			error_log("Error removing H-tags from the string.");
			return $string;
		}

		return $cleanedString;
	}

	function calculateDirectoryHash($directory) {
		// Überprüfen, ob der Ordner existiert und lesbar ist
		if (!is_dir($directory) || !is_readable($directory)) {
			return false; // Fehler, Ordner existiert nicht oder ist nicht lesbar
		}

		// Rekursive Funktion zum Abrufen aller Dateien im Ordner und Unterordnern
		function getFilesRecursive($dir)
		{
			$files = [];
			$dirIterator = new RecursiveDirectoryIterator($dir, RecursiveDirectoryIterator::SKIP_DOTS);
			$iterator = new RecursiveIteratorIterator($dirIterator, RecursiveIteratorIterator::SELF_FIRST);

			foreach ($iterator as $file) {
				if ($file->isFile()) {
					$files[] = $file->getPathname();
				}
			}

			// Alphabetisch sortieren
			sort($files);
			return $files;
		}

		// Alle Dateien im Ordner und Unterordner holen
		$files = getFilesRecursive($directory);

		// Falls keine Dateien gefunden wurden
		if (empty($files)) {
			return false; // Keine Dateien im Ordner
		}

		$combinedHashes = '';

		// Für jede Datei den SHA256-Hash berechnen und an die Hash-Liste anhängen
		foreach ($files as $file) {
			$fileContent = file_get_contents($file);
			if ($fileContent === false) {
				return false; // Fehler beim Lesen der Datei
			}

			$combinedHashes .= hash('sha256', $fileContent);
		}

		// Endgültigen SHA256-Hash des kombinierten Hash-Strings berechnen
		return hash('sha256', $combinedHashes);
	}

	function die_with_time() {
		$time_end = microtime(true);
		dier("Runtime: " . abs($time_end - $GLOBALS["time_start"]));
	}

	function get_header_file($file) {
		$replaced_file = preg_replace("/.*\//", "", $file);

		$names = array(
			"best_result.txt" => "Best results",
			"results.csv" => "Results",
			"job_infos.csv" => "Job-Infos",
			"parameters.txt" => "Parameter",
			"get_next_trials.csv" => "Next trial got/requested",
			"cpu_ram_usage.csv" => "CPU/RAM-usage",
			"evaluation_errors.log" => "Evaluation Errors",
			"oo_errors.txt" => "OmniOpt2-Errors",
			"worker_usage.csv" => "Number of workers (time, wanted, got, percentage)"
		);

		if (isset($names[$replaced_file])) {
			return "<h2>" . $names[$replaced_file] . " (<samp>$replaced_file</samp>)</h2>";
		} else {
			return "<h2>" . $replaced_file . "</h2>";
		}
	}

	function loadCsvToJsonByResult($file) {
		assert(file_exists($file), "CSV file does not exist.");

		$csvData = [];
		try {
			$fileHandle = fopen($file, "r");
			assert($fileHandle !== false, "Failed to open the file.");

			$headers = fgetcsv($fileHandle);
			assert($headers !== false, "Failed to read the headers.");

			if (!$headers) {
				return json_encode($csvData);
			}

			$result_column_id = array_search("result", $headers);

			while (($row = fgetcsv($fileHandle)) !== false) {
				if ($row[$result_column_id]) {
					while (count($row) < count($headers)) {
						$row[] = "None";
					}
					$csvData[] = array_combine($headers, $row);
				}
			}

			fclose($fileHandle);
		} catch (Exception $e) {
			print("Error reading CSV: " . $e->getMessage());
			warn("Ensure the CSV file is correctly formatted.");
			throw $e;
		}

		$jsonData = json_encode($csvData);
		assert($jsonData !== false, "Failed to encode JSON.");

		return $jsonData;
	}

	function loadCsvToJson($file) {
		assert(file_exists($file), "CSV file does not exist.");

		$csvData = [];
		try {
			$fileHandle = fopen($file, "r");
			assert($fileHandle !== false, "Failed to open the file.");

			while (($row = fgetcsv($fileHandle)) !== false) {
				$csvData[] = $row;
			}

			fclose($fileHandle);
		} catch (Exception $e) {
			print("Error reading CSV: " . $e->getMessage());
			warn("Ensure the CSV file is correctly formatted.");
			throw $e;
		}

		$jsonData = json_encode($csvData);
		assert($jsonData !== false, "Failed to encode JSON.");

		return $jsonData;
	}

	function warn($message) {
		echo "Warning: " . $message . "\n";
	}

	function findMatchingUUIDRunFolder(string $targetUUID, $sharesPath): ?string {
		// Glob-Muster, um alle passenden Dateien zu finden

		$glob_str = "$sharesPath/*/*/*/run_uuid";
		$files = glob($glob_str);
		// dier($glob_str);
		// dier($files);

		foreach ($files as $file) {
			// Dateiinhalt lesen und Whitespace (Leerzeichen, Newlines, Tabs) entfernen
			$fileContent = preg_replace('/\s+/', '', file_get_contents($file));

			// Überprüfen, ob die UUID übereinstimmt
			if ($fileContent === $targetUUID) {
				// Ordnerpfad ohne 'state_files/run_uuid' zurückgeben
				return dirname($file);  // Zwei Ebenen zurück gehen
			}
		}

		// Wenn keine Übereinstimmung gefunden wurde, null zurückgeben
		return null;
	}

	function checkPermissions($path, $user_id) {
		// Überprüfen, ob der Ordner existiert und dem aktuellen Benutzer gehört
		if (!file_exists($path) || !is_dir($path)) {
			print("Ordner existiert nicht oder ist kein Verzeichnis.");
			exit(1);
		}

		$currentUserId = getCurrentUserId(); // Funktion zur Ermittlung der Benutzer-ID
		$currentUserGroup = getCurrentUserGroup(); // Funktion zur Ermittlung der Gruppenzugehörigkeit

		// Annahme: $currentUserId und $currentUserGroup sind die aktuellen Werte des Benutzers
		// Annahme: Die Berechtigungen werden entsprechend geprüft, ob der Benutzer Schreibrechte hat

		if (!hasWritePermission($path, $currentUserId, $currentUserGroup)) {
			dier("Benutzer hat keine Schreibrechte für diesen Ordner.");
		}
	}

	function deleteOldFolders($path) {
		$threshold = strtotime('-30 days');

		$folders = glob($path . '/*', GLOB_ONLYDIR);

		foreach ($folders as $folder) {
			if (filemtime($folder) < $threshold) {
				// Ordner und alle Inhalte rekursiv löschen
				deleteFolder($folder);
			}
		}
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
			mkdir($newFolder, 0777, true); // Rechte 0777 für volle Zugriffsberechtigungen setzen
		} catch (Exception $e) {
			print("Error trying to create directory $newFolder");
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
		} catch (AssertionError $e) {
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

	function remove_ansi_colors($contents) {
		$contents = preg_replace('#\\x1b[[][^A-Za-z]*[A-Za-z]#', '', $contents);
		return $contents;
	}

	function print_url($content) {
		$content = htmlentities($content);
		if (preg_match("/^https?:\/\//", $content)) {
			echo "<a target='_blank' href='$content'>Link to the GUI, preloaded with all options specified here.</a>";

			return 1;
		}

		return 0;
	}

	function removeMatchingLines(array $lines, string $pattern): array {
		// Überprüfen, ob das Pattern ein gültiges Regex ist
		if (@preg_match($pattern, null) === false) {
			throw new InvalidArgumentException("Ungültiges Regex-Muster: $pattern");
		}

		$filteredLines = [];

		foreach ($lines as $line) {
			// Wenn die Zeile nicht mit dem Regex übereinstimmt, fügen wir sie zum Ergebnis hinzu
			if (!preg_match($pattern, $line)) {
				$filteredLines[] = $line;
			}
		}

		return $filteredLines;
	}

	function convertStringToHtmlTable($inputString) {
		// Convert the input string into an array of lines
		$lines = explode("\n", trim($inputString));
		array_shift($lines); // Remove headline line above the table
		$lines = removeMatchingLines($lines, "/[┡┏└][━─]+[┓┩┘]/");

		// Initialize an empty array to hold table rows
		$tableData = [];

		// Loop through each line and extract data
		foreach ($lines as $line) {
			// Trim whitespace and split the line by the box-drawing characters
			$columns = array_map('trim', preg_split('/[│┃]+/', $line));

			// Filter out empty columns
			$columns = array_filter($columns, fn($column) => $column !== '');

			// If the line contains valid data, add it to the table data array
			if (!empty($columns)) {
				$tableData[] = $columns;
			}
		}

		// dier($tableData);

		$skip_next_row = false;

		$newTableData = [];

		foreach ($tableData as $rowIndex => $row) {
			$thisRow = $tableData[$rowIndex];
			if ($rowIndex > 0) {
				if (!$skip_next_row && isset($tableData[$rowIndex + 1])) {
					$nextRow = $tableData[$rowIndex + 1];
					if (count($thisRow) > count($nextRow)) {
						$next_row_keys = array_keys($nextRow);

						foreach ($next_row_keys as $nrk) {
							$thisRow[$nrk] .= " " . $nextRow[$nrk];
						}

						$skip_next_row = true;

						$newTableData[] = $thisRow;
					} else {
						$newTableData[] = $thisRow;
					}
				} else {
					$skip_next_row = true;
				}
			} else {
				$newTableData[] = $thisRow;
			}
		}

		// dier($newTableData);

		// Start building the HTML table
		$html = '<table border="1">';

		// Loop through the table data and generate HTML rows
		foreach ($newTableData as $rowIndex => $row) {
			$html .= '<tr>';

			// Use th for the header row and td for the rest
			$tag = $rowIndex === 0 ? 'th' : 'td';

			// Loop through the row columns and generate HTML cells
			foreach ($row as $column) {
				$html .= "<$tag>" . htmlentities($column) . "</$tag>";
			}

			$html .= '</tr>';
		}

		$html .= '</table>';

		return $html;
	}

	function get_results_cpu_ram_usage ($file, $tab_headers, $html_parts) {
		$jsonData = loadCsvToJson($file);

		if ($jsonData != "[]") {
			$_part = "<div id='cpuRamChart'></div>";
			$_part .= "<script>var cpu_ram_usage_json = convertToIntAndFilter($jsonData.map(Object.values)); replaceZeroWithNull(cpu_ram_usage_json); plot_cpu_gpu_graph(cpu_ram_usage_json);</script>";

			$header = get_header_file($file);
			$_hash = hash('md5', "$header - $file");

			$html_parts[$_hash] = $_part;

			$tab_headers[] = array("id" => $_hash, "header" => $header);
		}

		return [$tab_headers, $html_parts];
	}

	function get_results_csv_code ($file, $tab_headers, $html_parts) {
		$content = remove_ansi_colors(file_get_contents($file));
		$content_encoding = mb_detect_encoding($content);
		if ($content_encoding == "ASCII" || $content_encoding == "UTF-8") {
			$resultsCsvJson = loadCsvToJsonByResult($file);

			$header = get_header_file($file);

			$_hash = hash('md5', "$header - $file");

			$this_html = "";
			if ($resultsCsvJson != "[]") {
				$this_html .= "<pre class='stdout_file invert_in_dark_mode autotable'>" . htmlentities($content) . "</pre>";
				$this_html .= copy_button("stdout_file");
				$this_html .= "<script>\n";
				$this_html .= "    var job_infos_csv = $resultsCsvJson;\n";
				$this_html .= "    var results_csv_bare = `".htmlentities($content)."`;\n";
				$this_html .= "</script>";

				$this_html .= "<script>var results_csv_json = $resultsCsvJson; plot_all_possible(results_csv_json);</script>";

				$html_parts[$_hash] = $this_html;

				$tab_headers[] = array("id" => $_hash, "header" => $header);

				$nr_real_headers = countNonMatchingHeaders($file);

				if($nr_real_headers > 0) {
					$tab_headers[] = array("id" => "parallel_plot_container", "header" => "Parallel-Plot");
				}

				if($nr_real_headers >= 2) {
					$tab_headers[] = array("id" => "scatter_plot_2d_container", "header" => "2d-Scatter-Plots");
				}

				if($nr_real_headers >= 3) {
					$tab_headers[] = array("id" => "scatter_plot_3d_container", "header" => "3d-Scatter-Plots");
				}
			}
		}

		return [$tab_headers, $html_parts];
	}

	function get_results_worker_usage ($file, $tab_headers, $html_parts) {
		$jsonData = loadCsvToJson($file);
		$content = remove_ansi_colors(file_get_contents($file));

		$header = get_header_file($file);

		$_hash = hash('md5', "$header - $file");

		$tab_headers[] = array("id" => $_hash, "header" => $header);


		if ($jsonData == "[]") {
			return [$tab_headers, $html_parts];
		}

		$this_html = "<pre class='stdout_file invert_in_dark_mode autotable'>" . htmlentities($content) . "</pre>\n";
		$this_html .= copy_button("stdout_file");
		$this_html .= "<div id='worker_usage_plot'></div>\n";
		$this_html .= "<script>var worker_usage_csv = convertToIntAndFilter($jsonData.map(Object.values)); plot_planned_vs_real_worker_over_time(worker_usage_csv);</script>";

		$html_parts[$_hash] = $this_html;

		return [$tab_headers, $html_parts];
	}

	function get_results_best_results ($file, $tab_headers, $html_parts) {
		$content = remove_ansi_colors(file_get_contents($file));
		$content_encoding = mb_detect_encoding($content);
		if (!($content_encoding == "ASCII" || $content_encoding == "UTF-8")) {
			return [$tab_headers, $html_parts];
		}

		$header = get_header_file($file);

		$_hash = hash('md5', "$header - $file");

		$tab_headers[] = array("id" => $_hash, "header" => $header);

		$this_html = "<pre>" . htmlentities($content) . "</pre>";

		$html_parts[$_hash] = $this_html;

		return [$tab_headers, $html_parts];
	}

	function get_results_parameters ($file, $tab_headers, $html_parts) {
		$content = remove_ansi_colors(file_get_contents($file));
		$content_encoding = mb_detect_encoding($content);
		if (!($content_encoding == "ASCII" || $content_encoding == "UTF-8")) {
			return [$tab_headers, $html_parts];
		}

		$header = get_header_file($file);

		$_hash = hash('md5', "$header - $file");

		$tab_headers[] = array("id" => $_hash, "header" => $header);

		$this_html = "<pre>" . htmlentities($content) . "</pre>";

		$html_parts[$_hash] = $this_html;

		return [$tab_headers, $html_parts];
	}

	function get_results_evaluation_errors_and_oo_errors ($file, $tab_headers, $html_parts) {
		$content = remove_ansi_colors(file_get_contents($file));
		$content_encoding = mb_detect_encoding($content);
		if (!($content_encoding == "ASCII" || $content_encoding == "UTF-8")) {
			return [$tab_headers, $html_parts];
		}

		$header = get_header_file($file);

		$_hash = hash('md5', "$header - $file");

		$tab_headers[] = array("id" => $_hash, "header" => $header);

		$this_html = "";

		$this_html .= "<pre>" . htmlentities($content) . "</pre>";

		$html_parts[$_hash] = $this_html;

		return [$tab_headers, $html_parts];
	}

	function get_results_get_next_trial ($file, $tab_headers, $html_parts) {
		$content = remove_ansi_colors(file_get_contents($file));
		$content_encoding = mb_detect_encoding($content);
		if (!($content_encoding == "ASCII" || $content_encoding == "UTF-8")) {
			return [$tab_headers, $html_parts];
		}

		$header = get_header_file($file);

		$_hash = hash('md5', "$header - $file");

		$tab_headers[] = array("id" => $_hash, "header" => $header);

		$this_html = "<pre class='stdout_file invert_in_dark_mode autotable' data-header_columns='datetime,got,requested'>" . htmlentities($content) . "</pre>";
		$this_html .= copy_button("stdout_file");

		$html_parts[$_hash] = $this_html;

		return [$tab_headers, $html_parts];
	}

	function get_results_job_infos ($file, $tab_headers, $html_parts) {
		$content = remove_ansi_colors(file_get_contents($file));
		$content_encoding = mb_detect_encoding($content);
		if (!($content_encoding == "ASCII" || $content_encoding == "UTF-8")) {
			return [$tab_headers, $html_parts];
		}

		$jobInfosCsvJson = loadCsvToJsonByResult($file);

		$header = get_header_file($file);

		$_hash = hash('md5', "$header - $file");

		if ($jobInfosCsvJson == "[]") {
			return [$tab_headers, $html_parts];
		} else {
			$tab_headers[] = array("id" => $_hash, "header" => $header);

			$this_html = "<pre class='stdout_file invert_in_dark_mode autotable'>" . htmlentities($content) . "</pre>";
			$this_html .= copy_button("stdout_file");
			$this_html .= "<script>var job_infos_csv = $jobInfosCsvJson; plot_parallel_plot(job_infos_csv);</script>";
			$html_parts[$_hash] = $this_html;
		}

		return [$tab_headers, $html_parts];
	}

	function show_run($folder) {
		$run_files = glob("$folder/*");

		$html = "";

		if (file_exists("$folder/ui_url.txt")) {
			$content = remove_ansi_colors(file_get_contents("$folder/ui_url.txt"));
			$content_encoding = mb_detect_encoding($content);
			if (($content_encoding == "ASCII" || $content_encoding == "UTF-8")) {
				#$shown_data += print_url($content);
			}
		}

		$out_or_err_files = [];

		$tab_headers = array();

		$out_files_already_in_tab = 0;

		$html_parts = [];

		foreach ($run_files as $file) {
			if (preg_match("/\/\.\.\/?/", $file)) {
				print("Invalid file " . htmlentities($file) . " detected. It will be ignored.");

				continue;
			}

			if (preg_match("/results\.csv$/", $file)) {
				$tab_headers_and_html_parts = get_results_csv_code($file, $tab_headers, $html_parts);
				$tab_headers = $tab_headers_and_html_parts[0];
				$html_parts = $tab_headers_and_html_parts[1];
			} elseif (preg_match("/cpu_ram_usage\.csv$/", $file)) {
				$tab_headers_and_html_parts = get_results_cpu_ram_usage($file, $tab_headers, $html_parts);
				$tab_headers = $tab_headers_and_html_parts[0];
				$html_parts = $tab_headers_and_html_parts[1];
			} elseif (preg_match("/worker_usage\.csv$/", $file)) {
				$tab_headers_and_html_parts = get_results_worker_usage($file, $tab_headers, $html_parts);
				$tab_headers = $tab_headers_and_html_parts[0];
				$html_parts = $tab_headers_and_html_parts[1];
			} elseif (preg_match("/best_result\.txt$/", $file)) {
				$tab_headers_and_html_parts = get_results_best_results($file, $tab_headers, $html_parts);
				$tab_headers = $tab_headers_and_html_parts[0];
				$html_parts = $tab_headers_and_html_parts[1];
			} elseif (preg_match("/parameters\.txt$/", $file)) {
				$tab_headers_and_html_parts = get_results_parameters($file, $tab_headers, $html_parts);
				$tab_headers = $tab_headers_and_html_parts[0];
				$html_parts = $tab_headers_and_html_parts[1];
			} elseif (
				preg_match("/evaluation_errors\.log$/", $file)
				|| preg_match("/oo_errors\.txt$/", $file)
			) {
				$tab_headers_and_html_parts = get_results_evaluation_errors_and_oo_errors($file, $tab_headers, $html_parts);
				$tab_headers = $tab_headers_and_html_parts[0];
				$html_parts = $tab_headers_and_html_parts[1];
			} elseif (
				preg_match("/get_next_trials/", $file)
			) {
				$tab_headers_and_html_parts = get_results_get_next_trial($file, $tab_headers, $html_parts);
				$tab_headers = $tab_headers_and_html_parts[0];
				$html_parts = $tab_headers_and_html_parts[1];
			} elseif (preg_match("/job_infos\.csv$/", $file)) {
				$tab_headers_and_html_parts = get_results_job_infos($file, $tab_headers, $html_parts);
				$tab_headers = $tab_headers_and_html_parts[0];
				$html_parts = $tab_headers_and_html_parts[1];
			} elseif (
				preg_match("/state_files/", $file)
				|| preg_match("/failed_logs/", $file)
				|| preg_match("/single_runs/", $file)
				|| preg_match("/gpu_usage/", $file)
				|| preg_match("/hash\.md5$/", $file)
				|| preg_match("/ui_url\.txt$/", $file)
				|| preg_match("/run_uuid$/", $file)
			) {
				// do nothing
			} elseif (preg_match("/\/\d*_\d*_log\.(err|out)$/", $file)) {
				if(!$out_files_already_in_tab) {
					$tab_headers[] = array("id" => 'single_run_files_container', "header" => "Out-Files");
					$out_files_already_in_tab = 1;
				}

				$out_or_err_files[] = $file;
			} else {
				echo "<!-- Unknown file '$file' -->\n";
			}
		}

		if(count($tab_headers)) {
			$html .= "<div id='main_tabbed' style='width: fit-content'>\n";
			$html .= "<ul>\n";
			foreach ($tab_headers as $header) {
				$cleaned_header = removeHTags($header["header"]);
				$html .= "<li><a href='#" . $header["id"] . "'>$cleaned_header</a></li>\n";
			}
			$html .= "</ul>\n";
		}

		if (count($out_or_err_files)) {
			if (count($out_or_err_files) == 1) {
				$_file = $out_or_err_files[0];
				if (file_exists($_file)) {
					$content = file_get_contents($_file);
					$html .= "<pre class='stdout_file invert_in_dark_mode'>" . htmlentities($content) . "\n</pre>";
					$html .= copy_button("stdout_file");
				}
			} else {
				$html .= "<div id='single_run_files_container'>\n";
				$html .= "<h2 id='single_run_files'>Single run output files</h2>";
				$html .= '<div id="out_files_tabs">';
				$html .= '<ul style="max-height: 200px; overflow: auto;">';
				$ok = "&#9989;";
				$error = "&#10060;";

				foreach ($out_or_err_files as $out_or_err_file) {
					$content = remove_ansi_colors(file_get_contents($out_or_err_file));

					$_hash = hash('md5', $content);

					$ok_or_error = $ok;

					if (!checkForResult($content)) {
						$ok_or_error = $error;
					}

					$html .= "<li><a href='#$_hash'>" . preg_replace("/_0_.*/", "", preg_replace("/.*\/+/", "", $out_or_err_file)) . "<span class='invert_in_dark_mode'>$ok_or_error</span></a></li>";
				}
				$html .= "</ul>";
				foreach ($out_or_err_files as $out_or_err_file) {
					$content = remove_ansi_colors(file_get_contents($out_or_err_file));

					$_hash = hash('md5', $content);
					$html .= "<div id='$_hash'>";
					$html .= "<pre class='stdout_file invert_in_dark_mode'>" . htmlentities($content) . "\n</pre>";
					$html .= copy_button("stdout_file");
					$html .= "</div>";
				}
				$html .= "</div>";

				$html .= "<script>";
				$html .= "$(function() {";
				$html .= '    $("#out_files_tabs").tabs();';
				$html .= '    $("#main_tabbed").tabs();';
				$html .= "});";
				$html .= "</script>";
				$html .= "</div>";
			}
		}

		$html .= "<div id='scatter_plot_2d_container'>";
		$html .= "</div>";

		$html .= "<div id='scatter_plot_3d_container'>";
		$html .= "</div>";

		$html .= "<div id='parallel_plot_container'>";
		$html .= "</div>";

		if(count($html_parts)) {
			foreach ($html_parts as $id => $html_part) {
				$html .= "<div id='$id'>";
				$html .= $html_part;
				$html .= "</div>";
			}
		}

		if(count($tab_headers)) {
			$html .= "</div>\n";
		}

		if (count($html_parts) == 0 || count($tab_headers) == 0) {
			$html .= "<h2>No visualizable data could be found</h2>";
		}

		print $html;
	}

	function custom_sort($a, $b) {
		// Extrahiere numerische und alphabetische Teile
		$a_numeric = preg_replace('/[^0-9]/', '', $a);
		$b_numeric = preg_replace('/[^0-9]/', '', $b);

		// Falls beide numerisch sind, sortiere numerisch
		if (is_numeric($a_numeric) && is_numeric($b_numeric)) {
			if ((int)$a_numeric == (int)$b_numeric) {
				return strcmp($a, $b); // Wenn numerisch gleich, alphabetisch sortieren
			}
			return (int)$a_numeric - (int)$b_numeric;
		}

		// Falls nur einer numerisch ist, numerische Sortierung bevorzugen
		if (is_numeric($a_numeric)) {
			return -1;
		}

		if (is_numeric($b_numeric)) {
			return 1;
		}

		// Falls keine numerisch sind, alphabetisch sortieren
		return strcmp($a, $b);
	}

	function check_and_filter_folders($folders) {
		// Überprüfe, ob das übergebene Argument ein Array ist
		if (!is_array($folders)) {
			throw new InvalidArgumentException("Der übergebene Parameter muss ein Array sein.");
		}

		$filtered_folders = array_filter($folders, function($folder) {
			// Überprüfe, ob der Pfad ein Verzeichnis ist
			if (!is_dir($folder)) {
				// Wenn es kein Verzeichnis ist, gebe eine Warnung aus und behalte den Eintrag
				error_log("Warnung: '$folder' ist kein gültiges Verzeichnis.");
				return true;
			}

			// Öffne das Verzeichnis
			$files = scandir($folder);

			// Entferne "." und ".." aus der Liste der Dateien
			$files = array_diff($files, array('.', '..'));

			// Wenn Dateien vorhanden sind, behalte den Eintrag
			return count($files) > 0;
		});

		return $filtered_folders;
	}

	function show_run_selection($sharesPath, $user, $experiment_name) {
		$experiment_name = preg_replace("/.*\//", "", $experiment_name);
		$folder_glob = "$sharesPath/$user/$experiment_name/*";
		$experiment_subfolders = glob($folder_glob, GLOB_ONLYDIR);

		$experiment_subfolders = check_and_filter_folders($experiment_subfolders);

		if (count($experiment_subfolders) == 0) {
			echo "No runs found in $folder_glob";
			exit(1);
		}

		usort($experiment_subfolders, 'custom_sort');

		foreach ($experiment_subfolders as $run_nr) {
			$run_nr = preg_replace("/.*\//", "", $run_nr);
			$sharesPathLink = $sharesPath == "./shares/" ? "" : "&share_path=$sharesPath";
			echo "<!-- show_run_selection " . __LINE__ . " -->\n";
			echo "<a class='_share_link' href=\"share.php?user_id=$user&experiment_name=$experiment_name&run_nr=$run_nr$sharesPathLink\">$run_nr</a><br>";
		}
	}

	function print_script_and_folder($folder) {
		echo "\n<script>createBreadcrumb('./$folder');</script>\n";
	}

	function checkForResult($content) {
		// Regulärer Ausdruck, der nach "RESULT: " gefolgt von einer Zahl sucht (int, negativ, float)
		$pattern = '/RESULT:\s*(-?\d+(\.\d+)?)/';

		// Überprüfe, ob der String mit dem Muster übereinstimmt
		if (preg_match($pattern, $content, $matches)) {
			// Wenn ein Treffer gefunden wurde, gibt $matches[1] die Zahl zurück
			return $matches[1];
		} else {
			// Kein Treffer gefunden
			return false;
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

	function show_dir_view_or_plot($sharesPath, $user_id, $experiment_name, $run_nr) {
		if (isset($user_id) && $user_id != "" && (!isset($experiment_name) || $experiment_name == "")) {
			// given:
			//	user_id
			// missing:
			//	experiment_name
			//	run_nr
			$user = $user_id;
			if (preg_match("/\.\./", $user)) {
				print("Invalid user path");
				exit(1);
			}

			$user = preg_replace("/.*\//", "", $user);

			$experiment_subfolders = glob("$sharesPath/$user/*", GLOB_ONLYDIR);
			if (count($experiment_subfolders) == 0) {
				print("Did not find any experiments for $sharesPath/$user/*");
				exit(0);
			} else {
				foreach ($experiment_subfolders as $experiment) {
					$experiment = preg_replace("/.*\//", "", $experiment);
					$sharesPathLink = $sharesPath == "./shares/" ? "" : "&share_path=$sharesPath";
					echo "<!-- show_dir_view_or_plot A " . __LINE__ . " --><a class='_share_link' href=\"share.php?user_id=$user&experiment_name=$experiment$sharesPathLink\">$experiment</a><br>\n";
				}
				print("<!-- $user/$experiment_name/ -->");
				print_script_and_folder("$user/$experiment_name/");
			}
		} elseif (isset($user_id) && $user_id != "" && (isset($experiment_name) || $experiment_name != "") && (!isset($run_nr) || $run_nr == "")) {
			// given:
			//	user_id
			//	experiment_name
			// missing:
			//	run_nr
			show_run_selection($sharesPath, $user_id, $experiment_name);
			print("<!-- $user_id/$experiment_name/ -->");
			print_script_and_folder("$user_id/$experiment_name/");
		} elseif (isset($user_id) && $user_id != "" && isset($experiment_name) && $experiment_name != "" && isset($run_nr) && $run_nr != "") {
			// given:
			//	user_id
			//	experiment_name
			//	run_nr
			// missing:
			//	none
			$run_folder_without_shares = "$user_id/$experiment_name/$run_nr/";

			$run_folder = "$sharesPath/$run_folder_without_shares";
			if (isset($_GET["get_hash_only"])) {
				echo calculateDirectoryHash($run_folder);

				exit(0);
			} else {
				print("<!-- $run_folder_without_shares -->");

				print_script_and_folder($run_folder_without_shares);

				show_run($run_folder);
			}
		} else {
			// given:
			//	none
			// missing:
			//	user_id
			//	experiment_name
			//	run_nr
			$user_subfolders = glob($sharesPath . '*', GLOB_ONLYDIR);
			if (count($user_subfolders)) {
				foreach ($user_subfolders as $user) {
					$user = preg_replace("/.*\//", "", $user);
					$sharesPathLink = $sharesPath == "./shares/" ? "" : "&share_path=$sharesPath";

					echo "<!-- show_dir_view_or_plot B " . __LINE__ . " -->\n";
					echo "<a class='_share_link' href=\"share.php?user_id=$user$sharesPathLink\">$user</a><br>\n";
				}
			} else {
				echo "No users found";
			}

			print("<!-- startpage -->");
			print_script_and_folder("");
		}
	}


	function move_files($offered_files, $added_files, $userFolder, $msgUpdate, $msg) {
		$empty_files = [];

		foreach ($offered_files as $offered_file) {
			$file = $offered_file["file"];
			$filename = $offered_file["filename"];
			if ($file && file_exists($file)) {
				$content = file_get_contents($file);
				$content_encoding = mb_detect_encoding($content);
				if ($content_encoding == "ASCII" || $content_encoding == "UTF-8") {
					if (filesize($file)) {
						move_uploaded_file($file, "$userFolder/$filename");
						$added_files++;
					} else {
						$empty_files[] = $filename;
					}
				} else {
					dier("$filename: \$content was not ASCII, but $content_encoding");
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
			$old_url = remove_extra_slashes_from_url("$BASEURL/share.php?user_id=$user_id&experiment_name=$experiment_name&run_nr=$run_id");
			echo "This project already seems to have been uploaded. See $old_url\n";
			exit(0);
		} else {
			if (!$uuid_folder || !is_dir($uuid_folder)) {
				$url = remove_extra_slashes_from_url("$BASEURL/share.php?user_id=$user_id&experiment_name=$experiment_name&run_nr=$run_id");

				move_files(
					$offered_files,
					$added_files,
					$userFolder,
					"See $url&update=1 for a live-trace.\n",
					"Run was successfully shared. See $url\nYou can share the link. It is valid for 30 days.\n"
				);
			} else {
				$url = remove_extra_slashes_from_url("$BASEURL/share.php?user_id=$user_id&experiment_name=$experiment_name&run_nr=$run_id&update=1");

				move_files(
					$offered_files,
					$added_files,
					$uuid_folder,
					"See $url for a live-trace.\n",
					"See $url for a live-trace.\n"
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
