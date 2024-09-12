<?php
	error_reporting(E_ALL);
	set_error_handler(function ($severity, $message, $file, $line) {
		throw new \ErrorException($message, $severity, $severity, $file, $line);
	});

	ini_set('display_errors', 1);

	$BASEURL = dirname((isset($_SERVER["REQUEST_SCHEME"]) ? $_SERVER["REQUEST_SCHEME"] : "http")."://".(isset($_SERVER["SERVER_NAME"]) ? $_SERVER["SERVER_NAME"] : "localhost")."/".$_SERVER["SCRIPT_NAME"]);
	$sharesPath = './shares/';

	$user_id = $_GET['user_id'] ?? null;
	$share_on_list_publically = $_GET['share_on_list_publically'] ?? null;
	$experiment_name = $_GET['experiment_name'] ?? null;

	$acceptable_files = ["best_result", "job_infos", "parameters", "results", "ui_url", "cpu_ram_usage", "get_next_trials", "run_uuid"];
	$acceptable_file_names = ["best_result.txt", "job_infos.csv", "parameters.txt", "results.csv", "ui_url.txt", "cpu_ram_usage.csv", "get_next_trials.csv", "run_uuid"];

	$GLOBALS["time_start"] = microtime(true);

	function die_with_time() {
		$time_end = microtime(true);
		dier("Runtime: ".abs($time_end - $GLOBALS["time_start"]));
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
				if($row[$result_column_id]) {
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

	function dier($msg) {
		print("<pre>".print_r($msg, true)."</pre>");
		exit(1);
	}

	function findFileByUUID($uuid) {
		$baseDir = 'shares/';

		function searchDirectory($dir, $uuid) {
			$files = array_diff(scandir($dir), ['.', '..']);

			foreach ($files as $file) {
				$path = $dir . DIRECTORY_SEPARATOR . $file;

				// Wenn es ein Verzeichnis ist, rekursiv weitersuchen
				if (is_dir($path)) {
					$result = searchDirectory($path, $uuid);
					if ($result !== null) {
						return $result;
					}
				} elseif (basename($path) === "run_uuid") {
					// Wenn es eine Datei ist und die Datei heißt wie die UUID, geben wir den Pfad zurück (ohne den letzten Teil)
					return dirname($path);
				}
			}

			// Wenn nichts gefunden wurde, null zurückgeben
			return null;
		}

		$result = searchDirectory($baseDir, $uuid);

		if ($result !== null) {
			return $result;
		}

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
		do {
			$newFolder = $path . "/$user_id/$experiment_name/$i";
			$i++;
		} while (file_exists($newFolder));

		mkdir($newFolder, 0777, true); // Rechte 0777 für volle Zugriffsberechtigungen setzen
		return $newFolder;
	}

	function searchForHashFile($directory, $new_upload_md5, $userFolder) {
		$files = glob($directory);

		foreach ($files as $file) {
			try {
				$file_content = file_get_contents($file);

				if ($file_content === $new_upload_md5) {
					return [True, dirname($file)];
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

		return [False, null];
	}

	function extractPathComponents($found_hash_file_dir) {
		$pattern = '#^shares/([^/]+)/([^/]+)/(\d+)$#';

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

	function remove_ansi_colors ($contents) {
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
		array_shift($lines); # Remove headline line above the table
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

		#dier($tableData);

		$skip_next_row = false;

		$newTableData = [];

		foreach ($tableData as $rowIndex => $row) {
			$thisRow = $tableData[$rowIndex];
			if($rowIndex > 0) {
				if(!$skip_next_row && isset($tableData[$rowIndex + 1])) {
					$nextRow = $tableData[$rowIndex + 1];
					if(count($thisRow) > count($nextRow)) {
						$next_row_keys = array_keys($nextRow);

						foreach ($next_row_keys as $nrk) {
							$thisRow[$nrk] .= " ".$nextRow[$nrk];
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

		#dier($newTableData);

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

	function show_run($folder) {
		$run_files = glob("$folder/*");
		
		$shown_data = 0;

		$file = "";

		if(file_exists("$folder/ui_url.txt")) {
			$content = remove_ansi_colors(file_get_contents("$folder/ui_url.txt"));
			$content_encoding = mb_detect_encoding($content);
			if(($content_encoding == "ASCII" || $content_encoding == "UTF-8")) {
				$shown_data += print_url($content);
			}
		}

		$out_or_err_files = [];

		foreach ($run_files as $file) {
			if(preg_match("/\/\.\.\/?/", $file)) {
				print("Invalid file ".htmlentities($file)." detected. It will be ignored.");
			}

			if (preg_match("/results\.csv$/", $file)) {
				$content = remove_ansi_colors(file_get_contents($file));
				$content_encoding = mb_detect_encoding($content);
				if(!($content_encoding == "ASCII" || $content_encoding == "UTF-8")) {
					continue;
				}

				$jsonData = loadCsvToJsonByResult($file);

				echo "<h2>".preg_replace("/.*\//", "", $file)."</h2>";
				if($jsonData == "[]") {
					echo "Data is empty";
					continue;
				}

				echo "<textarea readonly class='textarea_csv'>" . htmlentities($content) . "</textarea>";
?>
				<script>
					var results_csv_json = <?php echo $jsonData ?>;

					plot_all_possible(results_csv_json);
				</script>
<?php
				$shown_data += 1;
			} else if (
				preg_match("/parameters\.txt$/", $file)
			) {
				$content = remove_ansi_colors(file_get_contents($file));
				$content_encoding = mb_detect_encoding($content);
				if(!($content_encoding == "ASCII" || $content_encoding == "UTF-8")) {
					continue;
				}
				echo "<h2>".preg_replace("/.*\//", "", $file)."</h2>";
				echo "<textarea readonly class='textarea_csv'>" . htmlentities($content) . "</textarea>";
				$shown_data += 1;
			} else if (
				preg_match("/cpu_ram_usage\.csv$/", $file)
			) {
				$jsonData = loadCsvToJson($file);
				$content = remove_ansi_colors(file_get_contents($file));

				if($jsonData == "[]") {
					echo "Data is empty";
					continue;
				}

?>
				<script>
					function replaceZeroWithNull(arr) {
						// Überprüfen, ob arr ein Array ist
						if (Array.isArray(arr)) {
							for (let i = 0; i < arr.length; i++) {
								// Wenn das aktuelle Element ein Array ist, rekursiv aufrufen
								if (Array.isArray(arr[i])) {
									replaceZeroWithNull(arr[i]);
								} else if (arr[i] === 0) {
									// Wenn das aktuelle Element 0 ist, durch null ersetzen
									arr[i] = null;
								}
							}
						}
					};

					var cpu_ram_usage_json = convertToIntAndFilter(<?php echo $jsonData ?>.map(Object.values));

					replaceZeroWithNull(cpu_ram_usage_json);

					plot_cpu_gpu_graph(cpu_ram_usage_json);
				</script>
<?php
			} else if (
				preg_match("/worker_usage\.csv$/", $file)
			) {
				$jsonData = loadCsvToJson($file);
				$content = remove_ansi_colors(file_get_contents($file));

				echo "<h2>".preg_replace("/.*\//", "", $file)."</h2>";
				if($jsonData == "[]") {
					echo "Data is empty";
					continue;
				}

				echo "<textarea readonly class='textarea_csv'>" . htmlentities($content) . "</textarea>";
?>
				<script>
					var worker_usage_csv = convertToIntAndFilter(<?php echo $jsonData ?>.map(Object.values));

					plotLineChart(worker_usage_csv);
				</script>
<?php
			} else if (
				preg_match("/evaluation_errors\.log$/", $file) || 
				preg_match("/oo_errors\.txt$/", $file) ||
				preg_match("/best_result\.txt$/", $file) ||
				preg_match("/get_next_trials/", $file) ||
				preg_match("/job_infos\.csv$/", $file)
			) {
				$content = remove_ansi_colors(file_get_contents($file));
				$content_encoding = mb_detect_encoding($content);
				if(!($content_encoding == "ASCII" || $content_encoding == "UTF-8")) {
					continue;
				}
				echo "<h2>".preg_replace("/.*\//", "", $file)."</h2>";
				echo "<textarea readonly class='textarea_csv'>" . htmlentities($content) . "</textarea>";
				$shown_data += 1;
			} else if (
				preg_match("/state_files/", $file) ||
				preg_match("/failed_logs/", $file) ||
				preg_match("/single_runs/", $file) ||
				preg_match("/gpu_usage/", $file) ||
				preg_match("/hash\.md5$/", $file) ||
				preg_match("/ui_url\.txt$/", $file) ||
				preg_match("/run_uuid$/", $file)
			) {
				// do nothing
			} else if (
				preg_match("/\/\d*_\d*_log\.(err|out)$/", $file)
			) {
				$out_or_err_files[] = $file;
			} else {
				echo "<h2 class='error'>Unknown file type $file</h2>";
			}
		}

		if(count($out_or_err_files)) {
			if(count($out_or_err_files) == 1) {
				$_file = $out_or_err_files[0];
				if(file_exists($_file)) {
					$content = file_get_contents($_file);
					echo "<h2>".preg_replace("/.*\/+/", "", $_file)."</h2>";
					print "<textarea readonly class='textarea_csv'>" . htmlentities($content) . "</textarea>";
				}
			} else {
?>
				<h2 id='single_run_files'>Single run output files</h2>
				<div id="out_files_tabs">
					<ul>
<?php
						$ok = "&#9989;";
						$error = "&#10060;";



						foreach($out_or_err_files as $out_or_err_file) {
							$content = remove_ansi_colors(file_get_contents($out_or_err_file));

							$_hash = hash('md5', $content);

							$ok_or_error = $ok;

							if(!checkForResult($content)) {
								$ok_or_error = $error;
								dier("content does not contain result:\n$content");
							}

?>
							<li><a href="#<?php print $_hash; ?>"><?php print preg_replace("/_0_.*/", "", preg_replace("/.*\/+/", "", $out_or_err_file)); ?><span class="invert_in_dark_mode"><?php print $ok_or_error; ?></span></a></li>
<?php
						}
?>
					</ul>
<?php
						foreach($out_or_err_files as $out_or_err_file) {
							$content = remove_ansi_colors(file_get_contents($out_or_err_file));

							$_hash = hash('md5', $content);
?>
							<div id="<?php print $_hash; ?>">
								<textarea readonly class='textarea_csv'><?php print htmlentities($content); ?></textarea>
							</div>
<?php
						}
?>
					</div>

				<script>
					$(function() {
						$("#out_files_tabs").tabs();
					});
				</script>
<?php
			}
		}

		if($shown_data == 0) {
			echo "<h2>No visualizable data could be found</h2>";
		}
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

	function show_run_selection ($sharesPath, $user, $experiment_name) {
		$experiment_name = preg_replace("/.*\//", "", $experiment_name);
		$folder_glob = "$sharesPath/$user/$experiment_name/*";
		$experiment_subfolders = glob($folder_glob, GLOB_ONLYDIR);

		if (count($experiment_subfolders) == 0) {
			echo "No runs found in $folder_glob";
			exit(1);
		} else if (count($experiment_subfolders) == 1) {
			$user_dir = preg_replace("/^\.\//", "", preg_replace("/\/\/*/", "/", preg_replace("/\.\/shares\//", "./", $experiment_subfolders[0])));

			print_script_and_folder($user_dir);
			show_run($experiment_subfolders[0]);
			exit(0);
		}

		usort($experiment_subfolders, 'custom_sort');

		foreach ($experiment_subfolders as $run_nr) {
			$run_nr = preg_replace("/.*\//", "", $run_nr);
			echo "<a href=\"share.php?user=$user&experiment=$experiment_name&run_nr=$run_nr\">$run_nr</a><br>";
		}
	}

	function print_script_and_folder ($folder) {
		echo "<script>createBreadcrumb('./$folder');</script>\n";
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

	$update_uuid = isset($_GET["update_uuid"]) ? $_GET["update_uuid"] : null;
	$uuid_folder = findFileByUUID($update_uuid);

	if ($user_id !== null && $experiment_name !== null) {
		if(!$uuid_folder) {
			$userFolder = createNewFolder($sharesPath, $user_id, $experiment_name);
		} else {
			$userFolder = $uuid_folder;
		}
		$run_id = preg_replace("/.*\//", "", $userFolder);

		$added_files = 0;

		$num_offered_files = 0;
		$new_upload_md5_string = "";

		$offered_files = [];
		$i = 0;
		foreach ($acceptable_files as $acceptable_file) {
			$offered_files[$acceptable_file] = array(
				"file" => $_FILES[$acceptable_file]['tmp_name'] ?? null,
				"filename" => $acceptable_file_names[$i]
			);
			$i++;
		}

		foreach ($_FILES as $_file) {
			if(preg_match("/log.(err|out)$/", $_file["name"])) {
				$_file_without_ending = pathinfo($_file["name"], PATHINFO_FILENAME);
				if(!isset($offered_files[$_file_without_ending])) {
					if(isset($_file["name"])) {
						if($_file["error"] != 0) {
							print("File ".htmlentities($_file["name"])." could not be uploaded. Error-Code: ".$_file["error"]);
						} else {
							if ($_file["size"] > 0) {
								$num_offered_files++;
								$offered_files[$_file_without_ending] = array(
									"file" => $_file["tmp_name"] ?? null,
									"filename" => $_file["name"]
								);

							} else {
								#print("File ".htmlentities($_file["name"])." had filesize 0 and will be ignored.\n");
							}
						}
					} else {
						print("Could not determine filename for at least one uploaded file");
					}
				} else {
					print("$_file_without_ending coulnd't be found in \$offered_files\n");
				}
			} else {
				#print("File ".htmlentities($_file["name"])." will be ignored.\n");
			}
		}

		foreach ($offered_files as $offered_file) {
			$filename = $offered_file["filename"];
			$file = $offered_file["file"];
			if($file) {
				$content = file_get_contents($file);
				$new_upload_md5_string = $new_upload_md5_string . "$filename=$content";
				$num_offered_files++;
			}
		}

		if ($num_offered_files == 0) {
			print("Error sharing job. No offered files could be found");
			exit(1);
		}

		$project_md5 = hash('md5', $new_upload_md5_string);

		$found_hash_file_data = searchForHashFile("shares/*/*/*/hash.md5", $project_md5, $userFolder);

		$found_hash_file = $found_hash_file_data[0];
		$found_hash_file_dir = $found_hash_file_data[1];

		if($found_hash_file && is_null($update_uuid)) {
			list($user, $experiment_name, $run_id) = extractPathComponents($found_hash_file_dir);
			echo "This project already seems to have been uploaded. See $BASEURL/share.php?user=$user_id&experiment=$experiment_name&run_nr=$run_id\n";
			exit(0);
		} else {
			if(!$uuid_folder || !is_dir($uuid_folder)) {
				foreach ($offered_files as $offered_file) {
					$file = $offered_file["file"];
					$filename = $offered_file["filename"];
					if ($file && file_exists($file)) {
						$content = file_get_contents($file);
						$content_encoding = mb_detect_encoding($content);
						if($content_encoding == "ASCII" || $content_encoding == "UTF-8") {
							if(filesize($file)) {
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
					echo "Run was successfully shared. See $BASEURL/share.php?user=$user_id&experiment=$experiment_name&run_nr=$run_id\nYou can share the link. It is valid for 30 days.\n";
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
			} else {
				foreach ($offered_files as $offered_file) {
					$file = $offered_file["file"];
					$filename = $offered_file["filename"];
					if ($file && file_exists($file)) {
						$content = file_get_contents($file);
						$content_encoding = mb_detect_encoding($content);
						if($content_encoding == "ASCII" || $content_encoding == "UTF-8") {
							if(filesize($file)) {
								move_uploaded_file($file, "$uuid_folder/$filename");
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
					echo "See $BASEURL/share.php?user=$user_id&experiment=$experiment_name&run_nr=$run_id&update_interval=30 for a live-trace.\n";
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
		}
	} else {
		include_once("_functions.php");

		$dir_path = ".";
		if(preg_match("/\/tutorials\/?$/", dirname($_SERVER["PHP_SELF"]))) {
			$dir_path = "..";
		}
?>
		<script src='plotly-latest.min.js'></script>
		<script src='share.js'></script>
		<script src='share_graphs.js'></script>
		<link href="<?php echo $dir_path; ?>/share.css" rel="stylesheet" />

		<div id="breadcrumb"></div>
<?php
	}

	// Liste aller Unterordner anzeigen
	if (isset($_GET["user"]) && !isset($_GET["experiment"])) {
		$user = $_GET["user"];
		if(preg_match("/\.\./", $user)) {
			print("Invalid user path");
			exit(1);
		}

		$user = preg_replace("/.*\//", "", $user);

		$experiment_subfolders = glob("$sharesPath/$user/*", GLOB_ONLYDIR);
		if (count($experiment_subfolders) == 0) {
			print("Did not find any experiments for $sharesPath/$user/*");
			exit(0);
		} else if (count($experiment_subfolders) == 1) {
			show_run_selection($sharesPath, $user, $experiment_subfolders[0]);
			$this_experiment_name = "$experiment_subfolders[0]";
			$this_experiment_name = preg_replace("/.*\//", "", $this_experiment_name);
			print("<!-- $user/$experiment_name/$this_experiment_name -->");
			print_script_and_folder("$user/$experiment_name/$this_experiment_name");
		} else {
			foreach ($experiment_subfolders as $experiment) {
				$experiment = preg_replace("/.*\//", "", $experiment);
				echo "<a href=\"share.php?user=$user&experiment=$experiment\">$experiment</a><br>";
			}
			print("<!-- $user/$experiment_name/ -->");
			print_script_and_folder("$user/$experiment_name/");
		}
	} else if (isset($_GET["user"]) && isset($_GET["experiment"]) && !isset($_GET["run_nr"])) {
		$user = $_GET["user"];
		$experiment_name = $_GET["experiment"];
		show_run_selection($sharesPath, $user, $experiment_name);
		print("<!-- $user/$experiment_name/ -->");
		print_script_and_folder("$user/$experiment_name/");
	} else if (isset($_GET["user"]) && isset($_GET["experiment"]) && isset($_GET["run_nr"])) {
		$user = $_GET["user"];
		$experiment_name = $_GET["experiment"];
		$run_nr = $_GET["run_nr"];

		$run_folder = "$sharesPath/$user/$experiment_name/$run_nr/";
		print("<!-- $user/$experiment_name/$run_nr -->");
		print_script_and_folder("$user/$experiment_name/$run_nr");
		show_run($run_folder);
	} else {
		$user_subfolders = glob($sharesPath . '*', GLOB_ONLYDIR);
		if(count($user_subfolders)) {
			foreach ($user_subfolders as $user) {
				$user = preg_replace("/.*\//", "", $user);
				echo "<a href=\"share.php?user=$user\">$user</a><br>";
			}
		} else {
			echo "No users found";
		}
		print("<!-- startpage -->");
		print_script_and_folder("");
	}
?>
</div>
</body>
</html>
