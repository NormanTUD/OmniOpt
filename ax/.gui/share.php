<?php
	error_reporting(E_ALL);
	set_error_handler(function ($severity, $message, $file, $line) {
		throw new \ErrorException($message, $severity, $severity, $file, $line);
	});

	ini_set('display_errors', 1);

	$BASEURL = dirname($_SERVER["REQUEST_SCHEME"]."://".$_SERVER["SERVER_NAME"]."/".$_SERVER["SCRIPT_NAME"]);

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

	// Pfad zum shares-Verzeichnis
	$sharesPath = './shares/'; // Hier den richtigen Pfad einfügen

	// Funktion zum Überprüfen der Berechtigungen
	function checkPermissions($path, $user_id) {
		// Überprüfen, ob der Ordner existiert und dem aktuellen Benutzer gehört
		if (!file_exists($path) || !is_dir($path)) {
			print("Ordner existiert nicht oder ist kein Verzeichnis.");
			exit(1);
		}

		// Überprüfen, ob der aktuelle Benutzer Schreibrechte hat
		// Hier muss die Logik eingefügt werden, um den aktuellen Benutzer und seine Berechtigungen zu überprüfen
		// Beispiel: $currentUserId = getCurrentUserId(); // Funktion zur Ermittlung der Benutzer-ID
		// Beispiel: $currentUserGroup = getCurrentUserGroup(); // Funktion zur Ermittlung der Gruppenzugehörigkeit

		// Annahme: $currentUserId und $currentUserGroup sind die aktuellen Werte des Benutzers
		// Annahme: Die Berechtigungen werden entsprechend geprüft, ob der Benutzer Schreibrechte hat

		// Beispiel für Berechtigungsüberprüfung
		// if (!hasWritePermission($path, $currentUserId, $currentUserGroup)) {
		//     exit("Benutzer hat keine Schreibrechte für diesen Ordner.");
		// }
	}

	// Funktion zum Löschen alter Ordner
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

	// Rekursive Löschfunktion für Ordner und deren Inhalte
	function deleteFolder($folder) {
		$files = array_diff(scandir($folder), array('.', '..'));

		foreach ($files as $file) {
			(is_dir("$folder/$file")) ? deleteFolder("$folder/$file") : unlink("$folder/$file");
		}

		return rmdir($folder);
	}

	// Funktion zum Erstellen eines neuen Ordners
	function createNewFolder($path, $user_id, $experiment_name) {
		$i = 0;
		do {
			$newFolder = $path . "/$user_id/$experiment_name/$i";
			$i++;
		} while (file_exists($newFolder));

		mkdir($newFolder, 0777, true); // Rechte 0777 für volle Zugriffsberechtigungen setzen
		return $newFolder;
	}

	// Verarbeitung von GET- und POST-Parametern
	$user_id = $_GET['user_id'] ?? null;
	$share_on_list_publically = $_GET['share_on_list_publically'] ?? null;
	$experiment_name = $_GET['experiment_name'] ?? null;

	// Parameter per POST entgegennehmen
	$acceptable_files = ["best_result", "job_infos", "parameters", "results", "ui_url"];
	$acceptable_file_names = ["best_result.txt", "job_infos.csv", "parameters.txt", "results.csv", "ui_url.txt"];

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


	// Erstelle neuen Ordner basierend auf den Parametern
	if ($user_id !== null && $experiment_name !== null) {
		$userFolder = createNewFolder($sharesPath, $user_id, $experiment_name);
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

		if($found_hash_file) {
			list($user, $experiment_name, $run_id) = extractPathComponents($found_hash_file_dir);
			echo "This project already seems to have been uploaded. See $BASEURL/share.php?user=$user_id&experiment=$experiment_name&run_nr=$run_id\n";
			exit(0);
		} else {
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
					implode(", ", $empty_files);
					echo "Error sharing the job. The following files were empty: $empty_files_string. \n";
				} else {
					echo "Error sharing the job. No Files were found. \n";
				}
				exit(1);
			}
		}
	} else {
		include("_header_base.php");
?>
	<script>
		var log = console.log;

		var current_folder = ""

		function parsePathAndGenerateLink(path) {
			// Define the regular expression to capture the different parts of the path
			var regex = /\/([^\/]+)\/?([^\/]*)\/?(\d+)?\/?$/;
			var match = path.match(regex);

			// Check if the path matches the expected format
			if (match) {
				var user = match[1] || '';
				var experiment = match[2] || '';
				var runNr = match[3] || '';


				// Construct the query string
				var queryString = 'share.php?user=' + encodeURIComponent(user);
				if (experiment) {
					queryString += '&experiment=' + encodeURIComponent(experiment);
				}
				if (runNr) {
					queryString += '&run_nr=' + encodeURIComponent(runNr);
				}

				return queryString;
			} else {
				console.error(`Invalid path format: ${path}, regex: {regex}`);
			}
		}

		function createBreadcrumb(currentFolderPath) {
			var breadcrumb = document.getElementById('breadcrumb');
			breadcrumb.innerHTML = '';

			var pathArray = currentFolderPath.split('/');
			var fullPath = '';

			var currentPath = "/Start/"

			pathArray.forEach(function(folderName, index) {
				if (folderName == ".") {
					folderName = "Start";
				}
				if (folderName !== '') {
					var originalFolderName = folderName;
					fullPath += originalFolderName + '/';

					var link = document.createElement('a');
					link.classList.add("breadcrumb_nav");
					link.classList.add("box-shadow");
					link.textContent = decodeURI(folderName);

					var parsedPath = "";

					if (folderName == "Start") {
						eval(`$(link).on("click", async function () {
								window.location.href = "share.php";
							});
						`);
					} else {
						currentPath += `/${folderName}`;
						parsedPath = parsePathAndGenerateLink(currentPath)

						eval(`$(link).on("click", async function () {
								window.location.href = parsedPath;
							});
						`);
					}

					breadcrumb.appendChild(link);

					// Füge ein Trennzeichen hinzu, außer beim letzten Element
					breadcrumb.appendChild(document.createTextNode(' / '));
				}
			});
		}
	</script>
	<script src='plotly-latest.min.js'></script>
	<script src='share_graphs.js'></script>
	<style>
		.textarea_csv {
			width: 80%;
			height: 150px;
		}
		.scatter-plot {
			width: 1200px;
			width: 800px;
		}

		.box-shadow {
			box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
			transition: 0.3s;
		}

		.box-shadow:hover {
			box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
		}


			#breadcrumb {
				font-size: 2.7vw;
				padding: 10px;
			}

			.breadcrumb_nav {
				background-color: #fafafa;
				text-decoration: none;
				color: black;
				border: 1px groove darkblue;
				border-radius: 5px;
				margin: 3px;
				padding: 3px;
				height: 3vw;
				display: inline-block;
				min-height: 30px;
				font-size: calc(12px + 1.5vw);;
			}

			.box-shadow {
				box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
				transition: 0.3s;
			}

			.box-shadow:hover {
				box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
			}
	</style>

	<div id="breadcrumb"></div>
<?php
	}

	function remove_ansi_colors ($contents) {
		$contents = preg_replace('#\\x1b[[][^A-Za-z]*[A-Za-z]#', '', $contents);
		return $contents;
	}

	function print_url($content) {
		$content = htmlentities($content);
		if (preg_match("/^https?:\/\//", $content)) {
			print "<a target='_blank' href='$content'>Link to the GUI, preloaded with all options specified here.</a>";

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

		foreach ($run_files as $file) {
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

				print "<textarea readonly class='textarea_csv'>" . htmlentities($content) . "</textarea>";
?>
				<script>
					var results_csv_json = <?php print $jsonData ?>;

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
				print "<textarea readonly class='textarea_csv'>" . htmlentities($content) . "</textarea>";
				$shown_data += 1;
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

				print "<textarea readonly class='textarea_csv'>" . htmlentities($content) . "</textarea>";
?>
				<script>
					var worker_usage_csv = convertToIntAndFilter(<?php print $jsonData ?>.map(Object.values));

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
				print "<textarea readonly class='textarea_csv'>" . htmlentities($content) . "</textarea>";
				$shown_data += 1;
			} else if (
				preg_match("/state_files/", $file) ||
				preg_match("/failed_logs/", $file) ||
				preg_match("/single_runs/", $file) ||
				preg_match("/gpu_usage/", $file) ||
				preg_match("/hash\.md5$/", $file) ||
				preg_match("/ui_url\.txt$/", $file)
			) {
				// do nothing
			} else {
				print "<h2 class='error'>Unknown file type $file</h2>";
			}
		}

		if($shown_data == 0) {
			print "<h2>No visualizable data could be found</h2>";
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
		print "<script>createBreadcrumb('./$folder');</script>\n";
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
		foreach ($user_subfolders as $user) {
			$user = preg_replace("/.*\//", "", $user);
			echo "<a href=\"share.php?user=$user\">$user</a><br>";
		}
		print("<!-- startpage -->");
		print_script_and_folder("");
	}
?>
<script>
	if(current_folder) {
		log(`Creating breadcrumb from current_folder: ${current_folder}`);
		//createBreadcrumb(current_folder);
	}
</script>
