<?php
	error_reporting(E_ALL);
	set_error_handler(function ($severity, $message, $file, $line) {
		throw new \ErrorException($message, $severity, $severity, $file, $line);
	});

	ini_set('display_errors', 1);

	function loadCsvToJson($file) {
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

	$offered_files = [];
	$i = 0;
	foreach ($acceptable_files as $acceptable_file) {
		$offered_files[$acceptable_file] = array(
			"file" => $_FILES[$acceptable_file]['tmp_name'] ?? null,
			"filename" => $acceptable_file_names[$i]
		);
		$i++;
	}

	// Erstelle neuen Ordner basierend auf den Parametern
	if ($user_id !== null && $experiment_name !== null) {
		$userFolder = createNewFolder($sharesPath, $user_id, $experiment_name);
		$run_id = preg_replace("/.*\//", "", $userFolder);

		$added_files = 0;

		foreach ($offered_files as $offered_file) {
			$file = $offered_file["file"];
			$filename = $offered_file["filename"];
			if ($file && file_exists($file)) {
				$content = file_get_contents($file);
				$content_encoding = mb_detect_encoding($content);
				if($content_encoding == "ASCII" || $content_encoding == "UTF-8") {
					move_uploaded_file($file, "$userFolder/$filename");
					$added_files++;
				} else {
					dier("$filename: \$content was not ASCII, but $content_encoding");
				}
			}
		}

		if ($added_files) {
			echo "Run was successfully shared. See https://imageseg.scads.de/omniax/share.php?user=$user_id&experiment=$experiment_name&run_nr=$run_id\nYou can share the link. It is valid for 30 days.\n";
			exit(0);
		} else {
			echo "Error sharing the job. No Files were found.\n";
			exit(1);
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

	function show_run($folder) {
		$run_files = glob("$folder/*");
		
		$shown_data = 0;
		foreach ($run_files as $file) {
			if (preg_match("/results.csv/", $file)) {
				$content = remove_ansi_colors(file_get_contents($file));
				$content_encoding = mb_detect_encoding($content);
				if(!($content_encoding == "ASCII" || $content_encoding == "UTF-8")) {
					continue;
				}

				$jsonData = loadCsvToJson($file);

				if($jsonData == "[]") {
					continue;
				}

				echo "<h2>".preg_replace("/.*\//", "", $file)."</h2>";
				print "<textarea readonly class='textarea_csv'>" . htmlentities($content) . "</textarea>";
?>
				<script src='plotly-latest.min.js'></script>
				<script src='share_graphs.js'></script>
				<script>
					var results_csv_json = <?php print $jsonData ?>;

					// Extract parameter names
					var paramKeys = Object.keys(results_csv_json[0]).filter(function(key) {
						return !['trial_index', 'arm_name', 'trial_status', 'generation_method', 'result'].includes(key);
					});

					// Get result values for color mapping
					var resultValues = results_csv_json.map(function(row) { return parseFloat(row.result); });
					var minResult = Math.min.apply(null, resultValues);
					var maxResult = Math.max.apply(null, resultValues);

					// 2D Scatter Plot
					for (var i = 0; i < paramKeys.length; i++) {
						for (var j = i + 1; j < paramKeys.length; j++) {
							var xValues = results_csv_json.map(function(row) { return parseFloat(row[paramKeys[i]]); });
							var yValues = results_csv_json.map(function(row) { return parseFloat(row[paramKeys[j]]); });
							var colors = resultValues.map(getColor);

							var trace2d = {
								x: xValues,
								y: yValues,
								mode: 'markers',
								type: 'scatter',
								marker: {
									color: colors
								}
							};

							var layout2d = {
								title: `Scatter Plot: ${paramKeys[i]} vs ${paramKeys[j]}`,
								xaxis: { title: paramKeys[i] },
								yaxis: { title: paramKeys[j] }
							};

							var new_plot_div = $(`<div class='scatter-plot' id='scatter-plot-${i}_${j}' style='width:1200px;height:800px;'></div>`);
							$('body').append(new_plot_div);
							Plotly.newPlot(`scatter-plot-${i}_${j}`, [trace2d], layout2d);
						}
					}

					// 3D Scatter Plot
					if (paramKeys.length >= 3 && paramKeys.length <= 6) {
						for (var i = 0; i < paramKeys.length; i++) {
							for (var j = i + 1; j < paramKeys.length; j++) {
								for (var k = j + 1; k < paramKeys.length; k++) {
									var xValues = results_csv_json.map(function(row) { return parseFloat(row[paramKeys[i]]); });
									var yValues = results_csv_json.map(function(row) { return parseFloat(row[paramKeys[j]]); });
									var zValues = results_csv_json.map(function(row) { return parseFloat(row[paramKeys[k]]); });
									var colors = resultValues.map(getColor);

									var trace3d = {
										x: xValues,
										y: yValues,
										z: zValues,
										mode: 'markers',
										type: 'scatter3d',
										marker: {
											color: colors
										}
									};

									var layout3d = {
										title: `3D Scatter Plot: ${paramKeys[i]} vs ${paramKeys[j]} vs ${paramKeys[k]}`,
										width: 1200,
										height: 800,
										autosize: false,
										margin: {
											l: 50,
											r: 50,
											b: 100,
											t: 100,
											pad: 4
										},
										scene: {
											xaxis: { title: paramKeys[i] },
											yaxis: { title: paramKeys[j] },
											zaxis: { title: paramKeys[k] }
										}
									};

									var new_plot_div = $(`<div class='scatter-plot' id='scatter-plot-3d-${i}_${j}_${k}' style='width:1200px;height:800px;'></div>`);
									$('body').append(new_plot_div);
									Plotly.newPlot(`scatter-plot-3d-${i}_${j}_${k}`, [trace3d], layout3d);
								}
							}
						}
					}

					parallel_plot(paramKeys, results_csv_json);
				</script>
<?php
				$shown_data += 1;
			} else if (
				preg_match("/evaluation_errors.log/", $file) || 
				preg_match("/oo_errors.txt/", $file) ||
				preg_match("/best_result.txt/", $file) ||
				preg_match("/parameters.txt/", $file) ||
				preg_match("/get_next_trials/", $file) ||
				preg_match("/job_infos.csv/", $file) ||
				preg_match("/worker_usage.csv/", $file)
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
				preg_match("/ui_url/", $file)
			) {
				$content = remove_ansi_colors(file_get_contents($file));
				$content_encoding = mb_detect_encoding($content);
				if(!($content_encoding == "ASCII" || $content_encoding == "UTF-8")) {
					continue;
				}

				$shown_data += print_url($content);
			} else if (
				preg_match("/state_files/", $file) ||
				preg_match("/failed_logs/", $file) ||
				preg_match("/single_runs/", $file) ||
				preg_match("/gpu_usage/", $file)
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
		print("<!-- user only -->");
		print_script_and_folder("");
	}

	// Beispiel für den CURL-Befehl zum Hochladen von Dateien
	// curl -F "best_result=@../runs/__main__tests__/12/parameters.txt" http://example.com/upload.php
?>
<script>
	if(current_folder) {
		log(`Creating breadcrumb from current_folder: ${current_folder}`);
		//createBreadcrumb(current_folder);
	}
</script>
