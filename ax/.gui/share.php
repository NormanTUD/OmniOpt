<?php
	error_reporting(E_ALL);
	set_error_handler(function ($severity, $message, $file, $line) {
		throw new \ErrorException($message, $severity, $severity, $file, $line);
	});

	ini_set('display_errors', 1);


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
	$acceptable_files = ["best_result", "job_infos", "parameters", "results"];
	$acceptable_file_names = ["best_result.txt", "job_infos.csv", "parameters.txt", "results.csv"];

	$offered_files = [];
	$i = 0;
	foreach ($acceptable_files as $acceptable_file) {
		$offered_files[$acceptable_file] = array(
			"file" => $_FILES[$acceptable_file]['tmp_name'] ?? null,
			"filename" => $acceptable_file_names[$i]
		);
		$i++;
	}

	$best_result = $_FILES['best_result']['tmp_name'] ?? null;
	$job_infos = $_FILES['job_infos']['tmp_name'] ?? null;
	$parameters = $_FILES['parameters']['tmp_name'] ?? null;
	$results = $_FILES['results']['tmp_name'] ?? null;

	// Erstelle neuen Ordner basierend auf den Parametern
	if ($user_id !== null && $experiment_name !== null) {
		$userFolder = createNewFolder($sharesPath, $user_id, $experiment_name);
		$run_id = preg_replace("/.*\//", "", $userFolder);

		$added_files = 0;

		foreach ($offered_files as $offered_file) {
			$file = $offered_file["file"];
			$filename = $offered_file["filename"];
			if ($file) {
				move_uploaded_file($file, "$userFolder/$filename");
				$added_files++;
			}
		}

		if ($added_files) {
			echo "Job was successfully shared. See localhost/oo2_gui/share.php?user=$user_id&experiment=$experiment_name&run_nr=$run_id\n";
			exit(0);
		} else {
			echo "Error sharing the job. No Files were found";
			exit(1);
		}
	}

	function remove_ansi_colors ($contents) {
		$contents = preg_replace('#\\x1b[[][^A-Za-z]*[A-Za-z]#', '', $contents);
		return $contents;
	}

	function show_run($folder) {
		print("show_run: $folder");
		$run_files = glob("$folder/*");
		
		foreach ($run_files as $file) {
			if (preg_match("/best_result.txt/", $file)) {
				$content = remove_ansi_colors(file_get_contents($file));
				echo "<h2>".preg_replace("/.*\//", "", $file)."</h2>";
				print "<pre>$content</pre>";
			} else if (preg_match("/parameters.txt/", $file)) {
				$content = remove_ansi_colors(file_get_contents($file));
				echo "<h2>".preg_replace("/.*\//", "", $file)."</h2>";
				print "<pre>$content</pre>";
			} else if (preg_match("/job_infos.csv/", $file)) {
				$content = remove_ansi_colors(file_get_contents($file));
				echo "<h2>".preg_replace("/.*\//", "", $file)."</h2>";
				print "<pre>$content</pre>";
			} else if (preg_match("/results.csv/", $file)) {
				$content = remove_ansi_colors(file_get_contents($file));
				echo "<h2>".preg_replace("/.*\//", "", $file)."</h2>";
				print "<pre>$content</pre>";
			} else {
				print "<h2 class='error'>Unknown file type $file</h2>";
			}
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
			show_run($experiment_subfolders[0]);
			exit(0);
		}

		foreach ($experiment_subfolders as $run_nr) {
			$run_nr = preg_replace("/.*\//", "", $run_nr);
			echo "<a href=\"share.php?user=$user&experiment=$experiment_name&run_nr=$run_nr\">$user/$experiment_name/$run_nr</a><br>";
		}
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
		} else {
			foreach ($experiment_subfolders as $experiment) {
				$experiment = preg_replace("/.*\//", "", $experiment);
				echo "<a href=\"share.php?user=$user&experiment=$experiment\">$experiment</a><br>";
			}
		}
	} else if (isset($_GET["user"]) && isset($_GET["experiment"]) && !isset($_GET["run_nr"])) {
		print("show_run_selection 2:<br>");
		$user = $_GET["user"];
		$experiment_name = $_GET["experiment"];
		show_run_selection($sharesPath, $user, $experiment_name);
	} else if (isset($_GET["user"]) && isset($_GET["experiment"]) && isset($_GET["run_nr"])) {
		$user = $_GET["user"];
		$experiment_name = $_GET["experiment"];
		$run_nr = $_GET["run_nr"];

		$run_folder = "$sharesPath/$user/$experiment_name/$run_nr/";
		show_run($run_folder);
	} else {
		$user_subfolders = glob($sharesPath . '*', GLOB_ONLYDIR);
		foreach ($user_subfolders as $user) {
			echo "<a href=\"share.php?user=$user\">$user</a><br>";
		}
	}

	// Beispiel für den CURL-Befehl zum Hochladen von Dateien
	// curl -F "best_result=@../runs/__main__tests__/12/parameters.txt" http://example.com/upload.php
?>
