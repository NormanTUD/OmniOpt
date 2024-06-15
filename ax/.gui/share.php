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
	$best_result = $_FILES['best_result']['tmp_name'] ?? null;
	$job_infos = $_FILES['job_infos']['tmp_name'] ?? null;
	$parameters = $_FILES['parameters']['tmp_name'] ?? null;
	$results = $_FILES['results']['tmp_name'] ?? null;

	// Erstelle neuen Ordner basierend auf den Parametern
	if ($user_id !== null && $experiment_name !== null) {
		$userFolder = createNewFolder($sharesPath, $user_id, $experiment_name);

		// Hier können die Dateien in den neuen Ordner verschoben oder gespeichert werden
		// Beispiel:
		if ($best_result) {
			move_uploaded_file($best_result, "$userFolder/best_result.txt");
		}
		if ($results) {
			move_uploaded_file($results, "$userFolder/results.csv");
		}
		if ($parameters) {
			move_uploaded_file($parameters, "$userFolder/parameters.txt");
		}
		if ($job_infos) {
			move_uploaded_file($job_infos, "$userFolder/job_infos.csv");
		}
		// usw.
	}

	// Liste aller Unterordner anzeigen
	$subfolders = glob($sharesPath . '*', GLOB_ONLYDIR);
	foreach ($subfolders as $folder) {
		// Sicherheitsüberprüfung für den Ordnerlink
		$relativePath = ltrim(str_replace($sharesPath, '', $folder), '/');
		$safeLink = htmlspecialchars($relativePath);

		echo "<a href=\"share.php?folder=$safeLink\">$safeLink</a><br>";
	}

	// Beispiel für den CURL-Befehl zum Hochladen von Dateien
	// curl -F "best_result=@../runs/__main__tests__/12/parameters.txt" http://example.com/upload.php
?>
