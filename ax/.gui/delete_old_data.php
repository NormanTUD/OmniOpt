<?php
	header('Content-Type: application/json');

	function checkOldDirectories($dir) {
		$oldDirectories = [];
		$currentTime = time();

		// Durchlaufe alle Unterordner in der Struktur
		foreach (glob("$dir/*/*/*", GLOB_ONLYDIR) as $subdir) {
			$pathParts = explode('/', $subdir); // Zerlege den Pfad in seine Komponenten
			$secondDir = $pathParts[1] ?? ''; // Finde den zweiten Ordner im Pfad (index 1)

			// Setze das Zeitlimit auf 2 Stunden, wenn der zweite Ordner "runner" ist, ansonsten auf 30 Tage
			$threshold = ($secondDir === 'runner') ? 2 * 60 * 60 : 30 * 24 * 60 * 60; // 2 Stunden oder 30 Tage

			$dir_date = filemtime($subdir); // Hol dir das Änderungsdatum des Ordners
			if (is_dir($subdir) && ($dir_date < ($currentTime - $threshold))) {
				$oldDirectories[] = $subdir; // Füge alte Ordner der Liste hinzu
			}
		}

		return $oldDirectories;
	}

	$directoryToCheck = 'shares'; // Wurzelverzeichnis
	$oldDirs = checkOldDirectories($directoryToCheck);

	echo json_encode($oldDirs, JSON_PRETTY_PRINT); // Ausgabe der alten Verzeichnisse als JSON mit Formatierung
	echo "\n";
?>
