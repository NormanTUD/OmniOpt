<?php
	header('Content-Type: application/json');

	function checkOldDirectories($dir) {
		$oldDirectories = [];
		$currentTime = time();
		$threshold = 30 * 24 * 60 * 60; // 30 Tage in Sekunden

		// Durchlaufe alle Unterordner in der angegebenen Struktur
		foreach (glob("$dir/*/*/*", GLOB_ONLYDIR) as $subdir) {
			// Prüfe das Alter des Ordners
			$dir_date = filemtime($subdir);
			if (is_dir($subdir) && ($dir_date < ($currentTime - $threshold))) {
				// Ordner ist älter als 30 Tage
				// Füge den Ordner zur Liste der zu löschenden Ordner hinzu
				$oldDirectories[] = $subdir;

				// Kommentierter Code, um den Ordner zu löschen:
				// rmdir($subdir);
			}
		}

		return $oldDirectories;
	}

	$directoryToCheck = 'shares'; // Setze hier den Hauptordner
	$oldDirs = checkOldDirectories($directoryToCheck);

	echo json_encode($oldDirs);
	echo "\n";
?>
