<?php
	header('Content-Type: application/json');

	function checkOldDirectories($dir) {
		$oldDirectories = [];
		$currentTime = time();

		foreach (glob("$dir/*/*/*", GLOB_ONLYDIR) as $subdir) {
			$pathParts = explode('/', $subdir);
			$secondDir = $pathParts[1] ?? '';

			$threshold = ($secondDir === 'runner') ? (2 * 3600) : (30 * 24 * 3600);

			$dir_date = filemtime($subdir);

			if (is_dir($subdir) && ($dir_date < ($currentTime - $threshold))) {
				$oldDirectories[] = $subdir;
				rmdir($subdir);
			}
		}

		return $oldDirectories;
	}

	$directoryToCheck = 'shares';
	$oldDirs = checkOldDirectories($directoryToCheck);

	echo json_encode($oldDirs, JSON_PRETTY_PRINT);
	echo "\n";
?>
