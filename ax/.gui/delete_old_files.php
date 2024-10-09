<?php
	function rrmdir($dir) {
		if (is_dir($dir)) {
			$objects = scandir($dir);

			foreach ($objects as $object) {
				if ($object != '.' && $object != '..') {
					if (filetype($dir.'/'.$object) == 'dir') {
						rrmdir($dir.'/'.$object);
					} else {
						unlink($dir.'/'.$object);
					}
				}
			}

			reset($objects);
			rmdir($dir);
		}
	}

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
				rrmdir($subdir);
			}
		}

		return $oldDirectories;
	}

	$directoryToCheck = 'shares';
	$oldDirs = checkOldDirectories($directoryToCheck);

	header('Content-Type: application/json');
	echo json_encode($oldDirs, JSON_PRETTY_PRINT);
	echo "\n";
?>
