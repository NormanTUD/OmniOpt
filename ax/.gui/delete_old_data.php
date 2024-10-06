<?php
	header('Content-Type: application/json');

	function checkOldDirectories($dir) {
		$oldDirectories = [];
		$currentTime = time();
		$threshold = 30 * 24 * 60 * 60; // 30 Tage in Sekunden

		foreach (glob("$dir/*/*/*", GLOB_ONLYDIR) as $subdir) {
			$dir_date = filemtime($subdir);
			if (is_dir($subdir) && ($dir_date < ($currentTime - $threshold))) {
				$oldDirectories[] = $subdir;
			}
		}

		return $oldDirectories;
	}

	$directoryToCheck = 'shares';
	$oldDirs = checkOldDirectories($directoryToCheck);

	echo json_encode($oldDirs);
	echo "\n";
?>
