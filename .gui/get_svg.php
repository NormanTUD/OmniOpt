<?php
	require_once __DIR__ . '/_functions.php';
	require_once __DIR__ . '/share_functions.php';

	$file = get_get("file");

	if($file) {
		if(preg_match("/^[a-zA-Z0-9_\/]+$/", $file)) {
			header('Content-Type: image/svg+xml');
			echo file_get_contents($file);
		} else {
			echo "Invalid path";
		}
	}
?>
