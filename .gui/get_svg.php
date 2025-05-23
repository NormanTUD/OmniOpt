<?php
	if(isset($_GET["file"])) {
		$file = $_GET["file"];

		if(preg_match("/^[a-zA-Z0-9_\/]+$/", $file)) {
			header('Content-Type: image/svg+xml');
			echo file_get_contents($file);
		} else {
			echo "Invalid path";
		}
	}
?>
