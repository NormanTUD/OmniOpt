<?php
	include_once("share_functions.php");

	$oldDir = delete_old_shares();

	header('Content-Type: application/json');
	echo json_encode($oldDirs, JSON_PRETTY_PRINT);
	echo "\n";
?>
