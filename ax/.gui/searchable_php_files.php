<?php
	require_once "_functions.php";

	$GLOBALS["files"] = array(
		"tutorials" => array(
			"name" => "Tutorials&Help",
			"entries" => array()
		),
		"gui" => "GUI",
		"share" => "Share",
		"usage_stats" => "Statistics"
	);

	$_files = scandir('_tutorials/');
	foreach ($_files as $file) {
		if ($file != ".." && $file != "." && $file != "favicon.ico" and preg_match("/\.php/", $file)) {
			$name = $file;

			$heading_content = get_first_heading_content("_tutorials/$file");

			if ($heading_content !== null) {
				$name = $heading_content;
			}

			$file = preg_replace("/\.php$/", "", $file);

			$GLOBALS["files"]["tutorials"]["entries"][$file] = $name;
		}
	}
