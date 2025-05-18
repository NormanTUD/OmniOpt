<?php
	require_once "_functions.php";

	$GLOBALS["files"] = array(
		"tutorials" => array(
			"name" => "<span class='invert_in_dark_mode'>📚</span> Tutorials&Help",
			"entries" => array()
		),
		"gui" => "<span class='invert_in_dark_mode'>⚙️</span> GUI",
		"share" => "<span class='invert_in_dark_mode'>🌍</span> Share",
		"usage_stats" => "<span class='invert_in_dark_mode'>📊</span> Statistics"
	);

	$_files = scandir('_tutorials/');

	foreach ($_files as $file) {
		if ($file != ".." && $file != "." && $file != "favicon.ico" and preg_match("/\.(md|php)$/", $file)) {
			$name = $file;

			$heading_content = get_first_heading_content("_tutorials/$file");

			if ($heading_content !== null) {
				$name = $heading_content;
			}

			$file = preg_replace("/\.(md|php)$/", "", $file);

			$GLOBALS["files"]["tutorials"]["entries"][$file] = $name;
		}
	}
?>
