<?php
	require_once "_functions.php";

	$GLOBALS["files"] = array(
		"tutorials" => array(
			"name" => "<span>📚</span> Tutorials&Help",
			"entries" => array()
		),
		"gui" => "<span>⚙️</span> GUI",
		"share" => "<span>🌍</span> Share",
		"usage_stats" => "<span>📊</span> Statistics"
	);

	if (isset($GLOBALS["index_tutorials"])) {
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
	}
?>
