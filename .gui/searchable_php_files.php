<?php
	require_once "_functions.php";

	$GLOBALS["files"] = array(
		"tutorials" => array(
			"name" => "📚&nbsp;Tutorials&Help",
			"entries" => array()
		),
		"gui" => "🧩&nbsp;GUI",
		"share" => "🌍&nbsp;Share",
		"conceptdrift/index" => "💡&nbsp;Example",
		"usage_stats" => "📊&nbsp;Statistics"
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
