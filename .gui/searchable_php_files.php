<?php
	require_once "_functions.php";

	$GLOBALS["files"] = array(
		"tutorials" => array(
			"name" => "<img class='emoji_nav' src='emojis/books.svg' />&nbsp;HowTo",
			"entries" => array()
		),
		"gui" => "<img class='emoji_nav' src='emojis/memo.svg' />&nbsp;GUI",
		"share" => "<img class='emoji_nav' src='emojis/world.svg' />&nbsp;Share",
		"conceptdrift/index" => "<img class='emoji_nav' src='emojis/bulb.svg' />&nbsp;Example",
		"usage_stats" => "<img class='emoji_nav' src='emojis/chart.svg' />&nbsp;Statistics"
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
