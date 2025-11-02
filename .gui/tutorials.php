<?php
require "_header_base.php";

function is_valid_tutorial_name($name) {
	return preg_match("/^[a-z_]+$/", $name);
}

function get_valid_tutorial_file($base_name) {
	$paths = ["_tutorials/$base_name.md", "_tutorials/$base_name.php"];
	foreach ($paths as $path) {
		if (file_exists($path)) {
			return basename($path);
		}
	}
	return null;
}

function is_valid_file($file, $ext) {
	return preg_match("/^[a-z_]+\.$ext$/", $file) && file_exists("_tutorials/$file");
}

function render_tutorial($file) {
	$path = "_tutorials/$file";
	if (str_ends_with($file, '.md')) {
		convert_file_to_html($path);
	} elseif (str_ends_with($file, '.php')) {
		include($path);
	} else {
		echo "Invalid file format.";
	}
}

function list_tutorials() {
	$files = scandir('_tutorials/');
	$categories = [];
	$uncategorized = [];
	$has_any_category = false;

	foreach ($files as $file) {
		if (!preg_match("/\.(md|php)$/", $file)) continue;

		$path = "_tutorials/$file";
		$name = get_first_heading_content($path) ?? $file;
		$label = highlight_backticks(preg_replace("/\.(md|php)$/", "", $name));
		$file_link = preg_replace("/\.(md|php)$/", "", $file);

		$category = get_html_category_comment($path);
		$comment = get_html_comment($path);
		$entry = [
			'label' => $label,
			'link' => "tutorials?tutorial=$file_link",
			'comment' => $comment
		];

		if ($category !== null) {
			$has_any_category = true;
			$categories[$category][] = $entry;
		} else {
			$uncategorized[] = $entry;
		}
	}

	if (!$has_any_category) {
		echo "<ul>\n";
		$all_lists = array_values($categories);
		$all_lists[] = $uncategorized;
		foreach (array_merge(...$all_lists) as $entry) {
			$comment = "";
			if ($entry["comment"]) {
				$comment = " &mdash; ".$entry["comment"];
			}
			echo "<li class='li_list'><a href='{$entry['link']}'>{$entry['label']}</a>$comment</li>\n";
		}
		echo "</ul>\n";
		return;
	}

	$category_icons = [
		"Preparations, Basics and Setup" => "<img class='emoji_nav' src='emojis/box.svg' />",
		"Advanced Usage"                 => "<img class='emoji_nav' src='emojis/gear.svg' />",
		"Developing"                     => "<img class='emoji_nav' src='emojis/test_tube.svg' />",
		"Models"                         => "<img class='emoji_nav' src='emojis/robot.svg' />",
		"Multiple Objectives"            => "<img class='emoji_nav' src='emojis/direct_hit.svg' />",
		"Plotting and Sharing Results"   => "<img class='emoji_nav' src='emojis/chart_up.svg' />"
	];

	foreach ($categories as $cat => $entries) {
		$icon = $category_icons[$cat] ?? "<img class='emoji_nav' src='emojis/books.svg' />"; // Fallback-Icon
		echo "<h2><span>" . $icon . "</span> " . $cat . "</h2>\n";

		if (count($entries) === 1) {
			$e = $entries[0];
			$comment = "";
			if ($e["comment"]) {
				$comment = " &mdash; " . $e["comment"];
			}
			echo "<p><a href='{$e['link']}'>" . $e['label'] . "</a>$comment</p>\n";
		} else {
			echo "<ul>\n";
			foreach ($entries as $e) {
				$comment = "";
				if ($e["comment"]) {
					$comment = " &mdash; " . $e["comment"];
				}
				echo "<li class='li_list'><a href='{$e['link']}'>" . $e['label'] . "</a>$comment</li>\n";
			}
			echo "</ul>\n";
		}
	}

	if (!empty($uncategorized)) {
		echo "<h2>No category</h2>\n";
		echo "<ul>\n";
		foreach ($uncategorized as $e) {
			$comment = "";
			if ($e["comment"]) {
				$comment = " &mdash; ".$e["comment"];
			}
			echo "<li class='li_list'><a href='{$e['link']}'>{$e['label']}</a>$comment</li>\n";
		}
		echo "</ul>\n";
	}
}

if (isset($_GET["tutorial"])) {
	$tutorial_base = $_GET["tutorial"];
	if (is_valid_tutorial_name($tutorial_base)) {
		$valid_file = get_valid_tutorial_file($tutorial_base);
		if ($valid_file !== null && (is_valid_file($valid_file, "md") || is_valid_file($valid_file, "php"))) {
			render_tutorial($valid_file);
		} else {
			echo "Invalid file: $tutorial_base";
		}
	} else {
		echo "Invalid tutorial name.";
	}
} else {
	list_tutorials();
}
?>

</div>

<?php include("footer.php"); ?>
