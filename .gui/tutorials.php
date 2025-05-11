<?php
require "_header_base.php";

function is_valid_tutorial_name($name) {
	return preg_match("/^[a-z_]+$/", $name);
}

function get_valid_tutorial_file($base_name) {
	$paths = ["_tutorials/$base_name.md", "_tutorials/$base_name.php"];
	foreach ($paths as $path) {
		if (file_exists($path)) {
			return basename($path); // return filename with extension
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
		convertFileToHtml($path);
	} elseif (str_ends_with($file, '.php')) {
		include($path);
	} else {
		echo "Invalid file format.";
	}
}

function list_tutorials() {
	$files = scandir('_tutorials/');
	echo "<ul>\n";
	foreach ($files as $file) {
		if (!preg_match("/\.(md|php)$/", $file)) continue;

		$path = "_tutorials/$file";
		$name = get_first_heading_content($path) ?? $file;
		$label = highlightBackticks(preg_replace("/\.(md|php)$/", "", $name));
		$file_link = preg_replace("/\.(md|php)$/", "", $file);

		$comment = get_html_comment($path);
		$comment_display = $comment ? " &mdash; $comment" : "";

		echo "<li class='li_list'><a href='tutorials?tutorial=$file_link'>$label</a>$comment_display</li>\n";
	}
	echo "</ul>\n";
}

// Main logic
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
