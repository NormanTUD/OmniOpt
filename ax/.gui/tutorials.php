<?php
	include("_header_base.php");

	function get_first_heading_content($file_path) {
		// Read the content of the file
		$file_content = file_get_contents($file_path);

		if ($file_content === false) {
			return null; // Return null if the file could not be read
		}

		// Define a regular expression to match the first <h1> to <h6> tag and capture its content
		$heading_pattern = '/<h[1-6][^>]*>(.*?)<\/h[1-6]>/i';

		// Search for the first matching heading tag
		if (preg_match($heading_pattern, $file_content, $matches)) {
			return $matches[1]; // Return the captured content of the first heading tag
		}

		return null; // Return null if no heading tag was found
	}

	if (isset($_GET["tutorial"])) {
		$tutorial_file = $_GET["tutorial"];
		if(preg_match("/^[a-z_]+$/", $tutorial_file)) {
			if(file_exists("tutorials/$tutorial_file.php")) {
				$tutorial_file = "$tutorial_file.php";
			}
		}

		if (preg_match("/^[a-z_]+\.php$/", $tutorial_file) && file_exists("tutorials/$tutorial_file")) {
			$load_file = "tutorials/$tutorial_file";
			include($load_file);
		} else {
			echo "Invalid file: $tutorial_file";
		}
	} else {
?>
		<h1>Tutorials</h1>

		<p>Available tutorials/help files:</p>

		<ul>
<?php
		$files = scandir('tutorials/');
		foreach($files as $file) {
			if($file != ".." && $file != "." && $file != "favicon.ico" and preg_match("/\.php/", $file)) {
				$name = $file;

				$heading_content = get_first_heading_content("tutorials/$file");

				if ($heading_content !== null) {
					$name = $heading_content;
				}

				$file = preg_replace("/\.php$/", "", $file);

				print "<li class='li_list'><a href='tutorials.php?tutorial=$file'>$name</a></li>\n";
			}
		}
?>
		</ul>
<?php
	}
?>
	<script src="<?php print $dir_path; ?>/prism.js"></script>
	<script src="<?php print $dir_path; ?>/footer.js"></script>
</body>
</html>
