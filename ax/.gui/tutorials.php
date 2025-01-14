<?php
	require "_header_base.php";

	if (isset($_GET["tutorial"])) {
		$tutorial_file = $_GET["tutorial"];
		if (preg_match("/^[a-z_]+$/", $tutorial_file)) {
			if (file_exists("tutorials/$tutorial_file.php")) {
				$tutorial_file = "$tutorial_file.php";
			}
		}

		if (preg_match("/^[a-z_]+\.php$/", $tutorial_file) && file_exists("tutorials/$tutorial_file")) {
			$load_file = "tutorials/$tutorial_file";
			include $load_file;
		} else {
			echo "Invalid file: $tutorial_file";
		}
	} else {
?>
		<h1>Tutorials</h1>

		<ul>
<?php
			$files = scandir('tutorials/');
			foreach ($files as $file) {
				if ($file != ".." && $file != "." && $file != "favicon.ico" and preg_match("/\.php/", $file)) {
					$name = $file;

					$file_path = "tutorials/$file";

					$heading_content = get_first_heading_content($file_path);

					if ($heading_content !== null) {
						$name = $heading_content;
					}

					$file = preg_replace("/\.php$/", "", $file);

					$comment = "";
					$_comment = get_html_comment($file_path);

					if($_comment) {
						$comment = " &mdash; $_comment";
					}


					print "<li class='li_list'><a href='tutorials.php?tutorial=$file'>$name</a>$comment</li>\n";
				}
			}
?>
		</ul>
<?php
	}
?>
	</div>
<?php
	include("footer.php");
?>
