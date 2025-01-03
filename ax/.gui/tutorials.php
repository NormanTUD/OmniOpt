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

		<p>Available tutorials/help files:</p>

		<ul>
<?php
			$files = scandir('tutorials/');
			foreach ($files as $file) {
				if ($file != ".." && $file != "." && $file != "favicon.ico" and preg_match("/\.php/", $file)) {
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
	</div>
</body>
</html>
