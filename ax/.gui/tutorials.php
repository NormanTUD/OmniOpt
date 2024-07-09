<?php
	include("_header_base.php");

	if (isset($_GET["tutorial"])) {
		$tutorial_file = $_GET["tutorial"];
		if(preg_match("/^[a-z_]+$/", $tutorial_file)) {
			if(file_exists("tutorials/$tutorial_file.php")) {
				$tutorial_file = "$tutorial_file.php";
			}
		}

		if (preg_match("/^[a-z_]+\.php$/", $tutorial_file) && file_exists("tutorials/$tutorial_file")) {
			include("tutorials/$tutorial_file");
		} else {
			echo "Invalid file: $tutorial_file";
		}
	} else {
?>
		<h1>Tutorials</h1>

		<p>Available tutorials:</p>

		<ul>
<?php
		$files = scandir('tutorials/');
		foreach($files as $file) {
			if($file != ".." && $file != ".") {
				print "<ul><a href='tutorials.php?tutorial=$file'>$file</a></ul>\n";
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
