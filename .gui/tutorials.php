<?php
	require "_header_base.php";

	function convertMarkdownToHtml($markdown) {
		$markdown = preg_replace('/^###### (.*)$/m', '<h6>$1</h6>', $markdown);
		$markdown = preg_replace('/^##### (.*)$/m', '<h5>$1</h5>', $markdown);
		$markdown = preg_replace('/^#### (.*)$/m', '<h4>$1</h4>', $markdown);
		$markdown = preg_replace('/^### (.*)$/m', '<h3>$1</h3>', $markdown);
		$markdown = preg_replace('/^## (.*)$/m', '<h2>$1</h2>', $markdown);
		$markdown = preg_replace('/^# (.*)$/m', '<h1>$1</h1>', $markdown);

		$markdown = preg_replace('/\n\n/', '</p><p>', $markdown);
		$markdown = preg_replace('/^(?!<h[1-6]>)(.*)$/m', '<p>$1</p>', $markdown);

		$markdown = preg_replace('/\*\*(.*?)\*\*/', '<strong>$1</strong>', $markdown);
		$markdown = preg_replace('/__(.*?)__/', '<strong>$1</strong>', $markdown);

		$markdown = preg_replace('/\*(.*?)\*/', '<em>$1</em>', $markdown);
		$markdown = preg_replace('/_(.*?)_/', '<em>$1</em>', $markdown);

		$markdown = preg_replace('/\[(.*?)\]\((.*?)\)/', '<a href="$2">$1</a>', $markdown);

		$markdown = preg_replace('/!\[(.*?)\]\((.*?)\)/', '<img src="$2" alt="$1">', $markdown);

		$markdown = preg_replace('/`(.*?)`/', '<code>$1</code>', $markdown);

		$markdown = preg_replace('/```(.*?)```/s', '<pre><code>$1</code></pre>', $markdown);

		$markdown = preg_replace('/^\* (.*)$/m', '<ul><li>$1</li></ul>', $markdown);
		$markdown = preg_replace('/^- (.*)$/m', '<ul><li>$1</li></ul>', $markdown);

		$markdown = preg_replace('/^\d+\. (.*)$/m', '<ol><li>$1</li></ol>', $markdown);

		$markdown = preg_replace('/  \n/', '<br>', $markdown);

		return $markdown;
	}

	function convertFileToHtml($filePath) {
		if (!file_exists($filePath)) {
			echo "Die Datei wurde nicht gefunden: " . $filePath;
			return;
		}

		$markdownContent = file_get_contents($filePath);
		$htmlContent = convertMarkdownToHtml($markdownContent);

		echo $htmlContent;
	}

	if (isset($_GET["tutorial"])) {
		$tutorial_file = $_GET["tutorial"];
		if (preg_match("/^[a-z_]+$/", $tutorial_file)) {
			if (file_exists("_tutorials/$tutorial_file.php")) {
				$tutorial_file = "$tutorial_file.php";
			} else if (file_exists("_tutorials/$tutorial_file.md")) {
				$tutorial_file = "$tutorial_file.md";
			}
		}

		if (preg_match("/^[a-z_]+\.php$/", $tutorial_file) && file_exists("_tutorials/$tutorial_file")) {
			$load_file = "_tutorials/$tutorial_file";
			include($load_file);
		} else if (preg_match("/^[a-z_]+\.md$/", $tutorial_file) && file_exists("_tutorials/$tutorial_file")) {
			convertFileToHtml($tutorial_file);
		} else {
			echo "Invalid file: $tutorial_file";
		}
	} else {
?>
		<h1>Tutorials</h1>

		<ul>
<?php
			$files = scandir('_tutorials/');
			foreach ($files as $file) {
				if ($file != ".." && $file != "." && $file != "favicon.ico" and preg_match("/\.php/", $file)) {
					$name = $file;

					$file_path = "_tutorials/$file";

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


					print "<li class='li_list'><a href='tutorials?tutorial=$file'>$name</a>$comment</li>\n";
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
