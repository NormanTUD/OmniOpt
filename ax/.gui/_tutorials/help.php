<div id="toc"></div>

<?php
	function parse_arguments($file_path) {
		$groups = [];
		$current_group = "Ungrouped"; // Default group if no add_argument_group is found

		// Regex patterns
		$pattern_group = "/add_argument_group\(['\"](.+?)['\"],\s*['\"](.+?)['\"]/";
		$pattern_argument = "/add_argument\(\s*['\"]([^'\"]+)['\"](?:,\s*[^)]*help=['\"]([^'\"]+)['\"])?(?:,\s*type=([\w]+))?(?:,\s*default=([^,\)]+))?/";

		// Check if file exists
		if (!file_exists($file_path)) {
			echo "<p><strong>ERROR:</strong> File not found: $file_path</p>";
			return [];
		}

		#echo "<p>Loading file: <code>$file_path</code></p>";

		// Read the file
		$file = file($file_path, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES);

		if (!$file) {
			echo "<p><strong>ERROR:</strong> Unable to read file.</p>";
			return [];
		}

		#echo "<p>File loaded successfully. Parsing lines...</p>";

		foreach ($file as $line) {
			// Detect argument groups
			if (preg_match($pattern_group, $line, $matches)) {
				$current_group = trim($matches[2]); // Use full group description
				$groups[$current_group] = []; // Ensure the group is initialized
				#echo "<p>Detected group: <strong>$current_group</strong></p>";
			}

			// Detect arguments
			if (preg_match($pattern_argument, $line, $matches)) {
				$arg_name = trim($matches[1]);
				$description = isset($matches[2]) ? trim($matches[2]) : "No description available.";
				$default = isset($matches[4]) ? trim($matches[4], "\"'") : "-";

				$groups[$current_group][] = [$arg_name, $description, $default];

				#echo "<p>Detected argument: <code>$arg_name</code> (Group: <strong>$current_group</strong>)</p>";
			}
		}

		if (empty($groups)) {
			echo "<p><strong>WARNING:</strong> No arguments detected.</p>";
		}

		return $groups;
	}

	function generate_html_table($arguments) {
		if (empty($arguments)) {
			return "<p><strong>No arguments found.</strong></p>";
		}

		$html = "<h2>Available Parameters (--help)</h2>\n";
		$html .= "<table border='1'>\n<thead>\n<tr class='invert_in_dark_mode'>\n<th>Parameter</th>\n<th>Description</th>\n<th>Default Value</th>\n</tr>\n</thead>\n<tbody>\n";

		foreach ($arguments as $group => $args) {
			if (!empty($args)) {
				$html .= "<tr class='section-header invert_in_dark_mode'>\n<td colspan='3'><strong>$group</strong></td>\n</tr>\n";
				foreach ($args as [$name, $desc, $default]) {
					$html .= "<tr>\n<td><samp>$name</samp></td>\n<td>$desc</td>\n<td><samp>$default</samp></td>\n</tr>\n";
				}
			}
		}

		$html .= "</tbody>\n</table>";
		return $html;
	}

	$file_path = "../.omniopt.py";
	$arguments = parse_arguments($file_path);
	echo generate_html_table($arguments);
?>
