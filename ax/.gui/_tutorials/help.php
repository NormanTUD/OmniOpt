<div id="toc"></div>

<?php
	function parse_arguments($file_path) {
		$groups = [];
		$current_group = "Ungrouped"; // Default group if no add_argument_group is found

		// Regex patterns
		$pattern_group = "/(\w+)\s*=\s*self\.parser\.add_argument_group\(\s*['\"](.+?)['\"],\s*['\"](.+?)['\"]/";
		$pattern_argument = "/\.add_argument\(\s*['\"]([^'\"]+)['\"](?:,\s*[^)]*help=['\"]([^'\"]+)['\"])?(?:,\s*type=([\w]+))?(?:,\s*default=([^,\)]+))?/";

		// Check if file exists
		if (!file_exists($file_path)) {
			echo "<p><strong>ERROR:</strong> File not found: $file_path</p>";
			return [];
		}

		// Read the file
		$file = file($file_path, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES);
		if (!$file) {
			echo "<p><strong>ERROR:</strong> Unable to read file.</p>";
			return [];
		}

		$group_vars = []; // Stores variable names linked to group names

		foreach ($file as $line) {
			// Detect argument groups
			if (preg_match($pattern_group, $line, $matches)) {
				$group_var = trim($matches[1]);  // The variable name used for the group
				$group_name = trim($matches[2]); // Short name (e.g., "Required")
				$group_desc = trim($matches[3]); // Full description

				$group_vars[$group_var] = $group_name; // Map variable to group name
				$groups[$group_name] = ["desc" => $group_desc, "args" => []];

			} elseif (preg_match($pattern_argument, $line, $matches)) {
				$arg_name = trim($matches[1]);
				$description = isset($matches[2]) ? trim($matches[2]) : "No description available.";
				$default = isset($matches[4]) ? trim($matches[4], "\"'") : "-";

				// Try to detect the last known group (by variable name reference)
				foreach (array_reverse($group_vars) as $var => $group) {
					if (strpos($line, $var . ".add_argument") !== false) {
						$groups[$group]["args"][] = [$arg_name, htmlentities($description), $default];
						break;
					}
				}
			}
		}

		return $groups;
	}

	function generate_html_table($arguments) {
		if (empty($arguments)) {
			return "<p><strong>No arguments found.</strong></p>";
		}

		$html = "<h2>Available Parameters (--help)</h2>\n";
		$html .= "<table border='1'>\n<thead>\n<tr class='invert_in_dark_mode'>\n<th>Parameter</th>\n<th>Description</th>\n<th>Default Value</th>\n</tr>\n</thead>\n<tbody>\n";

		foreach ($arguments as $group => $data) {
			if (!empty($data["args"])) {
				$html .= "<tr class='section-header invert_in_dark_mode'>\n<td colspan='3'><strong>$group</strong> - {$data['desc']}</td>\n</tr>\n";
				foreach ($data["args"] as [$name, $desc, $default]) {
					$html .= "<tr>\n<td><pre>$name</pre></td>\n<td>$desc</td>\n<td><pre>$default</pre></td>\n</tr>\n";
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
