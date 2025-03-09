<div id="toc"></div>

<!-- Which python-files exists, which functions do they have and which types? -->
<h2 id="functions_and_types">Functions and types</h2>

<p>This is probably not interesting to anyone who is not planning on editing the source code of OmniOpt2.</p>

<?php
	$directory = "..";  // Parent directory
	$files = scandir($directory);

	foreach ($files as $file) {
		if (preg_match('/^\.[^\/]+\.py$/', $file)) { // Only .py files starting with "."
			$filepath = "$directory/$file";
			$content = file_get_contents($filepath);

			// Find function definitions
			preg_match_all('/^def\s+(\w+)\s*\(([^)]*)\)\s*(->\s*([\w\[\], ]+))?:/m', $content, $matches, PREG_SET_ORDER);

			echo "<h3>Functions in " . htmlentities($file) . "</h3>";
			echo "<table border='1'>";
			echo "<tr><th>Function Name</th><th>Parameters</th><th>Return Type</th></tr>";

			foreach ($matches as $match) {
				$func_name = htmlentities($match[1]);
				$params_raw = trim($match[2]);
				$return_type = htmlentities($match[4] ?? 'unknown');

				// Format parameters
				if ($params_raw === '') {
					$params = '<i>None</i>';
				} else {
					$param_list = explode(',', $params_raw);
					$param_list = array_map('trim', $param_list);

					if (count($param_list) > 1) {
						$params = "<ul>";
						foreach ($param_list as $param) {
							$params .= "<li><pre>" . htmlentities($param) . "</pre></li>";
						}
						$params .= "</ul>";
					} else {
						$params = "<pre>" . htmlentities($param_list[0]) . "</pre>";
					}
				}

				echo "<tr><td><pre>$func_name</pre></td><td>$params</td><td><pre>$return_type</pre></td></tr>";
			}

			echo "</table><br>";
		}
	}
?>
