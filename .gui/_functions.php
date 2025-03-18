<?php
	chdir(__DIR__);

	if (!function_exists("dier")) {
		function dier($data, $enable_html = 0, $exception = 0) {
			$print = "";

			$print .= "<pre>\n";
			ob_start();
			print_r($data);
			$buffer = ob_get_clean();
			if ($enable_html) {
				$print .= $buffer;
			} else {
				$print .= htmlentities($buffer);
			}
			$print .= "</pre>\n";

			$print .= "Backtrace:\n";
			$print .= "<pre>\n";
			foreach (debug_backtrace() as $trace) {
				$print .= htmlentities(sprintf("\n%s:%s %s", $trace['file'], $trace['line'], $trace['function']));
			}
			$print .= "</pre>\n";

			if (!$exception) {
				print $print;
				exit();
			} else {
				throw new Exception($print);
			}
		}
	}

	function _isCurl(){
		return function_exists('curl_version');
	}

	function getFirstTagName($url) {
		if(!_isCurl()) {
			return "";
		}

		$ch = curl_init();

		curl_setopt($ch, CURLOPT_URL, $url);
		curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
		curl_setopt($ch, CURLOPT_USERAGENT, 'PHP'); // GitHub API benötigt einen User-Agent

		$response = curl_exec($ch);

		if (curl_errno($ch)) {
			curl_close($ch);
			return null; // cURL-Fehler
		}

		$http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
		curl_close($ch);

		if ($http_code == 200) {
			$data = json_decode($response, true);

			if (isset($data[0]['name'])) {
				return $data[0]['name'];
			}
		}

		return null;
	}

	function get_current_tag() {
		$url = "https://api.github.com/repos/NormanTUD/OmniOpt/tags";
		$tagName = getFirstTagName($url);

		return $tagName;
	}

	function get_or_env ($name) {
		$var = isset($_GET[$name]) ? $_GET[$name] : null;

		if(is_null($var) || !strlen($var) && strlen(getenv($name))) {
			$var = getenv($name);
		}

		return $var;
	}

	function get_html_comment($file_path) {
		$file_content = file_get_contents($file_path);

		if ($file_content === false) {
			return null;
		}

		$heading_pattern = '/<!--\s*(.*?)\s*-->/i';

		if (preg_match($heading_pattern, $file_content, $matches)) {
			return $matches[1];
		}

		return null;
	}

	function get_first_heading_content($file_path) {
		$file_content = file_get_contents($file_path);

		if ($file_content === false) {
			return null;
		}

		$heading_pattern = '/<h[1-6][^>]*>(.*?)<\/h[1-6]>/i';

		if (preg_match($heading_pattern, $file_content, $matches)) {
			return $matches[1];
		}

		$markdown_heading_pattern = '/^#{1,6}\s*(.*?)\s*$/m';

		if (preg_match($markdown_heading_pattern, $file_content, $matches)) {
			return $matches[1];
		}

		return null;
	}

	function highlightBackticks($text) {
		return preg_replace_callback('/`([^`]+)`/', function ($matches) {
			return '<tt>' . htmlspecialchars($matches[1]) . '</tt>';
		}, $text);
	}

	function convertMarkdownToHtml($markdown) {
		$markdown = preg_replace_callback('/(?:^[-*] .*(?:\n(?!\n)[-*] .*)*)/m', function ($matches) {
			$items = preg_replace('/^[-*] (.*)$/m', '<li>$1</li>', $matches[0]);
			return "<ul>\n$items\n</ul>";
		}, $markdown);

		$markdown = preg_replace_callback('/```csv\R*(.*?)```/s', function($matches) {
			return '<<<CODEBLOCK_CSV>>>'. base64_encode($matches[1]) .'<<<CODEBLOCK_CSV>>>';
		}, $markdown);

		$markdown = preg_replace_callback('/```json\R*(.*?)```/s', function($matches) {
			return '<<<CODEBLOCK_JSON>>>'. base64_encode($matches[1]) .'<<<CODEBLOCK_JSON>>>';
		}, $markdown);

		$markdown = preg_replace_callback('/```yaml\R*(.*?)```/s', function($matches) {
			return '<<<CODEBLOCK_YAML>>>'. base64_encode($matches[1]) .'<<<CODEBLOCK_YAML>>>';
		}, $markdown);

		$markdown = preg_replace_callback('/```toml\R*(.*?)```/s', function($matches) {
			return '<<<CODEBLOCK_TOML>>>'. base64_encode($matches[1]) .'<<<CODEBLOCK_TOML>>>';
		}, $markdown);

		$markdown = preg_replace_callback('/```python\R*(.*?)```/s', function($matches) {
			return '<<<CODEBLOCK_PYTHON>>>'. base64_encode($matches[1]) .'<<<CODEBLOCK_PYTHON>>>';
		}, $markdown);

		$markdown = preg_replace_callback('/```bash\R*(.*?)```/s', function($matches) {
			return '<<<CODEBLOCK_BASH>>>'. base64_encode($matches[1]) .'<<<CODEBLOCK_BASH>>>';
		}, $markdown);

		$markdown = preg_replace_callback('/```\R*(.*?)```/s', function($matches) {
			return '<<<CODEBLOCK>>>'. base64_encode($matches[1]) .'<<<CODEBLOCK>>>';
		}, $markdown);

		$markdown = preg_replace('/^###### (.*)$/m', "<h6>$1</h6>\n", $markdown);
		$markdown = preg_replace('/^##### (.*)$/m', "<h5>$1</h5>\n", $markdown);
		$markdown = preg_replace('/^#### (.*)$/m', "<h4>$1</h4>\n", $markdown);
		$markdown = preg_replace('/^### (.*)$/m', "<h3>$1</h3>\n", $markdown);
		$markdown = preg_replace('/^## (.*)$/m', "<h2>$1</h2>\n", $markdown);
		$markdown = preg_replace('/^# (.*)$/m', "<h1>$1</h1>\n", $markdown);

		$markdown = preg_replace('/\n\n/', "<br>\n", $markdown);

		#$markdown = preg_replace('/^(?!<h[1-6]>)(?!<pre>)(?!<code>)(.*)$/m', '<p>$1</p>', $markdown);

		$markdown = preg_replace('/\*\*(.*?)\*\*/', '<strong>$1</strong>', $markdown);
		$markdown = preg_replace('/__(.*?)__/', '<strong>$1</strong>', $markdown);
		$markdown = preg_replace('/\*(.*?)\*/', "<em>$1</em>\n", $markdown);

		$markdown = preg_replace('/\[(.*?)\]\((.*?)\)/', '<a href="$2">$1</a>', $markdown);
		$markdown = preg_replace('/!\[(.*?)\]\((.*?)\)/', '<img src="$2" alt="$1">', $markdown);

		$markdown = preg_replace('/`(.*?)`/', "<span class='invert_in_dark_mode'><code class='language-bash'>$1</code></span>\n", $markdown);

		$markdown = preg_replace_callback('/<<<CODEBLOCK>>>(.*?)<<<CODEBLOCK>>>/s', function($matches) {
			return "<pre class='invert_in_dark_mode'><code>" . htmlentities(base64_decode($matches[1])) . "</code></pre>\n";
		}, $markdown);

		$markdown = preg_replace_callback('/<<<CODEBLOCK_YAML>>>(.*?)<<<CODEBLOCK_YAML>>>/s', function($matches) {
			return "<pre class='invert_in_dark_mode'><code class='language-yaml'>" . htmlentities(base64_decode($matches[1])) . "</code></pre>\n";
		}, $markdown);

		$markdown = preg_replace_callback('/<<<CODEBLOCK_CSV>>>(.*?)<<<CODEBLOCK_CSV>>>/s', function($matches) {
			return "<pre class='invert_in_dark_mode'><code class='language-csv'>" . htmlentities(base64_decode($matches[1])) . "</code></pre>\n";
		}, $markdown);

		$markdown = preg_replace_callback('/<<<CODEBLOCK_JSON>>>(.*?)<<<CODEBLOCK_JSON>>>/s', function($matches) {
			return "<pre class='invert_in_dark_mode'><code class='language-json'>" . htmlentities(base64_decode($matches[1])) . "</code></pre>\n";
		}, $markdown);

		$markdown = preg_replace_callback('/<<<CODEBLOCK_TOML>>>(.*?)<<<CODEBLOCK_TOML>>>/s', function($matches) {
			return "<pre class='invert_in_dark_mode'><code class='language-toml'>" . htmlentities(base64_decode($matches[1])) . "</code></pre>\n";
		}, $markdown);

		$markdown = preg_replace_callback('/<<<CODEBLOCK_PYTHON>>>(.*?)<<<CODEBLOCK_PYTHON>>>/s', function($matches) {
			return "<pre class='invert_in_dark_mode'><code class='language-python'>" . htmlentities(base64_decode($matches[1])) . "</code></pre>\n";
		}, $markdown);

		$markdown = preg_replace_callback('/<<<CODEBLOCK_BASH>>>(.*?)<<<CODEBLOCK_BASH>>>/s', function($matches) {
			return "<pre class='invert_in_dark_mode'><code class='language-bash'>" . htmlentities(base64_decode($matches[1])) . "</code></pre>\n";
		}, $markdown);

		$markdown = preg_replace('/^\d+\. (.*)$/m', "<ol><li>$1</li></ol>\n", $markdown);

		$markdown = preg_replace('/  \n/', '<br>', $markdown);

		return $markdown;
	}

	function convertFileToHtml($filePath) {
		if (!file_exists($filePath)) {
			echo "File not found: " . $filePath;
			return;
		}

		$markdownContent = file_get_contents($filePath);
		$htmlContent = convertMarkdownToHtml($markdownContent);

		echo $htmlContent;
	}

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

	function generate_argparse_html_table($arguments) {
		if (empty($arguments)) {
			return "<p><strong>No arguments found.</strong></p>";
		}

		$html = "<table border='1'>\n<thead>\n<tr class='invert_in_dark_mode'>\n<th>Parameter</th>\n<th>Description</th>\n<th>Default Value</th>\n</tr>\n</thead>\n<tbody>\n";

		foreach ($arguments as $group => $data) {
			if (!empty($data["args"])) {
				$html .= "<tr class='section-header invert_in_dark_mode'>\n<td colspan='3'><strong>$group</strong> - {$data['desc']}</td>\n</tr>\n";
				foreach ($data["args"] as [$name, $desc, $default]) {
					$html .= "<tr>\n<td>\n";
					$html .= "<pre class='invert_in_dark_mode'><code class='language-bash'>$name</code></pre>\n";
					$html .= "</td>\n<td>";
					$html .= "$desc";
					$html .= "</td>\n<td>";
					$html .= "<pre class='invert_in_dark_mode'><code class='language-bash'>$default</code></pre>\n";
					$html .= "</td>\n</tr>\n";
				}
			}
		}

		$html .= "</tbody>\n</table>";
		return $html;
	}

	function parse_arguments_and_print_html_table ($file_path) {
		$arguments = parse_arguments($file_path);
		echo generate_argparse_html_table($arguments);
	}
?>
