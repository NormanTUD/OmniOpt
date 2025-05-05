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

		$markdown = preg_replace_callback('/```python\[(.*?)\]?\R*(.*?)```/s', function($matches) {
			$filePath = $matches[2];

			$filePath = preg_replace("/\]/", "", trim($filePath));

			if (file_exists($filePath)) {
				$fileContent = file_get_contents($filePath);
				return '<<<CODEBLOCK_PYTHON>>>'. base64_encode($fileContent) .'<<<CODEBLOCK_PYTHON>>>';
			} else {
				return $matches[0];
			}
		}, $markdown);

		$markdown = preg_replace_callback('/```python\R*(.*?)```/s', function($matches) {
			return '<<<CODEBLOCK_PYTHON>>>'. base64_encode($matches[1]) .'<<<CODEBLOCK_PYTHON>>>';
		}, $markdown);

		$markdown = preg_replace_callback('/```run_php\R*(.*?)```/s', function($matches) {
			return '<<<PHP_RUN_BLOCK>>>'. base64_encode($matches[1]) .'<<<PHP_RUN_BLOCK>>>';
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

		$markdown = preg_replace_callback('/<<<PHP_RUN_BLOCK>>>(.*?)<<<PHP_RUN_BLOCK>>>/s', function($matches) {
			ob_start();

			eval(base64_decode($matches[1]));

			$output = ob_get_clean();

			return $output;
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

	function replace_python_placeholders($input, $replacements) {
		if (!is_string($input)) {
			throw new InvalidArgumentException("Input must be a string.");
		}

		if (!is_array($replacements)) {
			throw new InvalidArgumentException("Replacements must be an associative array.");
		}

		// Nutze preg_replace_callback um alle Platzhalter zu ersetzen
		$result = preg_replace_callback('/\{([^\{\}]+)\}/', function ($matches) use ($replacements) {
			$key = $matches[1];

			if (array_key_exists($key, $replacements)) {
				return $replacements[$key];
			}

			// Wenn kein Ersatz gefunden wurde, gib den Original-Platzhalter zurück
			return $matches[0];
		}, $input);

		if ($result === null) {
			throw new RuntimeException("Regex replacement failed.");
		}

		return $result;
	}

	function extract_and_join_python_list($file_path, $variable_name) {
		if (!is_readable($file_path)) {
			return null;
		}

		$lines = file($file_path, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES);
		if ($lines === false) {
			return null;
		}

		foreach ($lines as $line) {
			$trimmed = trim($line);

			$pattern = '/^' . preg_quote($variable_name, '/') . '\s*:\s*list\s*=\s*(\[.*\])\s*$/';

			if (preg_match($pattern, $trimmed, $matches)) {
				$list_literal = $matches[1];

				// Optional: basic validation on list syntax
				$elements = eval("return $list_literal;");

				if (!is_array($elements)) {
					return null;
				}

				$escaped_elements = array_map(function ($item) {
					return '"' . addslashes($item) . '"';
				}, $elements);

				$joined = implode(', ', $escaped_elements);
				$new_variable = "joined_" . $variable_name;

				return $joined;
			}
		}

		return null;
	}

	function parse_arguments($file_path) {
		$groups = [];
		$current_group = "Ungrouped";

		$pattern_group = "/(\w+)\s*=\s*(?:self\.)?parser\.add_argument_group\(\s*'(.+?)',\s*'(.+?)'/";
		$pattern_argument = "/\.add_argument\(\s*'([^']+)'(?:,\s*[^)]*help=(f?)'([^']+)')?(?:,\s*type=([\w]+))?(?:,\s*default=([^,\)]+))?/";

		if (!file_exists($file_path)) {
			echo "<p><strong>ERROR:</strong> File not found: $file_path</p>";
			return [];
		}

		$file = file($file_path, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES);
		if (!$file) {
			echo "<p><strong>ERROR:</strong> Unable to read file.</p>";
			return [];
		}

		$replacements = array(
			"Path.home()" => "\$HOME",
			"joined_valid_occ_types" => extract_and_join_python_list($file_path, "valid_occ_types"),
			"joined_supported_models" => extract_and_join_python_list($file_path, "SUPPORTED_MODELS")
		);

		$group_vars = [];

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
				$f = $matches[2];
				$description = isset($matches[3]) ? htmlentities(trim($matches[3])) : "<i style='color: red;'>No description available.</i>";
				$default = isset($matches[5]) ? trim($matches[5], "\"'") : "";

				if($default == "Path.home(") {
					$default = "\$HOME";
				}

				if($f == "f") {
					$description = replace_python_placeholders($description, $replacements);
				}

				// Try to detect the last known group (by variable name reference)
				if(count($group_vars)) {
					foreach (array_reverse($group_vars) as $var => $group) {
						if (strpos($line, $var . ".add_argument") !== false) {
							$groups[$group]["args"][] = [$arg_name, $description, $default];
							break;
						}
					}
				} else {
					$groups["Ungrouped"]["args"][] = [$arg_name, $description, $default];
				}
			}
		}

		return $groups;
	}

	function generate_argparse_html_table($arguments) {
		if (empty($arguments)) {
			return "<p><strong>No arguments found.</strong></p>";
		}

		$html = "<table>\n<thead>\n<tr class='invert_in_dark_mode'>\n<th>Parameter</th>\n<th>Description</th>\n<th>Default Value</th>\n</tr>\n</thead>\n<tbody>\n";

		foreach ($arguments as $group => $data) {
			if (!empty($data["args"])) {
				$desc = "";
				if (isset($data["desc"])) {
					$desc =  " - {$data['desc']}";
				}
				$html .= "<tr class='section-header invert_in_dark_mode'>\n<td colspan='3'><strong>$group</strong>$desc</td>\n</tr>\n";
				foreach ($data["args"] as [$name, $desc, $default]) {
					$html .= "<tr>\n<td>\n";
					$html .= "<pre class='invert_in_dark_mode'><code class='language-bash'>$name</code></pre>\n";
					$html .= "</td>\n<td>";
					$html .= "$desc";
					$html .= "</td>\n<td>";
					if($default) {
						$html .= "<pre class='invert_in_dark_mode'><code class='language-bash'>$default</code></pre>\n";
					}
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

	function extract_magic_comment($file_path, $descriptionKey) {
		if (!file_exists($file_path)) {
			return null;
		}

		$pattern = "/#\s*" . preg_quote($descriptionKey, "/") . ":\s*(.*)/";

		foreach (file($file_path) as $line) {
			if (preg_match($pattern, $line, $matches)) {
				return trim($matches[1]);
			}
		}

		return null;
	}

	function extract_help_params_from_bash($file_path, $show_help = 0) {
		if (!file_exists($file_path)) {
			return "File not found: " . htmlspecialchars($file_path);
		}

		$file = fopen($file_path, "r");
		if (!$file) {
			return "Error when trying to open the file $file.";
		}

		$inside_help = false;
		$help_lines = [];

		while (($line = fgets($file)) !== false) {
			$trimmed = trim($line);

			if (preg_match('/^function\s+help\s*{/', $trimmed)) {
				$inside_help = true;
				continue;
			}

			if ($inside_help) {
				if (preg_match('/^\s*exit\s+/', $trimmed)) {
					break;
				}
				if (preg_match('/^\s*echo\s+"(.*?)";?$/', $trimmed, $matches)) {
					$help_lines[] = $matches[1];
				}
			}
		}

		fclose($file);

		$html = "<table><tr><th>Option</th><th>Description</th></tr>";

		foreach ($help_lines as $line) {
			if (preg_match('/^(--[^\s=]+)(?:=([^ ]+))?\s+(.*)$/', trim($line), $parts)) {
				$option = htmlspecialchars($parts[1]);
				$value = isset($parts[2]) ? htmlspecialchars($parts[2]) : "";
				$description = htmlspecialchars($parts[3]);
				if($show_help == 1 && $option == "--help" || $option != "--help") {
					$html .= "<tr><td>{$option}</td><td>{$description}</td></tr>";
				}
			}
		}

		$html .= "</table>";
		return $html;
	}
?>
