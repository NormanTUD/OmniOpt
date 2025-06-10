<?php
	chdir(__DIR__);

	$GLOBALS["main_script_dir"] = null;

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

	function get_html_category_comment($file_path) {
		$file_content = file_get_contents($file_path);

		if ($file_content === false) {
			return null;
		}

		$heading_pattern = '/<!-- Category: \s*(.*?)\s*-->/i';

		if (preg_match($heading_pattern, $file_content, $matches)) {
			return $matches[1];
		}

		return null;
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
		if (!file_exists($file_path)) {
			return null;
		}

		$file_handle = fopen($file_path, 'r');

		if ($file_handle === false) {
			return null;
		}

		$heading_pattern = '/<h[1-6][^>]*>(.*?)<\/h[1-6]>/i';
		$markdown_heading_pattern = '/^#{1,6}\s*(.*?)\s*$/m';

		while (($line = fgets($file_handle)) !== false) {
			if (preg_match($heading_pattern, $line, $matches)) {
				fclose($file_handle);
				return $matches[1];
			}

			if (preg_match($markdown_heading_pattern, $line, $matches)) {
				fclose($file_handle);
				return $matches[1];
			}
		}

		fclose($file_handle);
		return null;
	}

	function highlight_backticks($text) {
		return preg_replace_callback('/`([^`]+)`/', function ($matches) {
			return '<tt>' . htmlspecialchars($matches[1]) . '</tt>';
		}, $text);
	}

	function convert_markdown_to_html($markdown) {
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

		$markdown = preg_replace('/\[(.*?)\]\((https?:\/\/.*?)\)/', '<a href="$2" target="_blank">$1</a>', $markdown);
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

		$markdown = preg_replace('/<!--.*?-->/s', '', $markdown);

		$markdown = preg_replace('/\n\s*\n/', "\n", $markdown);

		$markdown = preg_replace('/(<br\s*\/?>\s*)+/', "<br />\n", $markdown);;

		$markdown = preg_replace('/(<(?:h[1-6]|ul|ol|li)[^>]*>.*?<\/(?:h[1-6]|ul|ol|li)>)(\s*<br\s*\/?>)+/is', '$1', $markdown);

		$pattern = '#<br\s*/?\s*>(?=\s*<h[1-6]\b)#i';

		$markdown = preg_replace($pattern, '', $markdown);

		return $markdown;
	}

	function convert_file_to_html($filePath) {
		if (!file_exists($filePath)) {
			echo "File not found: " . $filePath;
			return;
		}

		$markdownContent = file_get_contents($filePath);
		$htmlContent = convert_markdown_to_html($markdownContent);

		echo $htmlContent;
	}

	function replace_python_placeholders($input, $replacements) {
		if (!is_string($input)) {
			throw new InvalidArgumentException("Input must be a string.");
		}

		if (!is_array($replacements)) {
			throw new InvalidArgumentException("Replacements must be an associative array.");
		}

		$result = preg_replace_callback('/\{([^\{\}]+)\}/', function ($matches) use ($replacements) {
			$key = $matches[1];

			if (array_key_exists($key, $replacements)) {
				return $replacements[$key];
			}

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
		if (preg_match("/\.py$/", $file_path)) {
			$groups = [];
			$current_group = "Ungrouped";

			$pattern_group = "/(\w+)\s*=\s*(?:self\.)?parser\.add_argument_group\(\s*'(.+?)',\s*'(.+?)'/";
			$pattern_argument = "/\.add_argument\(\s*'([^']+)'(.*)\)/";

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
				"joined_valid_acquisition_classes" => extract_and_join_python_list($file_path, "VALID_ACQUISITION_CLASSES"),
				"joined_valid_occ_types" => extract_and_join_python_list($file_path, "valid_occ_types"),
				"joined_supported_models" => extract_and_join_python_list($file_path, "SUPPORTED_MODELS")
			);

			$group_vars = [];

			foreach ($file as $line) {
				if (preg_match($pattern_group, $line, $matches)) {
					$group_var = trim($matches[1]);
					$group_name = trim($matches[2]);
					$group_desc = trim($matches[3]);

					$group_vars[$group_var] = $group_name;
					$groups[$group_name] = ["desc" => $group_desc, "args" => []];
				} elseif (preg_match($pattern_argument, $line, $matches)) {
					$arg_name = trim($matches[1]);
					$raw_params = trim($matches[2]);

					$description = "<i style='color: red;'>No description available.</i>";
					$default = "";
					$type = "";
					$action = "";

					preg_match_all("/(\w+)\s*=\s*f?(['\"])(.*?)\\2/s", $raw_params, $param_matches, PREG_SET_ORDER);

					foreach ($param_matches as $pm) {
						$key = $pm[1];
						$value = trim($pm[3]);

						if ($key === "help") {
							$description = htmlentities($value);
							if (strpos($raw_params, "f'") !== false) {
								$description = replace_python_placeholders($description, $replacements);
							}
						} elseif ($key === "default") {
							$default = $value;
							if (is_numeric($default)) {
								$default = (int)$default;
							} elseif ($default === "Path.home(") {
								$default = "\$HOME";
							} elseif ($default === "None") {
								$default = "";
							}
						} elseif ($key === "type") {
							$type = $value;
						} elseif ($key === "action") {
							$action = $value;
						}
					}

					if (preg_match("/default\s*=\s*(\d+)/", $raw_params, $default_match)) {
						$default = (int)$default_match[1];
					}

					if (preg_match("/type\s*=\s*([\w\.]+)/", $raw_params, $type_match)) {
						$type = $type_match[1];
					}

					if ($action !== "") {
						if ($action == "store_false") {
							$default = "True";
							$type = "bool";
						}

						if ($action == "store_true") {
							$default = "False";
							$type = "bool";
						}
					}

					$arg_entry = [$arg_name, $description, $default];
					if ($type !== "") {
						$arg_entry[] = "type: $type";
					}
					if ($action !== "") {
						$arg_entry[] = "action: $action";
					}

					if (count($group_vars)) {
						foreach (array_reverse($group_vars) as $var => $group) {
							if (strpos($line, $var . ".add_argument") !== false) {
								$groups[$group]["args"][] = $arg_entry;
								break;
							}
						}
					} else {
						$groups["Ungrouped"]["args"][] = $arg_entry;
					}
				}
			}

			return $groups;
		} else {
			// Bash help parser
			$groups = [];
			$help_found = false;

			if (!file_exists($file_path)) {
				echo "<p><strong>ERROR:</strong> File not found: $file_path</p>";
				return [];
			}

			$file = file($file_path, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES);
			if (!$file) {
				echo "<p><strong>ERROR:</strong> Unable to read file.</p>";
				return [];
			}

			$in_help_function = false;
			$args = [];

			foreach ($file as $line) {
				$line = trim($line);

				if (preg_match("/^function\s+help\s*\(\)\s*{/", $line) || preg_match("/^help\s*\(\)\s*{/", $line)) {
					$in_help_function = true;
					$help_found = true;
					continue;
				}
				if ($in_help_function && preg_match("/^}/", $line)) {
					$in_help_function = false;
					break;
				}

				if ($in_help_function && preg_match('/echo\s+"?\s*--([a-zA-Z0-9_\-]+)(=[^)\s]+)?\s+(.*?)"?$/', $line, $matches)) {
					$arg_name = "--" . $matches[1];
					$default = "";
					$type = "";

					if (!empty($matches[2])) {
						$equal_part = trim($matches[2], "=() ");
						if (preg_match('/0\|1/i', $equal_part)) {
							$type = "0 or 1";
						} elseif (preg_match('/INT|NR|NUM|[0-9]/i', $equal_part)) {
							$type = "int";
						} elseif (preg_match('/FLOAT/i', $equal_part)) {
							$type = "float";
						} elseif (preg_match('/STR|string/i', $equal_part)) {
							$type = "str";
						}
					} else {
						$type = "bool";
						$default = "False";
					}

					$description = htmlentities(trim($matches[3]));

					$args[] = [$arg_name, $description, $default, ($type !== "" ? "type: $type" : "")];
				}
			}

			if (!$help_found || count($args) === 0) {
				return [];
			}

			$groups["Ungrouped"] = [
				"desc" => "Extracted from <tt>--help</tt>",
				"args" => $args
			];

			return $groups;
		}
	}

	function generate_argparse_html_table($arguments, $no_msg_when_empty) {
		if (empty($arguments)) {
			if($no_msg_when_empty) {
				return "";
			}

			return "<p><strong>No arguments found.</strong></p>";
		}

		$html = "<table>\n<thead>\n<tr class='invert_in_dark_mode'>\n";
		$html .= "<th>Parameter</th>\n<th>Description</th>\n<th>Type</th>\n<th>Action</th>\n<th>Default Value</th>\n";
		$html .= "</tr>\n</thead>\n<tbody>\n";

		foreach ($arguments as $group => $data) {
			if (!empty($data["args"])) {
				$desc = "";
				if (isset($data["desc"])) {
					$desc =  " - {$data['desc']}";
				}

				$html .= "<tr class='section-header invert_in_dark_mode'>\n";
				$html .= "<td colspan='5'><strong>$group</strong>$desc</td>\n</tr>\n";

				foreach ($data["args"] as $arg_entry) {
					$name = $arg_entry[0];
					$description = $arg_entry[1];
					$default = $arg_entry[2];

					$type = "";
					$action = "";

					for ($i = 3; $i < count($arg_entry); $i++) {
						if (strpos($arg_entry[$i], "type:") === 0) {
							$type = trim(substr($arg_entry[$i], strlen("type:")));
						} elseif (strpos($arg_entry[$i], "action:") === 0) {
							$action = trim(substr($arg_entry[$i], strlen("action:")));
						}
					}

					$html .= "<tr>\n<td>\n";
					$html .= "<pre class='invert_in_dark_mode'><code class='language-bash'>$name</code></pre>\n";
					$html .= "</td>\n";

					$html .= "<td>$description</td>\n";

					$html .= "<td>";
					if ($type !== "") {
						$html .= "<pre class='invert_in_dark_mode'><code class='language-python'>$type</code></pre>";
					}
					$html .= "</td>\n";

					$html .= "<td>";
					if ($action !== "") {
						$html .= "<pre class='invert_in_dark_mode'><code class='language-python'>$action</code></pre>";
					}
					$html .= "</td>\n";

					$html .= "<td>";
					if ($default !== "") {
						$html .= "<pre class='invert_in_dark_mode'><code class='language-bash'>$default</code></pre>";
					}
					$html .= "</td>\n";

					$html .= "</tr>\n";
				}
			}
		}

		$html .= "</tbody>\n</table>";
		return $html;
	}

	function parse_arguments_and_print_html_table ($file_path, $no_msg_when_empty = 0) {
		$arguments = parse_arguments($file_path);
		echo generate_argparse_html_table($arguments, $no_msg_when_empty);
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
