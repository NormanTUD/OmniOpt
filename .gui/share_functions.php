<?php
	error_reporting(E_ALL);
	set_error_handler(
		function ($severity, $message, $file, $line) {
			throw new \ErrorException($message, $severity, $severity, $file, $line);
		}
	);

	$GLOBALS["modificationCache"] = [];
	$GLOBALS["recursiveModificationCache"] = [];
	$GLOBALS["ascii_or_utf8_cache"] = [];

	require_once 'libs/AnsiConverter/Theme/Theme.php';
	require_once 'libs/AnsiConverter/AnsiToHtmlConverter.php';
	require_once '_functions.php';
	require_once 'results_analyzer.php';

	ini_set('display_errors', 1);

	$GLOBALS["sharesPath"] = "shares/";

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
				$print .= my_htmlentities($buffer);
			}
			$print .= "</pre>\n";

			$print .= "Backtrace:\n";
			$print .= "<pre>\n";
			foreach (debug_backtrace() as $trace) {
				$file = array_key_exists('file', $trace) ? $trace['file'] : '[internal function]';
				$line = array_key_exists('line', $trace) ? $trace['line'] : '?';
				$function = array_key_exists('function', $trace) ? $trace['function'] : '[unknown]';
				$print .= my_htmlentities(sprintf("\n%s:%s %s", $file, $line, $function));
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

	function respond_with_error($error_message) {
		header("HTTP/1.1 500 Internal Server Error");
		header('Content-Type: application/json');

		print json_encode(array("error" => $error_message));
		exit(1);
	}

	function validate_param($param_name, $pattern, $error_message) {
		$value = get_get($param_name);
		if (!preg_match($pattern, $value)) {
			throw new Exception($error_message);
		}
		return $value;
	}

	function validate_directory($dir_path) {
		if (!is_dir($dir_path)) {
			throw new Exception("$dir_path not found");
		}
	}

	function utf8ize($d) {
		if (is_array($d)) {
			foreach ($d as $k => $v) {
				$d[$k] = utf8ize($v);
			}
		} else if (is_string($d)) {
			return mb_convert_encoding($d, 'UTF-8', 'auto');
		}
		return $d;
	}

	function respond_with_json($data) {
		header('Content-Type: application/json');

		$hash = hash("md5", json_encode($data));

		$json_data = array(
			"data" => $data,
			"hash" => $hash
		);

		$json_encoded_data = json_encode(utf8ize($json_data));

		print $json_encoded_data;

		exit(0);
	}

	function build_run_folder_path($user_id, $experiment_name, $run_nr) {
		return "$user_id/$experiment_name/$run_nr/";
	}

	function get_get($name, $default = null) {
		if(isset($_GET[$name])) {
			return $_GET[$name];
		}

		return $default;
	}

	function ansi_to_html($string) {
		if(!isset($GLOBALS["ansi_to_html_converter"])) {
			$GLOBALS["ansi_to_html_converter"] = new \SensioLabs\AnsiConverter\AnsiToHtmlConverter;
		}

		$ret = $GLOBALS["ansi_to_html_converter"]->convert($string);

		return html_entity_decode($ret);
	}

	function convert_sixel($output) {
		$command_v_sixel2png = shell_exec('command -v sixel2png');
		$has_sixel2png = is_string($command_v_sixel2png) && trim($command_v_sixel2png) !== '';

		$ESC = "\x1b";
		$start_marker = $ESC . "P";
		$end_marker = $ESC . "\\";

		$length = strlen($output);
		$new_output = '';
		$pos = 0;
		$last_pos = 0;

		while ($pos < $length) {
			$start_pos = strpos($output, $start_marker, $pos);
			if ($start_pos === false) {
				$new_output .= substr($output, $last_pos);
				break;
			}

			$end_pos = strpos($output, $end_marker, $start_pos + 2);
			if ($end_pos === false) {
				$new_output .= substr($output, $last_pos);
				break;
			}

			$end_pos_incl = $end_pos + 2;

			$new_output .= substr($output, $last_pos, $start_pos - $last_pos);

			$sixel_sequence = substr($output, $start_pos, $end_pos_incl - $start_pos);

			$img_html = "<br>";

			if ($has_sixel2png && strlen($sixel_sequence) > 0) {
				$sixel = html_entity_decode($sixel_sequence, ENT_QUOTES | ENT_HTML5, 'UTF-8');

				$tmp_sixel = tempnam(sys_get_temp_dir(), "sixel_") . ".sixel";
				$tmp_png = tempnam(sys_get_temp_dir(), "sixel_") . ".png";

				try {
					$bytes_written = file_put_contents($tmp_sixel, $sixel);
					if ($bytes_written !== false && $bytes_written > 0) {
						$cmd = "sixel2png -i " . escapeshellarg($tmp_sixel) . " -o " . escapeshellarg($tmp_png);
						shell_exec($cmd);

						if (file_exists($tmp_png) && filesize($tmp_png) > 0) {
							clean_black_lines_inplace($tmp_png);
							$data = file_get_contents($tmp_png);
							if ($data !== false && strlen($data) <= 5 * 1024 * 1024) {
								$base64 = base64_encode($data);
								$img_html = '<img src="data:image/png;base64,' . $base64 . '" alt="SIXEL Image"/>';
							}
						}
					}
				} finally {
					if (file_exists($tmp_sixel)) {
						my_unlink($tmp_sixel);
					}
					if (file_exists($tmp_png)) {
						my_unlink($tmp_png);
					}
				}
			}

			$new_output .= $img_html;

			$pos = $end_pos_incl;
			$last_pos = $pos;
		}

		return $new_output;
	}

	function clean_black_lines_inplace($filepath) {
		if (!file_exists($filepath)) {
			error_log("clean_black_lines_inplace: file not found: $filepath");
			return;
		}

		$img = @imagecreatefrompng($filepath);
		if (!$img) {
			error_log("clean_black_lines_inplace: could not load image: $filepath");
			return;
		}

		$width = imagesx($img);
		$height = imagesy($img);
		$new_y = 0;

		$buffer = imagecreatetruecolor($width, $height);
		if (!$buffer) {
			imagedestroy($img);
			error_log("clean_black_lines_inplace: failed to create working buffer");
			return;
		}

		for ($y = 0; $y < $height; $y++) {
			$is_black = true;

			for ($x = 0; $x < $width; $x++) {
				$rgb = imagecolorat($img, $x, $y);
				if (($rgb & 0xFFFFFF) !== 0) {
					$is_black = false;
					break;
				}
			}

			if ($is_black) {
				continue;
			}

			if (!imagecopy($buffer, $img, 0, $new_y, 0, $y, $width, 1)) {
				imagedestroy($img);
				imagedestroy($buffer);
				error_log("clean_black_lines_inplace: failed to copy line $y to buffer row $new_y");
				return;
			}

			$new_y++;
		}

		if ($new_y === 0) {
			imagedestroy($img);
			imagedestroy($buffer);
			error_log("clean_black_lines_inplace: image is fully black");
			return;
		}

		$trimmed = imagecreatetruecolor($width, $new_y);
		if (!$trimmed) {
			imagedestroy($img);
			imagedestroy($buffer);
			error_log("clean_black_lines_inplace: failed to create trimmed image");
			return;
		}

		if (!imagecopy($trimmed, $buffer, 0, 0, 0, 0, $width, $new_y)) {
			imagedestroy($img);
			imagedestroy($buffer);
			imagedestroy($trimmed);
			error_log("clean_black_lines_inplace: failed to copy from buffer to trimmed image");
			return;
		}

		if (!imagepng($trimmed, $filepath)) {
			error_log("clean_black_lines_inplace: failed to save image: $filepath");
		}

		imagedestroy($img);
		imagedestroy($buffer);
		imagedestroy($trimmed);
	}

	function my_unlink($path) {
		return unlink($path);
	}

	function read_file_as_array($filePath) {
		if (!is_readable($filePath)) {
			return [];
		}

		$contents = file_get_contents($filePath);

		if ($contents === false) {
			return [];
		}

		if (!mb_check_encoding($contents, 'UTF-8')) {
			return [];
		}

		$isAsciiOnly = preg_match('//u', $contents) && !preg_match('/[^\x00-\x7F]/', $contents);

		if (!$isAsciiOnly && !mb_detect_encoding($contents, ['UTF-8'], true)) {
			return [];
		}

		$lines = explode("\n", str_replace("\r", '', $contents));
		$lines = array_filter(array_map('trim', $lines), function ($line) {
			return $line !== '';
		});

		return array_values($lines);
	}

	function get_status_for_results_csv($csvFilePath) {
		if (!file_exists($csvFilePath) || !is_readable($csvFilePath)) {
			return json_encode(["error" => "File not found or not readable"], JSON_PRETTY_PRINT);
		}

		$statuses = [
			"failed" => 0,
			"succeeded" => 0,
			"running" => 0,
			"total" => 0
		];

		$k = 0;

		$status_index = null;

		if (($handle = fopen($csvFilePath, "r")) !== false) {
			while (($data = fgetcsv($handle, 0, ",", "\"", "\\")) !== false) {
				if (count($data) < 3) continue;

				if ($k == 0) {
					$status_index = array_search('trial_status', $data);
				}


				if ($k > 0) {
					$statuses["total"]++;

					$status = strtolower(trim($data[$status_index]));

					if ($status === "completed") {
						$statuses["succeeded"]++;
					} elseif ($status === "failed") {
						$statuses["failed"]++;
					} elseif ($status === "running") {
						$statuses["running"]++;
					}
				}

				$k++;
			}
			fclose($handle);
		}

		return $statuses;
	}

	function copy_id_to_clipboard_string($id, $filename) {
		$filename = basename($filename);

		$str = "<button class='copy_clipboard_button' onclick='copy_to_clipboard_from_id(\"".$id."\")'><img src='i/clipboard.svg' style='height: 1em'> Copy raw data to clipboard</button>\n";
		$str .= "<button onclick='download_as_file(\"".$id."\", \"".my_htmlentities($filename)."\")'><img src='i/download.svg' style='height: 1em'> Download &raquo;".my_htmlentities($filename)."&laquo; as file</button>\n";

		return $str;
	}

	function add_worker_cpu_ram_from_file($tabs, $warnings, $filename, $name, $id) {
		if(is_file($filename) && filesize($filename) && is_ascii_or_utf8($filename)) {
			$worker_info = file_get_contents($filename);
			$min_max_table = extract_min_max_ram_cpu_from_worker_info($worker_info);

			if($min_max_table) {
				$warnings[] = my_htmlentities($filename)." does not contain valid worker info";
				return [$tabs, $warnings];
			}

			$html = $min_max_table;
			$html .= "<button onclick='plot_worker_cpu_ram()' id='plot_worker_cpu_ram_button'>Plot this data (may be slow)</button>\n";
			$html .= '<div class="invert_in_dark_mode" id="cpuRamWorkerChartContainer"></div><br>';
			$html .= copy_id_to_clipboard_string("worker_cpu_ram_pre", $filename);
			$html .= '<pre id="worker_cpu_ram_pre">'.my_htmlentities($worker_info).'</pre>';
			$html .= copy_id_to_clipboard_string("worker_cpu_ram_pre", $filename);

			$svg_icon = get_icon_html("plot.svg");

			$tabs["$svg_icon$name"] = [
				'id' => $id,
				'content' => $html
			];
		} else {
			if(!is_file($filename)) {
				$warnings[] = "$filename does not exist";
			} else if(!filesize($filename)) {
				$warnings[] = "$filename is empty";
			} else if(!is_ascii_or_utf8($filename)) {
				$warnings[] = "$filename is not Ascii or UTF8";
			}
		}

		return [$tabs, $warnings];
	}

	function add_debug_log_from_file($tabs, $warnings, $filename, $name, $id) {
		if(is_file($filename) && filesize($filename)) {
			$output = "<div id='debug_log_spinner' class='spinner'></div>";
			$output .= "<div id='here_debuglogs_go'></div>";

			$svg_icon = get_icon_html("debug.svg");

			$tabs["$svg_icon$name"] = [
				'id' => $id,
				'content' => $output,
				'onclick' => "load_debug_log()"
			];
		} else {
			if(!is_file($filename)) {
				$warnings[] = "$filename does not exist";
			} else if(!filesize($filename)) {
				$warnings[] = "$filename is empty";
			}
		}

		return [$tabs, $warnings];
	}

	function add_cpu_ram_usage_main_worker_from_file($tabs, $warnings, $filename, $name, $id) {
		if(is_file($filename) && filesize($filename) && is_ascii_or_utf8($filename)) {
			$html = "<div class='invert_in_dark_mode' id='mainWorkerCPURAM'></div>";
			$html .= copy_id_to_clipboard_string("pre_$id", $filename);
			$html .= '<pre id="pre_' . $id . '">'.my_htmlentities(remove_ansi_colors(file_get_contents($filename))).'</pre>';
			$html .= copy_id_to_clipboard_string("pre_$id", $filename);

			$csv_contents = get_csv_data_as_array($filename);
			$headers = $csv_contents[0];
			$csv_contents_no_header = $csv_contents;
			array_shift($csv_contents_no_header);

			$csv_json = $csv_contents_no_header;
			$headers_json = $headers;

			$GLOBALS["json_data"]["{$id}_csv_json"] = $csv_json;
			$GLOBALS["json_data"]["{$id}_headers_json"] = $headers_json;

			$svg_icon = get_icon_html("cpu.svg");

			$tabs["$svg_icon$name"] = [
				'id' => $id,
				'content' => $html,
				"onclick" => "plotCPUAndRAMUsage();"
			];
		} else {
			if(!is_file($filename)) {
				$warnings[] = "$filename does not exist";
			} else if(!filesize($filename)) {
				$warnings[] = "$filename is empty";
			} else if(!is_ascii_or_utf8($filename)) {
				$warnings[] = "$filename is neither Ascii nor UTF8";
			}
		}

		return [$tabs, $warnings];
	}

	function add_scatter_3d_plots($tabs, $filename, $name, $id) {
		if(is_file($filename)) {
			$html = "<div class='invert_in_dark_mode' id='plotScatter3d'></div>";

			$svg_icon = get_icon_html("3d_scatter.svg");

			$tabs["$svg_icon$name"] = [
				'id' => $id,
				'content' => $html,
				"onclick" => "plotScatter3d()"
			];
		}

		return $tabs;
	}

	function add_scatter_2d_plots($tabs, $filename, $name, $id) {
		if(is_file($filename)) {
			$html = "<div class='invert_in_dark_mode' id='plotScatter2d'></div>";

			$svg_icon = get_icon_html("scatter.svg");

			$tabs["$svg_icon$name"] = [
				'id' => $id,
				'content' => $html,
				"onclick" => "plotScatter2d();"
			];
		}

		return $tabs;
	}

	function add_parameter_distribution_by_type_plot($tabs, $warnings, $filename, $name, $id) {
		if(is_file($filename) && filesize($filename) && is_ascii_or_utf8($filename)) {
			$html = "<div class='invert_in_dark_mode' id='parameter_by_status_distribution'></div>";

			$svg_icon = get_icon_html("plot.svg");

			$tabs["$svg_icon$name"] = [
				'id' => $id,
				'content' => $html,
				"onclick" => "plotParameterDistributionsByStatus();"
			];
		} else {
			if(!is_file($filename)) {
				$warnings[] = "$filename does not exist";
			} else if(!filesize($filename)) {
				$warnings[] = "$filename is empty";
			} else if(!is_ascii_or_utf8($filename)) {
				$warnings[] = "$filename is not Ascii or UTF8";
			}
		}

		return [$tabs, $warnings];
	}

	function add_worker_usage_plot_from_file($tabs, $warnings, $filename, $name, $id) {
		if(is_file($filename) && filesize($filename) && is_ascii_or_utf8($filename)) {
			$html = "<div class='invert_in_dark_mode' id='workerUsagePlot'></div>";
			$html .= copy_id_to_clipboard_string("pre_$id", $filename);
			$html .= '<pre id="pre_'.$id.'">'.my_htmlentities(remove_ansi_colors(file_get_contents($filename))).'</pre>';
			$html .= copy_id_to_clipboard_string("pre_$id", $filename);

			$csv_contents = get_csv_data_as_array($filename);

			$GLOBALS["json_data"]["{$id}_csv_json"] = $csv_contents;

			$svg_icon = get_icon_html("plot.svg");

			$tabs["$svg_icon$name"] = [
				'id' => $id,
				'content' => $html,
				"onclick" => "plotWorkerUsage();"
			];
		} else {
			if(!is_file($filename)) {
				$warnings[] = "$filename does not exist";
			} else if(!filesize($filename)) {
				$warnings[] = "$filename is empty";
			} else if(!is_ascii_or_utf8($filename)) {
				$warnings[] = "$filename is not Ascii or UTF8";
			}
		}

		return [$tabs, $warnings];
	}

	function add_simple_table_from_ascii_table_file($tabs, $warnings, $filename, $name, $id, $remove_ansi_colors = false) {
		if(is_file($filename) && filesize($filename) > 0 && is_ascii_or_utf8($filename)) {
			$contents = file_get_contents($filename);
			if(!$remove_ansi_colors) {
				$contents = remove_ansi_colors($contents);
			} else {
				$contents = htmlspecialchars($contents);
				$ansi_to_htmled = ansi_to_html($contents);
				$contents = remove_ansi_escape_sequences($ansi_to_htmled);
			}

			if(!$remove_ansi_colors) {
				$contents = my_htmlentities($contents);
			} else {
				$contents = convert_sixel($contents);
			}

			$html_table = ascii_table_to_html($contents);
			$html = $html_table;

			$svg_icon = get_icon_html("table.svg");

			$tabs["$svg_icon$name"] = [
				'id' => $id,
				'content' => $html
			];
		} else {
			if(!is_file($filename)) {
				$warnings[] = "$filename does not exist";
			} else if(!filesize($filename)) {
				$warnings[] = "$filename is empty";
			} else if(!is_ascii_or_utf8($filename)) {
				$warnings[] = "$filename is not Ascii or UTF8";
			}
		}

		return [$tabs, $warnings];
	}

	function is_valid_svg_file(string $filepath): bool {
		if (!is_readable($filepath)) {
			return false;
		}

		$contents = file_get_contents($filepath, false, null, 0, 2 * 1024 * 1024);
		if ($contents === false) {
			return false;
		}

		$contents = preg_replace('/^\xEF\xBB\xBF/', '', $contents);

		if (stripos($contents, '<svg') === false) {
			return false;
		}

		$start = trim(substr($contents, 0, 4096));
		if (!preg_match('/<svg[^>]*xmlns\s*=\s*["\']http:\/\/www\.w3\.org\/2000\/svg["\']/', $start)) {
			return false;
		}

		return true;
	}


	function add_flame_svg_file ($tabs, $warnings, $filename, $name, $id, $remove_ansi_colors = false) {
		if(is_file($filename) && filesize($filename) > 0 && is_valid_svg_file($filename)) {
			$svg_icon = get_icon_html("flame.svg");

			$svg = file_get_contents($filename);
			if (preg_match('/<svg[^>]*\bwidth="(\d+)"[^>]*\bheight="(\d+)"/', $svg, $matches)) {
				$width = intval($matches[1]) + 5;
				$height = intval($matches[2]) + 5;
				$html = "<iframe width='{$width}' height='{$height}' src='get_svg?file=$filename'>Your browser does not support iframes</iframe>";
			} else {
				$html = "<iframe width='100%' height='100%' src='get_svg?file=$filename'>Your browser does not support iframes</iframe>";
			}


			$tabs["$svg_icon$name"] = [
				'id' => $id,
				'content' => $html
			];
		} else {
			if(!is_file($filename)) {
				$warnings[] = "$filename does not exist";
			} else if(!filesize($filename)) {
				$warnings[] = "$filename is empty";
			} else if (!is_valid_svg_file($filename)) {
				$warnings[] = "$filename is not a valid SVG file";
			}
		}

		return [$tabs, $warnings];
	}

	function sanitize_safe_html($html) {
		$allowed_tags = ['img', 'span', 'b', 'i', 'u', 'pre', 'code', 'br'];
		$allowed_attrs = ['src', 'class', 'style'];

		$allowed_css_properties = [
			'color',
			'background-color',
			'font-weight',
			'font-style',
			'text-decoration',
			'font-size',
			'font-family',
			'margin',
			'margin-left',
			'margin-right',
			'padding',
			'padding-left',
			'padding-right',
			'text-align',
			'white-space',
			'display'
		];

		$html = preg_replace_callback(
			'#<\s*(/?)\s*([a-z0-9]+)([^<>]*?)(/?)\s*>#i',
			function ($matches) use ($allowed_tags, $allowed_attrs, $allowed_css_properties) {
				$closing_slash = $matches[1];
				$tag_name = strtolower($matches[2]);
				$attrs_string = $matches[3];
				$self_closing_slash = $matches[4];

				if (!in_array($tag_name, $allowed_tags)) {
					return '';
				}

				if ($closing_slash === '/') {
					return '</' . $tag_name . '>';
				}

				$safe_attrs = '';

				preg_match_all('/([a-z0-9\-\:_]+)\s*=\s*(["\'])(.*?)\2/si', $attrs_string, $attr_matches, PREG_SET_ORDER);
				foreach ($attr_matches as $attr) {
					$attr_name = strtolower($attr[1]);
					$attr_value = $attr[3];

					if (!in_array($attr_name, $allowed_attrs)) {
						continue;
					}

					if ($attr_name === 'src') {
						if (!preg_match('#^(https?|data):#i', $attr_value)) {
							continue;
						}
					}

					if ($attr_name === 'style') {
						$clean_styles = [];

						// Split style string into individual declarations
						$style_parts = explode(';', $attr_value);
						foreach ($style_parts as $style_part) {
							$style_part = trim($style_part);
							if (empty($style_part)) {
								continue;
							}

							if (!preg_match('/^([a-z\-]+)\s*:\s*(.+)$/i', $style_part, $style_match)) {
								continue;
							}

							$css_prop = strtolower(trim($style_match[1]));
							$css_val = trim($style_match[2]);

							// Skip disallowed properties
							if (!in_array($css_prop, $allowed_css_properties)) {
								continue;
							}

							// Reject dangerous values
							if (preg_match('#expression\s*\(|javascript:|url\s*\(\s*[\'"]?\s*javascript:#i', $css_val)) {
								continue;
							}

							$clean_styles[] = $css_prop . ': ' . htmlspecialchars($css_val, ENT_QUOTES);
						}

						if (!empty($clean_styles)) {
							$safe_value = implode('; ', $clean_styles) . ';';
							$safe_attrs .= ' style="' . $safe_value . '"';
						}

						continue;
					}

					$safe_value = htmlspecialchars($attr_value, ENT_QUOTES);
					$safe_attrs .= ' ' . $attr_name . '="' . $safe_value . '"';
				}

				if ($self_closing_slash === '/' || in_array($tag_name, ['br', 'img'])) {
					return '<' . $tag_name . $safe_attrs . ' />';
				}

				return '<' . $tag_name . $safe_attrs . '>';
			},
			$html
		);

		return $html;
	}

	function add_simple_pre_tab_from_file ($tabs, $warnings, $filename, $name, $id, $remove_ansi_colors = false) {
		if(is_file($filename) && filesize($filename) > 0 && is_ascii_or_utf8($filename)) {
			$contents = file_get_contents($filename);

			if(!$remove_ansi_colors) {
				$contents = remove_ansi_colors($contents);
			} else {
				$ansi_to_htmled = ansi_to_html($contents);
				$contents = remove_ansi_escape_sequences($ansi_to_htmled);
			}

			$html = copy_id_to_clipboard_string("simple_pre_tab_$id", $filename);
			if(!$remove_ansi_colors) {
				$contents = my_htmlentities($contents);
			} else {
				$contents = convert_sixel($contents);
			}

			$contents = sanitize_safe_html($contents);

			$html .= "<pre id='simple_pre_tab_$id'>$contents</pre>";

			$html .= copy_id_to_clipboard_string("simple_pre_tab_$id", $filename);

			$svg_icon = get_icon_html("main_log.svg");

			$tabs["$svg_icon$name"] = [
				'id' => $id,
				'content' => $html
			];
		} else {
			if(!is_file($filename)) {
				$warnings[] = "$filename does not exist";
			} else if(!filesize($filename)) {
				$warnings[] = "$filename is empty";
			} else if(!is_ascii_or_utf8($filename)) {
				$warnings[] = "$filename is not Ascii or UTF8";
			}
		}

		return [$tabs, $warnings];
	}

	function add_exit_codes_pie_plot($tabs) {
		$html = '<div class="invert_in_dark_mode" id="plotExitCodesPieChart"></div>';

		$svg_icon = get_icon_html("piechart.svg");

		$tabs["{$svg_icon}Exit-Codes"] = [
			'id' => 'tab_exit_codes_plot',
			'content' => $html,
			"onclick" => "plotExitCodesPieChart();"
		];

		return $tabs;
	}

	function add_violin_plot ($tabs) {
		$html = '<div class="invert_in_dark_mode" id="plotViolin"></div>';

		$svg_icon = get_icon_html("violin.svg");

		$tabs["{$svg_icon}Violin"] = [
			'id' => 'tab_violin',
			'content' => $html,
			"onclick" => "plotViolin();"
		];

		return $tabs;
	}

	function add_result_evolution_tab ($tabs, $warnings, $result_names) {
		if(count($result_names)) {
			if (isset($GLOBALS["json_data"]["tab_results_headers_json"])) {
				$html = '<div class="invert_in_dark_mode" id="plotResultEvolution"></div>';

				$svg_icon = get_icon_html("evolution.svg");

				$tabs["{$svg_icon}Evolution"] = [
					'id' => 'tab_hyperparam_evolution',
					'content' => $html,
					"onclick" => "plotResultEvolution();"
				];
			} else {
				$warnings[] = "tab_results_headers_json not found in global json_data";
			}
		} else {
			$warnings[] = "Not adding evolution tab because no result names could be found";
		}

		return [$tabs, $warnings];
	}

	function add_plot_result_pairs ($tabs) {
		$html = '<div class="invert_in_dark_mode" id="plotResultPairs"></div>';

		$svg_icon = get_icon_html("scatter.svg");

		$tabs["{$svg_icon}Result-Pairs"] = [
			'id' => 'tab_result_pairs',
			'content' => $html,
			"onclick" => "plotResultPairs();"
		];

		return $tabs;
	}

	function add_histogram_plot ($tabs) {
		$html = '<div class="invert_in_dark_mode" id="plotHistogram"></div>';

		$svg_icon = get_icon_html("plot.svg");

		$tabs["{$svg_icon}Histogram"] = [
			'id' => 'tab_histogram',
			'content' => $html,
			"onclick" => "plotHistogram();"
		];

		return $tabs;
	}

	function add_heatmap_plot_tab ($tabs) {
		$explanation = "
    <h1>Correlation Heatmap Explanation</h1>

    <p>
        This is a heatmap that visualizes the correlation between numerical columns in a dataset. The values represented in the heatmap show the strength and direction of relationships between different variables.
    </p>

    <h2>How It Works</h2>
    <p>
        The heatmap uses a matrix to represent correlations between each pair of numerical columns. The calculation behind this is based on the concept of \"correlation,\" which measures how strongly two variables are related. A correlation can be positive, negative, or zero:
    </p>
    <ul>
        <li><strong>Positive correlation</strong>: Both variables increase or decrease together (e.g., if the temperature rises, ice cream sales increase).</li>
        <li><strong>Negative correlation</strong>: As one variable increases, the other decreases (e.g., as the price of a product rises, the demand for it decreases).</li>
        <li><strong>Zero correlation</strong>: There is no relationship between the two variables (e.g., height and shoe size might show zero correlation in some contexts).</li>
    </ul>

    <h2>Color Scale: Yellow to Purple (Viridis)</h2>
    <p>
        The heatmap uses a color scale called \"Viridis,\" which ranges from yellow to purple. Here's what the colors represent:
    </p>
    <ul>
        <li><strong>Yellow (brightest)</strong>: A strong positive correlation (close to +1). This indicates that as one variable increases, the other increases in a very predictable manner.</li>
        <li><strong>Green</strong>: A moderate positive correlation. Variables are still positively related, but the relationship is not as strong.</li>
        <li><strong>Blue</strong>: A weak or near-zero correlation. There is a small or no discernible relationship between the variables.</li>
        <li><strong>Purple (darkest)</strong>: A strong negative correlation (close to -1). This indicates that as one variable increases, the other decreases in a very predictable manner.</li>
    </ul>

    <h2>What the Heatmap Shows</h2>
    <p>
        In the heatmap, each cell represents the correlation between two numerical columns. The color of the cell is determined by the correlation coefficient: from yellow for strong positive correlations, through green and blue for weaker correlations, to purple for strong negative correlations.
    </p>
";
		$html = '<div class="invert_in_dark_mode" id="plotHeatmap"></div><br>'.$explanation;

		$svg_icon = get_icon_html("plot.svg");

		$tabs["{$svg_icon}Heatmap"] = [
			'id' => 'tab_heatmap',
			'content' => $html,
			"onclick" => "plotHeatmap();"
		];

		return $tabs;
	}

	function add_job_status_distribution($tabs) {
		$html = '<div class="invert_in_dark_mode" id="plotJobStatusDistribution"></div>';

		$svg_icon = get_icon_html("plot.svg");

		$tabs["{$svg_icon}Job Status Distribution"] = [
			'id' => 'tab_plot_job_status_distribution',
			'content' => $html,
			"onclick" => "plotJobStatusDistribution();"
		];

		return $tabs;
	}

	function add_results_distribution_by_generation_method ($tabs) {
		$html = '<div class="invert_in_dark_mode" id="plotResultsDistributionByGenerationMethod"></div>';

		$svg_icon = get_icon_html("plot.svg");

		$tabs["{$svg_icon}Results by Generation Method"] = [
			'id' => 'tab_plot_results_distribution_by_generation_method',
			'content' => $html,
			"onclick" => "plotResultsDistributionByGenerationMethod();"
		];

		return $tabs;
	}

	function add_box_plot_tab ($tabs) {
		$html = '<div class="invert_in_dark_mode" id="plotBoxplot"></div>';

		$svg_icon = get_icon_html("plot.svg");

		$tabs["{$svg_icon}Boxplots"] = [
			'id' => 'tab_boxplots',
			'content' => $html,
			"onclick" => "plotBoxplot();"
		];

		return $tabs;
	}

	function add_gpu_plots ($tabs) {
		$html = '<div class="invert_in_dark_mode" id="gpu-plot"></div>';

		$svg_icon = get_icon_html("gpu.svg");

		$tabs["{$svg_icon}GPU Usage"] = [
			'id' => 'tab_gpu_usage',
			'content' => $html,
			"onclick" => "plotGPUUsage();"
		];

		return $tabs;
	}

	function add_timeline ($tabs, $warnings, $csv_file, $name, $tab_name) {
		$html = '<div class="invert_in_dark_mode" id="plot_timeline"></div>';

		$svg_icon = get_icon_html("timeline.svg");

		$tabs["{$svg_icon}$name"] = [
			'id' => $tab_name,
			'content' => $html,
			"onclick" => "plotTimelineFromGlobals();"
		];

		return [$tabs, $warnings];
	}


	function add_parallel_plot_tab ($tabs) {
		$html = '<input type="checkbox" id="enable_slurm_id_if_exists" onchange="createParallelPlot(tab_results_csv_json, tab_results_headers_json, result_names, special_col_names, true)" /> Show SLURM-Job-ID (if it exists)<br><div class="invert_in_dark_mode" id="parallel-plot"></div>';

		$svg_icon = get_icon_html("parallel.svg");

		$tabs["{$svg_icon}Parallel Plot"] = [
			'id' => 'tab_parallel',
			'content' => $html,
			"onclick" => "createParallelPlot(tab_results_csv_json, tab_results_headers_json, result_names, special_col_names);"
		];

		return $tabs;
	}

	function has_real_char($filename) {
		return file_exists($filename) && preg_match('/\S/', file_get_contents($filename));
	}

	function normalize_csv_value($field) {
		if (trim($field) === '') {
			return '';
		}

		if (is_numeric($field)) {
			if (strpos($field, '.') !== false) {
				$number = floatval($field);

				if (fmod($number, 1.0) === 0.0) {
					return strval(intval($number));
				} else {
					$normalized = rtrim(rtrim(number_format($number, 30, '.', ''), '0'), '.');
					return $normalized;
				}
			} else {
				return strval(intval($field));
			}
		}

		return $field;
	}

	function normalize_csv_file_contents($contents) {
		$lines = explode("\n", $contents);
		$normalized_lines = [];

		foreach ($lines as $line) {
			// Skip completely empty lines
			if (trim($line) === '') {
				continue;
			}

			$fields = str_getcsv($line, ",", "\"", "\\");
			$normalized_fields = array_map('normalize_csv_value', $fields);
			$normalized_lines[] = implode(',', $normalized_fields);
		}

		return implode("\n", $normalized_lines);
	}

	function add_simple_csv_tab_from_file ($tabs, $warnings, $filename, $name, $id, $header_line = null) {
		if(is_file($filename) && filesize($filename) && has_real_char($filename) && is_ascii_or_utf8($filename)) {
			$csv_contents = get_csv_data_as_array($filename, ",", $header_line);
			$headers = $csv_contents[0];
			$csv_contents_no_header = $csv_contents;
			array_shift($csv_contents_no_header);

			$csv_json = $csv_contents_no_header;
			$headers_json = $headers;

			$GLOBALS["json_data"]["{$id}_headers_json"] = $headers_json;
			$GLOBALS["json_data"]["{$id}_csv_json"] = $csv_json;


			$content = file_get_contents($filename);

			$content = normalize_csv_file_contents($content);

			$content = my_htmlentities($content);

			if($content && $header_line) {
				$content = implode(",", $header_line)."\n$content";
			}

			$results_html = "<div id='{$id}_csv_table'></div>\n";
			$results_html .= copy_id_to_clipboard_string("{$id}_csv_table_pre", $filename);
			$results_html .= "<pre id='{$id}_csv_table_pre'>".$content."</pre>\n";
			$results_html .= copy_id_to_clipboard_string("{$id}_csv_table_pre", $filename);
			$results_html .= "<script>\n\tcreateTable({$id}_csv_json, {$id}_headers_json, '{$id}_csv_table');</script>\n";

			$svg_icon = get_icon_html("csv.svg");

			$tabs["$svg_icon$name"] = [
				'id' => $id,
				'content' => $results_html,
				'onclick' => 'colorize_table_entries();'
			];
		} else {
			if(!is_file($filename)) {
				$warnings[] = "$filename does not exist";
			} else if(!is_ascii_or_utf8($filename)) {
				$warnings[] = "$filename is not Ascii or UTF8";
			} else if(!is_readable($filename)) {
				$warnings[] = "$filename is not a real char";
			} else if(!filesize($filename)) {
				$warnings[] = "$filename is empty";
			}
		}

		return [$tabs, $warnings];
	}

	function get_log_files($run_dir) {
		$log_files = [];

		if (!is_dir($run_dir)) {
			error_log("Error: Directory does not exist - $run_dir");
			return $log_files;
		}

		$files = scandir($run_dir);
		if ($files === false) {
			error_log("Error: Could not read Directory- $run_dir");
			return $log_files;
		}

		foreach ($files as $file) {
			if (preg_match('/^(\d+)_0_log\.out$/', $file, $matches)) {
				$nr = $matches[1];
				$log_files[$nr] = $file;
			}
		}

		return $log_files;
	}

	function get_csv_data_as_array($filePath, $delimiter = ",", $header_line = null) {
		if (!file_exists($filePath) || !is_readable($filePath)) {
			error_log("CSV file not found or not readable: " . $filePath);
			return [];
		}

		$data = [];

		if($header_line != null) {
			$data[] = $header_line;
		}

		$enclosure = "\"";
		$escape = "\\";

		if (($handle = fopen($filePath, "r")) !== false) {
			while (($row = fgetcsv($handle, 0, $delimiter, $enclosure, $escape)) !== false) {
				foreach ($row as &$value) {
					if (is_numeric($value)) {
						if (strpos($value, '.') !== false || stripos($value, 'e') !== false) {
							$value = (float)$value;
						} else {
							$value = (int)$value;
						}
					}
				}
				unset($value);
				$data[] = $row;
			}
			fclose($handle);
		} else {
			error_log("Failed to open CSV file: " . $filePath);
		}

		return $data;
	}

	function remove_ansi_colors($contents) {
		$contents = preg_replace('#\\x1b[[][^A-Za-z]*[A-Za-z]#', '', $contents);
		$contents = preg_replace('#\[(?:0|91)m#', '', $contents);
		return $contents;
	}

	function analyze_results_csv($res_csv) {
		if (!file_exists($res_csv)) {
			return "";
		}

		$handle = fopen($res_csv, "r");
		if (!$handle) {
			return "";
		}

		$delimiter = ",";
		$enclosure = "\"";
		$escape = "\\";

		$header = fgetcsv($handle, 0, $delimiter, $enclosure, $escape);
		$statusIndex = array_search("trial_status", $header);

		if ($statusIndex === false) {
			fclose($handle);
			return "";
		}

		$statusCounts = [
			"COMPLETED" => 0,
			"FAILED" => 0,
			"RUNNING" => 0
		];

		while (($row = fgetcsv($handle, 0, $delimiter, $enclosure, $escape)) !== false) {
			$status = $row[$statusIndex];
			if (isset($statusCounts[$status])) {
				$statusCounts[$status]++;
			}
		}

		fclose($handle);

		$parts = [];
		foreach ($statusCounts as $status => $count) {
			if ($count > 0) {
				$parts[] = ucfirst(strtolower($status)) . ": $count";
			}
		}

		return implode(", ", $parts);
	}

	function count_subfolders_or_files(string $path): int {
		$count = 0;

		if (!is_dir($path)) {
			return 0;
		}

		foreach (scandir($path) as $item) {
			if ($item === '.' || $item === '..') continue;
			if (is_dir($path . DIRECTORY_SEPARATOR . $item) || is_file($path . DIRECTORY_SEPARATOR . $item)) {
				$count++;
			}
		}

		return $count;
	}

	function get_latest_modification_time($folderPath) {
		if (isset($GLOBALS["modificationCache"][$folderPath])) {
			return $GLOBALS["modificationCache"][$folderPath];
		}

		$latestTime = 0;

		try {
			$iterator = new RecursiveIteratorIterator(
				new RecursiveDirectoryIterator($folderPath, FilesystemIterator::SKIP_DOTS),
				RecursiveIteratorIterator::SELF_FIRST
			);

			foreach ($iterator as $fileInfo) {
				if ($fileInfo->isFile()) {
					$mtime = $fileInfo->getMTime();
					if ($mtime > $latestTime) {
						$latestTime = $mtime;
					}
				}
			}
		} catch (Exception $e) {
			error_log("Error at reading '$folderPath': " . $e->getMessage());
		}

		$GLOBALS["modificationCache"][$folderPath] = $latestTime;
		return $latestTime;
	}

	function generate_folder_buttons($folderPath, $new_param_name) {
		if (!isset($_SERVER["REQUEST_URI"])) {
			return;
		}

		$sort = isset($_GET['sort']) ? $_GET['sort'] : 'time_desc';

		echo get_sort_options();

		if (is_dir($folderPath)) {
			$dir = opendir($folderPath);
			$currentUrl = $_SERVER['REQUEST_URI'];
			$folders = [];

			while (($folder = readdir($dir)) !== false) {
				if ($folder != "." && $folder != ".." && is_dir($folderPath . '/' . $folder) && preg_match("/^[a-zA-Z0-9-_]+$/", $folder)) {
					$folders[] = $folder;
				}
			}
			closedir($dir);

			switch ($sort) {
				case 'time_asc':
					usort($folders, function($a, $b) use ($folderPath) {
						$timeA = get_latest_modification_time($folderPath . '/' . $a);
						$timeB = get_latest_modification_time($folderPath . '/' . $b);
						return $timeA - $timeB;
					});
					break;
				case 'time_desc':
					usort($folders, function($a, $b) use ($folderPath) {
						$timeA = get_latest_modification_time($folderPath . '/' . $a);
						$timeB = get_latest_modification_time($folderPath . '/' . $b);
						return $timeB - $timeA;
					});
					break;
				case 'nr_asc':
					sort($folders);
					break;
				case 'nr_desc':
					rsort($folders);
					break;
			}

			$shown_folders = 0;

			if (count($folders)) {
				foreach ($folders as $folder) {
					$folderPathWithFile = $folderPath . '/' . $folder;
					$url = $currentUrl . (strpos($currentUrl, '?') === false ? '?' : '&') . $new_param_name . '=' . urlencode($folder);
					if ($sort != 'nr_asc') {
						$url .= '&sort=' . urlencode($sort);
					}

					$timestamp = get_latest_modification_time($folderPathWithFile);
					$lastModified = date("F d Y H:i:s", $timestamp);
					$timeSince = time_since($timestamp);

					$key_string = "";

					if($new_param_name == "run_nr") {
						$pw_file = "$folderPathWithFile/password.sha256";
						if(file_exists($pw_file)) {
							$key_string = "&nbsp;&#128272;";
						}
					}

					if(has_non_empty_folder($folderPathWithFile)) {
						$folder = htmlspecialchars($folder);
						$url = htmlspecialchars($url);

						$bracket_string = "$lastModified | $timeSince";

						$res_csv = "$folderPath/$folder/results.csv";

						$show = 1;

						if(file_exists($res_csv)) {
							$analyzed = analyze_results_csv($res_csv);
							if($analyzed) {
								$bracket_string .= " | ".$analyzed;
							}
						} else {
							$counted_subfolders = count_subfolders_or_files($folderPathWithFile);

							if($counted_subfolders != 0) {
								if($new_param_name != "run_nr") {
									if($counted_subfolders == 1) {
										$bracket_string .= " | ".$counted_subfolders. " subfolder";
									} else {
										$bracket_string .= " | ".$counted_subfolders. " subfolders";
									}
								}
							} else {
								$show = 0;
							}
						}

						if($show) {
							echo "<a class='share_folder_buttons' href='$url'>";
							echo "<button type='button'>$folder ($bracket_string)$key_string</button>";
							echo '</a><br>';
						}
						$shown_folders++;
					}
				}
			}
			
			if($shown_folders == 0) {
				print "<h2>Sorry, no jobs have been uploaded yet.</h2>";
			}
		} else {
			echo "The specified folder does not exist.";
		}
	}

	function time_since($timestamp) {
		$diff = time() - $timestamp;

		$units = [
			31536000 => 'year',
			2592000  => 'month',
			86400    => 'day',
			3600     => 'hour',
			60       => 'minute',
			1        => 'second'
		];

		foreach ($units as $unitSeconds => $unitName) {
			if ($diff >= $unitSeconds) {
				$count = floor($diff / $unitSeconds);
				return "$count $unitName" . ($count > 1 ? 's' : '') . " ago";
			}
		}

		return "just now";
	}

	function get_sort_options() {
		$sort = isset($_GET['sort']) ? $_GET['sort'] : 'time_desc';

		$currentUrl = $_SERVER['REQUEST_URI'];
		$urlParts = parse_url($currentUrl);
		parse_str($urlParts['query'] ?? '', $queryParams);
		unset($queryParams['sort']);

		return '
			<form id="sortForm" method="get">
				<select name="sort" onchange="update_url()">
					<option value="time_asc"' . ($sort == 'time_asc' ? ' selected' : '') . '>Time (ascending)</option>
					<option value="time_desc"' . ($sort == 'time_desc' ? ' selected' : '') . '>Time (descending)</option>
					<option value="nr_asc"' . ($sort == 'nr_asc' ? ' selected' : '') . '>Name (ascending)</option>
					<option value="nr_desc"' . ($sort == 'nr_desc' ? ' selected' : '') . '>Name (descending)</option>
				</select>
			</form>
			<script>
				function update_url() {
					const currentUrl = window.location.href;
					const url = new URL(currentUrl);

					const sortValue = document.querySelector("select[name=\'sort\']").value;

					url.searchParams.set("sort", sortValue);

					window.location.href = url.toString();
				}
			</script>
		';
	}

	function is_valid_user_or_experiment_name ($name) {
		if(preg_match("/^[a-zA-Z0-9_-]+$/", $name)) {
			return true;
		}

		return false;
	}

	function string_is_numeric ($name) {
		if(preg_match("/^\d+$/", $name)) {
			return true;
		}

		return false;
	}

	$tabs = [];

	function file_string_contains_results($file_as_string, $names) {
		if (!is_string($file_as_string) || strlen($file_as_string) === 0) {
			return false;
		}

		$names = array_map('strtolower', $names);
		$remaining = array_flip($names);

		$length = strlen($file_as_string);
		$pos = $length - 1;
		$buffer = '';

		while ($pos >= 0 && count($remaining) > 0) {
			$char = $file_as_string[$pos];
			$buffer = $char . $buffer;
			$pos--;

			if ($char === "\n" || ($length - $pos >= 8192) || $pos < 0) {
				$lines = explode("\n", $buffer);
				foreach ($lines as $line) {
					$line_lc = strtolower($line);
					foreach ($remaining as $name => $_) {
						if (strpos($line_lc, $name . ':') !== false) {
							if (preg_match('/\b' . preg_quote($name, '/') . '\s*:\s*[-+]?\d+(?:\.\d+)?/i', $line)) {
								unset($remaining[$name]);
								if (count($remaining) === 0) {
									return true;
								}
							}
						}
					}
				}
				$buffer = '';
			}
		}

		return count($remaining) === 0;
	}

	function ends_with_submitit_info($string) {
		$ret = preg_match('/submitit INFO \(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}\) - Exiting after successful completion$/', $string) === 1;

		return $ret;
	}

	function contains_slurm_time_limit_error($input) {
		if (!is_string($input)) {
			return false;
		}

		$pattern = '/slurmstepd:\s+error:\s+\*\*\*\s+JOB\s+\d+\s+ON\s+\S+\s+CANCELLED\s+AT\s+\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\s+DUE\s+TO\s+TIME\s+LIMIT\s+\*\*\*/';

		$match_found = preg_match($pattern, $input, $matches);

		if ($match_found === false) {
			error_log("Regex evaluation failed.");
			return false;
		}

		return $match_found === 1;
	}

	function extract_results_dict(string $file_as_string): array {
		$matches = [];
		preg_match_all('/^(\w+)\s*:\s*([+-]?\d+(?:\.\d+)?)/m', $file_as_string, $matches, PREG_SET_ORDER);

		$results = [];
		foreach ($matches as $m) {
			$results[$m[1]] = $m[2]; // überschreibt ältere Werte
		}

		return $results;
	}

	function format_results_from_dict(array $results_dict, array $result_names): string {
		$parts = [];
		foreach ($result_names as $name) {
			if (isset($results_dict[$name])) {
				$parts[] = "$name: {$results_dict[$name]}";
			}
		}

		return $parts ? implode(', ', $parts) : '';
	}

	function extract_trial_index($text, $nr) {
		if (preg_match('/\bTrial-Index:\s*(\d+)/', $text, $matches)) {
			return (int)$matches[1];
		}
		return $nr;
	}

	function generate_log_tabs($run_dir, $log_files, $result_names) {
		$red_cross = "<span>&#10060;</span>";
		$green_checkmark = "<span>&#9989;</span>";
		$gear = "<span><img style='height: 1em' src='i/gear.svg' /></span>";
		$memory = "<span><img style='height: 1em' src='i/memory.svg' /></span>";
		$time_warning = "<span><img style='height: 1em' src='i/timeout.svg' /></span>";

		$output = '<section class="tabs" style="width: 100%"><menu role="tablist" aria-label="Single-Runs">';

		$i = 0;

		foreach ($log_files as $nr => $file) {
			$file_path = "$run_dir/$file";
			$checkmark = $red_cross;
			if(is_file($file_path) && is_readable($file_path) && is_ascii_or_utf8($file_path)) {
				$file_as_string = file_get_contents($file_path);

				$status = "";

				if (file_string_contains_results($file_as_string, $result_names)) {
					$status = "success";
					$checkmark = $green_checkmark;
				} else {
					if(preg_match("/(?:(?:oom_kill\s+event)|(?:CUDA out of memory))/i", $file_as_string)) {
						$status = "oom";
						$checkmark = $memory;
					} else if(ends_with_submitit_info($file_as_string)) {
						$status = "failed";
						$checkmark = $red_cross;
					} else if(contains_slurm_time_limit_error($file_as_string)) {
						$status = "time_warning";
						$checkmark = $time_warning;
					} else {
						$status = "still_working";
						$checkmark = $gear;
					}
				}

				$runtime = get_runtime($file_as_string);
				$runtime_string = get_runtime_human_format($runtime);

				$brackets = [];

				if($runtime_string == "0s" || !$runtime_string) {
					$runtime_string = "";
				} else {
					$brackets[] = $runtime_string;
				}

				$exit_code_from_file = get_exit_code_from_outfile($file_as_string);

				if($exit_code_from_file != 0 && $exit_code_from_file != "") {
					$brackets[] = "exit-code: $exit_code_from_file";
				}

				$brackets_string = "";

				$results_dict = extract_results_dict($file_as_string);

				$res_strings = format_results_from_dict($results_dict, $result_names);

				if($res_strings) {
					$brackets[] = $res_strings;
				}

				if(count($brackets)) {
					$brackets_string = " (".implode(", ", $brackets).")";
				}

				$trial_index_or_nr = extract_trial_index($file_as_string, $nr);

				$tabname = "$trial_index_or_nr$brackets_string $checkmark";

				$data_array = [
					"trial_index=$nr",
					"trial_index_or_nr=$trial_index_or_nr",
					"exit_code=$exit_code_from_file",
					"runtime=$runtime",
					"status=$status"
				];

				if (!empty($result_names)) {
					foreach ($result_names as $name) {
						if (isset($results_dict[$name])) {
							$data_array[] = "$name={$results_dict[$name]}";
						}
					}
				}

				$data = " data-".implode(" data-", $data_array);

				$aria_selected_= $i == 0 ? 'aria-selected="true"' : '';

				$output .= '<button '.$data.' onclick="load_log_file('.$i.', \''.$file.'\')" role="tab" '.$aria_selected_.' aria-controls="single_run_'.$i.'">'.$tabname."</button>\n";
				$i++;
			}
		}

		$output .= '</menu>';

		$j = 0;
		foreach ($log_files as $nr => $file) {
			$file_path = $run_dir . '/' . $file;
			$output .= '<article role="tabpanel" id="single_run_' . $j . '">';
			if($j != 0) {
				$output .= "<div id='spinner_log_$j' class='spinner'></div>";
			}

			$output .= copy_id_to_clipboard_string("single_run_{$j}_pre", $file_path);

			if ($j == 0) {
				$content = file_get_contents($file_path);
				$output .= '<pre id="single_run_'.$j.'_pre" data-loaded="true">' . highlight_debug_info(ansi_to_html(htmlspecialchars($content))) . '</pre>';
			} else {
				$output .= '<pre id="single_run_'.$j.'_pre"></pre>';
			}
			$output .= copy_id_to_clipboard_string("single_run_{$j}_pre", $file_path);
			$output .= '</article>';
			$j++;
		}

		$output .= '</section>';
		return [$output, $i];
	}

	function is_valid_user_id($value) {
		if($value === null) {
			return false;
		}
		return !!preg_match('/^[a-zA-Z0-9_]+$/', $value);
	}

	function is_valid_experiment_name($value) {
		if($value === null) {
			return false;
		}
		return !!preg_match('/^[a-zA-Z-0-9_]+$/', $value);
	}

	function is_valid_run_nr($value) {
		if($value === null) {
			return false;
		}
		return !!preg_match('/^\d+$/', $value);
	}

	function remove_ansi_escape_sequences($string) {
		$lines = explode("\n", $string);
		$result = array();
		$in_sixel = false;

		foreach ($lines as $line) {
			if (preg_match('/^\x1bP.*q/', $line)) {
				$in_sixel = true;
				$result[] = $line;
				continue;
			}

			if ($in_sixel && strpos($line, "\x1b\\") !== false) {
				$in_sixel = false;
				$result[] = $line;
				continue;
			}

			if ($in_sixel) {
				$result[] = $line;
				continue;
			}

			$line = preg_replace('/\x1b\[[0-9;]*[A-Za-z]/', '', $line);
			$line = preg_replace('/\x1b\][^\x07]*\x07/', '', $line);
			$line = preg_replace('/\x1b[\(\)][0-9A-Za-z]/', '', $line);
			$line = preg_replace('/\x1b=|\x1b>/', '', $line);
			$line = preg_replace('/\x1b./', '', $line);

			$result[] = $line;
		}

		return implode("\n", $result);
	}

	function highlight_debug_info($log) {
		$log = preg_replace('/(E[0-9]{4}.*?)(?=\n|$)/', '<span style="color:red;">$1</span>', $log);

		$log = preg_replace('/(WARNING:.*?)(?=\n|$)/', '<span style="color:orange;">$1</span>', $log);

		$log = preg_replace('/(INFO.*?)(?=\n|$)/', '<span style="color:green;">$1</span>', $log);

		$log = preg_replace_callback('/(DEBUG INFOS START.*?DEBUG INFOS END)/s', function($matches) {
			$debugInfo = $matches[0];

			$debugInfo = preg_replace('/(Program-Code:.*?)(?=\n|$)/', '<span style="color:green;">$1</span>', $debugInfo);

			$debugInfo = preg_replace('/(File:.*?)(?=\n|$)/', '<span style="color:blue;">$1</span>', $debugInfo);

			$debugInfo = preg_replace('/(UID:.*?)(?=\n|$)/', '<span style="color:gray;">$1</span>', $debugInfo);
			$debugInfo = preg_replace('/(GID:.*?)(?=\n|$)/', '<span style="color:gray;">$1</span>', $debugInfo);

			$debugInfo = preg_replace('/(Status-Change-Time:.*?)(?=\n|$)/', '<span style="color:blue;">$1</span>', $debugInfo);
			$debugInfo = preg_replace('/(Last access:.*?)(?=\n|$)/', '<span style="color:blue;">$1</span>', $debugInfo);
			$debugInfo = preg_replace('/(Last modification:.*?)(?=\n|$)/', '<span style="color:blue;">$1</span>', $debugInfo);

			$debugInfo = preg_replace('/(Size:.*?)(?=\n|$)/', '<span style="color:purple;">$1</span>', $debugInfo);
			$debugInfo = preg_replace('/(Permissions:.*?)(?=\n|$)/', '<span style="color:purple;">$1</span>', $debugInfo);

			$debugInfo = preg_replace('/(Owner:.*?)(?=\n|$)/', '<span style="color:green;">$1</span>', $debugInfo);

			$debugInfo = preg_replace('/(Hostname:.*?)(?=\n|$)/', '<span style="color:orange;">$1</span>', $debugInfo);

			return '<div style="background-color:#f0f0f0;padding:10px;border:1px solid #ddd;">' . $debugInfo . '</div>';
		}, $log);

		$log = preg_replace('/(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})/', '<span style="color:blue;">$1</span>', $log);

		return $log;
	}

	function get_exit_code_from_outfile(string $input): ?int {
		// Pattern sucht nach "EXIT_CODE: " gefolgt von einer oder mehreren Ziffern am Ende des Strings
		$pattern = '/EXIT_CODE:\s*(\d+)\s*/';

		if (preg_match($pattern, $input, $matches) === 1) {
			$exitCode = intval($matches[1]);
			return $exitCode;
		}

		return null;
	}

	function get_runtime($string) {
		if (!$string) {
			return null;
		}

		$pattern = '/submitit INFO \((\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\)/';
		preg_match_all($pattern, $string, $matches);
		$dates = $matches[1];

		$unixTimes = [];
		foreach ($dates as $date) {
			$formattedDate = str_replace(',', '.', $date);
			$unixTimes[] = strtotime($formattedDate);
		}

		if (!count($unixTimes)) {
			return 0;
		}

		return max($unixTimes) - min($unixTimes);
	}

	function get_runtime_human_format($seconds) {
		if ($seconds <= 0) {
			return "0s";
		}

		$hours = floor($seconds / 3600);
		$minutes = floor(($seconds % 3600) / 60);
		$remainingSeconds = $seconds % 60;

		$result = [];
		if ($hours > 0) $result[] = "{$hours}h";
		if ($minutes > 0) $result[] = "{$minutes}m";
		if ($remainingSeconds > 0 || empty($result)) $result[] = "{$remainingSeconds}s";

		return implode(":", $result);
	}

	function extract_min_max_ram_cpu_from_worker_info($data) {
		preg_match_all('/CPU: ([\d\.]+)%, RAM: ([\d\.]+) MB/', $data, $matches);

		$cpu_values = $matches[1];
		$ram_values = $matches[2];

		if (empty($cpu_values) || empty($ram_values)) {
			echo "";
			return "";
		}

		function calculate_average($values) {
			return array_sum($values) / count($values);
		}

		function calculate_median($values) {
			sort($values);
			$count = count($values);
			$middle = floor($count / 2);
			if ($count % 2) {
				return $values[$middle];
			} else {
				return ($values[$middle - 1] + $values[$middle]) / 2;
			}
		}

		$min_cpu = min($cpu_values);
		$max_cpu = max($cpu_values);
		$avg_cpu = calculate_average($cpu_values);
		$median_cpu = calculate_median($cpu_values);

		$min_ram = min($ram_values);
		$max_ram = max($ram_values);
		$avg_ram = calculate_average($ram_values);
		$median_ram = calculate_median($ram_values);

		$html = '<table>';
		$html .= '<tr><th>Min RAM (MB)</th><th>Max RAM (MB)</th><th>Avg RAM (MB)</th><th>Median RAM (MB)</th>';
		$html .= '<th>Min CPU (%)</th><th>Max CPU (%)</th><th>Avg CPU (%)</th><th>Median CPU (%)</th></tr>';
		$html .= '<tr>';
		$html .= '<td>' . htmlspecialchars($min_ram) . '</td>';
		$html .= '<td>' . htmlspecialchars($max_ram) . '</td>';
		$html .= '<td>' . htmlspecialchars(round($avg_ram, 2)) . '</td>';
		$html .= '<td>' . htmlspecialchars($median_ram) . '</td>';
		$html .= '<td>' . htmlspecialchars($min_cpu) . '</td>';
		$html .= '<td>' . htmlspecialchars($max_cpu) . '</td>';
		$html .= '<td>' . htmlspecialchars(round($avg_cpu, 2)) . '</td>';
		$html .= '<td>' . htmlspecialchars($median_cpu) . '</td>';
		$html .= '</tr>';
		$html .= '</table>';

		return $html;
	}

	function check_folder_permissions($directory, $expectedUser, $expectedGroup, $alternativeUser, $alternativeGroup, $expectedPermissions) {
		if (getenv('CI') !== false) {
			return false;
		}

		if (!is_dir($directory)) {
			echo "<i>Error: '$directory' is not a valid directory</i>\n";
			return true;
		}

		$stat = stat($directory);
		if ($stat === false) {
			echo "<i>Error: Unable to retrieve information for '$directory'</i><br>\n";
			return;
		}

		$currentUser = posix_getpwuid($stat['uid'])['name'] ?? 'unknown';
		$currentGroup = posix_getgrgid($stat['gid'])['name'] ?? 'unknown';
		$currentPermissions = substr(sprintf('%o', $stat['mode']), -4);

		$issues = false;

		if ($currentUser !== $expectedUser) {
			if ($currentUser !== $alternativeUser) {
				$issues = true;
				echo "<i>Ownership issue: Current user is '$currentUser'. Expected user is '$expectedUser'</i><br>\n";
				echo "<samp>chown $expectedUser $directory</samp>\n<br>";
			}
		}

		if ($currentGroup !== $expectedGroup) {
			if ($currentGroup !== $alternativeGroup) {
				$issues = true;
				echo "<i>Ownership issue: Current group is '$currentGroup'. Expected group is '$expectedGroup'</i><br>\n";
				echo "<samp>chown :$expectedGroup $directory</samp><br>\n";
			}
		}

		if (intval($currentPermissions, 8) !== $expectedPermissions) {
			$issues = true;
			echo "<i>Permissions issue: Current permissions are '$currentPermissions'. Expected permissions are '" . sprintf('%o', $expectedPermissions) . "'</i><br>\n";
			echo "<samp>chmod " . sprintf('%o', $expectedPermissions) . " $directory\n</samp><br>";
		}

		return $issues;
	}

	function warn($message) {
		echo "Warning: " . $message . "\n";
	}

	function find_matching_uuid_run_folder(string $targetUUID, $sharesPath, $user_id, $experiment_name): ?string {
		if (!preg_match("/^[a-zA-Z0-9_-]+$/", $user_id)) {
			return null;
		}

		if (!preg_match("/^[a-zA-Z0-9_-]+$/", $experiment_name)) {
			return null;
		}

		$glob_str = "$sharesPath/$user_id/$experiment_name/*/run_uuid";
		$files = glob($glob_str);

		foreach ($files as $file) {
			$fileContent = preg_replace('/\s+/', '', file_get_contents($file));

			if ($fileContent === $targetUUID) {
				return dirname($file);
			}
		}

		return null;
	}

	function delete_folder($folder) {
		$files = array_diff(scandir($folder), array('.', '..'));

		foreach ($files as $file) {
			(is_dir("$folder/$file")) ? delete_folder("$folder/$file") : my_unlink("$folder/$file");
		}

		return rmdir($folder);
	}

	function create_new_folder($path, $user_id, $experiment_name) {
		$i = 0;

		$newFolder = $path . "/$user_id/$experiment_name/$i";

		do {
			$newFolder = $path . "/$user_id/$experiment_name/$i";
			$i++;
		} while (file_exists($newFolder));

		try {
			mkdir($newFolder, 0777, true);
		} catch (Exception $e) {
			print("Error trying to create directory $newFolder. Error:\n\n$e\n\n");
			exit(1);
		}
		return $newFolder;
	}

	function search_for_hash_file($directory, $new_upload_md5, $userFolder) {
		$files = glob($directory);

		foreach ($files as $file) {
			try {
				$file_content = file_get_contents($file);

				if ($file_content === $new_upload_md5) {
					return [true, dirname($file)];
				}
			} catch (AssertionError $e) {
				print($e->getMessage());
			}
		}

		try {
			$destinationPath = "$userFolder/hash.md5";
			assert(is_writable(dirname($destinationPath)), "Directory is not writable: " . dirname($destinationPath));

			file_put_contents($destinationPath, $new_upload_md5);
		} catch (\Throwable $e) {
			print("\n" . $e->getMessage() . "\n");
		}

		return [false, null];
	}

	function extract_path_components($found_hash_file_dir, $sharesPath) {
		$pattern = "#^$sharesPath/([^/]+)/([^/]+)/(\d+)$#";

		if (preg_match($pattern, $found_hash_file_dir, $matches)) {
			assert(isset($matches[1]), "Failed to extract user from path: $found_hash_file_dir");
			assert(isset($matches[2]), "Failed to extract experiment name from path: $found_hash_file_dir");
			assert(isset($matches[3]), "Failed to extract run ID from path: $found_hash_file_dir");

			$user = $matches[1];
			$experiment_name = $matches[2];
			$run_dir = $matches[3];

			return [$user, $experiment_name, $run_dir];
		} else {
			warn("The provided path does not match the expected pattern: $found_hash_file_dir");
			return [null, null, null];
		}
	}

	function get_user_folder($sharesPath, $_uuid_folder, $user_id, $experiment_name, $run_nr="") {
		$probe_dir = "$sharesPath/$user_id/$experiment_name/$run_nr";

		if($run_nr != "" && $run_nr >= 0 && is_dir($probe_dir)) {
			return $probe_dir;
		}

		if(getenv("disable_folder_creation")) {
			return;
		}

		if (!$_uuid_folder) {
			$userFolder = create_new_folder($sharesPath, $user_id, $experiment_name);
		} else {
			$userFolder = $_uuid_folder;
		}

		return $userFolder;
	}

	function is_valid_zip_file($path) {
		if (!file_exists($path) || !is_readable($path)) {
			return false;
		}

		$handle = fopen($path, 'rb');
		if (!$handle) {
			return false;
		}

		$signature = fread($handle, 4);
		fclose($handle);

		return $signature === "PK\x03\x04";
	}

	function move_files($offered_files, $added_files, $userFolder, $msgUpdate, $msg) {
		$empty_files = [];

		foreach ($offered_files as $offered_file) {
			$file = $offered_file["file"];
			$filename = $offered_file["filename"];

			if ($file) {
				if(file_exists($file)) {
					$content = file_get_contents($file);
					$content_encoding = 'UTF-8';

					if (mb_check_encoding($content, 'UTF-8')) {
						$content_encoding = 'UTF-8';
					} elseif (mb_check_encoding($content, 'ASCII')) {
						$content_encoding = 'ASCII';
					} else {
						$content_encoding = mb_detect_encoding($content);
					}

					if ($content_encoding == "ASCII" || $content_encoding == "UTF-8" || is_valid_zip_file($file)) {
						if (filesize($file)) {
							if(preg_match("/\.svg$/", $filename)) {
								$filename = "profile_svg";
							}

							try {
								move_uploaded_file($file, "$userFolder/$filename");
								$added_files++;
							} catch (Exception $e) {
								error_log("\nAn exception occured trying to move $file to $userFolder/$filename: $e\n");
							}
						} else {
							$empty_files[] = $filename;
						}
					} else {
						dier("$filename: \$content was not ASCII, UTF8-file or zip, but $content_encoding");
					}
				}
			}
		}

		if ($added_files) {
			if (isset($_GET["update"])) {
				eval('echo "$msgUpdate";');
			} else {
				eval('echo "$msg";');
			}
			exit(0);
		} else {
			if (count($empty_files)) {
				$empty_files_string = implode(", ", $empty_files);
				echo "Error sharing the job. The following files were empty: $empty_files_string. \n";
			} else {
				echo "Error sharing the job. No Files were found. \n";
			}
			exit(1);
		}
	}

	function remove_extra_slashes_from_url($string) {
		$pattern = '/(?<!:)(\/{2,})/';

		$cleaned_string = preg_replace($pattern, '/', $string);

		return $cleaned_string;
	}

	function warn_if_low_disk_space($userFolder, $minFreeMB = 100) {
		$freeBytes = disk_free_space($userFolder);
		if ($freeBytes === false) {
			error_log("Could not determine free space for: $userFolder");
			return false;
		}

		$freeMB = $freeBytes / (1024 * 1024);

		if ($freeMB < $minFreeMB) {
			error_log("WARNING: Low disk space in '$userFolder': only " . round($freeMB, 2) . " MB left.");
			return true;
		}

		return false;
	}

	function move_files_if_not_already_there($new_upload_md5_string, $update_uuid, $BASEURL, $user_id, $experiment_name, $run_id, $offered_files, $userFolder, $uuid_folder, $sharesPath) {
		$added_files = 0;
		$project_md5 = hash('md5', $new_upload_md5_string);

		$found_hash_file_data = search_for_hash_file("$sharesPath/$user_id/$experiment_name/*/hash.md5", $project_md5, $userFolder);

		$found_hash_file = $found_hash_file_data[0];
		$found_hash_file_dir = $found_hash_file_data[1];

		$password_hash_file = "$userFolder/password.sha256";
		$get_password = "";

		if(isset($_GET["password"]) && $_GET["password"]) {
			$get_password = hash("sha256", $_GET["password"]);
		}

		if ($found_hash_file && is_null($update_uuid)) {
			list($user, $experiment_name, $run_id) = extract_path_components($found_hash_file_dir, $sharesPath);
			$old_url = remove_extra_slashes_from_url("$BASEURL/share?user_id=$user_id&experiment_name=$experiment_name&run_nr=$run_id");
			echo "This project already seems to have been uploaded. See $old_url\n";
			exit(0);
		} else {
			$url = remove_extra_slashes_from_url("$BASEURL/share?user_id=$user_id&experiment_name=$experiment_name&run_nr=$run_id");

			$first_message = "See $url for live-results.\n";
			$second_message = "Run was successfully shared. See $url\nYou can share the link. It is valid for 90 days.\n";

			if (!(!$uuid_folder || !is_dir($uuid_folder))) {
				$second_message = $first_message;
			}

			if($get_password) {
				file_put_contents($password_hash_file, $get_password);
			}

			move_files(
				$offered_files,
				$added_files,
				$userFolder,
				$first_message,
				$second_message
			);
		}
	}

	function get_offered_files($acceptable_files, $acceptable_file_names, $i) {
		foreach ($acceptable_files as $acceptable_file) {
			$offered_files[$acceptable_file] = array(
				"file" => $_FILES[$acceptable_file]['tmp_name'] ?? null,
				"filename" => $acceptable_file_names[$i]
			);
			$i++;
		}

		return [$offered_files, $i];
	}

	function rrmdir($dir) {
		if (is_dir($dir)) {
			$objects = scandir($dir);

			foreach ($objects as $object) {
				if ($object != '.' && $object != '..') {
					$object_path = $dir.'/'.$object;
					if (filetype($object_path) == 'dir') {
						rrmdir($object_path);
					} else {
						if (file_exists($object_path)) {
							try {
								my_unlink($object_path);
							} catch (Exception $e) {
								error_log("\nAn exception occured trying to move $file to $userFolder/$filename: $e\n");
							}
						}
					}
				}
			}

			reset($objects);

			if(is_dir($dir)) {
				rmdir($dir);
			}
		}
	}

	function delete_empty_directories(string $directory): bool {
		if (!is_dir($directory)) {
			return false;
		}

		$iterator = new RecursiveIteratorIterator(
			new RecursiveDirectoryIterator(
				$directory,
				FilesystemIterator::SKIP_DOTS | FilesystemIterator::CURRENT_AS_SELF
			),
			RecursiveIteratorIterator::CHILD_FIRST
		);

		$anyDeleted = false;
		$now = time();

		foreach ($iterator as $fileInfo) {
			if (!$fileInfo->isDir()) {
				continue;
			}

			$path = $fileInfo->getPathname();

			// Schnell prüfen, ob leer
			$isEmpty = true;
			$handle = @opendir($path);
			if ($handle !== false) {
				while (($entry = readdir($handle)) !== false) {
					if ($entry !== '.' && $entry !== '..') {
						$isEmpty = false;
						break;
					}
				}
				closedir($handle);
			}

			// Wenn leer und älter als 1 Tag, löschen
			if ($isEmpty && $fileInfo->getMTime() < $now - 86400) {
				if (@rmdir($path)) {
					$anyDeleted = true;
				}
			}
		}

		return $anyDeleted;
	}

	function _delete_old_shares($dir) {
		$oldDirectories = [];
		$currentTime = time();

		function is_dir_empty($dir) {
			return (is_readable($dir) && count(scandir($dir)) == 2);
		}

		foreach (glob("$dir/*/*/*", GLOB_ONLYDIR) as $subdir) {
			$pathParts = explode('/', $subdir);
			$username_dir = $pathParts[1] ?? '';

			if ($username_dir != "s4122485" && $username_dir != "pwinkler") {
				$threshold = ($username_dir === 'runner' || $username_dir === "defaultuser" || $username_dir === "admin") ? 3600 : (3 * 30 * 24 * 3600);

				if(is_dir($subdir)) {
					$dir_date = filemtime($subdir);

					if (is_dir($subdir) && ($dir_date < ($currentTime - $threshold))) {
						$oldDirectories[] = $subdir;
						rrmdir($subdir);
					}

					if (is_dir($subdir) && is_dir_empty($subdir)) {
						$oldDirectories[] = $subdir;
						rrmdir($subdir);
					}
				}
			}
		}

		return $oldDirectories;
	}

	function delete_old_shares () {
		try {
			$oldDirs = _delete_old_shares($GLOBALS["sharesPath"]);
			delete_empty_directories($GLOBALS["sharesPath"], false);
			return $oldDirs;
		} catch (e) {
			error_log(e);
		}
	}

	function ascii_table_to_html($asciiTable) {
		$lines = explode("\n", trim($asciiTable));

		while (!empty($lines) && trim($lines[0]) === '') {
			array_shift($lines);
		}

		$headerText = null;
		if (!empty($lines) && !preg_match('/^[\s]*[┏━┡┩└─]+/u', $lines[0])) {
			$headerText = array_shift($lines);
		}

		$lines = array_filter($lines, function ($line) {
			return !preg_match('/^[\s]*[┏━┡┩└─]+/u', $line);
		});

		if (empty($lines)) return '<p>Error: No valid table found.</p>';

		$headerLine = array_shift($lines);
		$headerCells = preg_split('/\s*[┃│]\s*/u', trim($headerLine, "┃│"));

		$html = $headerText ? "<h2>$headerText</h2>" : '';
		$html .= '<table cellspacing="0" cellpadding="5"><thead><tr>';
		foreach ($headerCells as $cell) {
			$html .= '<th>' . $cell . '</th>';
		}
		$html .= '</tr></thead><tbody>';

		foreach ($lines as $line) {
			$cells = preg_split('/\s*[┃│]\s*/u', trim($line, "┃│"));
			if ($cells !== false && count($cells) === count($headerCells)) {
				$html .= '<tr>';
				foreach ($cells as $cell) {
					$html .= '<td>' . $cell . '</td>';
				}
				$html .= '</tr>';
			}
		}

		$html .= '</tbody></table>';
		return $html;
	}

	function analyze_column_types($csv_data, $column_indices) {
		$column_analysis = [];

		foreach ($column_indices as $index => $column_name) {
			$has_string = false;
			$has_numeric = false;

			foreach ($csv_data as $row) {
				if (!isset($row[$index])) {
					continue;
				}
				if (is_numeric($row[$index])) {
					$has_numeric = true;
				} else {
					$has_string = true;
				}

				if ($has_numeric && $has_string) {
					break;
				}
			}

			$column_analysis[$column_name] = [
				'numeric' => $has_numeric,
				'string' => $has_string
			];
		}

		return $column_analysis;
	}

	function count_column_types($column_analysis) {
		$nr_numerical = 0;
		$nr_string = 0;

		foreach ($column_analysis as $column => $types) {
			if (!empty($types['numeric']) && empty($types['string'])) {
				$nr_numerical++;
			} elseif (!empty($types['string'])) {
				$nr_string++;
			}
		}

		return [$nr_numerical, $nr_string];
	}

	if (!function_exists('str_starts_with')) {
		function str_starts_with(string $haystack, string $needle): bool {
			return substr($haystack, 0, strlen($needle)) === $needle;
		}
	}

	function add_pareto_from_from_file($tabs, $warnings, $run_dir) {
		$pareto_front_txt_file = "$run_dir/pareto_front_table.txt";
		$pareto_front_json_file = "$run_dir/pareto_front_data.json";
		$pareto_front_json_points_file = "$run_dir/pareto_idxs.json";

		$svg_icon = get_icon_html("pareto.svg");

		if(file_exists($pareto_front_json_points_file) && filesize($pareto_front_json_points_file)) {
			$pareto_idxs_json_content = file_get_contents($pareto_front_json_points_file);

			$GLOBALS["json_data"]["pareto_idxs"] = json_decode(json_encode(json_decode($pareto_idxs_json_content)), true);

			$pareto_front_html = "<div id='pareto_front_idxs_plot_container'></div><div id='pareto_from_idxs_table'></div>\n";

			$tabs["{$svg_icon}Pareto-Fronts-Estimation"] = [
				'id' => 'tab_pareto_fronts',
				'content' => $pareto_front_html,
				'onclick' => "load_pareto_graph_from_idxs();"
			];

		} else if(file_exists($pareto_front_json_file) && file_exists($pareto_front_txt_file) && filesize($pareto_front_json_file) && filesize($pareto_front_txt_file)) {
			$pareto_front_html = "";

			$pareto_front_text = remove_ansi_colors(my_htmlentities(file_get_contents($pareto_front_txt_file)));

			if($pareto_front_text) {
				$pareto_front_html .= "<pre>$pareto_front_text</pre>";
			}

			$pareto_json_content = file_get_contents($pareto_front_json_file);

			$GLOBALS["json_data"]["pareto_front_data"] = json_decode($pareto_json_content);

			if($pareto_front_html) {
				$pareto_front_html = "<div class='caveat warning'>The old algorithm for calculating the pareto-front was buggy. Please re-calculate them using <tt>bash omniopt --calculate_pareto_front_of_job runs/path_to_job/run_nr --live_share</tt>.</div><div id='pareto_front_graphs_container'></div>\n$pareto_front_html";

				$tabs["{$svg_icon}Pareto-Fronts-Estimation"] = [
					'id' => 'tab_pareto_fronts',
					'content' => $pareto_front_html
				];
			}
		} else {
			if(!file_exists($pareto_front_json_points_file)) {
				$warnings[] = "$pareto_front_json_points_file not found";
			} else if(!filesize($pareto_front_json_points_file)) {
				$warnings[] = "$pareto_front_json_points_file is empty";
			}

			if(!file_exists($pareto_front_json_file)) {
				$warnings[] = "$pareto_front_json_file not found";
			} else if(!filesize($pareto_front_json_file)) {
				$warnings[] = "$pareto_front_json_file is empty";
			}

			if(!file_exists("$pareto_front_txt_file")) {
				$warnings[] = "$pareto_front_txt_file not found";
			} else if(!filesize("$pareto_front_txt_file")) {
				$warnings[] = "$pareto_front_txt_file is empty";
			}
		}

		return [$tabs, $warnings];
	}

	function get_outfiles_tab_from_run_dir ($run_dir, $tabs, $warnings, $result_names) {
		$out_files = get_log_files($run_dir);

		if(count($out_files)) {
			[$content, $i] = generate_log_tabs($run_dir, $out_files, $result_names);

			if ($i > 0) {
				$svg_icon = get_icon_html("tabs.svg");

				$tabs["{$svg_icon}Single Logs"] = [
					'id' => 'tab_logs',
					'content' => $content
				];
			}
		} else {
			$warnings[] = "No out-files found";
		}

		return [$tabs, $warnings];
	}

	function add_tabs_to_string($inputString, $numTabs) {
		$lines = explode("\n", $inputString);
		$tabs = str_repeat("\t", $numTabs);
		$inPre = false;

		foreach ($lines as &$line) {
			$trimmed = strtolower(trim($line));

			if (strpos($trimmed, '<pre') !== false) {
				$inPre = true;
			}

			if (!$inPre) {
				$line = $tabs . $line;
			}

			if (strpos($trimmed, '</pre>') !== false) {
				$inPre = false;
			}
		}

		return implode("\n", $lines);
	}

	function remove_font_face_rules($cssContent) {
		$pattern = '/@font-face\s*\{[^}]*\}/s';
		$cleanedCss = preg_replace($pattern, '', $cssContent);

		return $cleanedCss;
	}

	function remove_excessive_newlines($string) {
		$cleanedString = preg_replace('/\n{3,}/', "\n", $string);

		return $cleanedString;
	}

	function generate_css_style_tag($filePath, $indentLevel = 3) {
		$file_content = file_get_contents($filePath);
		$file_content = remove_font_face_rules($file_content);

		$cssContent = remove_excessive_newlines($file_content);

		if ($cssContent === false) {
			error_log("Error: The file '$filePath' Could not be read.");
			return '';
		}

		//$cssContentWithTabs = add_tabs_to_string($cssContent, $indentLevel);
		//return $cssContentWithTabs."\n";

		return $cssContent;
	}

	function add_param_to_current_url(string $key, string $value): string {
		$scheme = (!empty($_SERVER['HTTPS']) && $_SERVER['HTTPS'] === 'on') ? 'https' : 'http';
		$host   = $_SERVER['HTTP_HOST'];
		$uri    = $_SERVER['REQUEST_URI'];

		$parts = parse_url($scheme.'://'.$host.$uri);
		parse_str($parts['query'] ?? '', $query);
		$query[$key] = $value;

		return $parts['scheme'].'://'.$parts['host']
			. ($parts['path'] ?? '')
			. '?'.http_build_query($query);
	}

	function get_export_tab ($tabs, $warnings, $run_dir) {
		if(!file_exists("js/share_functions.js")) {
			$warnings[] = "js/share_functions not found!";

			return [$tabs, $warnings];
		}

		$svg_icon = get_icon_html("export.svg");

		if(!isset($_GET["export"])) {
			$export_url = add_param_to_current_url('export', '1');

			$tabs["{$svg_icon}Export"] = [
				'id' => 'tab_export',
				'content' => "<a href='$export_url#tab_export'><button>Click here to enable the export (reloads the site)</button></a>"
			];

			return [$tabs, $warnings];
		}

		$run_dir = preg_replace('/^'.preg_replace("/\//", "\\/", $GLOBALS["sharesPath"]).'*/', "", $run_dir);

		$special_col_names = "var special_col_names = ".json_encode($GLOBALS["SPECIAL_COL_NAMES"]);

		$json_data_str = "";
		if(count($GLOBALS["json_data"])) {
			foreach ($GLOBALS["json_data"] as $json_name => $json_data) {
				if(!preg_match("/gpu/", $json_name)) {
					$json_data_str .= "var $json_name = " . implode("\n", array_map(fn($i, $l) => $i === 0 ? $l : "$l", array_keys(explode("\n", json_encode($json_data, JSON_PRETTY_PRINT))), explode("\n", json_encode($json_data, JSON_PRETTY_PRINT)))) . ";\n";
				}
			}
		}

		//$json_data_str = add_tabs_to_string($json_data_str, 3);

		$onclicks = [];
		$html_parts = [];

		$skipped_tab_names = [];

		foreach ($tabs as $tabname => $tab) {
			if(!preg_match("/(?:Single Logs|Main-Log|Debug-Logs|Job-Infos|GPU)/", $tabname)) {
				if (isset($tab['content'])) {
					$this_content = "<h1>$tabname</h1>\n".$tab["content"];

					$html_parts[] = $this_content;

					if (isset($tab['onclick'])) {
						$onclicks[] = $tab['onclick'];
					}
				}
			} else {
				$skipped_tab_names[] = $tabname;
			}
		}

		$uniqueOnclicks = array_unique($onclicks);

		$onclick_string = implode(";\n", $uniqueOnclicks);
		if (substr($onclick_string, -1) !== ';') {
			$onclick_string .= ';';
		}

		//$onclick_string = add_tabs_to_string($onclick_string, 4);

		//$html_parts_str = add_tabs_to_string(implode("\n", $html_parts), 3);
		$html_parts_str = implode("\n", $html_parts);

		$js_dir = "js";
		$js_functions = "";

		// Alle JS-Dateien im Ordner holen, alphabetisch sortiert
		$js_files = glob("$js_dir/*.js");
		sort($js_files);

		// share_functions.js und pareto_from_idxs.js später hinzufügen
		$defer_files = ["$js_dir/share_functions.js", "$js_dir/pareto_from_idxs.js"];
		$js_files = array_diff($js_files, $defer_files);

		// Erst alle anderen JS-Dateien anhängen
		foreach ($js_files as $file) {
			$js_functions .= "\n" . file_get_contents($file);
		}

		// Dann explizit share_functions.js und pareto_from_idxs.js anhängen
		foreach ($defer_files as $file) {
			if (file_exists($file)) {
				$js_functions .= "\n" . file_get_contents($file);
			}
		}

		//$js_functions = add_tabs_to_string($js_functions, 3);

		$share_css = generate_css_style_tag("css/share.css");
		$style_css = generate_css_style_tag("style.css");
		$xp_css = generate_css_style_tag("css/xp.css");

		$export_content = "<!DOCTYPE html>
<html lang='en'>
	<head>
		<meta charset='UTF-8'>
		<meta name='viewport' content='width=device-width, initial-scale=1.0'>
		<title>Exported &raquo;$run_dir&laquo; from OmniOpt2-Share</title>
		<script src='https://code.jquery.com/jquery-3.7.1.js'></script>
		<script src='https://cdnjs.cloudflare.com/ajax/libs/gridjs/6.2.0/gridjs.production.min.js'></script>
		<script src='https://cdn.jsdelivr.net/npm/plotly.js-dist@3.0.1/plotly.min.js'></script>
		<link rel='stylesheet' href='https://cdnjs.cloudflare.com/ajax/libs/gridjs/6.2.0/theme/mermaid.css'>
		<style>
$share_css
$style_css
$xp_css
		</style>
	</head>
	<body>
		<script>
			var log = console.log;
			var theme = 'light';

			$special_col_names

$json_data_str
$js_functions

			$(document).ready(function() {
$onclick_string

				colorize_table_entries();
			});
		</script>

		$html_parts_str
	</body>
</html>
";

		$buttons = copy_id_to_clipboard_string("export_tab_content", 'export.html');

		ini_set('memory_limit', '1024M');

		$skipped_tab_names_string = "";
		if(count($skipped_tab_names)) {
			$skipped_tab_names_string = "Skipped tabs: <ul>\n\t<li>".implode("</li>\n\t<li>", $skipped_tab_names)."</li>\n</ul>";
		}

		if(!isset($_GET["export_and_exit"])) {
			$export_content = "$skipped_tab_names_string$buttons<pre class='no-highlight' id='export_tab_content'><!-- export.html -->".my_htmlentities($export_content)."\n<!-- export.html --></pre>$buttons";
		}

		$tabs["{$svg_icon}Export"] = [
			'id' => 'tab_export',
			'content' => $export_content
		];

		if(isset($_GET["export_and_exit"])) {
			print $export_content;
			exit(0);
		}

		return [$tabs, $warnings];
	}

	function clean_result_name_lines(array $lines) {
		$cleaned = [];

		foreach ($lines as $line) {
			$cleaned_line = preg_replace('/[^a-zA-ZäöüÖÄÜß_0-9]/u', '', $line);
			$cleaned[] = $cleaned_line;
		}

		return $cleaned;
	}


	function get_result_names_and_min_max ($run_dir, $warnings) {
		$result_names_file = "$run_dir/result_names.txt";
		$result_min_max_file = "$run_dir/result_min_max";

		if (!file_exists($result_min_max_file)) {
			$result_min_max_file = "$run_dir/result_min_max.txt";
		}

		$result_names = [];
		$result_min_max = [];

		if(is_file($result_names_file)) {
			$result_names = read_file_as_array($result_names_file);
		} else {
			$warnings[] = "$result_names_file not found";
		}

		if(is_file($result_min_max_file)) {
			$result_min_max = read_file_as_array($result_min_max_file);
		} else {
			$warnings[] = "$result_min_max_file not found";
		}

		$result_names = clean_result_name_lines($result_names);
		$result_min_max = clean_result_name_lines($result_min_max);

		return [$result_names, $result_min_max, $warnings];
	}

	function add_ui_url_from_file_to_overview($run_dir, $overview_html, $warnings) {
		$ui_url_txt = "$run_dir/ui_url.txt";
		if(is_file($ui_url_txt)) {
			$firstLine = fgets(fopen($ui_url_txt, 'r'));

			if (filter_var($firstLine, FILTER_VALIDATE_URL) && (strpos($firstLine, 'http://') === 0 || strpos($firstLine, 'https://') === 0)) {
				$overview_html .= "<button onclick=\"window.open('".htmlspecialchars($firstLine)."', '_blank')\">GUI page with all the settings of this job</button><br><br>";
			}
		} else {
			$warnings[] = "$ui_url_txt not found";
		}

		return [$overview_html, $warnings];
	}

	function add_constraints_to_overview ($run_dir, $overview_html, $warnings) {
		$constraints = "$run_dir/constraints.txt";
		if(file_exists($constraints) && filesize($constraints) && is_ascii_or_utf8($constraints)) {
			$constraints_table = ascii_table_to_html(remove_ansi_colors(my_htmlentities(file_get_contents($constraints))));
			if($constraints_table) {
				$constraints .= $constraints_table;

				$overview_html .= "<h2>Constraints</h2>";

				$overview_html .= $constraints_table;
			} else {
				$warnings[] = "Could not create \$constraints_table";
			}
		} else {
			if(!file_exists($constraints)) {
				$warnings[] = "$constraints not found";
			} else if(!filesize($constraints)) {
				$warnings[] = "$constraints is empty";
			} else if(!is_ascii_or_utf8($constraints)) {
				$warnings[] = "$constraints is not a ascii or utf8 file";
			}
		}

		return [$overview_html, $warnings];
	}

	function is_ascii_or_utf8($filepath) {
		if (isset($GLOBALS["ascii_or_utf8_cache"][$filepath])) {
			return $GLOBALS["ascii_or_utf8_cache"][$filepath];
		}

		if (!is_readable($filepath) || !is_file($filepath)) {
			return false;
		}

		$content = file_get_contents($filepath);
		if ($content === false) {
			return false;
		}

		$valid = mb_check_encoding($content, 'UTF-8');

		$GLOBALS["ascii_or_utf8_cache"][$filepath] = $valid;

		return $valid;
	}

	function add_experiment_overview_to_overview ($run_dir, $overview_html, $warnings) {
		$experiment_overview = "$run_dir/experiment_overview.txt";
		if(file_exists($experiment_overview) && filesize($experiment_overview) && is_ascii_or_utf8($experiment_overview)) {
			$experiment_overview_table = ascii_table_to_html(remove_ansi_colors(my_htmlentities(file_get_contents($experiment_overview))));
			if($experiment_overview_table) {
				$experiment_overview .= $experiment_overview_table;

				$overview_html .= $experiment_overview_table;
			} else {
				$warnings[] = "Could not create \$experiment_overview_table";
			}
		} else {
			if(!file_exists($experiment_overview)) {
				$warnings[] = "$experiment_overview not found";
			} else if(!filesize($experiment_overview)) {
				$warnings[] = "$experiment_overview is empty";
			} else if(!is_ascii_or_utf8($experiment_overview)) {
				$warnings[] = "$experiment_overview is not Ascii or UTF8";
			}
		}

		return [$overview_html, $warnings];
	}

	function add_best_results_to_overview ($run_dir, $overview_html, $warnings) {
		$best_results_txt = "$run_dir/best_result.txt";
		if(is_file($best_results_txt) && filesize($best_results_txt) && is_ascii_or_utf8($best_results_txt)) {
			$overview_html .= ascii_table_to_html(remove_ansi_colors(my_htmlentities(file_get_contents($best_results_txt))));
		} else {
			if(!is_file($best_results_txt)) {
				$warnings[] = "$best_results_txt not found";
			} else if (!filesize($best_results_txt)) {
				$warnings[] = "$best_results_txt is empty";
			} else if (!is_ascii_or_utf8($best_results_txt)) {
				$warnings[] = "$best_results_txt is not Ascii or UTF8";
			}
		}

		return [$overview_html, $warnings];
	}

	function add_parameters_to_overview ($run_dir, $overview_html, $warnings) {
		$parameters_txt_file = "$run_dir/parameters.txt";
		if(is_file($parameters_txt_file) && filesize($parameters_txt_file) && is_ascii_or_utf8($parameters_txt_file)) {
			$overview_html .= ascii_table_to_html(remove_ansi_colors(my_htmlentities(file_get_contents("$run_dir/parameters.txt"))));
		} else {
			if(!is_file($parameters_txt_file)) {
				$warnings[] = "$run_dir/parameters.txt not found";
			} else if (!filesize($parameters_txt_file)) {
				$warnings[] = "$run_dir/parameters.txt is empty";
			} else if (!is_ascii_or_utf8($parameters_txt_file)) {
				$warnings[] = "$run_dir/parameters.txt is not Ascii or utf8";
			}
		}

		return [$overview_html, $warnings];
	}

	function add_progressbar_to_overview($run_dir, $overview_html, $warnings) {
		$progressbar_file = "$run_dir/progressbar";
		if(file_exists($progressbar_file) && filesize($progressbar_file) && is_ascii_or_utf8($progressbar_file)) {
			$lastLine = trim(array_slice(file($progressbar_file), -1)[0]);

			$overview_html .= "<h2>Last progressbar status</h2>\n";
			$overview_html .= "<tt>".my_htmlentities(remove_ansi_colors($lastLine))."</tt>";
		} else {
			if(!is_file($progressbar_file)) {
				$warnings[] = "$progressbar_file not found";
			} else if(!filesize($progressbar_file)) {
				$warnings[] = "$progressbar_file is empty";
			} else if(!is_ascii_or_utf8($progressbar_file)) {
				$warnings[] = "$progressbar_file is not Ascii or UTF8";
			}
		}

		return [$overview_html, $warnings];
	}

	function add_result_names_table_to_overview ($result_names, $result_min_max, $overview_html, $warnings) {
		if(count($result_names)) {
			$result_names_table = '<h2>Result names and types</h2>'."\n";
			$result_names_table .= '<table>'."\n";
			$result_names_table .= '<tr><th>name</th><th>min/max</th></tr>'."\n";
			for ($i = 0; $i < count($result_names); $i++) {
				$min_or_max = "min";

				if(isset($result_min_max[$i])) {
					$min_or_max = $result_min_max[$i];
				}

				$result_names_table .= '<tr>'."\n";
				$result_names_table .= '<td>' . htmlspecialchars($result_names[$i]) . '</td>'."\n";
				$result_names_table .= '<td>' . htmlspecialchars($min_or_max) . '</td>'."\n";
				$result_names_table .= '</tr>'."\n";
			}
			$result_names_table .= '</table>'."\n";

			$overview_html .= $result_names_table;
		} else {
			$warnings[] = "No result-names could be found";
		}

		return [$overview_html, $warnings];
	}

	function add_overview_for_jobs_and_generation_stuff($run_dir, $overview_html, $warnings) {
		$results_csv_file = "$run_dir/results.csv";

		if (!is_ascii_or_utf8($results_csv_file)) {
			$warnings[] = "Is not Ascii or utf8: $results_csv_file";
			return [$overview_html, $warnings];
		}

		if (!file_exists($results_csv_file)) {
			$warnings[] = "Missing file: $results_csv_file";
			return [$overview_html, $warnings];
		}

		if (!filesize($results_csv_file)) {
			$warnings[] = "File is empty: $results_csv_file";
			return [$overview_html, $warnings];
		}

		if (!is_readable($results_csv_file)) {
			$warnings[] = "File not readable: $results_csv_file";
			return [$overview_html, $warnings];
		}

		$handle = fopen($results_csv_file, 'r');
		if ($handle === false) {
			$warnings[] = "Failed to open file: $results_csv_file";
			return [$overview_html, $warnings];
		}

		$delimiter = ",";
		$enclosure = "\"";
		$escape = "\\";

		$header = fgetcsv($handle, 0, $delimiter, $enclosure, $escape);
		if ($header === false) {
			fclose($handle);
			$warnings[] = "Missing header line in file: $results_csv_file";
			return [$overview_html, $warnings];
		}

		$index_generation = array_search('generation_node', $header);
		$index_status = array_search('trial_status', $header);

		if ($index_generation === false || $index_status === false) {
			fclose($handle);
			$warnings[] = "Required columns 'generation_node' or 'trial_status' not found in file: $results_csv_file";
			return [$overview_html, $warnings];
		}

		$summary = array();
		$all_statuses = array();

		while (($row = fgetcsv($handle, 0, $delimiter, $enclosure, $escape)) !== false) {
			if (!isset($row[$index_generation]) || !isset($row[$index_status])) {
				continue;
			}

			$gen = trim($row[$index_generation]);
			$status = strtoupper(trim($row[$index_status]));

			if ($gen === '') {
				$gen = 'UNKNOWN';
			}

			if (!isset($summary[$gen])) {
				$summary[$gen] = array('total' => 0);
			}

			if (!isset($summary[$gen][$status])) {
				$summary[$gen][$status] = 0;
				$all_statuses[$status] = true;
			}

			$summary[$gen]['total'] += 1;
			$summary[$gen][$status] += 1;
		}

		fclose($handle);

		$status_columns = array_keys($all_statuses);
		sort($status_columns, SORT_STRING | SORT_FLAG_CASE);

		$overview_html .= "<h2>Job Summary per Generation Node</h2>\n";
		$overview_html .= "<table border='1' cellpadding='5' cellspacing='0'>\n";
		$overview_html .= "<thead><tr><th>Generation Node</th><th>Total</th>";

		foreach ($status_columns as $status) {
			$overview_html .= "<th>" . htmlspecialchars($status) . "</th>";
		}

		$overview_html .= "</tr></thead>\n";
		$overview_html .= "<tbody>\n";

		foreach ($summary as $gen => $counts) {
			$overview_html .= "<tr>";
			$overview_html .= "<td>" . htmlspecialchars($gen) . "</td>";
			$overview_html .= "<td>" . htmlspecialchars((string)$counts['total']) . "</td>";

			foreach ($status_columns as $status) {
				$count = isset($counts[$status]) ? $counts[$status] : 0;
				$overview_html .= "<td>" . htmlspecialchars((string)$count) . "</td>";
			}

			$overview_html .= "</tr>\n";
		}

		$overview_html .= "</tbody></table>\n";

		return [$overview_html, $warnings];
	}

	function add_git_version_to_overview ($run_dir, $overview_html, $warnings) {
		$git_version_file = "$run_dir/git_version";
		if(file_exists($git_version_file) && filesize($git_version_file) && is_ascii_or_utf8($git_version_file)) {
			$lastLine = my_htmlentities(file_get_contents($git_version_file));

			$overview_html .= "<br>\n";
			$overview_html .= "<h2>Git-Version</h2>\n";
			$overview_html .= "<tt>".my_htmlentities($lastLine)."</tt>";
		} else {
			if(!is_file($git_version_file)) {
				$warnings[] = "$git_version_file not found";
			} else if (!filesize($git_version_file)) {
				$warnings[] = "$git_version_file empty";
			} else if (!is_ascii_or_utf8($git_version_file)) {
				$warnings[] = "$git_version_file is not Ascii or UTF8";
			}
		}

		return [$overview_html, $warnings];
	}

	function add_insights_from_file($tabs, $warnings, $run_dir, $result_names, $result_min_max) {
		$results_csv_file = "$run_dir/results.csv";

		if(is_file($results_csv_file) && filesize($results_csv_file)) {
			$status_data = get_status_for_results_csv($results_csv_file);

			if($status_data["total"]) {
				$natural_language_markdown = nl2br(create_insights($results_csv_file, $result_names, $result_min_max));
				$svg_icon = get_icon_html("insights.svg");

				$html = convert_markdown_to_html($natural_language_markdown);

				$tabs["{$svg_icon}Insights"] = [
					'id' => 'tab_insights',
					'content' => $html,
					"onclick" => "initializeResultParameterVisualizations()"
				];


			} else {
				$warnings[] = "No evaluations detected";
			}
		} else {
			if(!is_file($results_csv_file)) {
				$warnings[] = "$results_csv_file not found";
			} else if (!filesize($results_csv_file)) {
				$warnings[] = "$results_csv_file is empty";
			}
		}

		return [$tabs, $warnings];
	}

	function add_overview_table_to_overview_and_get_status_data ($run_dir, $status_data, $overview_html, $warnings) {
		$results_csv_file = "$run_dir/results.csv";

		if(is_file($results_csv_file) && filesize($results_csv_file) && is_ascii_or_utf8($results_csv_file)) {
			$status_data = get_status_for_results_csv($results_csv_file);

			if($status_data["total"]) {
				$overview_table = '<h2>Number of evaluations</h2>'."\n";
				$overview_table .= '<table>'."\n";
				$overview_table .= '<tbody>'."\n";
				$overview_table .= '<tr>'."\n";

				foreach ($status_data as $key => $value) {
					$capitalizedKey = ucfirst($key);
					$overview_table .= '<th>' . $capitalizedKey . '</th>'."\n";
				}
				$overview_table .= '</tr>'."\n";

				$overview_table .= '<tr>'."\n";

				foreach ($status_data as $value) {
					$overview_table .= '<td>' . $value . '</td>'."\n";
				}
				$overview_table .= '</tr>'."\n";

				$overview_table .= '</tbody>'."\n";
				$overview_table .= '</table>'."\n";

				$overview_html .= "$overview_table";
			} else {
				$warnings[] = "No evaluations detected";
			}
		} else {
			if(!is_file($results_csv_file)) {
				$warnings[] = "$results_csv_file not found";
			} else if (!filesize($results_csv_file)) {
				$warnings[] = "$results_csv_file is empty";
			} else if (!is_ascii_or_utf8($results_csv_file)) {
				$warnings[] = "$results_csv_file is not Ascii or UTF8";
			}
		}

		return [$overview_html, $warnings, $status_data];
	}

	function add_overview_tab($tabs, $warnings, $run_dir, $status_data, $result_names, $result_min_max) {
		$overview_html = "";

		[$overview_html, $warnings] = add_ui_url_from_file_to_overview($run_dir, $overview_html, $warnings);
		[$overview_html, $warnings] = add_experiment_overview_to_overview($run_dir, $overview_html, $warnings);
		[$overview_html, $warnings] = add_best_results_to_overview($run_dir, $overview_html, $warnings);
		[$overview_html, $warnings] = add_overview_for_jobs_and_generation_stuff($run_dir, $overview_html, $warnings);
		[$overview_html, $warnings] = add_parameters_to_overview($run_dir, $overview_html, $warnings);
		[$overview_html, $warnings] = add_constraints_to_overview($run_dir, $overview_html, $warnings);
		[$overview_html, $warnings, $status_data] = add_overview_table_to_overview_and_get_status_data($run_dir, $status_data, $overview_html, $warnings);
		[$overview_html, $warnings] = add_result_names_table_to_overview($result_names, $result_min_max, $overview_html, $warnings);
		[$overview_html, $warnings] = add_progressbar_to_overview($run_dir, $overview_html, $warnings);
		[$overview_html, $warnings] = add_git_version_to_overview($run_dir, $overview_html, $warnings);

		if($overview_html != "") {
			$svg_icon = get_icon_html("overview.svg");

			$tabs["{$svg_icon}Overview"] = [
				'id' => 'tab_overview',
				'content' => $overview_html
			];
		} else {
			$warnings[] = "\$overview_html was empty";
		}

		return [$tabs, $warnings, $status_data];
	}

	function find_gpu_usage_files($run_dir) {
		if (!is_dir($run_dir)) {
			error_log("Error: Directory '$run_dir' does not exist or is not accessible.");
			return [];
		}

		$pattern = $run_dir . DIRECTORY_SEPARATOR . 'gpu_usage__*.csv';
		$files = glob($pattern);

		if ($files === false) {
			return [];
		}

		return $files;
	}

	function parse_gpu_usage_files($files) {
		$gpu_usage_data = [];

		$headers = [
			"timestamp", "name", "pci.bus_id", "driver_version", "pstate",
			"pcie.link.gen.max", "pcie.link.gen.current", "temperature.gpu",
			"utilization.gpu", "utilization.memory", "memory.total",
			"memory.free", "memory.used"
		];

		$keep_cols = ['timestamp', 'utilization.gpu', 'temperature.gpu'];

		$headerIndexes = [];
		foreach ($headers as $idx => $colName) {
			if (in_array($colName, $keep_cols)) {
				$headerIndexes[$colName] = $idx;
			}
		}

		foreach ($files as $file) {
			$basename = basename($file);
			if (is_file($file) && is_ascii_or_utf8($file) && preg_match('/gpu_usage__i(\d+)\.csv/', $basename, $matches)) {
				$index = $matches[1];
				$gpu_usage_data[$index] = [];

				$handle = fopen($file, "r");
				if ($handle !== false) {
					while (($data = fgetcsv($handle, 0, ",", '"', "\\")) !== false) {
						if (count($data) !== count($headers)) {
							continue;
						}

						$timestampRaw = trim($data[$headerIndexes['timestamp']]);
						$utilGpuRaw = str_replace('%', '', trim($data[$headerIndexes['utilization.gpu']]));
						$tempGpuRaw = str_replace('MiB', '', trim($data[$headerIndexes['temperature.gpu']]));

						$ts = strtotime($timestampRaw);
						if ($ts === false) {
							continue;
						}

						$utilGpu = intval($utilGpuRaw);
						$tempGpu = intval($tempGpuRaw);

						$gpu_usage_data[$index][] = [$ts, $utilGpu, $tempGpu];
					}
					fclose($handle);
				} else {
					error_log("Error: Could not open file '$file'.");
				}
			}
		}

		return $gpu_usage_data;
	}

	function has_non_empty_folder($dir) {
		if (!is_dir($dir)) {
			return false;
		}

		$files = new RecursiveIteratorIterator(new RecursiveDirectoryIterator($dir), RecursiveIteratorIterator::LEAVES_ONLY);

		foreach ($files as $file) {
			if (!$file->isDir()) {
				return true;
			}
		}

		return false;
	}

	function get_icon_html ($name) {
		return "<img class='invert_icon' src='i/$name' style='height: 1em' />&nbsp;";
	}

	function check_and_filter_tabs($re, $tabs, $warnings) {
		if (!preg_match('/^[()a-zA-Z0-9|\s]+$/', $re)) {
			dier('Wrong Regex format: Only a-z, A-Z, space, parentheses, and | allowed');
		}

		$pattern = '/' . $re . '/i';

		foreach ($tabs as $key => $value) {
			if (preg_match($pattern, $key)) {
				$warnings[] = "Filtered out Tab '$key'";
				unset($tabs[$key]);
			}
		}

		return [$tabs, $warnings];
	}

	function get_latest_recursive_modification_time($folderPath) {
		if (isset($GLOBALS["recursiveModificationCache"][$folderPath])) {
			return $GLOBALS["recursiveModificationCache"][$folderPath];
		}
		$latestTime = 0;

		$iterator = new RecursiveIteratorIterator(
			new RecursiveDirectoryIterator($folderPath, FilesystemIterator::SKIP_DOTS),
			RecursiveIteratorIterator::SELF_FIRST
		);

		foreach ($iterator as $item) {
			$modTime = $item->getMTime();
			if ($modTime > $latestTime) {
				$latestTime = $modTime;
			}
		}

		if ($latestTime === 0 && is_dir($folderPath)) {
			$latestTime = filemtime($folderPath);
		}

		$GLOBALS["recursiveModificationCache"][$folderPath] = $latestTime;
		return $latestTime;
	}

	function sort_folders_by_modification_time($basePath, &$folders) {
		usort($folders, function($a, $b) use ($basePath) {
			$timeA = get_latest_recursive_modification_time("$basePath/$a");
			$timeB = get_latest_recursive_modification_time("$basePath/$b");
			return $timeB <=> $timeA;
		});
	}

	function get_valid_folders($path) {
		$folders = [];
		if (!is_dir($path)) return $folders;

		$dir = opendir($path);
		while (($entry = readdir($dir)) !== false) {
			if ($entry === '.' || $entry === '..') continue;
			$full = $path . '/' . $entry;
			if (is_dir($full) && preg_match('/^[a-zA-Z0-9-_]+$/', $entry)) {
				$folders[] = $entry;
			}
		}
		closedir($dir);
		return $folders;
	}

	function generate_folder_tree_view($basePath) {
		if (!is_dir($basePath)) {
			echo "Base path does not exist.";
			return;
		}

		$currentUser = $_GET['user_id'] ?? null;
		$currentExp  = $_GET['experiment_name'] ?? null;
		$currentRun  = $_GET['run_nr'] ?? null;

		echo '<ul class="tree-view">';

		$users = get_valid_folders($basePath);
		sort_folders_by_modification_time($basePath, $users);

		foreach ($users as $user) {
			$userPath = "$basePath/$user";
			$experiments = get_valid_folders($userPath);
			sort_folders_by_modification_time($userPath, $experiments);

			$hasValidRun = false;
			foreach ($experiments as $experiment) {
				$experimentPath = "$userPath/$experiment";
				$runs = get_valid_folders($experimentPath);
				sort_folders_by_modification_time($experimentPath, $runs);

				foreach ($runs as $run) {
					$runPath = "$experimentPath/$run";
					if (has_non_empty_folder($runPath)) {
						$hasValidRun = true;
						break 2;
					}
				}
			}

			if (!$hasValidRun) continue;

			$userIsOpen = ($currentUser === $user);
			$userLink = htmlspecialchars("share?user_id=$user");
			echo '<li><details' . ($userIsOpen ? ' open' : '') . '><summary><a href="' . $userLink . '">' . htmlspecialchars($user) . '</a></summary><ul>';

			foreach ($experiments as $experiment) {
				$experimentPath = "$userPath/$experiment";
				$runs = get_valid_folders($experimentPath);
				sort_folders_by_modification_time($experimentPath, $runs);

				$validRunItems = [];

				foreach ($runs as $run) {
					$runPath = "$experimentPath/$run";

					if (!has_non_empty_folder($runPath)) continue;

					$timestamp = get_latest_modification_time($runPath);
					$lastModified = date("F d Y H:i:s", $timestamp);
					$timeSince = time_since($timestamp);
					$bracket_string = "$lastModified | $timeSince";

					$res_csv = "$runPath/results.csv";
					$show = 1;

					if (file_exists($res_csv)) {
						$analyzed = analyze_results_csv($res_csv);
						if ($analyzed) {
							$bracket_string .= " | $analyzed";
						}
					} else {
						$counted_subfolders = count_subfolders_or_files($runPath);
						if ($counted_subfolders > 0) {
							$bracket_string .= " | $counted_subfolders " . ($counted_subfolders === 1 ? "subfolder" : "subfolders");
						} else {
							$show = 0;
						}
					}

					if ($show) {
						$href = htmlspecialchars("share?user_id=$user&experiment_name=$experiment&run_nr=$run");
						$validRunItems[] = '<li><a href="' . $href . '">' . htmlspecialchars($run) . ' (' . htmlspecialchars($bracket_string) . ')</a></li>';
					}
				}

				if (count($validRunItems) > 0) {
					$experimentIsOpen = ($currentUser === $user && $currentExp === $experiment);
					$expLink = htmlspecialchars("share?user_id=$user&experiment_name=$experiment");
					echo '<li><details' . ($experimentIsOpen ? ' open' : '') . '><summary><a href="' . $expLink . '">' . htmlspecialchars($experiment) . '</a></summary><ul>';
					echo implode('', $validRunItems);
					echo '</ul></details></li>';
				}
			}

			echo '</ul></details></li>';
		}

		echo '</ul>';
	}

	function my_htmlentities ($str) {
		return htmlentities($str, ENT_QUOTES, 'utf-8');
	}
?>
