<?php
	include_once("share_functions.php");

	try {
		$run_nr = validate_param("run_nr", "/^\d+$/", "Invalid run_nr");
		$user_id = validate_param("user_id", "/^[a-zA-Z0-9_]+$/", "Invalid user_id");
		$experiment_name = validate_param("experiment_name", "/^[a-zA-Z0-9_-]+$/", "Invalid experiment_name");
		// Allow nested paths (e.g. single_runs/<jobid>/<file>) which the new
		// share protocol stores under single_runs/.  Reject traversal etc.
		$filename = validate_param("filename", "/^[a-zA-Z0-9\._\/]+$/", "Invalid filename");
		if(strpos($filename, "..") !== false || strpos($filename, "\\") !== false || strpos($filename, "\0") !== false) {
			respond_with_error("Invalid filename");
		}
		$run_folder_without_shares = build_run_folder_path($user_id, $experiment_name, $run_nr);
		$run_folder = $GLOBALS["sharesPath"]."/$run_folder_without_shares";

		validate_directory($run_folder);

		// Fallback: unpack bundle.zip if the requested file isn't there yet.
		if (!file_exists("$run_folder/$filename")) {
			ensure_share_bundle_extracted($run_folder);
		}

		$path = "$run_folder/$filename";

		if(file_exists($path)) {
			// Large or ANSI/backspace-heavy logs must never be converted
			// server-side: the converter builds several full-size arrays and
			// can exhaust PHP's memory limit.  Return at most a bounded tail
			// and let the browser convert ANSI to HTML.
			$max_bytes = 2 * 1024 * 1024;
			$server_convert_max_bytes = 256 * 1024;
			$max_control_chars = 2000;
			$size = filesize($path);
			$raw = isset($_GET["raw"]) && $_GET["raw"] !== "0";

			if($raw || $size > $max_bytes) {
				$fh = @fopen($path, "rb");
				if(!$fh) {
					respond_with_error("Cannot read $path");
				}
				$truncated = $size > $max_bytes;
				if($truncated) {
					fseek($fh, -$max_bytes, SEEK_END);
				}
				$content = stream_get_contents($fh);
				fclose($fh);
				if($content === false) {
					respond_with_error("Cannot read $path");
				}
				if($truncated) {
					$content = "... [log truncated: showing the last "
						. round($max_bytes / (1024 * 1024), 1)
						. " MiB of " . round($size / (1024 * 1024), 1)
						. " MiB] ...\n\n" . $content;
				}
				respond_with_json([
					"content" => $content,
					"mode" => "ansi",
					"truncated" => $truncated,
				]);
			} else {
				$content = file_get_contents($path);
				if($content === false) {
					respond_with_error("Cannot read $path");
				}

				// Logs with many escape/backspace sequences make the server
				// converter allocate huge token arrays.  Let the browser
				// handle those as well.
				$control_chars = substr_count($content, "\x1b") + substr_count($content, "\x08");
				if($size > $server_convert_max_bytes || $control_chars > $max_control_chars) {
					respond_with_json([
						"content" => $content,
						"mode" => "ansi",
						"truncated" => false,
					]);
				}

				$out_file = ansi_to_html(htmlentities($content));
				$out_file = highlight_debug_info($out_file);
				respond_with_json([
					"content" => $out_file,
					"mode" => "html",
					"truncated" => false,
				]);
			}
		} else {
			respond_with_json([
				"content" => build_missing_log_help_html($run_folder, $filename),
				"mode" => "html",
				"truncated" => false,
			]);
		}
	} catch (Exception $e) {
		respond_with_error($e->getMessage());
	}
?>
