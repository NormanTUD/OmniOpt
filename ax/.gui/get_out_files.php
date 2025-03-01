<?php
	error_reporting(E_ALL);
	set_error_handler(
		function ($severity, $message, $file, $line) {
			throw new \ErrorException($message, $severity, $severity, $file, $line);
		}
	);

	ini_set('display_errors', 1);

	include_once "_functions.php";
	include_once "share_functions.php";

	$sharesPath = "./shares/";

	try {
		$run_nr = validate_param("run_nr", "/^\d+$/", "Invalid run_nr");
		$user_id = validate_param("user_id", "/^[a-zA-Z0-9_]+$/", "Invalid user_id");
		$experiment_name = validate_param("experiment_name", "/^[a-zA-Z0-9_-]+$/", "Invalid experiment_name");
		$environment_share_path = get_or_env("share_path");
		$fn = get_or_env("fn");
		$sharesPath = determine_share_path($environment_share_path, $sharesPath);
		$run_folder_without_shares = build_run_folder_path($user_id, $experiment_name, $run_nr);
		$run_folder = "$sharesPath/$run_folder_without_shares";

		validate_directory($run_folder);

		if($fn) {
			$path = "$run_folder/$fn";

			if(file_exists($path)) {
				$out_file = get_out_file($path);
				respond_with_json($out_file);
			} else {
				respond_with_error("Invalid path $path found");
			}
		} else {
			$out_or_err_files = find_log_files($run_folder);

			respond_with_json($out_or_err_files);
		}
	} catch (Exception $e) {
		respond_with_error($e->getMessage());
	}

	function validate_param($param_name, $pattern, $error_message) {
		$value = get_or_env($param_name);
		if (!preg_match($pattern, $value)) {
			throw new Exception($error_message);
		}
		return $value;
	}

	function determine_share_path($environment_share_path, $default_share_path) {
		if ($environment_share_path && is_dir($environment_share_path) && !preg_match("/\.\./", $environment_share_path)) {
			return $environment_share_path;
		}
		return $default_share_path;
	}

	function build_run_folder_path($user_id, $experiment_name, $run_nr) {
		return "$user_id/$experiment_name/$run_nr/";
	}

	function validate_directory($dir_path) {
		if (!is_dir($dir_path)) {
			throw new Exception("$dir_path not found");
		}
	}

	function find_log_files($run_folder) {
		$run_files = glob("$run_folder/*");
		$out_or_err_files = [];

		foreach ($run_files as $file) {
			if (preg_match("/\/\.\.\/?/", $file)) {
				print("Invalid file " . htmlentities($file) . " detected. It will be ignored.");
				continue;
			}
			if (preg_match("/\/\d*_\d*_log\.(err|out)$/", $file)) {
				$out_or_err_files[] = $file;
			}
		}

		return $out_or_err_files;
	}

	function respond_with_json($data) {
		header('Content-Type: application/json');

		print json_encode(array(
			"data" => $data,
			"hash" => hash("md5", json_encode($data))
		));
		exit(0);
	}

	function respond_with_error($error_message) {
		header('Content-Type: application/json');

		print json_encode(array("error" => $error_message));
		exit(1);
	}
