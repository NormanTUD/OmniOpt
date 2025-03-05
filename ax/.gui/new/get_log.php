<?php
	error_reporting(E_ALL);
	set_error_handler(
		function ($severity, $message, $file, $line) {
			throw new \ErrorException($message, $severity, $severity, $file, $line);
		}
	);

	ini_set('display_errors', 1);

	function get_get($name, $default = null) {
		if(isset($_GET[$name])) {
			return $_GET[$name];
		}

		return $default;
	}

	$sharesPath = "../shares/";

	try {
		$run_nr = validate_param("run_nr", "/^\d+$/", "Invalid run_nr");
		$user_id = validate_param("user_id", "/^[a-zA-Z0-9_]+$/", "Invalid user_id");
		$experiment_name = validate_param("experiment_name", "/^[a-zA-Z0-9_-]+$/", "Invalid experiment_name");
		$filename = validate_param("filename", "/^[a-zA-Z0-9\._]+$/", "Invalid filename");
		$run_folder_without_shares = build_run_folder_path($user_id, $experiment_name, $run_nr);
		$run_folder = "$sharesPath/$run_folder_without_shares";

		validate_directory($run_folder);

		$path = "$run_folder/$filename";;

		if(file_exists($path)) {
			$out_file = file_get_contents($path);
			respond_with_json($out_file);
		} else {
			respond_with_error("Invalid path $path found");
		}
	} catch (Exception $e) {
		respond_with_error($e->getMessage());
	}

	function validate_param($param_name, $pattern, $error_message) {
		$value = get_get($param_name);
		if (!preg_match($pattern, $value)) {
			throw new Exception($error_message);
		}
		return $value;
	}

	function build_run_folder_path($user_id, $experiment_name, $run_nr) {
		return "$user_id/$experiment_name/$run_nr/";
	}

	function validate_directory($dir_path) {
		if (!is_dir($dir_path)) {
			throw new Exception("$dir_path not found");
		}
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
