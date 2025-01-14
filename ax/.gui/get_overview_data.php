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

	$run_nr = get_or_env("run_nr");
	$user_id = get_or_env("user_id");
	$experiment_name = get_or_env("experiment_name");
	$environment_share_path = get_or_env("share_path");

	if($environment_share_path && is_dir($environment_share_path) && !preg_match("/\.\./", $environment_share_path)) {
		$sharesPath = $environment_share_path;
	}

	if(!preg_match("/^\d+$/", $run_nr)) {
		print json_encode(array("error" => "Invalid run_nr"));
		exit(0);
	}

	if(!preg_match("/^[a-zA-Z0-9_-]+$/", $experiment_name)) {
		print json_encode(array("error" => "Invalid experiment_name"));
		exit(0);
	}

	if(!preg_match("/^[a-zA-Z0-9_]+$/", $user_id)) {
		print json_encode(array("error" => "Invalid user_id"));
		exit(0);
	}

	$run_folder_without_shares = "$user_id/$experiment_name/$run_nr/";

	$run_folder = "$sharesPath/$run_folder_without_shares";

	if(!is_dir($run_folder)) {
		print json_encode(array("error" => "$run_folder not found"));
		exit(1);
	}

	$run_files = glob("$run_folder/*");

	$out_or_err_files = [];

	foreach ($run_files as $file) {
		if (!preg_match("/\/\.\.\/?/", $file) && preg_match("/\/\d*_\d*_log\.(err|out)$/", $file)) {
			$out_or_err_files[] = $file;
		}
	}

	$stat = array(
		"failed" => 0,
		"succeeded" => 0,
		"total" => 0
	);

	foreach ($out_or_err_files as $file) {
		if(checkForResult(file_get_contents($file)) != false) {
			$stat["succeeded"]++;
		} else {
			$stat["failed"]++;
		}

		$stat["total"]++;
	}

	print json_encode($stat);
