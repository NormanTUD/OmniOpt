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

	if(!preg_match("/^\d+$/", $run_nr)) {
		die("Invalid run_nr");
	}

	if(!preg_match("/^[a-zA-Z0-9_]+$/", $experiment_name)) {
		die("Invalid experiment_name");
	}
	
	if(!preg_match("/^[a-zA-Z0-9_]+$/", $user_id)) {
		die("Invalid user_id");
	}

	$run_folder_without_shares = "$user_id/$experiment_name/$run_nr/";

	$run_folder = "$sharesPath/$run_folder_without_shares";

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

	$html = get_out_files_html($out_or_err_files);

	print json_encode(array("raw" => $html));
