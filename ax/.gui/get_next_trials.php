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

	$file = "$run_folder/get_next_trials.csv";

	if(!file_exists($file)) {
		echo json_encode(array("error" => "File $file not found"));
		exit(0);
	}

	$content = remove_ansi_colors(file_get_contents($file));
	$content_encoding = mb_detect_encoding($content);
	if (!($content_encoding == "ASCII" || $content_encoding == "UTF-8")) {
		echo json_encode(array("error" => "File $file is neither UTF-8 nor ASCII"));
		exit(1);
	}

	$header = get_header_file($file);

	$this_html = "<pre class='stdout_file invert_in_dark_mode autotable' data-header_columns='datetime,got,requested'>" . htmlentities($content) . "</pre>\n";
	$this_html .= copy_button("stdout_file");

	print json_encode(
		array(
			"raw" => $this_html,
			"hash" => hash("md5", $this_html)
		)
	);
