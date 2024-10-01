<?php
	include_once("_functions.php");
	include_once("share_functions.php");

	function has_invalid_chars($name, $val) {
		if($name == "run_nr") {
			if(!preg_match("/^\d+$/", $val)) {
				die("user_id must be a number!");
			}
		} else if ($name == "experiment_name" || $name == "user_id") {
			if(!preg_match("/^[a-zA-Z_0-9]+$/", $val)) {
				die("$name must consist of numbers, letters or underscore!");
			}
		} else if ($name == "filename") {
			if(!preg_match("/^[a-zA-Z_0-9]+\.(?:txt|csv|log)$/", $val)) {
				die("$name must consist of numbers, letters or underscore, and end with .{txt,csv,log}!");
			}
		} else {
			dier("Unknown name $name");
		}
	}

	$vars = array(
		"user_id" => get_or_env("user_id"),
		"experiment_name" => get_or_env("experiment_name"),
		"run_nr" => get_or_env("run_nr"),
		"filename" => get_or_env("filename")
	);

	$missing = array();

	foreach ($vars as $var_key => $var_value) {
		if(has_invalid_chars($var_key, $var_value)) {
			echo "Invalid chars in $var_key.";
			exit(1);
		}

		if(is_null($var_value) || strlen($var_value) == 0) {
			$missing[] = $var_key;
		}
	}

	if(count($missing)) {
		print json_encode(["error" => implode(", ", $missing)." undefined"]);
		exit(1);
	}

	$share_dir = "./shares/";
	if(!is_dir($share_dir)) {
		print json_encode(["error" => "$share_dir is not a directory"]);
		exit(1);
	}

	$share_dir = "$share_dir/" . $vars["user_id"] . "/";
	if(!is_dir($share_dir)) {
		print json_encode(["error" => "$share_dir is not a directory"]);
		exit(1);
	}

	$share_dir = "$share_dir/" . $vars["experiment_name"] . "/";
	if(!is_dir($share_dir)) {
		print json_encode(["error" => "$share_dir is not a directory"]);
		exit(1);
	}

	$share_dir = "$share_dir/" . $vars["run_nr"];
	if(!is_dir($share_dir)) {
		print json_encode(["error" => "$share_dir is not a directory"]);
		exit(1);
	}

	$share_file = "$share_dir/" . $vars["filename"];
	if(!file_exists($share_file)) {
		print json_encode(["error" => "$share_dir is not a directory"]);
		exit(1);
	}

	if(preg_match("\.csv$", $share_file)) {
		echo loadCsvToJson($share_file);
	} else {
		echo remove_ansi_colors(file_get_contents($share_file));
	}
?>
