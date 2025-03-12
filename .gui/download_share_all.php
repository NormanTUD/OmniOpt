<?php
	include_once "_functions.php";
	include_once "share_functions.php";

	$sharesPath = './shares/';

	if (getenv("share_path") || isset($_GET["share_path"])) {
		$sharesPath = getenv("share_path");
		if(!$sharesPath) {
			$sharesPath = "./" . $_GET["share_path"] . "/";
			$sharesPath = preg_replace("/\/*$/", "/", $sharesPath);
		}

		if (preg_match("/^\//", $sharesPath)) {
			print("Absolute path is not allowed.");
			exit(1);
		}

		if (preg_match("/\.\./", $sharesPath)) {
			print("It is not allowed to traverse upwards.");
			exit(2);
		}

		if (!is_dir($sharesPath)) {
			print("Share dir $sharesPath could not be found!");
			exit(3);
		}


		print("Using sharesPath $sharesPath\n");
	}


	$user_id = get_or_env("user_id");

	$experiment_name = get_or_env('experiment_name');

	$run_nr = get_or_env("run_nr");

	$errors = 0;

	if ($user_id === null || $user_id == "" || !preg_match("/^[\w\d_]+$/", $user_id)) {
		echo "Error: user_id is null or empty or invalid\n";
		$errors++;
	}

	if ($experiment_name === null || $experiment_name == "" || !preg_match("/^[\w\d_]+$/", $experiment_name)) {
		echo "Error: experiment_name is null or empty or invalid\n";
		$errors++;
	}

	if ($run_nr === null || $run_nr == "" || !preg_match("/^\d++$/", $run_nr)) {
		echo "Error: run_nr is null or empty or invalid\n";
		$errors++;
	}

	$dir = "$sharesPath/$user_id/$experiment_name/$run_nr";

	$zip_file_path = "$dir/everything_but_singleruns_zip";

	if($errors == 0) {
		if (!is_dir($dir)) {
			echo "$dir is not a dir\n";
			$errors++;
		}
	}

	if($errors == 0) {
		if (!is_file($zip_file_path)) {
			echo "$zip_file_path could not be found\n";
			$errors++;
		}	
	}

	if($errors == 0) {
		if (!is_valid_zip_file($zip_file_path)) {
			echo "$zip_file_path is not a valid zip file\n";
			$errors++;
		}	
	}

	if($errors) {
		echo "download_share_all: $errors errors in total\n";
		exit(1);
	}

	header('Content-Type: application/zip');
	header('Content-Disposition: attachment; filename="' . basename($zip_file_path) . '"');
	header('Content-Length: ' . filesize($zip_file_path));

	readfile($zip_file_path);
	exit(0);
?>
