<?php
	function getCsvStatusSummary($filePath) {
		if (!file_exists($filePath) || !is_readable($filePath)) {
			return json_encode(["error" => "File not found or not readable"], JSON_PRETTY_PRINT);
		}

		$statuses = [
			"failed" => 0,
			"succeeded" => 0,
			"running" => 0,
			"total" => 0
		];

		if (($handle = fopen($filePath, "r")) !== false) {
			$header = fgetcsv($handle, 0, ",", "\"", "\\");
			while (($data = fgetcsv($handle, 0, ",", "\"", "\\")) !== false) {
				if (count($data) < 3) continue; // Sicherstellen, dass es genug Spalten gibt

				$statuses["total"]++;
				$status = strtolower(trim($data[2]));

				if ($status === "completed") {
					$statuses["succeeded"]++;
				} elseif ($status === "failed") {
					$statuses["failed"]++;
				} elseif ($status === "running") {
					$statuses["running"]++;
				}
			}
			fclose($handle);
		}

		return json_encode($statuses, JSON_PRETTY_PRINT);
	}

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

	if(file_exists("$run_folder/results.csv")) {
		print(getCsvStatusSummary("$run_folder/results.csv"));
	} else {
		$statuses = [
			"failed" => 0,
			"succeeded" => 0,
			"running" => 0,
			"total" => 0
		];

		print(json_encode($statuses, JSON_PRETTY_PRINT));
	}
?>
