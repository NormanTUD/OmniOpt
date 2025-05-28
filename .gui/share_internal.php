<?php
	if(!defined('STDERR')) define('STDERR', fopen('php://stderr', 'wb'));

	$GLOBALS["time_start"] = microtime(true);

	error_reporting(E_ALL);
	set_error_handler(
		function ($severity, $message, $file, $line) {
			throw new \ErrorException($message, $severity, $severity, $file, $line);
		}
	);

	ini_set('display_errors', 1);

	include_once "_functions.php";
	include_once "share_functions.php";

	$sharesPath = './shares/';

	$port = $_SERVER["SERVER_PORT"] ?? 80;
	$scheme = ($port == 443) ? "https" : "http";
	$host = $_SERVER["SERVER_NAME"] ?? "localhost";
	$portPart = ($port != 80 && $port != 443) ? ":$port" : "";
	$script = $_SERVER["SCRIPT_NAME"];

	$BASEURL = dirname("$scheme://$host$portPart$script");

	try {
		delete_old_shares();
	} catch (\Throwable $e) {
		fwrite(STDERR, strval($e));
	}

	$user_id = get_or_env("user_id");

	$experiment_name = get_or_env('experiment_name');

	$acceptable_file_names = [
		"best_result.txt",
		"job_infos.csv",
		"parameters.txt",
		"results.csv",
		"ui_url.txt",
		"Constraints.txt",
		"cpu_ram_usage.csv",
		"get_next_trials.csv",
		"global_vars.json",
		"run_uuid",
		"outfile.txt",
		"oo_errors.txt",
		"evaluation_errors.log",
		"continue_from_run_uuid",
		"pareto_idxs.json",
		"outfile",
		"log",
		"install_errors",
		"progressbar",
		"trial_index_to_params",
		"worker_usage.csv",
		"job_start_time.txt",
		"pareto_front_table.txt",
		"pareto_front_data.json",
		"everything_but_singleruns.zip",
		"everything_but_singleruns_zip",
		"everything.zip",
		"args_overview.txt",
		"experiment_overview.txt",
		"eval_nodes_cpu_ram_logs.txt",
		"verbose_log.txt",
		"result_names.txt",
		"result_min_max.txt",
		"result_min_max",
		"job_submit_durations.txt",
		"generation_times.txt",
		"git_version"
	];

	$acceptable_files = array_map(function($file) {
		return preg_replace('/\.[^.]+$/', '', $file);
	}, $acceptable_file_names);

	$update_uuid = isset($_GET["update_uuid"]) ? $_GET["update_uuid"] : null;
	$uuid_folder = null;

	if ($update_uuid) {
		$uuid_folder = findMatchingUUIDRunFolder($update_uuid, $sharesPath, $user_id, $experiment_name);
	}

	$num_offered_files = 0;

	$offered_files_i = get_offered_files($acceptable_files, $acceptable_file_names, 0);

	$offered_files = $offered_files_i[0];
	$i = $offered_files_i[1];

	foreach ($_FILES as $_file) {
		$file_name = $_file["name"];
		$file_error = $_file["error"];

		$tmp_name = $_file['tmp_name'];

		if($tmp_name && file_exists($tmp_name)) {
			$contents = file_get_contents($tmp_name);

			if($contents) {
				$file_size = strlen($contents);
				$file_without_ending = pathinfo($file_name, PATHINFO_FILENAME);

				if($file_size > 0) {
					$num_offered_files++;
					$offered_files[$file_without_ending] = array(
						"file" => $_file["tmp_name"] ?? null,
						"filename" => $file_name,
						"file_size" => $file_size
					);
				}
			}
		}
	}

	if ($user_id !== null && $experiment_name !== null && ($num_offered_files > 0 || isset($_GET["update"]))) {
		$userFolder = get_user_folder($sharesPath, $uuid_folder, $user_id, $experiment_name);
		if(!$userFolder) {
			die("Could not create user folder");
		}

		$run_nr = preg_replace("/.*\//", "", $userFolder);

		if($run_nr != "" && getenv("run_nr") && preg_match("/^\d+$/", getenv("run_dir"))) {
			$run_nr = get_or_env("run_nr");
		}

		$new_upload_md5_string = "";

		foreach ($offered_files as $offered_file) {
			$filename = $offered_file["filename"];
			$file = $offered_file["file"];
			if ($file) {
				$content = file_get_contents($file);
				$new_upload_md5_string = $new_upload_md5_string . "$filename=$content";
				$num_offered_files++;
			}
		}

		if ($num_offered_files == 0 && !isset($_GET["update"])) {
			print("Error sharing job. No offered files could be found.");
			exit(1);
		}

		if($num_offered_files) {
			if (warnIfLowDiskSpace($userFolder)) {
				echo "Warning: The disk space is almost full. This may lead to error messages and you not being able to push jobs. If you want to see results anyway, check https://imageseg.scads.de/omniax/tutorials?tutorial=oo_share#run-locally-in-docker on how to install it locally (with docker). If you run on HPC, you may want to install this into a Research Cloud at the TU Dresden.\n";
			}

			move_files_if_not_already_there($new_upload_md5_string, $update_uuid, $BASEURL, $user_id, $experiment_name, $run_nr, $offered_files, $userFolder, $uuid_folder, $sharesPath);

			exit(0);
		}
	}
?>
