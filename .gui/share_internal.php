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

	$BASEURL = dirname((isset($_SERVER["REQUEST_SCHEME"]) ? $_SERVER["REQUEST_SCHEME"] : "http")  . "://" . (isset($_SERVER["SERVER_NAME"]) ? $_SERVER["SERVER_NAME"] : "localhost") . "/" . $_SERVER["SCRIPT_NAME"]);

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
		"cpu_ram_usage.csv",
		"get_next_trials.csv",
		"global_vars.json",
		"run_uuid",
		"outfile.txt",
		"oo_errors.txt",
		"evaluation_errors.log",
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
		"git_version"
	];

	$acceptable_files = array_map(function($file) {
		return preg_replace('/\.[^.]+$/', '', $file);
	}, $acceptable_file_names);

	$update_uuid = isset($_GET["update_uuid"]) ? $_GET["update_uuid"] : null;
	$uuid_folder = null;
	if ($update_uuid) {
		$uuid_folder = findMatchingUUIDRunFolder($update_uuid, $sharesPath);
	}

	$num_offered_files = 0;

	$offered_files_i = get_offered_files($acceptable_files, $acceptable_file_names, 0);

	$offered_files = $offered_files_i[0];
	$i = $offered_files_i[1];

	foreach ($_FILES as $_file) {
		$file_name = $_file["name"];
		$file_error = $_file["error"];
		$file_size = $_file["size"];
		$file_without_ending = pathinfo($file_name, PATHINFO_FILENAME);

		if (preg_match("/log.(zip|err|out)$/", $file_name)) {
			if (!isset($offered_files[$file_without_ending])) {
				if (isset($file_name)) {
					if ($file_error != 0) {
						print("File " . htmlentities($file_name) . " could not be uploaded. Error-Code: " . $file_error);
					} else {
						if ($file_size > 0 || isset($_GET["update"])) {
							$num_offered_files++;
							$offered_files[$file_without_ending] = array(
								"file" => $_file["tmp_name"] ?? null,
								"filename" => $file_name
							);
						}
					}
				} else {
					print("Could not determine filename for at least one uploaded file");
				}
			} else {
				print("$file_without_ending couldn't be found in \$offered_files\n");
			}
		} else {
			if(preg_match("/^gpu_usage__.*.csv$/", $file_name)) {
				if($file_size > 0) {
					if ($file_error == 0) {
						$offered_files[$file_without_ending] = array(
							"file" => $_file["tmp_name"] ?? null,
							"filename" => $file_name
						);
					}
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
			move_files_if_not_already_there($new_upload_md5_string, $update_uuid, $BASEURL, $user_id, $experiment_name, $run_nr, $offered_files, $userFolder, $uuid_folder, $sharesPath);

			exit(0);
		}
	}
?>
