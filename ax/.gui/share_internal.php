<?php
	error_reporting(E_ALL);
	set_error_handler(
		function ($severity, $message, $file, $line) {
			throw new \ErrorException($message, $severity, $severity, $file, $line);
		}
	);

	ini_set('display_errors', 1);

	$BASEURL = dirname((isset($_SERVER["REQUEST_SCHEME"]) ? $_SERVER["REQUEST_SCHEME"] : "http")  . "://" . (isset($_SERVER["SERVER_NAME"]) ? $_SERVER["SERVER_NAME"] : "localhost") . "/" . $_SERVER["SCRIPT_NAME"]);
	$sharesPath = './shares/';

	if (getenv("share_path") || isset($_GET["share_path"])) {
		$sharesPath = getenv("share_path");
		if(!$sharesPath) {
			$sharesPath = "./" . $_GET["share_path"] . "/";
			$sharesPath = preg_replace("/\/*$/", "/", $sharesPath);
		}

		if (!is_dir($sharesPath)) {
			print("Share dir $sharesPath could not be found!");
			exit(1);
		}

		if (preg_match("/^\//", $sharesPath)) {
			print("Absolute path is not allowed.");
			exit(2);
		}

		if (preg_match("/\.\./", $sharesPath)) {
			print("It is not allowed to traverse upwards.");
			exit(2);
		}

		print("Using sharesPath $sharesPath");
	}

	$user_id = $_GET['user_id'] ?? null;
	if(!$user_id && getenv("user_id")) {
		$user_id = getenv("user_id");
	}

	$share_on_list_publically = $_GET['share_on_list_publically'] ?? null;
	$experiment_name = $_GET['experiment_name'] ?? null;

	$acceptable_files = ["best_result", "job_infos", "parameters", "results", "ui_url", "cpu_ram_usage", "get_next_trials", "run_uuid"];
	$acceptable_file_names = ["best_result.txt", "job_infos.csv", "parameters.txt", "results.csv", "ui_url.txt", "cpu_ram_usage.csv", "get_next_trials.csv", "run_uuid"];

	$GLOBALS["time_start"] = microtime(true);

	require_once "share_functions.php";

	$update_uuid = isset($_GET["update_uuid"]) ? $_GET["update_uuid"] : null;
	$uuid_folder = null;
	if ($update_uuid) {
		$uuid_folder = findMatchingUUIDRunFolder($update_uuid, $sharesPath);
	}

	if ($user_id !== null && $experiment_name !== null) {
		$userFolder = get_user_folder($sharesPath, $uuid_folder, $user_id, $experiment_name);

		$run_id = preg_replace("/.*\//", "", $userFolder);

		$num_offered_files = 0;
		$new_upload_md5_string = "";

		$i = 0;

		$offered_files_i = get_offered_files($acceptable_files, $acceptable_file_names, $i);
		$offered_files = $offered_files_i[0];
		$i = $offered_files_i[1];


		foreach ($_FILES as $_file) {
			if (preg_match("/log.(err|out)$/", $_file["name"])) {
				$_file_without_ending = pathinfo($_file["name"], PATHINFO_FILENAME);
				if (!isset($offered_files[$_file_without_ending])) {
					if (isset($_file["name"])) {
						if ($_file["error"] != 0) {
							print("File " . htmlentities($_file["name"]) . " could not be uploaded. Error-Code: " . $_file["error"]);
						} else {
							if ($_file["size"] > 0) {
								$num_offered_files++;
								$offered_files[$_file_without_ending] = array(
									"file" => $_file["tmp_name"] ?? null,
									"filename" => $_file["name"]
								);
							}
						}
					} else {
						print("Could not determine filename for at least one uploaded file");
					}
				} else {
					print("$_file_without_ending coulnd't be found in \$offered_files\n");
				}
			}
		}

		foreach ($offered_files as $offered_file) {
			$filename = $offered_file["filename"];
			$file = $offered_file["file"];
			if ($file) {
				$content = file_get_contents($file);
				$new_upload_md5_string = $new_upload_md5_string . "$filename=$content";
				$num_offered_files++;
			}
		}

		if ($num_offered_files == 0) {
			print("Error sharing job. No offered files could be found.");
			exit(1);
		}

		move_files_if_not_already_there($new_upload_md5_string, $update_uuid, $BASEURL, $user_id, $experiment_name, $run_id, $offered_files, $userFolder, $uuid_folder, $sharesPath);
	} else {
		include_once "_functions.php";

		$dir_path = ".";
		if (preg_match("/\/tutorials\/?$/", dirname($_SERVER["PHP_SELF"]))) {
			$dir_path = "..";
		}
		if (!isset($_GET["get_hash_only"])) {
	?>
		    <script src='plotly-latest.min.js'></script>
		    <script src='share.js'></script>
		    <script src='share_graphs.js'></script>
		    <link href="<?php echo $dir_path; ?>/share.css" rel="stylesheet" />

		    <div id="breadcrumb"></div>
	<?php
		}
	}

	show_dir_view_or_plot($sharesPath, $user_id, $experiment_name);
