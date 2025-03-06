<?php
	include_once("new_share_functions.php");

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
			$out_file = ansi_to_html(file_get_contents($path));
			$out_file = highlightDebugInfo($out_file);
			respond_with_json($out_file);
		} else {
			respond_with_error("Invalid path $path found");
		}
	} catch (Exception $e) {
		respond_with_error($e->getMessage());
	}
?>
