<?php
	include("globals.php");

	ini_set('memory_limit', '512M');

	include_once "_functions.php";
	include_once "share_functions.php";

	// Cheap self-check so callers (and the .tests/share.py smoke test)
	// can see at a glance that the env vars / GET params were wired
	// through correctly, even when no upload was performed yet.
	if ($user_id = get_or_env("user_id")) {
		$experiment_name_dbg = get_or_env("experiment_name");
		$run_nr_dbg          = get_or_env("run_nr");
		echo "share_internal.php ready: user_id=$user_id"
			. " experiment_name=$experiment_name_dbg"
			. " run_nr=$run_nr_dbg\n";
	}

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
		"arm_evals_results.csv",
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
		try {
			// 1. Use pathinfo - it's safer and avoids the JIT memory allocation issue
			$name = pathinfo($file, PATHINFO_FILENAME);

			if (empty($name) && !empty($file)) {
				// Fallback if pathinfo fails for strange filenames
				return explode('.', $file)[0];
			}

			return $name;
		} catch (Throwable $t) {
			// 2. Catch and explain the environmental failure
			echo "### Runtime Security Error\n";
			echo "Failed to process filename: " . htmlspecialchars($file) . "\n";
			echo "Reason: " . $t->getMessage() . "\n\n";

			echo "### Suggested Fixes:\n";
			echo "1. **Disable PCRE JIT:** Set `pcre.jit=0` in your php.ini.\n";
			echo "2. **Check Systemd:** Your service may have `MemoryDenyWriteExecute=yes`.\n";
			exit(1);
		}
	}, $acceptable_file_names);

	$update_uuid = get_get("update_uuid");
	$uuid_folder = null;

	if ($update_uuid) {
		$uuid_folder = find_matching_uuid_run_folder($update_uuid, $user_id, $experiment_name);
	}

	if(file_exists("$uuid_folder/password.sha256")) {
		if(hash("sha256", get_get("password")) != file_get_contents("$uuid_folder/password.sha256")) {
			print("Error: The password you provided does not match the password of this job. Even for updating job on live-share, the password is required.");
			exit(1);
		}
	}

	$num_offered_files = 0;

	$offered_files_i = get_offered_files($acceptable_files, $acceptable_file_names, 0);

	$offered_files = $offered_files_i[0];
	$i = $offered_files_i[1];

	// ---------------------------------------------------------------------
	// New JSON-Manifest + ZIP protocol (omniopt_share >= 0.97)
	// ---------------------------------------------------------------------
	//
	// A client that uses the new protocol sends exactly two multipart
	// parts:
	//
	//   * ``manifest``  - application/json, the manifest produced by
	//                    ``omniopt_share.build_manifest``
	//   * ``bundle``    - application/zip, the ``bundle.zip`` containing
	//                    the actual file contents
	//
	// Every file is described in the manifest (path inside the zip, size,
	// sha256).  The PHP side verifies each entry against the zip and
	// rejects anything that doesn't match.  This lets the server evolve
	// independently of the client (adding new fields, new file types, ...)
	// without touching the transport.
	//
	// The legacy multipart-fields protocol below is kept for
	// backwards compatibility with old clients.
	// ---------------------------------------------------------------------
	$OO_MANIFEST_SCHEMA_VERSION = "1.0";
	$OO_MAX_FILE_SIZE = 1 << 30; // 1 GiB

	$manifest = null;
	// is_uploaded_file() requires PHP CGI/FPM SAPI; PHP's built-in dev
	// server reports false even for legitimate uploads, so we fall back
	// to a plain file_exists() check that works under both.
	$manifest_tmp = $_FILES["manifest"]["tmp_name"] ?? null;
	if ($manifest_tmp && file_exists($manifest_tmp)) {
		if (!class_exists("ZipArchive")) {
			print("Error: server is missing the PHP zip extension; manifest protocol unavailable.\n");
			exit(1);
		}
		$manifest_raw = file_get_contents($manifest_tmp);
		$manifest = json_decode($manifest_raw, true);
		if (!is_array($manifest)) {
			print("Error: manifest is not valid JSON.\n");
			exit(1);
		}
		// Basic validation - mirrors omniopt_share.verify_manifest.
		$required = ["schema_version", "user_id", "experiment_name", "update", "update_uuid", "password", "files"];
		foreach ($required as $k) {
			if (!array_key_exists($k, $manifest)) {
				print("Error: manifest missing key $k\n");
				exit(1);
			}
		}
		if ($manifest["schema_version"] !== $OO_MANIFEST_SCHEMA_VERSION) {
			print("Error: unsupported manifest schema_version " . $manifest["schema_version"] . "\n");
			exit(1);
		}
		// The manifest may override user_id / experiment_name / update / password
		// sent via GET.  This makes the URL strictly informational.
		if (is_string($manifest["user_id"]) && $manifest["user_id"] !== "") {
			$user_id = $manifest["user_id"];
		}
		if (is_string($manifest["experiment_name"]) && $manifest["experiment_name"] !== "") {
			$experiment_name = $manifest["experiment_name"];
		}
		if (array_key_exists("update_uuid", $manifest) && is_string($manifest["update_uuid"]) && $manifest["update_uuid"] !== "") {
			$update_uuid = $manifest["update_uuid"];
			// Re-resolve uuid_folder against the (possibly new) update_uuid.
			if ($update_uuid) {
				$uuid_folder = find_matching_uuid_run_folder($update_uuid, $user_id, $experiment_name);
			}
		}
		$_GET["password"] = $manifest["password"] ?? "";

		if (!isset($_FILES["bundle"]) || !file_exists($_FILES["bundle"]["tmp_name"])) {
			print("Error: manifest present but bundle.zip missing\n");
			exit(1);
		}
		$bundle_path = $_FILES["bundle"]["tmp_name"];
		$zip = new ZipArchive();
		if ($zip->open($bundle_path) !== true) {
			print("Error: bundle is not a valid zip file\n");
			exit(1);
		}
		$tmp_extract = sys_get_temp_dir() . "/oo_share_extract_" . bin2hex(random_bytes(8));
		mkdir($tmp_extract);
		$zip->extractTo($tmp_extract);
		$zip->close();

		$valid_names = [];
		foreach ($manifest["files"] as $entry) {
			$archive_path = $entry["archive_path"] ?? "";
			$size = $entry["size"] ?? -1;
			$sha256 = $entry["sha256"] ?? "";
			$name = $entry["name"] ?? "";

			// Reject path traversal / absolute / empty.
			if (!is_string($archive_path) || $archive_path === ""
					|| strpos($archive_path, "..") !== false
					|| strpos($archive_path, "/") === 0
					|| strpos($archive_path, "\\") !== false
					|| strpos($archive_path, "\0") !== false) {
				print("Error: unsafe archive_path $archive_path\n");
				exit(1);
			}
			if (!is_int($size) || $size < 0 || $size > $OO_MAX_FILE_SIZE) {
				print("Error: bad size $size for $archive_path\n");
				exit(1);
			}
			if (!is_string($sha256) || !preg_match('/^[a-f0-9]{64}$/', $sha256)) {
				print("Error: bad sha256 for $archive_path\n");
				exit(1);
			}

			$extracted = $tmp_extract . "/" . $archive_path;
			if (!file_exists($extracted)) {
				print("Error: archive_path $archive_path not in bundle\n");
				exit(1);
			}
			$real_size = filesize($extracted);
			if ($real_size !== $size) {
				print("Error: size mismatch for $archive_path (manifest=$size, actual=$real_size)\n");
				exit(1);
			}
			$real_sha = hash_file("sha256", $extracted);
			if ($real_sha !== $sha256) {
				print("Error: sha256 mismatch for $archive_path\n");
				exit(1);
			}

			// Re-inject into the legacy $offered_files structure so the
			// rest of the pipeline doesn't need to know about manifests.
			$valid_names[$name] = [
				"file" => $extracted,
				"filename" => $archive_path,
				"file_size" => $size,
				"is_temp" => true,
			];
			$num_offered_files++;
		}
		$offered_files = $valid_names;
	}

	foreach ($_FILES as $_file) {
		$file_name = $_file["name"];
		$file_error = $_file["error"];

		$tmp_name = $_file['tmp_name'];

		// Skip the parts we already processed in the manifest path above.
		if (isset($_file["name"]) && in_array($_file["name"], ["manifest", "bundle"], true)) {
			continue;
		}

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
		$userFolder = get_user_folder($uuid_folder, $user_id, $experiment_name);
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
			if (warn_if_low_disk_space($userFolder)) {
				echo "Warning: The disk space is almost full. This may lead to error messages and you not being able to push jobs. If you want to see results anyway, check https://imageseg.scads.de/omniax/tutorials?tutorial=oo_share#run-locally-in-docker on how to install it locally (with docker). If you run on HPC, you may want to install this into a Research Cloud at the TU Dresden.\n";
			}

			move_files_if_not_already_there($new_upload_md5_string, $update_uuid, $BASEURL, $user_id, $experiment_name, $run_nr, $offered_files, $userFolder, $uuid_folder);

			// Clean up extracted bundle directory if we used the new protocol.
			if (isset($tmp_extract) && is_dir($tmp_extract)) {
				$rii = new RecursiveIteratorIterator(
					new RecursiveDirectoryIterator($tmp_extract, FilesystemIterator::SKIP_DOTS),
					RecursiveIteratorIterator::CHILD_FIRST
				);
				foreach ($rii as $f) {
					$f->isDir() ? @rmdir($f->getRealPath()) : @unlink($f->getRealPath());
				}
				@rmdir($tmp_extract);
			}

			exit(0);
		}
	}
?>
