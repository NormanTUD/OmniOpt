<?php
	function warn($message) {
		echo "Warning: " . $message . "\n";
	}

	function findMatchingUUIDRunFolder(string $targetUUID, $sharesPath): ?string {
		$glob_str = "$sharesPath/*/*/*/run_uuid";
		$files = glob($glob_str);

		foreach ($files as $file) {
			$fileContent = preg_replace('/\s+/', '', file_get_contents($file));

			if ($fileContent === $targetUUID) {
				return dirname($file);
			}
		}

		return null;
	}

	function deleteFolder($folder) {
		$files = array_diff(scandir($folder), array('.', '..'));

		foreach ($files as $file) {
			(is_dir("$folder/$file")) ? deleteFolder("$folder/$file") : unlink("$folder/$file");
		}

		return rmdir($folder);
	}

	function createNewFolder($path, $user_id, $experiment_name) {
		$i = 0;

		$newFolder = $path . "/$user_id/$experiment_name/$i";

		do {
			$newFolder = $path . "/$user_id/$experiment_name/$i";
			$i++;
		} while (file_exists($newFolder));

		try {
			mkdir($newFolder, 0777, true);
		} catch (Exception $e) {
			print("Error trying to create directory $newFolder. Error:\n\n$e\n\n");
			exit(1);
		}
		return $newFolder;
	}

	function searchForHashFile($directory, $new_upload_md5, $userFolder) {
		$files = glob($directory);

		foreach ($files as $file) {
			try {
				$file_content = file_get_contents($file);

				if ($file_content === $new_upload_md5) {
					return [true, dirname($file)];
				}
			} catch (AssertionError $e) {
				print($e->getMessage());
			}
		}

		try {
			$destinationPath = "$userFolder/hash.md5";
			assert(is_writable(dirname($destinationPath)), "Directory is not writable: " . dirname($destinationPath));

			$write_success = file_put_contents($destinationPath, $new_upload_md5);
			assert($write_success !== false, "Failed to write to file: $destinationPath");
		} catch (\Throwable $e) {
			print($e->getMessage());
		}

		return [false, null];
	}

	function extractPathComponents($found_hash_file_dir, $sharesPath) {
		$pattern = "#^$sharesPath/([^/]+)/([^/]+)/(\d+)$#";

		if (preg_match($pattern, $found_hash_file_dir, $matches)) {
			assert(isset($matches[1]), "Failed to extract user from path: $found_hash_file_dir");
			assert(isset($matches[2]), "Failed to extract experiment name from path: $found_hash_file_dir");
			assert(isset($matches[3]), "Failed to extract run ID from path: $found_hash_file_dir");

			$user = $matches[1];
			$experiment_name = $matches[2];
			$run_dir = $matches[3];

			return [$user, $experiment_name, $run_dir];
		} else {
			warn("The provided path does not match the expected pattern: $found_hash_file_dir");
			return [null, null, null];
		}
	}

	function get_user_folder($sharesPath, $_uuid_folder, $user_id, $experiment_name, $run_nr="") {
		$probe_dir = "$sharesPath/$user_id/$experiment_name/$run_nr";

		if($run_nr != "" && $run_nr >= 0 && is_dir($probe_dir)) {
			return $probe_dir;
		}

		if(getenv("disable_folder_creation")) {
			return;
		}

		if (!$_uuid_folder) {
			$userFolder = createNewFolder($sharesPath, $user_id, $experiment_name);
		} else {
			$userFolder = $_uuid_folder;
		}

		return $userFolder;
	}

	function is_valid_zip_file($path) {
		if (!file_exists($path) || !is_readable($path)) {
			return false;
		}

		$handle = fopen($path, 'rb');
		if (!$handle) {
			return false;
		}

		$signature = fread($handle, 4);
		fclose($handle);

		// ZIP-Files begin with "PK\x03\x04"
		return $signature === "PK\x03\x04";
	}

	function move_files($offered_files, $added_files, $userFolder, $msgUpdate, $msg) {
		$empty_files = [];

		foreach ($offered_files as $offered_file) {
			$file = $offered_file["file"];
			$filename = $offered_file["filename"];

			if ($file) {
				if(file_exists($file)) {
					$content = file_get_contents($file);
					$content_encoding = mb_detect_encoding($content);
					if ($content_encoding == "ASCII" || $content_encoding == "UTF-8" || is_valid_zip_file($file)) {
						if (filesize($file)) {
							try {
								move_uploaded_file($file, "$userFolder/$filename");
								$added_files++;
							} catch (Exception $e) {
								echo "An exception occured trying to move $file to $userFolder/$filename";
							}
						} else {
							$empty_files[] = $filename;
						}
					} else {
						dier("$filename: \$content was not ASCII, but $content_encoding");
					}
				} else {
					print("$file does not exist");
				}
			}
		}

		if ($added_files) {
			if (isset($_GET["update"])) {
				eval('echo "$msgUpdate";');
			} else {
				eval('echo "$msg";');
			}
			exit(0);
		} else {
			if (count($empty_files)) {
				$empty_files_string = implode(", ", $empty_files);
				echo "Error sharing the job. The following files were empty: $empty_files_string. \n";
			} else {
				echo "Error sharing the job. No Files were found. \n";
			}
			exit(1);
		}
	}

	function remove_extra_slashes_from_url($string) {
		$pattern = '/(?<!:)(\/{2,})/';

		$cleaned_string = preg_replace($pattern, '/', $string);

		return $cleaned_string;
	}

	function move_files_if_not_already_there($new_upload_md5_string, $update_uuid, $BASEURL, $user_id, $experiment_name, $run_id, $offered_files, $userFolder, $uuid_folder, $sharesPath) {
		$added_files = 0;
		$project_md5 = hash('md5', $new_upload_md5_string);

		$found_hash_file_data = searchForHashFile("$sharesPath/*/*/*/hash.md5", $project_md5, $userFolder);

		$found_hash_file = $found_hash_file_data[0];
		$found_hash_file_dir = $found_hash_file_data[1];

		if ($found_hash_file && is_null($update_uuid)) {
			list($user, $experiment_name, $run_id) = extractPathComponents($found_hash_file_dir, $sharesPath);
			$old_url = remove_extra_slashes_from_url("$BASEURL/share?user_id=$user_id&experiment_name=$experiment_name&run_nr=$run_id");
			echo "This project already seems to have been uploaded. See $old_url\n";
			exit(0);
		} else {
			if (!$uuid_folder || !is_dir($uuid_folder)) {
				$url = remove_extra_slashes_from_url("$BASEURL/share?user_id=$user_id&experiment_name=$experiment_name&run_nr=$run_id");

				move_files(
					$offered_files,
					$added_files,
					$userFolder,
					"See $url for live-results.\n",
					"Run was successfully shared. See $url\nYou can share the link. It is valid for 30 days.\n"
				);
			} else {
				$url = remove_extra_slashes_from_url("$BASEURL/share?user_id=$user_id&experiment_name=$experiment_name&run_nr=$run_id");

				move_files(
					$offered_files,
					$added_files,
					$uuid_folder,
					"See $url for live-results.\n",
					"See $url for live-results.\n"
				);
			}
		}
	}

	function get_offered_files($acceptable_files, $acceptable_file_names, $i) {
		foreach ($acceptable_files as $acceptable_file) {
			$offered_files[$acceptable_file] = array(
				"file" => $_FILES[$acceptable_file]['tmp_name'] ?? null,
				"filename" => $acceptable_file_names[$i]
			);
			$i++;
		}

		return [$offered_files, $i];
	}

	function rrmdir($dir) {
		if (is_dir($dir)) {
			$objects = scandir($dir);

			foreach ($objects as $object) {
				if ($object != '.' && $object != '..') {
					$object_path = $dir.'/'.$object;
					if (filetype($object_path) == 'dir') {
						rrmdir($object_path);
					} else {
						if (file_exists($object_path)) {
							unlink($object_path);
						}
					}
				}
			}

			reset($objects);

			if(is_dir($dir)) {
				rmdir($dir);
			}
		}
	}

	function deleteEmptyDirectories(string $directory, bool $is_recursive_call): bool {
		if (!is_dir($directory)) {
			return false;
		}

		$files = array_diff(scandir($directory), ['.', '..']);

		foreach ($files as $file) {
			$path = $directory . DIRECTORY_SEPARATOR . $file;
			if (is_dir($path)) {
				deleteEmptyDirectories($path, true);
			}
		}

		$filesAfterCheck = array_diff(scandir($directory), ['.', '..']);
		if ($is_recursive_call && empty($filesAfterCheck)) {
			rmdir($directory);
			return true;
		}

		return false;
	}

	function _delete_old_shares($dir) {
		$oldDirectories = [];
		$currentTime = time();

		// Helper function to check if a directory is empty
		function is_dir_empty($dir) {
			return (is_readable($dir) && count(scandir($dir)) == 2); // Only '.' and '..' are present in an empty directory
		}

		foreach (glob("$dir/*/*/*", GLOB_ONLYDIR) as $subdir) {
			$pathParts = explode('/', $subdir);
			$secondDir = $pathParts[1] ?? '';

			// Skip Elias's project directory
			if ($secondDir != "s4122485") {
				$threshold = ($secondDir === 'runner') ? 86400 : (30 * 24 * 3600);

				if(is_dir($subdir)) {
					$dir_date = filemtime($subdir);

					// Check if the directory is older than the threshold and is either empty or meets the original condition
					if (is_dir($subdir) && ($dir_date < ($currentTime - $threshold))) {
						$oldDirectories[] = $subdir;
						rrmdir($subdir);
					}

					if (is_dir($subdir) && is_dir_empty($subdir)) {
						$oldDirectories[] = $subdir;
						rrmdir($subdir);
					}
				}
			}
		}

		return $oldDirectories;
	}

	function delete_old_shares () {
		$directoryToCheck = 'shares';
		deleteEmptyDirectories($directoryToCheck, false);
		$oldDirs = _delete_old_shares($directoryToCheck);
		deleteEmptyDirectories($directoryToCheck, false);

		return $oldDirs;
	}
?>
