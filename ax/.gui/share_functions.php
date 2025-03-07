<?php
	function calculateDirectoryHash($directory) {
		if (!is_dir($directory) || !is_readable($directory)) {
			return false;
		}

		function getFilesRecursive($dir)
		{
			$files = [];
			$dirIterator = new RecursiveDirectoryIterator($dir, RecursiveDirectoryIterator::SKIP_DOTS);
			$iterator = new RecursiveIteratorIterator($dirIterator, RecursiveIteratorIterator::SELF_FIRST);

			foreach ($iterator as $file) {
				if ($file->isFile()) {
					$files[] = $file->getPathname();
				}
			}

			sort($files);
			return $files;
		}

		$files = getFilesRecursive($directory);

		if (empty($files)) {
			return false;
		}

		$combinedHashes = '';

		foreach ($files as $file) {
			$fileContent = file_get_contents($file);
			if ($fileContent === false) {
				return false;
			}

			$combinedHashes .= hash('sha256', $fileContent);
		}

		return hash('sha256', $combinedHashes);
	}

	function convert_to_int_or_float_if_possible($var) {
		if (is_numeric($var)) {
			if (ctype_digit($var) || (is_numeric($var) && (float)$var == (int)$var)) {
				return (int)$var;
			}
			return (float)$var;
		}
		return $var;
	}

	function loadCsvToJson($file) {
		assert(file_exists($file), "CSV file does not exist.");

		$csvData = [];
		try {
			$fileHandle = fopen($file, "r");
			assert($fileHandle !== false, "Failed to open the file.");

			while (($row = fgetcsv($fileHandle)) !== false) {
				$new_row = [];
				foreach ($row as $r) {
					$new_row[] = convert_to_int_or_float_if_possible($r);
				}
				$csvData[] = $new_row;
			}

			fclose($fileHandle);
		} catch (Exception $e) {
			print("Error reading CSV: " . $e->getMessage());
			warn("Ensure the CSV file is correctly formatted.");
			throw $e;
		}

		$jsonData = json_encode($csvData);
		assert($jsonData !== false, "Failed to encode JSON.");

		return $jsonData;
	}

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

	function remove_ansi_colors($contents) {
		$contents = preg_replace('#\\x1b[[][^A-Za-z]*[A-Za-z]#', '', $contents);
		return $contents;
	}

	function get_out_file ($fn) {
		if(!file_exists($fn)) {
			die("Unknown file path $fn");
		}

		$c = file_get_contents($fn);

		#$c = htmlentities($c);

		return $c;
	}

	function show_run() {
		$html = "<button id='refresh_button' class='invert_in_dark_mode' onclick='refresh()'>Refresh</button>\n";
		$html .= "<div id='main_tabbed' style='width: max-content'>\n";
		$html .= '<ul style="max-height: 200px; overflow: auto;">' . "\n";
		$html .= "</ul>\n";
		$html .= "</div>";
		$html .= "<script>";
		$html .= "$(function() {";
		$html .= '    $("#out_files_tabs").tabs();';
		$html .= "});";
		$html .= "</script>";

		print $html;
	}

	function custom_sort($a, $b) {
		$a_numeric = preg_replace('/[^0-9]/', '', $a);
		$b_numeric = preg_replace('/[^0-9]/', '', $b);

		if (is_numeric($a_numeric) && is_numeric($b_numeric)) {
			if ((int)$a_numeric == (int)$b_numeric) {
				return strcmp($a, $b);
			}
			return (int)$a_numeric - (int)$b_numeric;
		}

		if (is_numeric($a_numeric)) {
			return -1;
		}

		if (is_numeric($b_numeric)) {
			return 1;
		}

		return strcmp($a, $b);
	}

	function check_and_filter_folders($folders) {
		if (!is_array($folders)) {
			throw new InvalidArgumentException("Der übergebene Parameter muss ein Array sein.");
		}

		$filtered_folders = array_filter($folders, function($folder) {
			if (!is_dir($folder)) {
				error_log("Warnung: '$folder' ist kein gültiges Verzeichnis.");
				return true;
			}

			$files = scandir($folder);

			$files = array_diff($files, array('.', '..'));

			return count($files) > 0;
		});

		return $filtered_folders;
	}

	function show_run_selection($sharesPath, $user, $experiment_name) {
		$experiment_name = preg_replace("/.*\//", "", $experiment_name);
		$folder_glob = "$sharesPath/$user/$experiment_name/*";
		$experiment_subfolders = glob($folder_glob, GLOB_ONLYDIR);

		$experiment_subfolders = check_and_filter_folders($experiment_subfolders);

		if (count($experiment_subfolders) == 0) {
			echo "No runs found in $folder_glob";
			exit(1);
		}

		usort($experiment_subfolders, 'custom_sort');

		foreach ($experiment_subfolders as $run_nr) {
			$run_nr = preg_replace("/.*\//", "", $run_nr);
			$sharesPathLink = $sharesPath == "./shares/" ? "" : "&share_path=$sharesPath";
			echo "<!-- show_run_selection " . __LINE__ . " -->\n";
			$name = $run_nr;

			$job_start_times_file = "$sharesPath/$user/$experiment_name/$run_nr/job_start_time.txt";

			if (file_exists($job_start_times_file) && is_readable($job_start_times_file)) {
				$file_content = file_get_contents($job_start_times_file);

				$date_pattern = '/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/m';

				if (preg_match($date_pattern, $file_content, $matches)) {
					$name .= ' (' . trim($matches[0]) . ')';
				}
			}

			echo "<a class='_share_link' href=\"share?user_id=$user&experiment_name=$experiment_name&run_nr=$run_nr$sharesPathLink\">$name</a><br>";
		}
	}

	function print_script_and_folder($folder) {
		echo "\n<script>createBreadcrumb('./$folder');</script>\n";
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

	# TODO: Is slow sometimes: parseAnsiToVirtualTerminal
	function parseAnsiToVirtualTerminal($input) {
		$rows = [[]];
		$maxWidth = 0;
		$cursorX = 0;
		$cursorY = 0;

		function ensureTerminalSize(&$rows, $y) {
			while (count($rows) <= $y) {
				$rows[] = [];
			}
		}

		function handleAnsiCode($code, &$cursorX, &$cursorY, &$rows) {
			if (preg_match('/^\[(\d+)?([A-Za-z])/', $code, $cursorMatch)) {
				$value = isset($cursorMatch[1]) ? intval($cursorMatch[1]) : 1;
				switch ($cursorMatch[2]) {
					case 'A':
						$cursorY = max($cursorY - $value, 0);
						break;
					case 'B':
						$cursorY += $value;
						ensureTerminalSize($rows, $cursorY);
						break;
					case 'C':
						$cursorX += $value;
						break;
					case 'D':
						$cursorX = max($cursorX - $value, 0);
						break;
					case 'K':
						$rows[$cursorY] = [];
						$cursorX = 0;
						break;
				}
			}
		}

		$i = 0;
		$len = strlen($input);
		while ($i < $len) {
			$char = $input[$i];

			if ($char === "\x1b" && isset($input[$i + 1]) && $input[$i + 1] === '[') {
				if (preg_match('/^[^a-zA-Z]*[a-zA-Z]/', substr($input, $i + 2), $ansiCode) && preg_match('/^[\d;]*[A-DK]$/', $ansiCode[0])) {
					handleAnsiCode($ansiCode[0], $cursorX, $cursorY, $rows);
					$i += strlen($ansiCode[0]) + 2;
					continue;
				}
			} elseif ($char === "\r") {
				$cursorX = 0;
			} elseif ($char === "\n") {
				$cursorY++;
				$cursorX = 0;
				ensureTerminalSize($rows, $cursorY);
			} else {
				ensureTerminalSize($rows, $cursorY);
				if ($cursorX >= count($rows[$cursorY])) {
					$rows[$cursorY][$cursorX] = $char;
				} else {
					$rows[$cursorY][$cursorX] = $char;
				}
				$cursorX++;
				if ($cursorX > $maxWidth) $maxWidth = $cursorX;
			}
			$i++;
		}

		$output = '';
		foreach ($rows as $line) {
			$lineStr = implode('', $line);
			$output .= str_pad($lineStr, $maxWidth, ' ') . "\n";
		}
		return rtrim($output);
	}

	function checkFolderPermissions($directory, $expectedUser, $expectedGroup, $alternativeUser, $alternativeGroup, $expectedPermissions) {
		if (getenv('CI') !== false) {
			return false;
		}

		if (!is_dir($directory)) {
			echo "<i>Error: '$directory' is not a valid directory</i>\n";
			return true;
		}

		// Get stat information
		$stat = stat($directory);
		if ($stat === false) {
			echo "<i>Error: Unable to retrieve information for '$directory'</i><br>\n";
			return;
		}

		// Get current ownership and permissions
		$currentUser = posix_getpwuid($stat['uid'])['name'] ?? 'unknown';
		$currentGroup = posix_getgrgid($stat['gid'])['name'] ?? 'unknown';
		$currentPermissions = substr(sprintf('%o', $stat['mode']), -4);

		$issues = false;

		// Check user
		if ($currentUser !== $expectedUser) {
			if ($currentUser !== $alternativeUser) {
				$issues = true;
				echo "<i>Ownership issue: Current user is '$currentUser'. Expected user is '$expectedUser'</i><br>\n";
				echo "<samp>chown $expectedUser $directory</samp>\n<br>";
			}
		}

		// Check group
		if ($currentGroup !== $expectedGroup) {
			if ($currentGroup !== $alternativeGroup) {
				$issues = true;
				echo "<i>Ownership issue: Current group is '$currentGroup'. Expected group is '$expectedGroup'</i><br>\n";
				echo "<samp>chown :$expectedGroup $directory</samp><br>\n";
			}
		}

		// Check permissions
		if (intval($currentPermissions, 8) !== $expectedPermissions) {
			$issues = true;
			echo "<i>Permissions issue: Current permissions are '$currentPermissions'. Expected permissions are '" . sprintf('%o', $expectedPermissions) . "'</i><br>\n";
			echo "<samp>chmod " . sprintf('%o', $expectedPermissions) . " $directory\n</samp><br>";
		}

		return $issues;
	}
