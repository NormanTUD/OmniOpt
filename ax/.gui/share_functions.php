<?php
	function copy_button($name_to_search_for) {
		return "<button class='copy_to_clipboard_button invert_in_dark_mode' onclick='find_closest_element_behind_and_copy_content_to_clipboard(this, \"$name_to_search_for\")'>📋 Copy raw data to clipboard</button>";
	}

	function calculateDirectoryHash($directory) {
		// Überprüfen, ob der Ordner existiert und lesbar ist
		if (!is_dir($directory) || !is_readable($directory)) {
			return false; // Fehler, Ordner existiert nicht oder ist nicht lesbar
		}

		// Rekursive Funktion zum Abrufen aller Dateien im Ordner und Unterordnern
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

			// Alphabetisch sortieren
			sort($files);
			return $files;
		}

		// Alle Dateien im Ordner und Unterordner holen
		$files = getFilesRecursive($directory);

		// Falls keine Dateien gefunden wurden
		if (empty($files)) {
			return false; // Keine Dateien im Ordner
		}

		$combinedHashes = '';

		// Für jede Datei den SHA256-Hash berechnen und an die Hash-Liste anhängen
		foreach ($files as $file) {
			$fileContent = file_get_contents($file);
			if ($fileContent === false) {
				return false; // Fehler beim Lesen der Datei
			}

			$combinedHashes .= hash('sha256', $fileContent);
		}

		// Endgültigen SHA256-Hash des kombinierten Hash-Strings berechnen
		return hash('sha256', $combinedHashes);
	}

	function get_header_file($file) {
		$replaced_file = preg_replace("/.*\//", "", $file);

		$names = array(
			"best_result.txt" => "Best results",
			"results.csv" => "Results",
			"job_infos.csv" => "Job-Infos",
			"parameters.txt" => "Parameter",
			"get_next_trials.csv" => "get_next_trials",
			"cpu_ram_usage.csv" => "CPU/RAM-usage of main worker",
			"evaluation_errors.log" => "Evaluation Errors",
			"oo_errors.txt" => "OmniOpt2-Errors",
			"worker_usage.csv" => "Number of workers"
		);

		if (isset($names[$replaced_file])) {
			return "<h2>" . $names[$replaced_file] . "</h2>";
		} else {
			return "<h2>" . $replaced_file . "</h2>";
		}
	}

	function convert_to_int_or_float_if_possible($var) {
		// Prüfen, ob die Eingabe ein numerischer Wert ist
		if (is_numeric($var)) {
			// Wenn es ein ganzzahliger Wert ist, nach int konvertieren
			if (ctype_digit($var) || (is_numeric($var) && (float)$var == (int)$var)) {
				return (int)$var;
			}
			// Sonst als float zurückgeben
			return (float)$var;
		}
		// Wenn die Eingabe nicht numerisch ist, den originalen Wert zurückgeben
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
		// Glob-Muster, um alle passenden Dateien zu finden

		$glob_str = "$sharesPath/*/*/*/run_uuid";
		$files = glob($glob_str);
		// dier($glob_str);
		// dier($files);

		foreach ($files as $file) {
			// Dateiinhalt lesen und Whitespace (Leerzeichen, Newlines, Tabs) entfernen
			$fileContent = preg_replace('/\s+/', '', file_get_contents($file));

			// Überprüfen, ob die UUID übereinstimmt
			if ($fileContent === $targetUUID) {
				// Ordnerpfad ohne 'state_files/run_uuid' zurückgeben
				return dirname($file);  // Zwei Ebenen zurück gehen
			}
		}

		// Wenn keine Übereinstimmung gefunden wurde, null zurückgeben
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
			mkdir($newFolder, 0777, true); // Rechte 0777 für volle Zugriffsberechtigungen setzen
		} catch (Exception $e) {
			print("Error trying to create directory $newFolder");
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
		} catch (AssertionError $e) {
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

	function get_out_files_html ($out_or_err_files) {
		$html = "";
		if (count($out_or_err_files)) {
			if (count($out_or_err_files) == 1) {
				$_file = $out_or_err_files[0];
				if (file_exists($_file)) {
					$content = file_get_contents($_file);
					$html .= "<pre class='stdout_file invert_in_dark_mode'>" . htmlentities($content) . "\n</pre>";
					$html .= copy_button("stdout_file");
				}
			} else {
				$html .= "<div id='single_run_files_container'>\n";
				$html .= "<h2 id='single_run_files'>Single run output files</h2>\n";
				$html .= '<div id="out_files_tabs">' . "\n";
				$html .= '<ul style="max-height: 200px; overflow: auto;">' . "\n";

				$ok = "&#9989;";
				$error = "&#10060;";

				foreach ($out_or_err_files as $out_or_err_file) {
					$content = remove_ansi_colors(file_get_contents($out_or_err_file));

					$_hash = hash('md5', "$out_or_err_file-$content");

					$ok_or_error = $ok;

					if (!checkForResult($content)) {
						$ok_or_error = $error;
					}

					$html .= "<li><a href='#$_hash'>" . preg_replace("/_0_.*/", "", preg_replace("/.*\/+/", "", $out_or_err_file)) . "<span class='invert_in_dark_mode'>$ok_or_error</span></a></li>";
				}
				$html .= "</ul>";

				$html_part = "";

				foreach ($out_or_err_files as $out_or_err_file) {
					$content = remove_ansi_colors(file_get_contents($out_or_err_file));
					$_hash = hash('md5', "$out_or_err_file-$content");
					$html_part .= "<div id='$_hash'>\n";  // Hier öffnest du einen neuen Container
					$html_part .= "<pre class='stdout_file invert_in_dark_mode'>" . htmlentities($content) . "\n</pre>\n";
					$html_part .= copy_button("stdout_file");
					$html_part .= "</div>\n";  // Und hier schließt du ihn
				}


				$html .= $html_part;

				$html .= "</div>\n";

				$html .= "<script>";
				$html .= "$(function() {";
				$html .= '    $("#out_files_tabs").tabs();';
				$html .= '    $("#main_tabbed").tabs();';
				$html .= "});";
				$html .= "</script>";
				$html .= "</div>\n";
			}
		}

		return $html;
	}

	function show_run($folder) {
		$html = "<div id='main_tabbed' style='width: fit-content'>\n";
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
		// Extrahiere numerische und alphabetische Teile
		$a_numeric = preg_replace('/[^0-9]/', '', $a);
		$b_numeric = preg_replace('/[^0-9]/', '', $b);

		// Falls beide numerisch sind, sortiere numerisch
		if (is_numeric($a_numeric) && is_numeric($b_numeric)) {
			if ((int)$a_numeric == (int)$b_numeric) {
				return strcmp($a, $b); // Wenn numerisch gleich, alphabetisch sortieren
			}
			return (int)$a_numeric - (int)$b_numeric;
		}

		// Falls nur einer numerisch ist, numerische Sortierung bevorzugen
		if (is_numeric($a_numeric)) {
			return -1;
		}

		if (is_numeric($b_numeric)) {
			return 1;
		}

		// Falls keine numerisch sind, alphabetisch sortieren
		return strcmp($a, $b);
	}

	function check_and_filter_folders($folders) {
		// Überprüfe, ob das übergebene Argument ein Array ist
		if (!is_array($folders)) {
			throw new InvalidArgumentException("Der übergebene Parameter muss ein Array sein.");
		}

		$filtered_folders = array_filter($folders, function($folder) {
			// Überprüfe, ob der Pfad ein Verzeichnis ist
			if (!is_dir($folder)) {
				// Wenn es kein Verzeichnis ist, gebe eine Warnung aus und behalte den Eintrag
				error_log("Warnung: '$folder' ist kein gültiges Verzeichnis.");
				return true;
			}

			// Öffne das Verzeichnis
			$files = scandir($folder);

			// Entferne "." und ".." aus der Liste der Dateien
			$files = array_diff($files, array('.', '..'));

			// Wenn Dateien vorhanden sind, behalte den Eintrag
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
			echo "<a class='_share_link' href=\"share.php?user_id=$user&experiment_name=$experiment_name&run_nr=$run_nr$sharesPathLink\">$run_nr</a><br>";
		}
	}

	function print_script_and_folder($folder) {
		echo "\n<script>createBreadcrumb('./$folder');</script>\n";
	}

	function checkForResult($content) {
		// Regulärer Ausdruck, der nach "RESULT: " gefolgt von einer Zahl sucht (int, negativ, float)
		$pattern = '/RESULT:\s*(-?\d+(\.\d+)?)/';

		// Überprüfe, ob der String mit dem Muster übereinstimmt
		if (preg_match($pattern, $content, $matches)) {
			// Wenn ein Treffer gefunden wurde, gibt $matches[1] die Zahl zurück
			return $matches[1];
		} else {
			// Kein Treffer gefunden
			return false;
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

	function show_dir_view_or_plot($sharesPath, $user_id, $experiment_name, $run_nr) {
		if (isset($user_id) && $user_id != "" && (!isset($experiment_name) || $experiment_name == "")) {
			// given:
			//	user_id
			// missing:
			//	experiment_name
			//	run_nr
			$user = $user_id;
			if (preg_match("/\.\./", $user)) {
				print("Invalid user path");
				exit(1);
			}

			$user = preg_replace("/.*\//", "", $user);

			$experiment_subfolders = glob("$sharesPath/$user/*", GLOB_ONLYDIR);
			if (count($experiment_subfolders) == 0) {
				print("Did not find any experiments for $sharesPath/$user/*");
				exit(0);
			} else {
				foreach ($experiment_subfolders as $experiment) {
					$experiment = preg_replace("/.*\//", "", $experiment);
					$sharesPathLink = $sharesPath == "./shares/" ? "" : "&share_path=$sharesPath";
					echo "<!-- show_dir_view_or_plot A " . __LINE__ . " --><a class='_share_link' href=\"share.php?user_id=$user&experiment_name=$experiment$sharesPathLink\">$experiment</a><br>\n";
				}
				print("<!-- $user/$experiment_name/ -->");
				print_script_and_folder("$user/$experiment_name/");
			}
		} elseif (isset($user_id) && $user_id != "" && (isset($experiment_name) || $experiment_name != "") && (!isset($run_nr) || $run_nr == "")) {
			// given:
			//	user_id
			//	experiment_name
			// missing:
			//	run_nr
			show_run_selection($sharesPath, $user_id, $experiment_name);
			print("<!-- $user_id/$experiment_name/ -->");
			print_script_and_folder("$user_id/$experiment_name/");
		} elseif (isset($user_id) && $user_id != "" && isset($experiment_name) && $experiment_name != "" && isset($run_nr) && $run_nr != "") {
			// given:
			//	user_id
			//	experiment_name
			//	run_nr
			// missing:
			//	none
			$run_folder_without_shares = "$user_id/$experiment_name/$run_nr/";

			$run_folder = "$sharesPath/$run_folder_without_shares";
			if (isset($_GET["get_hash_only"])) {
				echo calculateDirectoryHash($run_folder);

				exit(0);
			} else {
				print("<!-- $run_folder_without_shares -->");

				print_script_and_folder($run_folder_without_shares);

				show_run($run_folder);
			}
		} else {
			// given:
			//	none
			// missing:
			//	user_id
			//	experiment_name
			//	run_nr
			$user_subfolders = glob($sharesPath . '*', GLOB_ONLYDIR);
			if (count($user_subfolders)) {
				foreach ($user_subfolders as $user) {
					$user = preg_replace("/.*\//", "", $user);
					$sharesPathLink = $sharesPath == "./shares/" ? "" : "&share_path=$sharesPath";

					echo "<!-- show_dir_view_or_plot B " . __LINE__ . " -->\n";
					echo "<a class='_share_link' href=\"share.php?user_id=$user$sharesPathLink\">$user</a><br>\n";
				}
			} else {
				echo "No users found";
			}

			print("<!-- startpage -->");
			print_script_and_folder("");
		}
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
					if ($content_encoding == "ASCII" || $content_encoding == "UTF-8") {
						if (filesize($file)) {
							move_uploaded_file($file, "$userFolder/$filename");
							$added_files++;
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
			$old_url = remove_extra_slashes_from_url("$BASEURL/share.php?user_id=$user_id&experiment_name=$experiment_name&run_nr=$run_id");
			echo "This project already seems to have been uploaded. See $old_url\n";
			exit(0);
		} else {
			if (!$uuid_folder || !is_dir($uuid_folder)) {
				$url = remove_extra_slashes_from_url("$BASEURL/share.php?user_id=$user_id&experiment_name=$experiment_name&run_nr=$run_id");

				move_files(
					$offered_files,
					$added_files,
					$userFolder,
					"See $url&update=1 for a live-trace.\n",
					"Run was successfully shared. See $url\nYou can share the link. It is valid for 30 days.\n"
				);
			} else {
				$url = remove_extra_slashes_from_url("$BASEURL/share.php?user_id=$user_id&experiment_name=$experiment_name&run_nr=$run_id&update=1");

				move_files(
					$offered_files,
					$added_files,
					$uuid_folder,
					"See $url for a live-trace.\n",
					"See $url for a live-trace.\n"
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
					if (filetype($dir.'/'.$object) == 'dir') {
						rrmdir($dir.'/'.$object);
					} else {
						unlink($dir.'/'.$object);
					}
				}
			}

			reset($objects);
			rmdir($dir);
		}
	}

	function _delete_old_shares ($dir) {
		$oldDirectories = [];
		$currentTime = time();

		foreach (glob("$dir/*/*/*", GLOB_ONLYDIR) as $subdir) {
			$pathParts = explode('/', $subdir);
			$secondDir = $pathParts[1] ?? '';

			$threshold = ($secondDir === 'runner') ? (2 * 3600) : (30 * 24 * 3600);

			$dir_date = filemtime($subdir);

			if (is_dir($subdir) && ($dir_date < ($currentTime - $threshold))) {
				$oldDirectories[] = $subdir;
				rrmdir($subdir);
			}
		}

		return $oldDirectories;
	}

	function delete_old_shares () {
		$directoryToCheck = 'shares';
		$oldDirs = _delete_old_shares($directoryToCheck);

		return $oldDirs;
	}
