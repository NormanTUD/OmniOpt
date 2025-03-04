<?php
	include_once("_functions.php");
	include_once("share_functions.php");

	function has_invalid_chars($name, $val) {
		if($name == "run_nr") {
			if(!preg_match("/^\d+$/", $val)) {
				die("user_id must be a number!");
			}
		} else if ($name == "experiment_name" || $name == "user_id") {
			if(!preg_match("/^[a-zA-Z_0-9]+$/", $val)) {
				die("$name must consist of numbers, letters or underscore!");
			}
		} else if ($name == "filename") {
			if(!preg_match("/^[a-zA-Z_0-9]+(?:\.(?:txt|json|csv|log|out))?$/", $val)) {
				print json_encode(
					array(
						"raw" => null,
						"error" => "$name must consist of numbers, letters or underscore, and end with .{txt,csv,log}, if it has an ending!"
					)
				);

				exit(1);
			}
		} else {
			print json_encode(
				array(
					"raw" => null,
					"error" => "Unknown name $name"
				)
			);

			exit(1);
		}
	}

	$vars = array(
		"user_id" => get_or_env("user_id"),
		"experiment_name" => get_or_env("experiment_name"),
		"run_nr" => get_or_env("run_nr"),
		"filename" => get_or_env("filename")
	);

	$no_raw_data = get_or_env("no_raw_data");
	$only_raw_data = get_or_env("only_raw_data");

	$missing = array();

	foreach ($vars as $var_key => $var_value) {
		if(has_invalid_chars($var_key, $var_value)) {
			echo "Invalid chars in $var_key.";
			exit(1);
		}

		if(is_null($var_value) || strlen($var_value) == 0) {
			$missing[] = $var_key;
		}
	}

	if(count($missing)) {
		print json_encode(["error" => implode(", ", $missing)." undefined"]);
		exit(1);
	}

	$share_dir = "./shares/";
	if(!is_dir($share_dir)) {
		print json_encode(["error" => "A: $share_dir is not a directory"]);
		exit(1);
	}

	$share_dir = "$share_dir/" . $vars["user_id"] . "/";
	if(!is_dir($share_dir)) {
		print json_encode(["error" => "B: $share_dir is not a directory"]);
		exit(1);
	}

	$share_dir = "$share_dir/" . $vars["experiment_name"] . "/";
	if(!is_dir($share_dir)) {
		print json_encode(["error" => "C: $share_dir is not a directory"]);
		exit(1);
	}

	$share_dir = "$share_dir/" . $vars["run_nr"];
	if(!is_dir($share_dir)) {
		print json_encode(["error" => "D: $share_dir is not a directory"]);
		exit(1);
	}

	$share_file = "$share_dir/" . $vars["filename"];
	if(!file_exists($share_file)) {
		print json_encode(["error" => "E: $share_file is not a file"]);
		exit(1);
	}

	function removeDuplicateCsvRows($csvString) {
		$rows = explode("\n", $csvString);
		$uniqueRows = array();
		$header = null;

		foreach ($rows as $row) {
			$row = trim($row);
			if (empty($row)) {
				continue;
			}

			$rowArray = str_getcsv($row);

			if ($header === null) {
				$header = $rowArray;
				$uniqueRows[] = implode(',', $header);
			} else {
				$rowString = implode(',', $rowArray);
				if (!in_array($rowString, $uniqueRows)) {
					$uniqueRows[] = $rowString;
				}
			}
		}

		return implode("\n", $uniqueRows);
	}

	if(preg_match("/\.csv$/", $share_file)) {
		$raw_file = json_decode(loadCsvToJson($share_file));

		if(isset($_GET["remove_ansi"])) {
			$raw_file = remove_ansi_colors($raw_file);
		}

		$data = array(
			"data" => $raw_file,
			"hash" => hash("md5", file_get_contents($share_file))
		);

		if($no_raw_data != "1") {
			$data["raw"] = removeDuplicateCsvRows(remove_ansi_colors(file_get_contents($share_file)));
		}

		if ($only_raw_data == 1) {
			echo $data["raw"];
		} else {
			header('Content-Type: application/json; charset=utf-8');
			echo json_encode($data);
		}
	} else {
		$raw_file = file_get_contents($share_file);

		if(isset($_GET["remove_ansi"])) {
			$raw_file = parseAnsiToVirtualTerminal(remove_ansi_colors($raw_file));
		}

		$data = array(
			"hash" => hash("md5", file_get_contents($share_file))
		);

		if($no_raw_data != "1") {
			$data["raw"] = $raw_file;
		}

		if ($only_raw_data) {
			echo $data["raw"];
		} else {
			header('Content-Type: application/json; charset=utf-8');
			echo json_encode($data);
		}
	}
?>
