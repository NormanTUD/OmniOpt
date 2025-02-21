<?php
	ini_set('memory_limit', '256M');

	require "_usage_stats_header.php";
	require "_usage_stat_functions.php";

	$db_path = "stats/usage_statistics.db";

	if (!class_exists('SQLite3')) {
		die("Fatal error: SQLite3 extension is not installed. Try <tt>sudo apt-get install php-sqlite3</tt> on your host system.\n");
	}

	function check_database_path($db_path) {
		$dir = dirname($db_path);
		$user = posix_getpwuid(posix_geteuid())['name']; // Aktueller Benutzer

		// Prüfen, ob das Verzeichnis existiert
		if (!is_dir($dir)) {
			die("Error: The directory '$dir' does not exist. \nSolution: Create it with:\n  mkdir -p '$dir' && chown $user '$dir' && chmod 755 '$dir'\n");
		}

		// Prüfen, ob das Verzeichnis beschreibbar ist
		if (!is_writable($dir)) {
			die("Error: The directory '$dir' is not writable by user '$user'. \nSolution: Change permissions with:\n  chmod 775 '$dir'\nOr change the owner with:\n  chown $user '$dir'\n");
		}

		// Prüfen, ob die Datei existiert
		if (file_exists($db_path)) {
			// Prüfen, ob die Datei beschreibbar ist
			if (!is_writable($db_path)) {
				die("Error: The database file '$db_path' is not writable by user '$user'. \nSolution: Change permissions with:\n  chmod 664 '$db_path'\nOr change the owner with:\n  chown $user '$db_path'\n");
			}
		} else {
			// Falls die Datei nicht existiert, prüfen, ob sie erstellt werden kann
			if (!is_writable($dir)) {
				die("Error: The database file '$db_path' does not exist and cannot be created in '$dir'. \nSolution: Ensure the directory is writable using:\n  chmod 775 '$dir'\n");
			}
		}

		// Prüfen, ob SQLite3 verfügbar ist
		if (!class_exists('SQLite3')) {
			die("Error: SQLite3 is not available in PHP. \nSolution: Install SQLite3 with:\n  sudo apt install php-sqlite3\nOr enable the extension in 'php.ini'.\n");
		}
	}

	function initialize_database($db_path) {
		check_database_path($db_path);

		try {
			$db = new SQLite3($db_path);
			$db->exec("PRAGMA journal_mode=WAL;");
			$db->exec("CREATE TABLE IF NOT EXISTS usage_statistics (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				anon_user TEXT NOT NULL,
				has_sbatch INTEGER NOT NULL,
				run_uuid TEXT NOT NULL,
				git_hash TEXT NOT NULL,
				exit_code INTEGER NOT NULL,
				runtime REAL NOT NULL,
				time INTEGER NOT NULL
			);");
			$db->exec("CREATE INDEX IF NOT EXISTS idx_time ON usage_statistics(time);");
			$db->close();
		} catch (Exception $e) {
			die("Failed to initialize database '$db_path': " . $e->getMessage());
		}
	}

	function importCsvToDatabase($db_path) {
		$csvFile = __DIR__ . "/stats/usage_statistics.csv";

		$pdo = new PDO("sqlite:" . $db_path);
		$pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);

		if (($handle = fopen($csvFile, "r")) !== FALSE) {
			fgetcsv($handle);

			$stmt = $pdo->prepare("INSERT INTO usage_statistics 
				(anon_user, has_sbatch, run_uuid, git_hash, exit_code, runtime, time) 
				VALUES (?, ?, ?, ?, ?, ?, ?)");

			while (($data = fgetcsv($handle)) !== FALSE) {
				$stmt->execute($data);
			}
			fclose($handle);
		}

		echo "<br>Import done! Reload the page.";
	}

	function append_to_db($params, $db_path) {
		if (validate_parameters($params, $db_path)) {
			$params["time"] = time();

			try {
				$db = new SQLite3($db_path);
				$stmt = $db->prepare("INSERT INTO usage_statistics (anon_user, has_sbatch, run_uuid, git_hash, exit_code, runtime, time)
				                      VALUES (:anon_user, :has_sbatch, :run_uuid, :git_hash, :exit_code, :runtime, :time)");
				$stmt->bindValue(':anon_user', $params['anon_user'], SQLITE3_TEXT);
				$stmt->bindValue(':has_sbatch', $params['has_sbatch'], SQLITE3_INTEGER);
				$stmt->bindValue(':run_uuid', $params['run_uuid'], SQLITE3_TEXT);
				$stmt->bindValue(':git_hash', $params['git_hash'], SQLITE3_TEXT);
				$stmt->bindValue(':exit_code', $params['exit_code'], SQLITE3_INTEGER);
				$stmt->bindValue(':runtime', $params['runtime'], SQLITE3_FLOAT);
				$stmt->bindValue(':time', $params['time'], SQLITE3_INTEGER);
				$stmt->execute();
				$db->close();
			} catch (Exception $e) {
				log_error("Failed to write to database: " . $e->getMessage() . ". Check permissions for <tt>$db_path</tt>.");
				exit(1);
			}
		} else {
			log_error("Parameters contain wrong values. Cannot save.");
			exit(1);
		}
	}

	function get_group_data($db_path) {
		$id_map = [
			'affeaffeaffeaffeaffeaffeaffeaffe' => 'developer_ids',
			'affed00faffed00faffed00faffed00f' => 'test_ids'
		];

		$groups = ['developer_ids' => [], 'test_ids' => [], 'regular_data' => []];

		try {
			$db = new SQLite3($db_path);
			$query = "SELECT anon_user, has_sbatch, run_uuid, git_hash, exit_code, runtime, time FROM usage_statistics";
			$result = $db->query($query);

			while ($row = $result->fetchArray(SQLITE3_NUM)) {
				$key = $id_map[$row[0]] ?? 'regular_data';
				$groups[$key][] = $row;
			}

			$db->close();
		} catch (Exception $e) {
			die("Failed to fetch data: " . $e->getMessage());
		}

		return array_values($groups);
	}

	initialize_database($db_path);

	if (isset($_SERVER["REQUEST_METHOD"]) && isset($_GET["anon_user"])) {
		append_to_db($_GET, $db_path);
	}

	$data = fetch_data($db_path);

	if (!empty($data)) {
		list($developer_ids, $test_ids, $regular_data) = get_group_data($db_path);
?>
		<br>
		<div id="tabs">
			<ul>
<?php
				$links = ['regular_data' => 'Regular Users', 'test_ids' => 'Tests', 'developer_ids' => 'Developer'];
				foreach ($links as $key => $label) {
					if (count(${$key})) {
						echo '<li class="invert_in_dark_mode"><a href="#' . $key . '">' . $label . '</a></li>';
					}
				}
?>
				<li class="invert_in_dark_mode"><a href="#exit_codes">Exit-Codes</a></li>
			</ul>
<?php
		$sections = ['regular_data' => 'Regular Users', 'test_ids' => 'Test Users', 'developer_ids' => 'Developer Users'];
		foreach ($sections as $key => $title) {
			if (count(${$key})) {
				echo '<div id="' . $key . '">';
				echo "<h2>$title</h2>";
				display_plots(${$key}, explode('_', $key)[0], $db_path);
				echo '</div>';
			}
		}
?>
		<div id="exit_codes">
<?php
			include "exit_code_table.php";
?>
		</div>
<?php
	} else {
		echo "No valid data found in the database";

		importCsvToDatabase($db_path);
	}
	include("footer.php");
?>
