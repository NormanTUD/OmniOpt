<?php
	ini_set('memory_limit', '256M');

	require "_usage_stats_header.php";
	require "_usage_stat_functions.php";

	$db_path = "stats/usage_statistics.db";

	function die_with_error_text ($msg) {
		echo '<span class="error_text invert_in_dark_mode">' . $msg . '</span>';

		exit(1);
	}

	if (!class_exists('SQLite3')) {
		die_with_error_text("Fatal error: SQLite3 extension is not installed. Try <tt>sudo apt-get install php-sqlite3</tt> on your host system.\n");
	}

	function check_database_path($db_path) {
		$dir = dirname($db_path);
		$user = posix_getpwuid(posix_geteuid())['name'];

		if (!is_dir($dir)) {
			die_with_error_text("Error: The directory '$dir' does not exist. \nSolution: Create it with:\n  mkdir -p '$dir' && chown $user '$dir' && chmod 755 '$dir'\n");
		}

		if (!is_writable($dir)) {
			die_with_error_text("Error: The directory '$dir' is not writable by user '$user'. \nSolution: Change permissions with:\n  chmod 775 '$dir'\nOr change the owner with:\n  chown $user '$dir'\n");
		}

		if (file_exists($db_path)) {
			if (!is_writable($db_path)) {
				die_with_error_text("Error: The database file '$db_path' is not writable by user '$user'. \nSolution: Change permissions with:\n  chmod 664 '$db_path'\nOr change the owner with:\n  chown $user '$db_path'\n");
			}
		} else {
			if (!is_writable($dir)) {
				die_with_error_text("Error: The database file '$db_path' does not exist and cannot be created in '$dir'. \nSolution: Ensure the directory is writable using:\n  chmod 775 '$dir'\n");
			}
		}

		if (!class_exists('SQLite3')) {
			die_with_error_text("Error: SQLite3 is not available in PHP. \nSolution: Install SQLite3 with:\n  sudo apt install php-sqlite3\nOr enable the extension in 'php.ini'.\n");
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
			die_with_error_text("Failed to initialize database '$db_path': " . $e->getMessage());
		}
	}

	function wrapped_fgetcsv($handle) {
		return fgetcsv($handle, null, ",", "\"", "\\");
	}

	function import_csv_to_database($db_path) {
		$csvFile = __DIR__ . "/stats/usage_statistics.csv";

		if(file_exists($csvFile)) {
			echo "No valid data found in the database. Trying to import existing CSV data into the database, if possible.";
			$pdo = new PDO("sqlite:" . $db_path);
			$pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);

			if (($handle = fopen($csvFile, "r")) !== FALSE) {
				wrapped_fgetcsv($handle);

				$successful_inserts = 0;
				$failed_inserts = 0;

				$stmt = $pdo->prepare("INSERT INTO usage_statistics
					(anon_user, has_sbatch, run_uuid, git_hash, exit_code, runtime, time)
					VALUES (?, ?, ?, ?, ?, ?, ?)");

				while (($data = wrapped_fgetcsv($handle)) !== FALSE) {
					$ret = $stmt->execute($data);

					if(!$ret) {
						echo "<pre>";
						echo "Failed to insert:\n";
						print_r($data);
						echo "</pre>";

						$failed_inserts++;
					} else {
						$successful_inserts++;
					}
				}
				fclose($handle);

				print("Successful inserts: $successful_inserts, failed inserts: $failed_inserts\n");
			} else {
				print("Failed to load CSV file");
			}

			echo "<br>Import done! Reload the page.";

			return true;
		}

		return false;
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

	initialize_database($db_path);

	if (isset($_SERVER["REQUEST_METHOD"]) && isset($_GET["anon_user"])) {
		append_to_db($_GET, $db_path);
	}

	list($developer_ids, $test_ids, $regular_data) = get_group_data($db_path);

	$groups = array(
		'regular_data' => array(
			'label' => 'Usage overview',
			'data'  => $regular_data
		),
		'test_ids' => array(
			'label' => 'Test Users',
			'data'  => $test_ids
		),
		'developer_ids' => array(
			'label' => 'Developer Users',
			'data'  => $developer_ids
		)
	);

	$has_data = false;
	foreach ($groups as $group) {
		if (!empty($group['data'])) {
			$has_data = true;
			break;
		}
	}

	if ($has_data) {
		echo '<section class="tabs">';
		echo '  <menu role="tablist" aria-label="Data Sections">';

		$tab_buttons = '';
		$tab_panels  = '';
		$first = true;

		foreach ($groups as $key => $group) {
			if (!empty($group['data'])) {
				$selected_attr = $first ? ' aria-selected="true"' : '';
				$hidden_attr = $first ? '' : ' hidden';

				$tab_buttons .= '<button role="tab"' . $selected_attr . ' aria-controls="' . $key . '">' . htmlspecialchars($group['label']) . '</button>' . "\n";
				$tab_panels .= '<article role="tabpanel" id="' . $key . '"' . $hidden_attr . '>' . "\n";
				ob_start();
				display_plots($group['data'], explode('_', $key)[0], $db_path);
				$tab_panels .= ob_get_clean();
				$tab_panels .= '</article>' . "\n";

				$first = false;
			}
		}

		$tab_buttons .= '<button role="tab" aria-controls="exit_codes">Exit-Codes</button>' . "\n";
		$tab_panels .= '<article role="tabpanel" hidden id="exit_codes">' . "\n";
		$tab_panels .= '<h3>Exit-Codes</h3>' . "\n";
		ob_start();
		include "exit_code_table.php";
		$tab_panels .= ob_get_clean();
		$tab_panels .= '</article>' . "\n";

		echo $tab_buttons;
		echo '  </menu>';
		echo $tab_panels;
		echo '</section>';

		echo <<<EOT
<script>
document.addEventListener('DOMContentLoaded', function () {
    const tabs = document.querySelectorAll('[role="tab"]');
    const panels = document.querySelectorAll('[role="tabpanel"]');

    tabs.forEach(tab => {
	tab.addEventListener('click', function () {
	    tabs.forEach(t => t.setAttribute('aria-selected', 'false'));
	    panels.forEach(p => p.hidden = true);

	    tab.setAttribute('aria-selected', 'true');
	    const target = document.getElementById(tab.getAttribute('aria-controls'));
	    if (target) {
		target.hidden = false;
	    }
	});
    });
});
</script>
EOT;

	} else {
		if(import_csv_to_database($db_path)) {
			echo "Please reload the page to see the imported statistics";
		} else {
			echo "No statistics data found";
		}
	}

	include("footer.php");
?>
