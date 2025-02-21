<?php
	function fetch_data($db_path) {
		$db = new SQLite3($db_path);
		$result = $db->query("SELECT anon_user, has_sbatch, run_uuid, git_hash, exit_code, runtime, time FROM usage_statistics");
		$data = [];
		while ($row = $result->fetchArray(SQLITE3_NUM)) {
			$data[] = $row;
		}
		$db->close();
		return $data;
	}
?>
