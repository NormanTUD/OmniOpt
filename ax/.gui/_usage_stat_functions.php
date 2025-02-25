<?php
	function fetch_data($db_path, $element_id) {
		$db = new SQLite3($db_path);

		$query = "SELECT anon_user, has_sbatch, run_uuid, git_hash, exit_code, runtime, time FROM usage_statistics";

		if($element_id == "test") {
			$query .= " where anon_user = 'affed00faffed00faffed00faffed00f'";
		} else if ($element_id == "developer") {
			$query .= " where anon_user = 'affeaffeaffeaffeaffeaffeaffeaffe'";
		} else {
			$query .= " where anon_user != 'affeaffeaffeaffeaffeaffeaffeaffe' and anon_user != 'affed00faffed00faffed00faffed00f'";
		}

		$result = $db->query($query);
		$data = [];
		while ($row = $result->fetchArray(SQLITE3_NUM)) {
			$data[] = $row;
		}
		$db->close();
		return $data;
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
?>
