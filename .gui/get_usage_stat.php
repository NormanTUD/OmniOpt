<?php
	header('Content-Type: application/json');

	require "_usage_stat_functions.php";

	function assign_unique_ids(array $strings) {
		$idMap = [];
		$counter = 1;
		$result = [];

		foreach ($strings as $string) {
			if (!isset($idMap[$string])) {
				$idMap[$string] = $counter;
				$counter++;
			}
			$result[] = $idMap[$string];
		}

		return $result;
	}

	$db_path = "stats/usage_statistics.db";

	$element_id = $_GET["element_id"];

	$data = fetch_data($db_path, $element_id);

	$anon_users = array_column($data, 0);
	$has_sbatch = array_column($data, 1);
	$exit_codes = array_map('intval', array_column($data, 4));
	$runtimes = array_map('floatval', array_column($data, 5));

	$show_sbatch_plot = count(array_unique($has_sbatch)) > 1 ? 1 : 0;

	echo json_encode([
		'anon_users' => assign_unique_ids($anon_users),
		'has_sbatch' => $has_sbatch,
		'exit_codes' => $exit_codes,
		'runtimes' => $runtimes,
		'show_sbatch_plot' => $show_sbatch_plot
	]);
?>
