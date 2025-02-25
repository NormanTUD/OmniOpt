<?php
	header('Content-Type: application/json');

	require "_usage_stat_functions.php";

	function assignUniqueIds(array $strings) {
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

	$data = fetch_data($db_path);

	$anon_users = array_column($data, 0);
	$has_sbatch = array_column($data, 1);
	$exit_codes = array_map('intval', array_column($data, 4));
	$runtimes = array_map('floatval', array_column($data, 5));

	$show_sbatch_plot = count(array_unique($has_sbatch)) > 1 ? 1 : 0;

	echo json_encode([
		'anon_users' => assignUniqueIds($anon_users),
		'has_sbatch' => $has_sbatch,
		'exit_codes' => $exit_codes,
		'runtimes' => $runtimes,
		'show_sbatch_plot' => $show_sbatch_plot
	]);
?>
