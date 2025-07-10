<?php
function detectCorrelationsAll(array $stats): array {
	$keys = array_keys($stats);
	$correlations = [];

	for ($i = 0; $i < count($keys); $i++) {
		for ($j = $i + 1; $j < count($keys); $j++) {
			$x = $stats[$keys[$i]]['values'];
			$y = $stats[$keys[$j]]['values'];
			if (count($x) !== count($y)) continue;

			$r = pearsonCorrelation($x, $y);
			if (abs($r) > 0.3) {  // Schwellenwert 0.3 für Sichtbarkeit
				$correlations[] = [
					'param1' => $keys[$i],
					'param2' => $keys[$j],
					'correlation' => $r
				];
			}
		}
	}
	return $correlations;
}


function analyzeResultsCSV(string $csvPath, array $resultNames = [], array $resultMinMax = []): string {
	if (!file_exists($csvPath) || !is_readable($csvPath)) {
		return "## Error\nFile not found or not readable: `$csvPath`";
	}

	$data = loadCSV($csvPath);
	if ($data === null || count($data) < 2) {
		return "## Error\nCSV could not be read or contains no data.";
	}

	$header = $data[0];
	$rows = array_slice($data, 1);

	$columnStats = calculateStats($header, $rows);
	$correlationMatrix = computeCorrelationMatrix($columnStats, $resultNames);

	// Additional failure analysis
	$failureAnalysis = analyzeFailures($header, $rows, $columnStats);

	return renderMarkdownNarrative($columnStats, $correlationMatrix, $resultNames, $resultMinMax, $failureAnalysis);
}

function loadCSV(string $path): ?array {
	$rows = [];
	if (($handle = fopen($path, "r")) !== false) {
		while (($line = fgetcsv($handle)) !== false) {
			$rows[] = $line;
		}
		fclose($handle);
	}
	return $rows ?: null;
}

function calculateStats(array $header, array $rows): array {
	$stats = [];
	foreach ($header as $i => $name) {
		$vals = [];
		$isNumeric = true;
		foreach ($rows as $r) {
			if (!isset($r[$i]) || trim($r[$i]) === '') continue;
			if (is_numeric($r[$i])) {
				$vals[] = floatval($r[$i]);
			} else {
				$isNumeric = false;
				break;
			}
		}
		if ($isNumeric && count($vals) > 0) {
			$min = min($vals);
			$max = max($vals);
			$mean = array_sum($vals) / count($vals);
			$std = sqrt(array_sum(array_map(fn($v) => pow($v - $mean, 2), $vals)) / count($vals));
			$stats[$name] = [
				'type' => 'numeric',
				'min' => $min,
				'max' => $max,
				'mean' => $mean,
				'std' => $std,
				'count' => count($vals),
				'values' => $vals,
				'index' => $i,
			];
		} else {
			// treat as categorical
			$vals = [];
			foreach ($rows as $r) {
				if (!isset($r[$i]) || trim($r[$i]) === '') continue;
				$vals[] = $r[$i];
			}
			if (count($vals) > 0) {
				$counts = array_count_values($vals);
				arsort($counts);
				$stats[$name] = [
					'type' => 'categorical',
					'counts' => $counts,
					'unique_count' => count($counts),
					'values' => $vals,
					'index' => $i,
				];
			}
		}
	}
	return $stats;
}

function pearsonCorrelation(array $x, array $y): float {
	$n = count($x);
	if ($n !== count($y) || $n === 0) return 0;

	$meanX = array_sum($x) / $n;
	$meanY = array_sum($y) / $n;

	$num = 0;
	$denX = 0;
	$denY = 0;
	for ($i = 0; $i < $n; $i++) {
		$dx = $x[$i] - $meanX;
		$dy = $y[$i] - $meanY;
		$num += $dx * $dy;
		$denX += $dx * $dx;
		$denY += $dy * $dy;
	}
	return ($denX * $denY) == 0 ? 0 : $num / sqrt($denX * $denY);
}

function computeCorrelationMatrix(array $stats, array $resultNames): array {
	$matrix = [];
	foreach ($resultNames as $result) {
		if (!isset($stats[$result]) || $stats[$result]['type'] !== 'numeric') continue;
		$resultVals = $stats[$result]['values'];

		foreach ($stats as $name => $stat) {
			if ($name === $result) continue;
			if ($stat['type'] === 'numeric') {
				$corr = pearsonCorrelation($stat['values'], $resultVals);
				$matrix[$result][$name] = $corr;
			}
		}
	}
	return $matrix;
}

function analyzeFailures(array $header, array $rows, array $stats): array {
	// We assume column "trial_status" exists and marks FAILED or COMPLETED jobs
	$statusIndex = array_search('trial_status', $header);
	if ($statusIndex === false) {
		return ['error' => "No 'trial_status' column found."];
	}

	$failureFlags = [];
	$paramValues = [];

	// Build arrays of values per column, aligned with failure status 1 or 0
	foreach ($rows as $row) {
		if (!isset($row[$statusIndex])) continue;
		$fail = (strtoupper($row[$statusIndex]) === 'FAILED') ? 1 : 0;
		$failureFlags[] = $fail;
	}

	// We'll collect param values aligned to failureFlags
	foreach ($stats as $name => &$stat) {
		$i = $stat['index'];
		$vals = [];
		foreach ($rows as $row) {
			if (!isset($row[$i]) || trim($row[$i]) === '') {
				$vals[] = null;
				continue;
			}
			if ($stat['type'] === 'numeric') {
				$vals[] = floatval($row[$i]);
			} else {
				$vals[] = $row[$i];
			}
		}
		$paramValues[$name] = $vals;
	}
	unset($stat);

	// Analyze numeric params for correlation with failure
	$numericFailures = [];
	foreach ($stats as $name => $stat) {
		if ($stat['type'] !== 'numeric') continue;
		$vals = [];
		$failFlagsFiltered = [];
		for ($i=0; $i<count($failureFlags); $i++) {
			if ($paramValues[$name][$i] === null) continue; // skip missing
			$vals[] = $paramValues[$name][$i];
			$failFlagsFiltered[] = $failureFlags[$i];
		}
		if (count($vals) < 2) continue;
		$corr = pearsonCorrelation($vals, $failFlagsFiltered);
		// Significant correlation threshold > 0.3 or < -0.3 for example
		if (abs($corr) >= 0.3) {
			$numericFailures[$name] = $corr;
		}
	}

	// Analyze categorical params for failure association (using simple risk ratios)
	$categoricalFailures = [];
	foreach ($stats as $name => $stat) {
		if ($stat['type'] !== 'categorical') continue;
		$valueFailures = [];
		$valueCounts = [];
		for ($i=0; $i<count($failureFlags); $i++) {
			$val = $paramValues[$name][$i];
			if ($val === null) continue;
			if (!isset($valueFailures[$val])) {
				$valueFailures[$val] = 0;
				$valueCounts[$val] = 0;
			}
			$valueFailures[$val] += $failureFlags[$i];
			$valueCounts[$val]++;
		}
		// Compute failure rate per category value
		$failureRates = [];
		foreach ($valueFailures as $val => $failCount) {
			$count = $valueCounts[$val];
			$rate = $count > 0 ? $failCount / $count : 0;
			$failureRates[$val] = $rate;
		}
		// Find values with high failure rate > 0.5 and that have enough samples
		$highRiskValues = [];
		foreach ($failureRates as $val => $rate) {
			if ($rate > 0.5 && $valueCounts[$val] >= 3) { // minimum count 3 to reduce noise
				$highRiskValues[$val] = $rate;
			}
		}
		if (count($highRiskValues) > 0) {
			$categoricalFailures[$name] = $highRiskValues;
		}
	}

	return [
		'numeric_correlations_with_failure' => $numericFailures,
		'categorical_high_failure_values' => $categoricalFailures,
	];
}

function computeParameterCorrelations(array $stats, array $resultNames): array {
	$result = [];

	foreach ($resultNames as $res) {
		if (!isset($stats[$res]) || $stats[$res]['type'] !== 'numeric') continue;

		// Liste aller relevanten Parameter außer dem Result
		$params = array_filter($stats, fn($s, $k) => $k !== $res && $s['type'] === 'numeric', ARRAY_FILTER_USE_BOTH);
		$paramNames = array_keys($params);

		foreach ($paramNames as $i => $paramA) {
			foreach ($paramNames as $j => $paramB) {
				if ($j <= $i) continue; // vermeide doppelt + Eigenkorrelation

				$valA = $stats[$paramA]['values'];
				$valB = $stats[$paramB]['values'];

				$corr = pearsonCorrelation($valA, $valB);

				$result[$res]["$paramA-$paramB"] = $corr;
			}
		}
	}

	return $result;
}

function renderMarkdownNarrative(array $stats, array $correlations, array $result_names, array $resultMinMax): string {
	$md = "## 📊 Summary of CSV Data\n\n";

	$dont_show_col_overview = ["trial_index", "start_time", "end_time", "program_string", "exit_code", "hostname", "arm_name", "generation_node", "trial_status", "submit_time", "queue_time", "OO_Info_SLURM_JOB_ID"];

	foreach ($stats as $col => $s) {
		if (!in_array($col, $dont_show_col_overview)) {
			if (isset($s['min'], $s['max'], $s['mean'], $s['std'], $s['count'])) {
				$md .= "### 🔹 `$col`\n";
				$md .= "The values of **`$col`** range from <b>" . round($s['min'], 4) . "</b> to <b>" . round($s['max'], 4) . "</b>, with an average (mean) of <b>" . round($s['mean'], 4) . "</b> and a standard deviation of <b>" . round($s['std'], 4) . "</b>, based on <b>" . $s['count'] . "</b> data points.\n\n";
			} else {
				$md .= "### 🔹 `$col`\n";
				$md .= "No numerical statistics available for this column.\n\n";
			}

		}
	}

	if (count($correlations) > 0) {
		$cnt = 0;

		foreach ($correlations as $c) {
			if (!isset($c['param1'], $c['param2'], $c['correlation'])) {
				// Skip invalid entries
				continue;
			}

			if ($cnt == 0) {
				$md .= "## 🔍 Detected Correlations Between Parameters\n\n";
				$md .= "The following notable correlations between parameter pairs were found:\n\n";

				$cnt = $cnt + 1;
			}

			$r = round($c['correlation'], 3);
			$abs = abs($r);

			// Choose color based on strength and sign of correlation
			if ($r >= 0.7) {
				$color = "#006400";  // Dark Green for strong positive
			} elseif ($r >= 0.3) {
				$color = "#32CD32";  // Lime Green for moderate positive
			} elseif ($r <= -0.7) {
				$color = "#8B0000";  // Dark Red for strong negative
			} elseif ($r <= -0.3) {
				$color = "#FF4500";  // OrangeRed for moderate negative
			} else {
				$color = "#000000";  // Black for weak/no correlation
			}

			// Determine strength label
			if ($abs >= 0.85) {
				$strength = "very strong";
			} elseif ($abs >= 0.7) {
				$strength = "strong";
			} elseif ($abs >= 0.5) {
				$strength = "moderate";
			} else {
				$strength = "weak";
			}

			// Format correlation coefficient as simple text, no mathjax
			$md .= "<p style=\"color: $color;\">";
			$md .= "Parameters <b>`{$c['param1']}`</b> and <b>`{$c['param2']}`</b> show a <i>{$strength}</i> ";
			$md .= $r > 0 ? "positive" : "negative";
			$md .= " correlation with coefficient <code>r = {$r}</code>.</p>\n";
		}



		function computeDirectionalInfluenceFlat(array $correlations, array $resultMinMax): array {
			$interpretations = [];
			$ignoreKeys = ['trial_index', 'start_time', 'end_time', 'run_time', 'exit_code', 'RESULT1', 'RESULT2'];

			foreach ($correlations as $result => $paramCorrs) {
				if (!isset($resultMinMax[$result])) continue;
				$goal = strtolower($resultMinMax[$result]) === 'min' ? 'minimize' : 'maximize';

				foreach ($paramCorrs as $param => $r) {
					if (in_array($param, $ignoreKeys)) continue;

					$r = round($r, 3);
					$abs = abs($r);
					if ($abs < 0.3) continue;

					$certainty = $abs >= 0.85 ? "very high" :
						($abs >= 0.7 ? "high" :
						($abs >= 0.5 ? "moderate" : "low"));

					$positive_effect = ($goal === 'minimize') ? ($r < 0) : ($r > 0);
					$color = $positive_effect ? "#006400" : "#8B0000";

					$interpretations[] = [
						'html' => "<p style=\"color: $color;\">
						Increasing <code>$param</code> tends to lead to <b>better</b> results for <code>$result</code> (<i>$goal</i> goal),
						with <b>$certainty certainty</b> (r = $r).
						</p>",
			'certainty' => $certainty,
				'result' => $result,
				'param' => $param,
				'r' => $r,
					];
				}
			}
			return $interpretations;
		}



		$influences = computeDirectionalInfluenceFlat($correlations, array_combine($result_names, $resultMinMax));

		if (!empty($influences)) {
			$md .= "\n## 🔁 Parameter Influence on Result Quality\n\n";
			foreach ($influences as $info) {
				$md .= $info['html'] . "\n";
			}
		}



	} else {
		$md .= "## ❕ No notable correlations between parameters were found (threshold: |r| > 0.3).\n";
	}

	return $md;
}
