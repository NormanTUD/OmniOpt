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

	return renderMarkdownNarrative($csvPath, $columnStats, $correlationMatrix, $resultNames, $resultMinMax, $failureAnalysis);
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

	$validX = [];
	$validY = [];

	for ($i = 0; $i < $n; $i++) {
		if (is_numeric($x[$i]) && is_numeric($y[$i])) {
			$validX[] = (float)$x[$i];
			$validY[] = (float)$y[$i];
		}
	}

	$n = count($validX);
	if ($n === 0) return 0;

	$meanX = array_sum($validX) / $n;
	$meanY = array_sum($validY) / $n;

	$num = 0;
	$denX = 0;
	$denY = 0;

	for ($i = 0; $i < $n; $i++) {
		$dx = $validX[$i] - $meanX;
		$dy = $validY[$i] - $meanY;
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

function computeDirectionalInfluenceFromCsv(string $csvPath, array $resultMinMax, array $dont_show_col_overview = [], array $custom_params = []): array {
	if (!file_exists($csvPath)) {
		throw new Exception("CSV file not found: $csvPath");
	}

	$handle = fopen($csvPath, 'r');
	if (!$handle) {
		throw new Exception("Cannot open CSV file: $csvPath");
	}

	// Kopfzeile lesen
	$header = fgetcsv($handle);
	if (!$header) {
		throw new Exception("CSV file is empty or invalid");
	}

	// Spalteninitialisierung
	$columns = array_fill_keys($header, []);

	// CSV-Zeilen einlesen
	while (($row = fgetcsv($handle)) !== false) {
		foreach ($header as $i => $colName) {
			$value = $row[$i];
			// Versuche numerisch zu casten (falls möglich)
			if (is_numeric($value)) {
				$columns[$colName][] = (float)$value;
			} else {
				$columns[$colName][] = $value;
			}
		}
	}
	fclose($handle);

	$resultNames = array_keys($resultMinMax);
	$correlations = [];

	foreach ($resultNames as $result) {
		if (!isset($columns[$result])) continue;
		$resultValues = $columns[$result];

		// Nur wenn Result-Spalte numerisch ist
		if (!is_numeric($resultValues[0])) continue;

		foreach ($columns as $param => $values) {
			if ($param === $result) continue;
			if (in_array($param, $dont_show_col_overview)) continue;
			if (!is_numeric($values[0])) continue;

			$r = pearsonCorrelation($values, $resultValues);
			if (!is_finite($r)) continue;

			$correlations[$result][$param] = $r;
		}
	}

	return computeDirectionalInfluenceFlat($correlations, $resultMinMax, $dont_show_col_overview, $columns);
}

function computeDirectionalInfluenceFlat(array $correlations, array $resultMinMax, array $dont_show_col_overview, array $columns = []): array {
	$interpretations = [];

	if (empty($correlations)) {
		error_log("[Interpretation] ERROR: correlations are empty.");
		return [];
	}

	foreach ($correlations as $result => $paramCorrs) {
		if (!isset($resultMinMax[$result])) {
			error_log("[Interpretation] WARNING: Missing resultMinMax entry for '$result'. Skipping.");
			continue;
		}

		$goalType = strtolower(is_array($resultMinMax[$result]) ? reset($resultMinMax[$result]) : $resultMinMax[$result]);
		$goal = $goalType === 'min' ? 'minimize' : 'maximize';

		$resultValuesRaw = $columns[$result] ?? null;
		if (!$resultValuesRaw || !is_array($resultValuesRaw)) {
			error_log("[Interpretation] ERROR: Missing or invalid values for result '$result' in \$columns.");
			continue;
		}

		$resultValues = array_values(array_filter($resultValuesRaw, 'is_numeric'));
		if (count($resultValues) < 3) {
			error_log("[Interpretation] ERROR: Too few numeric result values for '$result' (only " . count($resultValues) . ").");
			continue;
		}

		$bestIndex = array_keys($resultValuesRaw, $goal === 'maximize' ? max($resultValues) : min($resultValues));
		if (!$bestIndex) {
			error_log("[Interpretation] ERROR: Could not find best index for '$result'.");
			continue;
		}
		$bestIndex = $bestIndex[0];
		$bestValue = $resultValuesRaw[$bestIndex];

$width = 200;
$height = count($paramInfos);
$im = imagecreatetruecolor($width, $height);
imagesavealpha($im, true);
$trans = imagecolorallocatealpha($im, 0, 0, 0, 127);
imagefill($im, 0, 0, $trans);
imagealphablending($im, true);

// Farbverlauf vorbereiten
for ($x = 0; $x < $width; $x++) {
	$t = $x / ($width - 1);
	$colorVal = function($t) use ($goal) {
		if ($goal === 'maximize') {
			return [
				(int)(255 * (1 - $t)), // R
				(int)(255 * $t),       // G
				0                      // B
			];
		} else {
			return [
				(int)(255 * $t),       // R
				(int)(255 * (1 - $t)), // G
				0                      // B
			];
		}
	};

	foreach ($paramInfos as $y => $info) {
		[$r, $g, $b] = $colorVal($x / ($width - 1));
		$col = imagecolorallocate($im, $r, $g, $b);
		imagesetpixel($im, $x, $y, $col);
	}
}

// Marker für bestParamVal
foreach ($paramInfos as $y => $info) {
	$min = min($columns[$info['param']]);
	$max = max($columns[$info['param']]);
	$val = $info['bestParamVal'];
	if (!is_numeric($val) || $max == $min) continue;
	$x = (int)(($val - $min) / ($max - $min) * ($width - 1));
	$dotColor = imagecolorallocate($im, 0, 0, 0); // Schwarz
	imagesetpixel($im, $x, $y, $dotColor);
}

// Als Base64 einbetten
ob_start();
imagepng($im);
$imgData = ob_get_clean();
imagedestroy($im);
$base64 = base64_encode($imgData);

// Bild als <img>
$html .= "<p><b>Parameter influence overview for <code>$result</code></b><br>";
$html .= "<img src='data:image/png;base64,$base64' width='$width' height='50' style='image-rendering: pixelated;'></p>";


		$html .= "</tbody></table>";

		$interpretations[] = [
			'html' => $html,
			'result' => $result,
			'goal' => $goal,
			'bestValue' => $bestValue,
			'parameters' => $paramInfos,
		];
	}

	return $interpretations;
}

// Helper function to compute quantile from sorted array
function quantile(array $sortedValues, float $q): float {
    $n = count($sortedValues);
    if ($n === 0) return NAN;
    $pos = ($n - 1) * $q;
    $floor = floor($pos);
    $ceil = ceil($pos);
    if ($floor == $ceil) {
        return $sortedValues[$floor];
    }
    $d0 = $sortedValues[$floor] * ($ceil - $pos);
    $d1 = $sortedValues[$ceil] * ($pos - $floor);
    return round($d0 + $d1, 5);
}

function renderMarkdownNarrative(string $csvPath, array $stats, array $correlations, array $result_names, array $resultMinMax): string {
	$md = "";

	$dont_show_col_overview = ["trial_index", "start_time", "end_time", "program_string", "exit_code", "hostname", "arm_name", "generation_node", "trial_status", "submit_time", "queue_time", "OO_Info_SLURM_JOB_ID"];

	$custom_params = [];

	if (count($correlations) > 0) {
		$influences = computeDirectionalInfluenceFromCsv($csvPath, array_combine($result_names, $resultMinMax), $dont_show_col_overview, $custom_params);

		if (!empty($influences)) {
			$md .= "\n## 🔁 Parameter Influence on Result Quality\n\n";
			foreach ($influences as $info) {
				$md .= $info['html'] . "";
			}
		}
	} else {
		$md .= "## ❕ No notable correlations between parameters were found (threshold: |r| > 0.3).";
	}

	$md .= "<h2>📊 Parameter statistics</h2>";

	$md .= "<table border='1' cellpadding='5' cellspacing='0'>";
	$md .= "<thead><tr><th>Parameter</th><th>Min</th><th>Max</th><th>Mean</th><th>Std Dev</th><th>Count</th></tr></thead>";
	$md .= "<tbody>";

	foreach ($stats as $col => $s) {
		if (!in_array($col, $dont_show_col_overview)) {
			$md .= "<tr>";
			$md .= "<td>" . htmlspecialchars($col) . "</td>";
			if (isset($s['min'], $s['max'], $s['mean'], $s['std'], $s['count'])) {
				$md .= "<td>" . round($s['min'], 4) . "</td>";
				$md .= "<td>" . round($s['max'], 4) . "</td>";
				$md .= "<td>" . round($s['mean'], 4) . "</td>";
				$md .= "<td>" . round($s['std'], 4) . "</td>";
				$md .= "<td>" . $s['count'] . "</td>";
			} else {
				$md .= "<td colspan='5' style='text-align:center;'>No numerical statistics available</td>";
			}
			$md .= "</tr>";
		}
	}

	$md .= "</tbody></table>";
	return $md;
}
