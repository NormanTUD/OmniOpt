<?php
function create_insights(string $csvPath, array $resultNames = [], array $resultMinMax = []): string {
	if (!file_exists($csvPath) || !is_readable($csvPath)) {
		return "## Error\nFile not found or not readable: `$csvPath`";
	}

	$data = load_csv($csvPath);
	if ($data === null || count($data) < 2) {
		return "## Error\nCSV could not be read or contains no data.";
	}

	$header = $data[0];
	$rows = array_slice($data, 1);

	$columnStats = calculate_stats($header, $rows);
	$correlationMatrix = compute_correlation_matrix($columnStats, $resultNames);

	// Additional failure analysis
	$failureAnalysis = analyze_failures($header, $rows, $columnStats);

	return render_markdown($csvPath, $columnStats, $correlationMatrix, $resultNames, $resultMinMax, $failureAnalysis);
}

function load_csv(string $path): ?array {
	$delimiter = ",";
	$enclosure = "\"";
	$escape = "\\";

	$rows = [];
	if (($handle = fopen($path, "r")) !== false) {
		while (($line = fgetcsv($handle, 0, $delimiter, $enclosure, $escape)) !== false) {
			$rows[] = $line;
		}
		fclose($handle);
	}
	return $rows ?: null;
}

function calculate_stats(array $header, array $rows): array {
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

function pearson_correlation(array $x, array $y): float {
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

function compute_correlation_matrix(array $stats, array $resultNames): array {
	$matrix = [];
	foreach ($resultNames as $result) {
		if (!isset($stats[$result]) || $stats[$result]['type'] !== 'numeric') continue;
		$resultVals = $stats[$result]['values'];

		foreach ($stats as $name => $stat) {
			if ($name === $result) continue;
			if ($stat['type'] === 'numeric') {
				$corr = pearson_correlation($stat['values'], $resultVals);
				$matrix[$result][$name] = $corr;
			}
		}
	}
	return $matrix;
}

function analyze_failures(array $header, array $rows, array $stats): array {
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
		$corr = pearson_correlation($vals, $failFlagsFiltered);
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

function compute_csv_insights(string $csvPath, array $resultMinMax, array $dont_show_col_overview = [], array $custom_params = []): array {
	if (!file_exists($csvPath)) {
		throw new Exception("CSV file not found: $csvPath");
	}

	$handle = fopen($csvPath, 'r');
	if (!$handle) {
		throw new Exception("Cannot open CSV file: $csvPath");
	}

	$delimiter = ",";
	$enclosure = "\"";
	$escape = "\\";

	// Kopfzeile lesen
	$header = fgetcsv($handle, 0, $delimiter, $enclosure, $escape);
	if (!$header) {
		throw new Exception("CSV file is empty or invalid");
	}

	// Spalteninitialisierung
	$columns = array_fill_keys($header, []);

	// CSV-Zeilen einlesen
	while (($row = fgetcsv($handle, 0, $delimiter, $enclosure, $escape)) !== false) {
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

			$r = pearson_correlation($values, $resultValues);
			if (!is_finite($r)) continue;

			$correlations[$result][$param] = $r;
		}
	}

	return compute_csv_insights_flat($csvPath, $correlations, $resultMinMax, $dont_show_col_overview, $columns);
}

function compute_csv_insights_flat(string $csvPath, array $correlations, array $resultMinMax, array $dont_show_col_overview, array $columns = []): array {
	$interpretations = [];

	if (empty($correlations)) {
		error_log("[Interpretation] ERROR: correlations are empty.");
		return [];
	}


	$delimiter = ",";
	$enclosure = "\"";
	$escape = "\\";

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

		// Collect parameter info
		$paramInfos = [];

		foreach ($paramCorrs as $param => $r) {
			if (in_array($param, $dont_show_col_overview)) continue;

			$r = round($r, 3);
			$abs = abs($r);

			if ($abs < 0.05) {
				// Too weak influence, ignore
				continue;
			}

			$paramValuesRaw = $columns[$param] ?? [];
			if (!is_array($paramValuesRaw)) {
				error_log("[Interpretation] ERROR: '$param' is missing or invalid in \$columns.");
				continue;
			}
			if (count($paramValuesRaw) != count($resultValuesRaw)) {
				error_log("[Interpretation] ERROR: Mismatch in length between result '$result' and parameter '$param' ("
					. count($resultValuesRaw) . " vs " . count($paramValuesRaw) . ").");
				continue;
			}

			// Extract aligned numeric values
			$paramValues = [];
			foreach ($resultValuesRaw as $i => $val) {
				if (is_numeric($val) && isset($paramValuesRaw[$i]) && is_numeric($paramValuesRaw[$i])) {
					$paramValues[] = $paramValuesRaw[$i];
				}
			}
			if (count($paramValues) < 3) {
				error_log("[Interpretation] ERROR: Too few valid numeric values for parameter '$param' (only " . count($paramValues) . ").");
				continue;
			}

			// Calculate min/max of full param range
			$paramFullMin = min($paramValues);
			$paramFullMax = max($paramValues);

			// Value of parameter at best index
			$bestParamVal = $paramValuesRaw[$bestIndex];

			// Number of top results (10%)
			$count = count($resultValues);
			$n_top = max(1, (int)round($count * 0.1));

			// Pair result and param values and sort by result (best first)
			$zipped = array_map(null, $resultValues, $paramValues);
			usort($zipped, function ($a, $b) use ($goal) {
				return $goal === 'maximize' ? $b[0] <=> $a[0] : $a[0] <=> $b[0];
			});

			// Get top 10% param values
			$topValues = array_slice($zipped, 0, $n_top);
			$topParamVals = array_column($topValues, 1);

			sort($topParamVals);
			$paramInfos[] = [
				'param' => $param,
				'bestParamVal' => $bestParamVal,
			];
		}

		if (empty($paramInfos)) {
			continue;
		}

		$html = "<h3>Interpretation for result: <code>$result</code> (goal: <b>$goal</b>)</h3>";
		$html .= "<p>Best value: <b>$bestValue</b><br>Achieved at:";
		foreach ($paramInfos as $info) {
			$html .= "<br>- <code>{$info['param']}</code> = {$info['bestParamVal']}";
		}
		$html .= "</p>";

		$html .= "<table border='1' cellpadding='4' cellspacing='0' style='border-collapse: collapse;'>";
		$html .= "<thead><tr><th>Parameter</th>"
			. "<th>Visualization in dependence of $result</th></tr></thead><tbody>";
		$csvData = [];
		if (($handle = fopen($csvPath, 'r')) !== false) {
			$header = fgetcsv($handle, 0, $delimiter, $enclosure, $escape);
			if ($header === false) {
				error_log("CSV file $csvPath appears empty or invalid header.");
			} else {
				while (($row = fgetcsv($handle, 0, $delimiter, $enclosure, $escape)) !== false) {
					$csvData[] = array_combine($header, $row);
				}
			}
			fclose($handle);
		} else {
			error_log("Could not open CSV file $csvPath");
		}

		foreach ($paramInfos as $info) {
			$html .= "<tr>";
			$html .= "<td><code>{$info['param']}</code></td>";

			$width = 1000;
			$height = count($paramInfos);
			$im = imagecreatetruecolor($width, $height);

			// Hintergrund weiß
			$white = imagecolorallocate($im, 255, 255, 255);
			imagefill($im, 0, 0, $white);

			$height = 1;
			$im = imagecreatetruecolor($width, $height);

			// Weißer Hintergrund
			$white = imagecolorallocate($im, 255, 255, 255);
			imagefill($im, 0, 0, $white);

			$param = $info['param'] ?? null;
			if ($param === null) {
				error_log("Param not set in info");
				imagedestroy($im);
				continue;
			}

			// Prüfe, ob $columns[$param] existiert und nicht leer
			if (empty($columns[$param])) {
				error_log("No data in columns for param '$param'");
				imagedestroy($im);
				continue;
			}

			$minVal = (float) min($columns[$param]);
			$maxVal = (float) max($columns[$param]);
			if ($maxVal == $minVal) {
				error_log("Max equals min for param '$param'");
				imagedestroy($im);
				continue;
			}

			// Werte aus CSV durchgehen und je 1px in der Breite zeichnen
			foreach ($csvData as $row) {
				if (!isset($row[$param])) continue;
				if (!isset($row[$result])) {
					error_log("Result column '$result' not found in CSV row");
					continue;
				}

				$val = (float) $row[$param];
				$resValRaw = $row[$result];

				if (!is_numeric($resValRaw)) {
					error_log("Non-numeric result value '$resValRaw' for param '$param'");
					continue;
				}
				$resVal = (float) $resValRaw;

				$x = (int)(($val - $minVal) / ($maxVal - $minVal) * ($width - 1));
				if ($x < 0) $x = 0;
				if ($x >= $width) $x = $width - 1;

				// Berechne t zwischen 0 und 1, je nachdem ob min oder max gewünscht ist
				if (!isset($resultMinMax[$result])) {
					error_log("No min/max info for result '$result'");
					$t = 0.5;
				} else {
					$allResultValsRaw = array_column($csvData, $result);
					$allResultVals = array_filter($allResultValsRaw, 'is_numeric');

					if (count($allResultVals) === 0) {
						error_log("No numeric values for result '$result'");
						$t = 0.5;
					} else {
						$minResult = min($allResultVals);
						$maxResult = max($allResultVals);

						if ($maxResult == $minResult) {
							$t = 0.5;
						} else {
							if ($resultMinMax[$result] === 'min') {
								$t = ($resVal - $minResult) / ($maxResult - $minResult);
							} else {
								$t = 1 - (($resVal - $minResult) / ($maxResult - $minResult));
							}
							$t = max(0, min(1, $t));
						}
					}
				}

				// Farbverlauf grün -> gelb -> rot
				if ($t <= 0.5) {
					$r = (int)(255 * ($t * 2));    // von 0 bis 255
					$g = 255;
				} else {
					$r = 255;
					$g = (int)(255 * (1 - 2 * ($t - 0.5)));  // von 255 runter bis 0
				}
				$b = 0;

				$color = imagecolorallocate($im, $r, $g, $b);
				imagesetpixel($im, $x, 0, $color);
			}

			// Bild als Base64 einbetten
			ob_start();
			imagepng($im);
			$imgData = ob_get_clean();
			imagedestroy($im);
			$base64 = base64_encode($imgData);

			$img_as_base64 = "<img src='data:image/png;base64,$base64' width='$width' height='50' style='image-rendering: pixelated;'>";

			$html .= "<td>{$img_as_base64}</td>";
			$html .= "</tr>";
		}

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

function render_markdown(string $csvPath, array $stats, array $correlations, array $result_names, array $resultMinMax): string {
	$md = "";

	$dont_show_col_overview = ["trial_index", "start_time", "end_time", "program_string", "exit_code", "hostname", "arm_name", "generation_node", "trial_status", "submit_time", "queue_time", "OO_Info_SLURM_JOB_ID"];

	$custom_params = [];

	if (count($correlations) > 0) {
		$influences = compute_csv_insights($csvPath, array_combine($result_names, $resultMinMax), $dont_show_col_overview, $custom_params);

		if (!empty($influences)) {
			$md .= "\n## <span class='invert_in_dark_mode'>🔁</span> Parameter Influence on Result Quality\n\n";
			foreach ($influences as $info) {
				$md .= $info['html'] . "";
			}
		}
	} else {
		$md .= "## <span class='invert_in_dark_mode'>❕</span> No notable correlations between parameters were found (threshold: |r| > 0.3).";
	}

	$md .= "<h2><span class='invert_in_dark_mode'>📊</span> Parameter statistics</h2>";

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
