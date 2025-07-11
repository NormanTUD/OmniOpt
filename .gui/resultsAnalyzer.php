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

        // Zieltyp (min oder max) ermitteln
        $goalTypeRaw = is_array($resultMinMax[$result]) ? reset($resultMinMax[$result]) : $resultMinMax[$result];
        $goalType = strtolower($goalTypeRaw);
        $goal = $goalType === 'min' ? 'minimize' : 'maximize';

        // Rohdaten für Ergebnis-Werte holen
        $resultValuesRaw = $columns[$result] ?? null;
        if (!$resultValuesRaw || !is_array($resultValuesRaw)) {
            error_log("[Interpretation] ERROR: Missing or invalid values for result '$result' in \$columns.");
            continue;
        }

        // Nur numerische Werte filtern (mind. 3 Werte)
        $resultValues = array_values(array_filter($resultValuesRaw, 'is_numeric'));
        if (count($resultValues) < 3) {
            error_log("[Interpretation] ERROR: Too few numeric result values for '$result' (only " . count($resultValues) . ").");
            continue;
        }

        // Index mit dem besten Ergebniswert finden (max oder min je nach Ziel)
        if ($goal === 'maximize') {
            $bestIndexCandidates = array_keys($resultValuesRaw, max($resultValues));
        } else {
            $bestIndexCandidates = array_keys($resultValuesRaw, min($resultValues));
        }
        if (empty($bestIndexCandidates)) {
            error_log("[Interpretation] ERROR: Could not find best index for '$result'.");
            continue;
        }
        $bestIndex = $bestIndexCandidates[0];
        $bestValue = $resultValuesRaw[$bestIndex];

        // Infos aller relevanten Parameter sammeln
        $paramInfos = [];

        foreach ($paramCorrs as $param => $r) {
            if (in_array($param, $dont_show_col_overview, true)) continue;

            $r = round($r, 3);
            $abs = abs($r);

            if ($abs < 0.05) {
                // Kaum Einfluss, ignorieren
                continue;
            }

            // Werte für Parameter holen und prüfen
            $paramValuesRaw = $columns[$param] ?? [];
            if (!is_array($paramValuesRaw)) {
                error_log("[Interpretation] ERROR: '$param' is missing or invalid in \$columns.");
                continue;
            }
            if (count($paramValuesRaw) !== count($resultValuesRaw)) {
                error_log("[Interpretation] ERROR: Length mismatch between result '$result' and parameter '$param' ("
                    . count($resultValuesRaw) . " vs " . count($paramValuesRaw) . ").");
                continue;
            }

            // Numerische Werte passend zu resultValuesRaw extrahieren (Index-gleich)
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

            // Min, Max und Median-Wert für den Parameter bestimmen
            $paramFullMin = min($paramValues);
            $paramFullMax = max($paramValues);
            $bestParamVal = $paramValuesRaw[$bestIndex];

            $midpoint = ($paramFullMin + $paramFullMax) / 2;
            $range = $paramFullMax - $paramFullMin;
            $medianThreshold = 0.15; // 15% Toleranz um Median

            // Richtung bestimmen mit neuer Logik:
            // - Korrelationsbetrag < 0.2 => "not relevant"
            // - Parameterwert nahe Median => "median"
            // - Sonst Richtung: steigender oder fallender Einfluss
            if ($abs < 0.2) {
                $direction = 'not relevant';
            } elseif ($range > 0 && abs($bestParamVal - $midpoint) <= $range * $medianThreshold) {
                $direction = 'median';
            } else {
                // Wenn Ziel maximize ist und r positiv => steigender Einfluss, sonst fallend
                // Oder umgekehrt für minimize
                $direction = ($goal === 'maximize') === ($r > 0) ? '&uarr; increasing' : '&darr; decreasing';
            }

            // Top 10% der Ergebniswerte bestimmen
            $count = count($resultValues);
            $n_top = max(1, (int)round($count * 0.1));

            // Paarweise sortieren nach Ergebniswert, top N Werte extrahieren
            $zipped = array_map(null, $resultValues, $paramValues);
            usort($zipped, function ($a, $b) use ($goal) {
                return $goal === 'maximize' ? $b[0] <=> $a[0] : $a[0] <=> $b[0];
            });

            $topValues = array_slice($zipped, 0, $n_top);
            $topParamVals = array_column($topValues, 1);

            $paramMin = round(min($topParamVals), 5);
            $paramMax = round(max($topParamVals), 5);

            // Parameterinfo speichern
            $paramInfos[] = [
                'param' => $param,
                'direction' => $direction,
                'certainty' => $abs >= 0.85 ? "very high" :
                               ($abs >= 0.7 ? "high" :
                               ($abs >= 0.5 ? "moderate" : "low")),
                'r' => $r,
                'range_min' => $paramMin,
                'range_max' => $paramMax,
                'bestParamVal' => $bestParamVal,
            ];
        }

        if (empty($paramInfos)) {
            // Keine relevanten Parameter gefunden
            continue;
        }

        // HTML-Ausgabe für Ergebnis erzeugen
        $html = "<h3>Interpretation for result: <code>$result</code> (goal: <b>$goal</b>)</h3>";
        $html .= "<p>Best value: <b>$bestValue</b><br>Achieved at:";
        foreach ($paramInfos as $info) {
            $html .= "<br>- <code>{$info['param']}</code> = {$info['bestParamVal']}";
        }
        $html .= "</p>";

        $html .= "<table border='1' cellpadding='4' cellspacing='0' style='border-collapse: collapse;'>";
        $html .= "<thead><tr><th>Parameter</th><th>Influence</th><th>Certainty</th><th>r</th><th>Typical good range (top 10%)</th></tr></thead><tbody>";
        foreach ($paramInfos as $info) {
            $html .= "<tr>";
            $html .= "<td><code>{$info['param']}</code></td>";
            $html .= "<td>{$info['direction']}</td>";
            $html .= "<td>{$info['certainty']}</td>";
            $html .= "<td>{$info['r']}</td>";
            $html .= "<td>{$info['range_min']} – {$info['range_max']}</td>";
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


function renderMarkdownNarrative(string $csvPath, array $stats, array $correlations, array $result_names, array $resultMinMax): string {
	$md = "## 📊 Summary of CSV Data\n\n";

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

	$md .= "<table border='1' cellpadding='5' cellspacing='0'>";
	$md .= "<thead><tr>
		<th>Column</th>
		<th>Min</th>
		<th>Max</th>
		<th>Mean</th>
		<th>Std Dev</th>
		<th>Count</th>
		</tr></thead>";
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
