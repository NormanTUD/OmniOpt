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
	$correlations = compute_correlation_matrix($columnStats, $resultNames);

	return bring_insights_data_together($csvPath, $header, $rows, $columnStats, $correlations, $resultNames, $resultMinMax);
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

function pearson_correlation(array $x, array $y): ?float {
	$n = count($x);
	if ($n < 3 || $n !== count($y)) return null;

	$meanX = array_sum($x) / $n;
	$meanY = array_sum($y) / $n;

	$sumXY = 0.0;
	$sumX2 = 0.0;
	$sumY2 = 0.0;

	for ($i = 0; $i < $n; $i++) {
		$dx = $x[$i] - $meanX;
		$dy = $y[$i] - $meanY;
		$sumXY += $dx * $dy;
		$sumX2 += $dx * $dx;
		$sumY2 += $dy * $dy;
	}

	$denom = sqrt($sumX2 * $sumY2);
	if ($denom == 0.0) return 0.0;

	return $sumXY / $denom;
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

function compute_correlation_matrix(array $stats, array $resultNames): array {
	$matrix = [];
	foreach ($resultNames as $result) {
		if (!isset($stats[$result]) || $stats[$result]['type'] !== 'numeric') continue;
		$resultVals = $stats[$result]['values'];

		foreach ($stats as $name => $stat) {
			if ($name === $result) continue;
			if ($stat['type'] === 'numeric') {
				$r = pearson_correlation($resultVals, $stat['values']);
				if ($r !== null) {
					$matrix[$result][$name] = $r;
				}
			}
		}
	}
	return $matrix;
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

	$header = fgetcsv($handle, 0, $delimiter, $enclosure, $escape);
	if (!$header) {
		throw new Exception("CSV file is empty or invalid");
	}

	$columns = array_fill_keys($header, []);

	while (($row = fgetcsv($handle, 0, $delimiter, $enclosure, $escape)) !== false) {
		foreach ($header as $i => $colName) {
			$value = $row[$i];
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

		$numericResult = array_filter($resultValues, 'is_numeric');
		if (count($numericResult) < 3) continue;
		$resultNumeric = array_values($numericResult);

		foreach ($columns as $param => $values) {
			if ($param === $result) continue;
			if (in_array($param, $dont_show_col_overview)) continue;

			$numericParam = array_filter($values, 'is_numeric');
			if (count($numericParam) < 3) continue;
			$paramNumeric = array_values($numericParam);

			$r = pearson_correlation($resultNumeric, $paramNumeric);
			if ($r !== null) {
				$correlations[$result][$param] = $r;
			}
		}
	}

	$ret = compute_csv_insights_flat($csvPath, $correlations, $resultMinMax, $dont_show_col_overview, $columns);

	return $ret;
}

function compute_csv_insights_flat(string $csvPath, array $correlations, array $resultMinMax, array $dont_show_col_overview, array $columns = []): array {
	$interpretations = [];

	if (empty($correlations)) {
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
			continue;
		}

		$bestValue = $goal === 'maximize' ? max($resultValues) : min($resultValues);
		$bestIndex = array_keys($resultValuesRaw, $bestValue);
		if (!$bestIndex) {
			error_log("[Interpretation] ERROR: Could not find best index for '$result'.");
			continue;
		}

		$bestIndex = $bestIndex[0];

		// Collect parameter info with actual correlation values
		$paramInfos = [];

		foreach ($paramCorrs as $param => $r) {
			if (in_array($param, $dont_show_col_overview)) {
				continue;
			}

			$paramValuesRaw = $columns[$param] ?? [];
			$bestParamVal = $paramValuesRaw[$bestIndex] ?? 'N/A';

			$paramInfos[] = [
				'param' => $param,
				'bestParamVal' => $bestParamVal,
				'correlation' => $r,
			];
		}

		if (empty($paramInfos)) {
			continue;
		}

		// Sort by absolute correlation strength
		usort($paramInfos, function($a, $b) {
			return abs($b['correlation']) <=> abs($a['correlation']);
		});

		$html = "<h2><code>" . htmlspecialchars($result) . "</code> (goal: <b>$goal</b>)</h2>";
		$html .= "<p>Best value: <b>" . htmlspecialchars((string)$bestValue) . "</b><br>Achieved at:";
		foreach ($paramInfos as $info) {
			$rVal = $info['correlation'];
			$rFormatted = round($rVal, 3);
			$strength = abs($rVal);
			$label = 'weak';
			if ($strength >= 0.7) $label = 'strong';
			elseif ($strength >= 0.4) $label = 'moderate';

			$direction = $rVal > 0 ? 'positive' : 'negative';
			$html .= "<br>- <code>" . htmlspecialchars($info['param']) . "</code> = " . htmlspecialchars((string)$info['bestParamVal']);
			$html .= " <small>(r=$rFormatted, $label $direction correlation)</small>";
		}
		$html .= "</p>";

		$html .= '<div class="result_parameter_visualization" data-resname="' . htmlspecialchars($result) . '"></div>';

		$interpretations[] = [
			'html' => $html,
			'result' => $result,
			'bestValue' => $bestValue,
			'parameters' => $paramInfos,
		];
	}

	return $interpretations;
}

function compute_top_k_results(array $header, array $rows, array $resultNames, array $resultMinMax, int $k = 5): string {
	if (empty($resultNames) || empty($resultMinMax)) return '';
	if (count($rows) < 2) return '';

	$html = "<h2>Top-$k Best Results</h2>";

	foreach ($resultNames as $resultName) {
		if (!isset($resultMinMax[$resultName])) continue;

		$resultIdx = array_search($resultName, $header);
		if ($resultIdx === false) continue;

		$goalType = strtolower(is_array($resultMinMax[$resultName]) ? reset($resultMinMax[$resultName]) : $resultMinMax[$resultName]);
		$goal = $goalType === 'min' ? 'minimize' : 'maximize';

		$validRows = [];
		foreach ($rows as $row) {
			if (isset($row[$resultIdx]) && is_numeric($row[$resultIdx])) {
				$validRows[] = $row;
			}
		}

		if (count($validRows) < 2) continue;

		usort($validRows, function($a, $b) use ($resultIdx, $goal) {
			$va = floatval($a[$resultIdx]);
			$vb = floatval($b[$resultIdx]);
			return $goal === 'minimize' ? $va <=> $vb : $vb <=> $va;
		});

		$topRows = array_slice($validRows, 0, $k);

		$html .= "<h3><code>" . htmlspecialchars($resultName) . "</code> ($goal)</h3>";
		$html .= "<table border='1' cellpadding='5' cellspacing='0'>";
		$html .= "<thead><tr><th>#</th><th>Value</th>";

		// Show only non-special, non-result parameter columns
		$paramCols = [];
		foreach ($header as $i => $col) {
			if ($col === $resultName) continue;
			if (in_array($col, ["trial_index", "start_time", "end_time", "program_string", "exit_code", "hostname", "arm_name", "generation_node", "trial_status", "submit_time", "queue_time", "worker_generator_uuid", "generation_method"])) continue;
			if (strpos($col, 'OO_Info_') === 0) continue;
			$paramCols[] = ['index' => $i, 'name' => $col];
		}

		foreach ($paramCols as $pc) {
			$html .= "<th>" . htmlspecialchars($pc['name']) . "</th>";
		}
		$html .= "</tr></thead><tbody>";

		$rank = 1;
		foreach ($topRows as $row) {
			$html .= "<tr>";
			$html .= "<td>$rank</td>";
			$html .= "<td><b>" . htmlspecialchars($row[$resultIdx]) . "</b></td>";
			foreach ($paramCols as $pc) {
				$val = isset($row[$pc['index']]) ? $row[$pc['index']] : '';
				$html .= "<td>" . htmlspecialchars($val) . "</td>";
			}
			$html .= "</tr>";
			$rank++;
		}

		$html .= "</tbody></table>";
	}

	return $html;
}

function compute_parameter_sensitivity(array $header, array $rows, array $resultNames, array $resultMinMax, array $dont_show_col_overview): string {
	if (empty($resultNames)) return '';

	$html = "<h2>Parameter Sensitivity Ranking</h2>";
	$html .= "<p>Parameters ranked by absolute Pearson correlation with each result. Stronger correlations (|r| &ge; 0.7) suggest the parameter has more influence on the outcome.</p>";

	$columns = [];
	foreach ($header as $i => $col) {
		$columns[$col] = [];
	}
	foreach ($rows as $row) {
		foreach ($header as $i => $col) {
			if (isset($row[$i])) {
				$columns[$col][] = is_numeric($row[$i]) ? (float)$row[$i] : $row[$i];
			}
		}
	}

	foreach ($resultNames as $resultName) {
		if (!isset($columns[$resultName]) || !isset($resultMinMax[$resultName])) continue;

		$resultVals = array_filter($columns[$resultName], 'is_numeric');
		if (count($resultVals) < 3) continue;
		$resultNumeric = array_values($resultVals);

		$goalType = strtolower(is_array($resultMinMax[$resultName]) ? reset($resultMinMax[$resultName]) : $resultMinMax[$resultName]);
		$goal = $goalType === 'min' ? 'minimize' : 'maximize';

		$sensitivities = [];
		foreach ($columns as $param => $values) {
			if ($param === $resultName) continue;
			if (in_array($param, $dont_show_col_overview)) continue;

			$numericVals = array_filter($values, 'is_numeric');
			if (count($numericVals) < 3) continue;
			$paramNumeric = array_values($numericVals);

			$r = pearson_correlation($resultNumeric, $paramNumeric);
			if ($r !== null) {
				$sensitivities[] = ['param' => $param, 'r' => $r, 'abs_r' => abs($r)];
			}
		}

		if (empty($sensitivities)) continue;

		usort($sensitivities, fn($a, $b) => $b['abs_r'] <=> $a['abs_r']);

		$html .= "<h3><code>" . htmlspecialchars($resultName) . "</code> (goal: $goal)</h3>";
		$html .= "<table border='1' cellpadding='5' cellspacing='0'>";
		$html .= "<thead><tr><th>Rank</th><th>Parameter</th><th>Pearson r</th><th>Strength</th><th>Direction</th></tr></thead><tbody>";

		$rank = 1;
		foreach ($sensitivities as $s) {
			$rVal = $s['r'];
			$rFormatted = round($rVal, 3);
			$absR = $s['abs_r'];

			$strength = 'Weak';
			$strengthColor = '#999';
			if ($absR >= 0.7) {
				$strength = 'Strong';
				$strengthColor = '#d32f2f';
			} elseif ($absR >= 0.4) {
				$strength = 'Moderate';
				$strengthColor = '#f57c00';
			}

			$direction = $rVal > 0 ? 'Positive (+)' : 'Negative (-)';

			$html .= "<tr>";
			$html .= "<td>$rank</td>";
			$html .= "<td><code>" . htmlspecialchars($s['param']) . "</code></td>";
			$html .= "<td><b>$rFormatted</b></td>";
			$html .= "<td style='color: $strengthColor; font-weight: bold;'>$strength</td>";
			$html .= "<td>$direction</td>";
			$html .= "</tr>";
			$rank++;
		}

		$html .= "</tbody></table>";
	}

	return $html;
}

function compute_convergence_analysis(array $header, array $rows, array $resultNames, array $resultMinMax): string {
	if (empty($resultNames)) return '';
	if (count($rows) < 5) return '';

	$html = "<h2>Convergence Analysis</h2>";

	$trialIdx = array_search('trial_index', $header);
	if ($trialIdx === false) {
		// Fall back to row order if no trial_index column
		$trialIdx = -1;
	}

	foreach ($resultNames as $resultName) {
		if (!isset($resultMinMax[$resultName])) continue;

		$resultIdx = array_search($resultName, $header);
		if ($resultIdx === false) continue;

		$goalType = strtolower(is_array($resultMinMax[$resultName]) ? reset($resultMinMax[$resultName]) : $resultMinMax[$resultName]);
		$goal = $goalType === 'min' ? 'minimize' : 'maximize';

		$dataPoints = [];
		foreach ($rows as $row) {
			if (!isset($row[$resultIdx]) || !is_numeric($row[$resultIdx])) continue;
			$trialNr = $trialIdx >= 0 && isset($row[$trialIdx]) && is_numeric($row[$trialIdx]) ? intval($row[$trialIdx]) : count($dataPoints);
			$dataPoints[] = ['trial' => $trialNr, 'value' => floatval($row[$resultIdx])];
		}

		if (count($dataPoints) < 5) continue;

		// Running best
		$runningBest = $dataPoints[0]['value'];
		$runningBestTrial = $dataPoints[0]['trial'];
		$improvements = [];

		foreach ($dataPoints as $dp) {
			$improved = false;
			if ($goal === 'minimize' && $dp['value'] < $runningBest) {
				$runningBest = $dp['value'];
				$runningBestTrial = $dp['trial'];
				$improved = true;
			} elseif ($goal === 'maximize' && $dp['value'] > $runningBest) {
				$runningBest = $dp['value'];
				$runningBestTrial = $dp['trial'];
				$improved = true;
			}
			if ($improved) {
				$improvements[] = ['trial' => $dp['trial'], 'value' => $runningBest];
			}
		}

		$totalTrials = end($dataPoints)['trial'];
		$totalEvaluated = count($dataPoints);

		$html .= "<h3><code>" . htmlspecialchars($resultName) . "</code> ($goal)</h3>";

		if (count($improvements) <= 1) {
			$html .= "<p>Best value ($runningBest) was found at trial <b>$runningBestTrial</b>.</p>";
			if (count($improvements) === 0) {
				$html .= "<p>No improvements were found during the optimization run.</p>";
			} else {
				$html .= "<p>The optimizer found an improvement early and did not find a better value in subsequent evaluations.</p>";
			}
		} else {
			$lastImprovementTrial = end($improvements)['trial'];
			$trialsSinceLastImprovement = $totalTrials - $lastImprovementTrial;
			$noImprovePct = $totalEvaluated > 0 ? round(($trialsSinceLastImprovement / max($totalEvaluated, 1)) * 100, 1) : 0;

			$html .= "<p>Best value <b>" . round($runningBest, 6) . "</b> found at trial <b>$runningBestTrial</b> out of $totalEvaluated evaluated.</p>";
			$html .= "<p>The optimizer made <b>" . count($improvements) . " improvements</b> over the course of $totalEvaluated trials.</p>";

			if ($trialsSinceLastImprovement > $totalEvaluated * 0.5) {
				$html .= "<p style='color: #f57c00;'>No improvement in the last $trialsSinceLastImprovement trials (~$noImprovePct% of run). The optimizer may have converged.</p>";
			} else {
				$html .= "<p>The optimizer was still finding improvements relatively late in the run.</p>";
			}

			if (count($improvements) >= 2) {
				$firstImp = $improvements[0]['trial'];
				$lastImp = end($improvements)['trial'];
				$avgSpacing = count($improvements) > 1 ? round(($lastImp - $firstImp) / (count($improvements) - 1), 1) : 'N/A';
				$html .= "<p>Average spacing between improvements: ~$avgSpacing trials.</p>";
			}
		}
	}

	return $html;
}

function compute_generation_method_comparison(array $header, array $rows, array $resultNames, array $resultMinMax): string {
	$genMethodIdx = array_search('generation_method', $header);
	if ($genMethodIdx === false) return '';

	$html = "<h2>Generation Method Comparison</h2>";

	foreach ($resultNames as $resultName) {
		if (!isset($resultMinMax[$resultName])) continue;

		$resultIdx = array_search($resultName, $header);
		if ($resultIdx === false) continue;

		$goalType = strtolower(is_array($resultMinMax[$resultName]) ? reset($resultMinMax[$resultName]) : $resultMinMax[$resultName]);
		$goal = $goalType === 'min' ? 'minimize' : 'maximize';

		$methodResults = [];
		foreach ($rows as $row) {
			$method = isset($row[$genMethodIdx]) ? trim($row[$genMethodIdx]) : '';
			$value = isset($row[$resultIdx]) && is_numeric($row[$resultIdx]) ? floatval($row[$resultIdx]) : null;
			if ($method !== '' && $value !== null) {
				if (!isset($methodResults[$method])) {
					$methodResults[$method] = [];
				}
				$methodResults[$method][] = $value;
			}
		}

		if (count($methodResults) < 2) continue;

		$html .= "<h3><code>" . htmlspecialchars($resultName) . "</code> ($goal)</h3>";
		$html .= "<table border='1' cellpadding='5' cellspacing='0'>";
		$html .= "<thead><tr><th>Method</th><th>Count</th><th>Best</th><th>Mean</th><th>Std Dev</th></tr></thead><tbody>";

		$methodStats = [];
		foreach ($methodResults as $method => $vals) {
			$n = count($vals);
			$best = $goal === 'minimize' ? min($vals) : max($vals);
			$mean = array_sum($vals) / $n;
			$std = $n > 1 ? sqrt(array_sum(array_map(fn($v) => pow($v - $mean, 2), $vals)) / $n) : 0;
			$methodStats[$method] = ['n' => $n, 'best' => $best, 'mean' => $mean, 'std' => $std];
		}

		// Sort by best value
		uasort($methodStats, function($a, $b) use ($goal) {
			return $goal === 'minimize' ? $a['best'] <=> $b['best'] : $b['best'] <=> $a['best'];
		});

		$overallBest = null;
		foreach ($methodStats as $ms) {
			if ($overallBest === null) {
				$overallBest = $ms['best'];
			}
		}

		foreach ($methodStats as $method => $ms) {
			$bestStr = round($ms['best'], 6);
			$meanStr = round($ms['mean'], 6);
			$stdStr = round($ms['std'], 6);

			$highlight = ($ms['best'] == $overallBest) ? ' style="background-color: #e8f5e9;"' : '';

			$html .= "<tr$highlight>";
			$html .= "<td><code>" . htmlspecialchars($method) . "</code></td>";
			$html .= "<td>" . $ms['n'] . "</td>";
			$html .= "<td><b>$bestStr</b></td>";
			$html .= "<td>$meanStr</td>";
			$html .= "<td>$stdStr</td>";
			$html .= "</tr>";
		}

		$html .= "</tbody></table>";

		// Narrative comparison
		$methods = array_keys($methodStats);
		if (count($methods) >= 2) {
			$bestMethod = $methods[0];
			$secondMethod = $methods[1];

			$improvement = 0;
			if ($methodStats[$secondMethod]['best'] != 0) {
				$improvement = abs(($methodStats[$bestMethod]['best'] - $methodStats[$secondMethod]['best']) / $methodStats[$secondMethod]['best']) * 100;
			}

			$html .= "<p><b>" . htmlspecialchars($bestMethod) . "</b> achieved the best result of <b>" . round($methodStats[$bestMethod]['best'], 6) . "</b>";
			if ($improvement > 0) {
				$html .= ", which is " . round($improvement, 1) . "% better than <b>" . htmlspecialchars($secondMethod) . "</b>";
			}
			$html .= ".</p>";
		}
	}

	return $html;
}

function bring_insights_data_together(string $csvPath, array $header, array $rows, array $stats, array $correlations, array $result_names, array $resultMinMax): string {
	$md = "";

	$dont_show_col_overview = ["trial_index", "start_time", "end_time", "program_string", "exit_code", "hostname", "arm_name", "generation_node", "trial_status", "submit_time", "queue_time", "OO_Info_SLURM_JOB_ID", "worker_generator_uuid"];

	$custom_params = [];

	// Existing: best result info
	if (count($correlations) > 0) {
		$influences = compute_csv_insights($csvPath, array_combine($result_names, $resultMinMax), $dont_show_col_overview, $custom_params);

		if (!empty($influences)) {
			foreach ($influences as $info) {
				$md .= $info['html'] . "";
			}
		}
	} else {
		$md .= "## <span class='invert_in_dark_mode'>!</span> No notable correlations between parameters were found (threshold: |r| > 0.3).";
	}

	// New: Top-K best results
	$md .= compute_top_k_results($header, $rows, $result_names, $resultMinMax, 5);

	// New: Parameter sensitivity ranking
	$md .= compute_parameter_sensitivity($header, $rows, $result_names, $resultMinMax, $dont_show_col_overview);

	// New: Generation method comparison
	$md .= compute_generation_method_comparison($header, $rows, $result_names, $resultMinMax);

	// New: Convergence analysis
	$md .= compute_convergence_analysis($header, $rows, $result_names, $resultMinMax);

	// Existing: Parameter statistics table
	$md .= "<h2>Parameter statistics</h2>";

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
