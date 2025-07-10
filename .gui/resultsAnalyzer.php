<?php
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

function renderMarkdownNarrative(array $stats, array $correlations, array $resultNames, array $resultMinMax, array $failureAnalysis): string {
    $md = "# Analysis Report\n\n";

    $md .= "## Summary Statistics\n\n";
    foreach ($stats as $name => $stat) {
        if ($stat['type'] === 'numeric') {
            $md .= "- `$name`: numeric, min={$stat['min']}, max={$stat['max']}, mean=".round($stat['mean'],3).", std=".round($stat['std'],3)."\n";
        } else {
            $md .= "- `$name`: categorical, unique values={$stat['unique_count']}\n";
        }
    }

    if (count($resultNames) > 0) {
        $md .= "\n## Correlations with Result Metrics\n\n";
        foreach ($correlations as $result => $corrs) {
            $md .= "- **$result** correlations:\n";
            arsort($corrs);
            foreach ($corrs as $param => $corr) {
                $sign = ($corr > 0) ? "+" : "";
                $md .= "  - $param: {$sign}".round($corr,3)."\n";
            }
            $md .= "\n";
        }
    } else {
        $md .= "\n_No result metrics provided for correlation analysis._\n";
    }

    // Add failure analysis
    $md .= "\n## Failure Analysis\n\n";
    if (isset($failureAnalysis['error'])) {
        $md .= "⚠️ " . $failureAnalysis['error'] . "\n\n";
    } else {
        // Numeric correlation with failure
        if (count($failureAnalysis['numeric_correlations_with_failure']) > 0) {
            $md .= "Numeric parameters showing moderate correlation (≥ 0.3 absolute) with failure:\n\n";
            foreach ($failureAnalysis['numeric_correlations_with_failure'] as $param => $corr) {
                $sign = ($corr > 0) ? "higher values correlate with failures" : "lower values correlate with failures";
                $md .= "- `$param`: correlation = ".round($corr,3)." → $sign\n";
            }
            $md .= "\n";
        } else {
            $md .= "No strong numeric correlations with failure detected.\n\n";
        }

        // Categorical values with high failure risk
        if (count($failureAnalysis['categorical_high_failure_values']) > 0) {
            $md .= "Categorical parameters with values strongly associated with failures (failure rate > 50% and ≥ 3 samples):\n\n";
            foreach ($failureAnalysis['categorical_high_failure_values'] as $param => $vals) {
                $md .= "- `$param`:\n";
                foreach ($vals as $val => $rate) {
                    $percent = round($rate*100,1);
                    $md .= "  - Value '$val' has failure rate ≈ {$percent}%\n";
                }
            }
            $md .= "\n";
        } else {
            $md .= "No categorical values with strongly elevated failure rates found.\n\n";
        }
    }

    $md .= "---\n\n*This report was generated automatically.*\n";

    return $md;
}

// Example usage:
// echo analyzeResultsCSV("results.csv", ["loss", "accuracy"], ["loss" => "min", "accuracy" => "max"]);

