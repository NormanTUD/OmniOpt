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

    return renderMarkdownNarrative($columnStats, $correlationMatrix, $resultNames, $resultMinMax);
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
        foreach ($rows as $r) {
            if (!isset($r[$i]) || trim($r[$i]) === '') continue;
            if (is_numeric($r[$i])) {
                $vals[] = floatval($r[$i]);
            }
        }
        if (count($vals) === 0) continue;

        $min = min($vals);
        $max = max($vals);
        $mean = array_sum($vals) / count($vals);
        $std = sqrt(array_sum(array_map(fn($v) => pow($v - $mean, 2), $vals)) / count($vals));
        $stats[$name] = [
            'min' => $min,
            'max' => $max,
            'mean' => $mean,
            'std' => $std,
            'count' => count($vals),
            'values' => $vals
        ];
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
    $keys = array_keys($stats);
    $correlations = [];

    $ignore = [
        'trial_index','start_time','end_time','run_time','program_string','exit_code',
        'signal','hostname','arm_name','trial_status','generation_method','generation_node'
    ];

    // Also ignore OO_Info... and all non-numeric combinations
    $ignore = array_merge($ignore, array_filter($keys, fn($k) => str_starts_with($k, "OO_Info")));

    for ($i = 0; $i < count($keys); $i++) {
        for ($j = $i + 1; $j < count($keys); $j++) {
            $xKey = $keys[$i];
            $yKey = $keys[$j];

            if (in_array($xKey, $ignore) || in_array($yKey, $ignore)) continue;
            if (!isset($stats[$xKey]) || !isset($stats[$yKey])) continue;

            $xVals = $stats[$xKey]['values'];
            $yVals = $stats[$yKey]['values'];
            if (count($xVals) !== count($yVals)) continue;

            $r = pearsonCorrelation($xVals, $yVals);
            if (abs($r) > 0.5) {
                $correlations[] = [
                    'x' => $xKey,
                    'y' => $yKey,
                    'r' => $r
                ];
            }
        }
    }
    return $correlations;
}

function renderMarkdownNarrative(array $stats, array $correlations, array $resultNames, array $resultMinMax): string {
    $md = "## 📊 Automatic CSV Analysis Report\n\n";
    $md .= "The following numerical columns were detected in the data:\n\n";

    foreach ($stats as $col => $s) {
        $md .= "- **`$col`**: $$ \\min = " . round($s['min'], 4) .
               ",\\ \\max = " . round($s['max'], 4) .
               ",\\ \\mu = " . round($s['mean'], 4) .
               ",\\ \\sigma = " . round($s['std'], 4) . " $$\n";
    }

    if (count($correlations) === 0) {
        $md .= "\n### 🔍 No significant correlations found\n";
        return $md;
    }

    $md .= "\n## 🔄 Statistically Significant Correlations\n";
    $md .= "The following relationships were discovered between parameters:\n\n";

    foreach ($correlations as $c) {
        $a = $c['x'];
        $b = $c['y'];
        $r = round($c['r'], 3);
        $sign = $r > 0 ? "positive" : "negative";
        $abs = abs($r);
        $strength = $abs > 0.85 ? "very strong" :
                    ($abs > 0.7 ? "strong" :
                    ($abs > 0.5 ? "moderate" : "weak"));

        $md .= "- There is a **$strength $sign linear correlation** between **`$a`** and **`$b`**:\n";
        $md .= "  $$ r_{\\text{" . $a . "}, \\text{" . $b . "}} = $r $$\n\n";

        // Interpretation
        $explanation = $r > 0
            ? "Higher values of `$a` tend to be associated with higher values of `$b`."
            : "Higher values of `$a` tend to be associated with lower values of `$b`.";

        // Add result optimization context if applicable
        $optimizeHint = "";
        if (in_array($a, $resultNames)) {
            $idx = array_search($a, $resultNames);
            if (isset($resultMinMax[$idx])) {
                $goal = $resultMinMax[$idx] === 'min' ? "lower is better" : "higher is better";
                $optimizeHint .= " (`$a`: $goal)";
            }
        }
        if (in_array($b, $resultNames)) {
            $idx = array_search($b, $resultNames);
            if (isset($resultMinMax[$idx])) {
                $goal = $resultMinMax[$idx] === 'min' ? "lower is better" : "higher is better";
                $optimizeHint .= " (`$b`: $goal)";
            }
        }

        $md .= "  → Interpretation: $explanation$optimizeHint\n\n";
    }

    return $md;
}
?>
