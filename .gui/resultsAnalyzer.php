<?php
function analyzeResultsCSV(string $csvPath): string {
    if (!file_exists($csvPath) || !is_readable($csvPath)) {
        return "## Fehler\nDatei nicht gefunden oder nicht lesbar: `$csvPath`";
    }

    $data = loadCSV($csvPath);
    if ($data === null || count($data) < 2) {
        return "## Fehler\nCSV konnte nicht gelesen werden oder enthält keine Daten.";
    }

    $header = $data[0];
    $rows = array_slice($data, 1);

    $columnStats = calculateStats($header, $rows);
    $correlationMatrix = computeCorrelationMatrix($columnStats);

    return renderMarkdownNarrative($columnStats, $correlationMatrix);
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

function computeCorrelationMatrix(array $stats): array {
    $keys = array_keys($stats);
    $correlations = [];

    for ($i = 0; $i < count($keys); $i++) {
        for ($j = $i + 1; $j < count($keys); $j++) {
            $xKey = $keys[$i];
            $yKey = $keys[$j];

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

function renderMarkdownNarrative(array $stats, array $correlations): string {
    $md = "## 📈 Automatische Analyse der CSV-Daten\n\n";
    $md .= "Die folgenden numerischen Spalten wurden in der Datei erkannt:\n\n";

    foreach ($stats as $col => $s) {
        $md .= "- **`$col`**: $$ \\min = " . round($s['min'], 4) .
               ",\\ \\max = " . round($s['max'], 4) .
               ",\\ \\mu = " . round($s['mean'], 4) .
               ",\\ \\sigma = " . round($s['std'], 4) . " $$\n";
    }

    if (count($correlations) === 0) {
        $md .= "\n### 🔍 Keine signifikanten Zusammenhänge erkannt\n";
        return $md;
    }

    $md .= "\n## 🔄 Statistisch signifikante Korrelationen\n";
    $md .= "Es wurden folgende Zusammenhänge zwischen den Parametern entdeckt:\n\n";

    foreach ($correlations as $c) {
        $a = $c['x'];
        $b = $c['y'];
        $r = round($c['r'], 3);
        $sign = $r > 0 ? "positiven" : "negativen";
        $abs = abs($r);
        $stärke = $abs > 0.85 ? "sehr starken" :
                  ($abs > 0.7 ? "starken" :
                  ($abs > 0.5 ? "moderaten" : "schwachen"));

        $md .= "- Zwischen **`$a`** und **`$b`** besteht eine $stärke $sign lineare Korrelation:\n";
        $md .= "  $$ r_{".$a.",".$b."} = $r $$\n\n";

        // Interpretation in Fließtext
        $richtung = $r > 0
            ? "Je höher der Wert von `$a`, desto höher ist tendenziell auch der Wert von `$b`."
            : "Ein höherer Wert von `$a` geht typischerweise mit einem niedrigeren Wert von `$b` einher.";

        $md .= "  → Interpretation: $richtung\n\n";
    }

    return $md;
}

// Beispiel-Nutzung:
print nl2br(analyzeResultsCSV("/home/norman/repos/OmniOpt/runs/__main__tests__BOTORCH_MODULAR___local_nogridsearch/0/results.csv"));
?>
