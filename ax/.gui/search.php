<?php
header('Content-Type: application/json');

// Funktion zum Lesen des Inhalts einer Datei
function read_file_content($file_path) {
    try {
        if (!file_exists($file_path)) {
            throw new Exception("Datei nicht gefunden: $file_path");
        }
        $content = file($file_path, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES);
        if ($content === false) {
            throw new Exception("Fehler beim Lesen der Datei: $file_path");
        }
        return $content;
    } catch (Exception $e) {
        log_error($e->getMessage());
        return false;
    }
}

// Funktion zum Extrahieren von HTML-Code aus PHP-Datei
function extract_html_from_php($file_content) {
    ob_start();
    eval('?>' . implode("\n", $file_content));
    $html_content = ob_get_clean();
    return $html_content;
}

// Funktion zum Entfernen von HTML-Tags
function strip_html_tags($html_content) {
    return strip_tags($html_content);
}

// Funktion zum Durchsuchen des Textes und Finden der Positionen
function search_text_with_context($text_lines, $regex) {
    $results = [];
    foreach ($text_lines as $line_number => $line) {
        $clean_line = strip_tags($line);
        if (preg_match($regex, $clean_line)) {
            $context = find_nearest_heading($text_lines, $line_number);
            $results[] = [
                'line' => trim($clean_line),
                'context' => $context
            ];
        }
    }
    return $results;
}

// Funktion zum Finden der nächsten vor der Zeile liegenden <h1>, <h2>, ... mit ID
function find_nearest_heading($text_lines, $current_line) {
    for ($i = $current_line; $i >= 0; $i--) {
        if (preg_match('/<(h[1-6])\s+[^>]*id=["\']([^"\']+)["\']/', $text_lines[$i], $matches)) {
            return [
                'tag' => $matches[1],
                'id' => $matches[2]
            ];
        }
    }
    return null;
}

// Funktion zum Loggen von Fehlern
function log_error($message) {
    error_log($message);
    echo json_encode(["error" => $message]);
    exit;
}

// Hauptprogramm
$files = ['folder_structure.php', 'plot.php']; // Liste der zu durchsuchenden Dateien
$default_search_term = 'done_jobs'; // Der Standardsuchbegriff

// Überprüfen und Validieren des regulären Ausdrucks
if (isset($_GET['regex'])) {
    $regex = $_GET['regex'];
    // Hinzufügen von "/" Begrenzer, wenn nicht vorhanden
    if (substr($regex, 0, 1) !== '/') {
        $regex = '/' . $regex;
    }
    if (substr($regex, -1) !== '/') {
        $regex = $regex . '/i';
    }
    if (@preg_match($regex, '') === false) {
        log_error("Ungültiger regulärer Ausdruck: $regex");
    }
} else {
    $regex = '/' . preg_quote($default_search_term, '/') . '/i'; // Fallback auf Standardsuchbegriff
}

$output = [];

foreach ($files as $file_path) {
    $file_content = read_file_content($file_path);
    if ($file_content !== false) {
        $html_content = extract_html_from_php($file_content);
        $text_lines = explode("\n", $html_content); // Hier HTML-Inhalt in Zeilen aufteilen

        $search_results = search_text_with_context($text_lines, $regex);
        if (!empty($search_results)) {
            foreach ($search_results as $result) {
                $entry = [
                    'Datei' => $file_path,
                    'Zeileninhalt' => $result['line']
                ];
                if ($result['context']) {
                    $entry['Link zur Heading'] = '#' . $result['context']['id'];
                }
                $output[] = $entry;
            }
        }
    }
}

echo json_encode($output);
?>
