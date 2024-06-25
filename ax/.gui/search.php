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
function search_text_with_context($text_lines, $search_term) {
    $results = [];
    foreach ($text_lines as $line_number => $line) {
        $clean_line = strip_tags($line);
        if (stripos($clean_line, $search_term) !== false) {
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
$file_path = 'folder_structure.php'; // Pfad zu Ihrer PHP-Datei im selben Ordner
$search_term = 'this'; // Der zu suchende Text

$file_content = read_file_content($file_path);
if ($file_content !== false) {
    $html_content = extract_html_from_php($file_content);
    $text_lines = explode("\n", $html_content); // Hier HTML-Inhalt in Zeilen aufteilen

    $search_results = search_text_with_context($text_lines, $search_term);
    $output = [];
    if (!empty($search_results)) {
        foreach ($search_results as $result) {
            $entry = [
                'content' => $result['line']
            ];
            if ($result['context']) {
                $entry['link'] = '#' . $result['context']['id'];
            }
            $output[] = $entry;
        }
    }

    echo json_encode($output);
}
?>

