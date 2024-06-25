<?php
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
        if (strpos($line, $search_term) !== false) {
            $context = find_nearest_heading($text_lines, $line_number);
            $results[] = [
                'line_number' => $line_number + 1, // 1-basierte Zeilennummerierung
                'line' => $line,
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
                'id' => $matches[2],
                'line_number' => $i + 1 // 1-basierte Zeilennummerierung
            ];
        }
    }
    return null;
}

// Funktion zum Loggen von Fehlern
function log_error($message) {
    error_log($message);
    echo "Fehler: $message\n";
}

// Hauptprogramm
$file_path = 'folder_structure.php'; // Pfad zu Ihrer PHP-Datei
$search_term = '_jobs'; // Der zu suchende Text

print "<pre>";
$file_content = read_file_content($file_path);
if ($file_content !== false) {
    $html_content = extract_html_from_php($file_content);
    $plain_text = strip_html_tags($html_content);
    $text_lines = explode("\n", $plain_text);

    $search_results = search_text_with_context($text_lines, $search_term);
    if (!empty($search_results)) {
        foreach ($search_results as $result) {
            echo "Gefundener Begriff in Datei: $file_path\n";
            echo "Zeilennummer: " . $result['line_number'] . "\n";
            echo "Zeileninhalt: " . $result['line'] . "\n";
            if ($result['context']) {
                echo "Gefundenes Heading: <" . $result['context']['tag'] . "> mit ID '" . $result['context']['id'] . "'\n";
                echo "Link zur Heading: #" . $result['context']['id'] . "\n";
                echo "Heading Zeilennummer: " . $result['context']['line_number'] . "\n";
            }
            echo "--------------------------------------------------\n";
        }
    } else {
        echo "Der Begriff '$search_term' wurde nicht gefunden.\n";
    }
}
?>
</pre>
