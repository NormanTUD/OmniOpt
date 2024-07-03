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
	$html_content = preg_replace("/<head>.*<\/head>/is", "", $html_content);
	return $html_content;
}

// Funktion zum Entfernen von HTML-Tags
function strip_html_tags($html_content) {
	$res = strip_tags($html_content);
	return $res;
}

// Funktion zum Durchsuchen des Textes und Finden der Positionen
function search_text_with_context($text_lines, $regex) {
	$results = [];
	foreach ($text_lines as $line_number => $line) {
		$clean_line = strip_html_tags($line);
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
	header('Content-Type: application/json');
	echo json_encode(["error" => $message]);
	exit;
}

// Hauptprogramm
$php_files = []; // Liste der zu durchsuchenden Dateien

include("searchable_php_files.php");

foreach ($files as $fn => $n) {
	if (is_array($n)) {
		foreach ($n["entries"] as $sub_fn => $sub_n) {
			$php_files[] = "$sub_fn.php";
		}
	} else {
		$php_files[] = "$fn.php";
	}
}

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
	header('Content-Type: application/json');
	print(json_encode(array("error" => "No 'regex' parameter given for search")));
	exit(0);
}

$output = [];

foreach ($php_files as $file_path) {
	if($file_path != "share.php" && $file_path != "usage_stats.php" && $file_path != "index.php") {
		$file_content = read_file_content($file_path);
		if ($file_content !== false) {
			$html_content = extract_html_from_php($file_content);
			$text_lines = explode("\n", $html_content); // Hier HTML-Inhalt in Zeilen aufteilen

			$search_results = search_text_with_context($text_lines, $regex);
			if (!empty($search_results)) {
				foreach ($search_results as $result) {
					$entry = [
						'content' => $result['line']
					];
					if ($result['context']) {
						$entry['link'] = $file_path . '#' . $result['context']['id'];
						$output[] = $entry;
					}
				}
			}
		}
	}
}

header('Content-Type: application/json');
echo json_encode($output);
?>
