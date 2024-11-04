<?php
	chdir(__DIR__);

	if (!function_exists("dier")) {
		function dier($data, $enable_html = 0, $exception = 0) {
			#$source_data = debug_backtrace()[0];
			#@$source = 'Aufgerufen von <b>' . debug_backtrace()[1]['file'] . '</b>::<i>' . debug_backtrace()[1]['function'] . '</i>, line ' . htmlentities($source_data['line']) . "<br>\n";
			#$print = $source;
			$print = "";

			$print .= "<pre>\n";
			ob_start();
			print_r($data);
			$buffer = ob_get_clean();
			if ($enable_html) {
				$print .= $buffer;
			} else {
				$print .= htmlentities($buffer);
			}
			$print .= "</pre>\n";

			$print .= "Backtrace:\n";
			$print .= "<pre>\n";
			foreach (debug_backtrace() as $trace) {
				$print .= htmlentities(sprintf("\n%s:%s %s", $trace['file'], $trace['line'], $trace['function']));
			}
			$print .= "</pre>\n";

			if (!$exception) {
				print $print;
				exit();
			} else {
				throw new Exception($print);
			}
		}
	}

	function _isCurl(){
		return function_exists('curl_version');
	}

	function getFirstTagName($url) {
		if(!_isCurl()) {
			return "";
		}

		$ch = curl_init();

		curl_setopt($ch, CURLOPT_URL, $url);
		curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
		curl_setopt($ch, CURLOPT_USERAGENT, 'PHP'); // GitHub API benötigt einen User-Agent

		$response = curl_exec($ch);

		if (curl_errno($ch)) {
			curl_close($ch);
			return null; // cURL-Fehler
		}

		$http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
		curl_close($ch);

		if ($http_code == 200) {
			$data = json_decode($response, true);

			if (isset($data[0]['name'])) {
				return $data[0]['name'];
			}
		}

		return null;
	}

	function get_current_tag() {
		$url = "https://api.github.com/repos/NormanTUD/OmniOpt/tags";
		$tagName = getFirstTagName($url);

		return $tagName;
	}

	function get_or_env ($name) {
		$var = isset($_GET[$name]) ? $_GET[$name] : null;

		if(is_null($var) || !strlen($var) && strlen(getenv($name))) {
			$var = getenv($name);
		}

		return $var;
	}

	function createPartitionOverviewTable($jsonFilePath) {
		// Check if file exists and is readable
		if (!file_exists($jsonFilePath) || !is_readable($jsonFilePath)) {
			return "<p>Error: JSON file not found or not readable.</p>";
		}

		// Decode JSON file
		$jsonData = file_get_contents($jsonFilePath);
		$partitions = json_decode($jsonData, true);

		// Check if JSON decoding was successful
		if ($partitions === null) {
			return "<p>Error: Invalid JSON data.</p>";
		}

		// Start table HTML
		$html = '<table border="1" cellpadding="5" cellspacing="0">';
		$html .= '<thead>';
		$html .= '<tr>';
		$html .= '<th>Partition Name</th>';
		$html .= '<th>Number of Workers</th>';
		$html .= '<th>Computation Time (min)</th>';
		$html .= '<th>Max GPUs</th>';
		$html .= '<th>Min GPUs</th>';
		$html .= '<th>Max Memory per Core (MB)</th>';
		$html .= '<th>Memory per CPU (MB)</th>';
		$html .= '<th>Warning</th>';
		$html .= '<th>Link</th>';
		$html .= '</tr>';
		$html .= '</thead>';
		$html .= '<tbody>';

		// Populate rows with data from each partition
		foreach ($partitions as $partitionKey => $partition) {
			$html .= '<tr>';
			$html .= '<td>' . htmlspecialchars($partition['name']) . '</td>';
			$html .= '<td>' . htmlspecialchars($partition['number_of_workers']) . '</td>';
			$html .= '<td>' . htmlspecialchars($partition['computation_time']) . '</td>';
			$html .= '<td>' . htmlspecialchars($partition['max_number_of_gpus']) . '</td>';
			$html .= '<td>' . htmlspecialchars($partition['min_number_of_gpus']) . '</td>';
			$html .= '<td>' . htmlspecialchars($partition['max_mem_per_core']) . '</td>';
			$html .= '<td>' . htmlspecialchars($partition['mem_per_cpu']) . '</td>';
			$html .= '<td>' . htmlspecialchars($partition['warning']) . '</td>';
			$html .= '<td><a href="' . htmlspecialchars($partition['link']) . '" target="_blank">Documentation</a></td>';
			$html .= '</tr>';
		}

		$html .= '</tbody>';
		$html .= '</table>';

		return $html;
	}
