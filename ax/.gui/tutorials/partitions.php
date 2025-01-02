<?php
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


?>
<h1>Available partitions</h1>

<div id="toc"></div>

<h2 id="available_partitions">Available Partitions</h2>

<?php
	echo createPartitionOverviewTable("partition_data.json");
?>
