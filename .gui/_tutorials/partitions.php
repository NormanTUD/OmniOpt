<!-- List of partitions and Memory, GPUs and so on on the HPC-Systems of TU-Dresden -->

<!-- Category: Preparations, Basics and Setup -->

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
		$html = '<table cellpadding="5" cellspacing="0">';
		$html .= '<thead>';
		$html .= '<tr class="invert_in_dark_mode">';
		$html .= '<th class="no_border"></th>';

		// Add a column header for each partition
		foreach ($partitions as $partition) {
			$html .= '<th class="no_border">' . htmlspecialchars($partition['name']) . '</th>';
		}

		$html .= '</tr>';
		$html .= '</thead>';
		$html .= '<tbody>';

		// Define the properties to display
		$properties = [
			'Number of Workers' => 'number_of_workers',
			'Computation Time (min)' => 'computation_time',
			'Max GPUs' => 'max_number_of_gpus',
			'Min GPUs' => 'min_number_of_gpus',
			'Max Memory per Core (MB)' => 'max_mem_per_core',
			'Memory per CPU (MB)' => 'mem_per_cpu',
			'Warning' => 'warning',
			'Link' => 'link'
		];

		// Loop over each property and create a row for it
		foreach ($properties as $label => $key) {
			$html .= '<tr>';
			$html .= '<th class="invert_in_dark_mode no_border">' . $label . '</th>';

			// Loop over each partition and display its value for this property
			foreach ($partitions as $partition) {
				$value = isset($partition[$key]) ? $partition[$key] : 'N/A';
				if ($key === 'link') {
					$html .= '<td><a href="' . htmlspecialchars($value) . '" target="_blank">Documentation</a></td>';
				} else {
					$html .= '<td>' . htmlspecialchars($value) . '</td>';
				}
			}

			$html .= '</tr>';
		}

		$html .= '</tbody>';
		$html .= '</table>';

		return $html;
	}
?>

<h1><span class="invert_in_dark_mode">🖥️</span> Available partitions</h1>

<div id="toc"></div>

<h2 id="available_partitions">Available Partitions</h2>

<?php
	echo createPartitionOverviewTable("partition_data.json");
?>
