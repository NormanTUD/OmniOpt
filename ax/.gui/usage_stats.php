<?php
	function dier($msg) {
		print("<pre>".print_r($msg, true)."</pre>");
		exit(1);
	}
?>
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Usage Statistics</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
        }
        th {
            padding-top: 12px;
            padding-bottom: 12px;
            text-align: left;
            background-color: #4CAF50;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
    </style>
</head>
<body>
    <?php
    function log_error($error_message) {
        error_log($error_message);
        echo "<p>Error: $error_message</p>";
    }

    function validate_parameters($params) {
        assert(is_array($params), "Parameters should be an array");

        $required_params = ['anon_user', 'has_sbatch', 'run_uuid', 'git_hash', 'exit_code', 'runtime'];
        $patterns = [
            'anon_user' => '/^[a-f0-9]{32}$/',
            'has_sbatch' => '/^[01]$/',
            'run_uuid' => '/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/',
            'git_hash' => '/^[0-9a-f]{40}$/',
            'exit_code' => '/^\d{1,3}$/',
            'runtime' => '/^\d+(\.\d+)?$/'  // Positive number (integer or decimal)
        ];

        foreach ($required_params as $param) {
            if (!isset($params[$param])) {
                log_error("Missing required parameter: $param");
                return false;
            }
            if (!preg_match($patterns[$param], $params[$param])) {
                log_error("Invalid format for parameter: $param");
                return false;
            }
        }

        $exit_code = intval($params['exit_code']);
        if ($exit_code < 0 || $exit_code > 255) {
            log_error("Invalid exit_code value: $exit_code");
            return false;
        }

        $runtime = floatval($params['runtime']);
        if ($runtime < 0) {
            log_error("Invalid runtime value: $runtime");
            return false;
        }

        return true;
    }

    function append_to_csv($params, $filepath) {
        assert(is_array($params), "Parameters should be an array");
        assert(is_string($filepath), "Filepath should be a string");

        $headers = ['anon_user', 'has_sbatch', 'run_uuid', 'git_hash', 'exit_code', 'runtime'];
        $file_exists = file_exists($filepath);

        try {
            $file = fopen($filepath, 'a');
            if (!$file_exists) {
                fputcsv($file, $headers);
            }
            fputcsv($file, $params);
            fclose($file);
        } catch (Exception $e) {
            log_error("Failed to write to CSV: " . $e->getMessage());
        }
    }

    function validate_csv($filepath) {
        if (!file_exists($filepath) || !is_readable($filepath)) {
            log_error("CSV file does not exist or is not readable.");
            return false;
        }

        try {
            $file = fopen($filepath, 'r');
            $content = fread($file, filesize($filepath));
            fclose($file);
            if (!mb_check_encoding($content, 'UTF-8') && !mb_check_encoding($content, 'ASCII')) {
                log_error("CSV file encoding is not valid UTF-8 or ASCII.");
                return false;
            }
        } catch (Exception $e) {
            log_error("Failed to read CSV file: " . $e->getMessage());
            return false;
        }

        return true;
    }

    function filter_data($data) {
        $developer_ids = [];
        $test_ids = [];
        $regular_data = [];

        foreach ($data as $row) {
            if ($row[0] == 'affeaffeaffeaffeaffeaffeaffeaffe') {
                $developer_ids[] = $row;
            } elseif ($row[0] == 'affed00faffed00faffed00faffed00f') {
                $test_ids[] = $row;
            } else {
                $regular_data[] = $row;
            }
        }

        return [$developer_ids, $test_ids, $regular_data];
    }

    function display_plots($data, $title, $element_id) {
        $anon_users = array_column($data, 0);
        $has_sbatch = array_column($data, 1);
        $exit_codes = array_map('intval', array_column($data, 4));
        $runtimes = array_map('floatval', array_column($data, 5));

        $unique_sbatch = array_unique($has_sbatch);
        $show_sbatch_plot = count($unique_sbatch) > 1 ? '1' : 0;

        echo "<div id='$element_id-exit-codes' style='height: 400px;'></div>";
        echo "<div id='$element_id-runs' style='height: 400px;'></div>";
        echo "<div id='$element_id-violins' style='height: 400px;'></div>";
        echo "<div id='$element_id-runtimes' style='height: 400px;'></div>";
        echo "<div id='$element_id-runtime-vs-exit-code' style='height: 400px;'></div>";

        if ($show_sbatch_plot) {
            echo "<div id='$element_id-sbatch' style='height: 400px;'></div>";
        }

        echo "<script>
            var anon_users_$element_id = " . json_encode($anon_users) . ";
            var has_sbatch_$element_id = " . json_encode($has_sbatch) . ";
            var exit_codes_$element_id = " . json_encode($exit_codes) . ";
            var runtimes_$element_id = " . json_encode($runtimes) . ";

            var exitCodePlot = {
                x: exit_codes_$element_id,
                type: 'histogram',
                marker: {
                    color: exit_codes_$element_id.map(function(code) { return code == 0 ? 'blue' : 'red'; })
                },
                name: 'Exit Codes'
            };

            var userPlot = {
                x: anon_users_$element_id,
                type: 'histogram',
                name: 'Runs per User'
            };

            var violinPlot = {
                y: exit_codes_$element_id,
                x: anon_users_$element_id,
                type: 'violin',
                points: 'all',
                jitter: 0.3,
                name: 'Exit Codes per User'
            };

            var runtimePlot = {
                x: runtimes_$element_id,
                type: 'histogram',
                name: 'Runtimes'
            };

            var runtimeVsExitCodePlot = {
                x: exit_codes_$element_id,
                y: runtimes_$element_id,
                mode: 'markers',
                type: 'scatter',
                marker: {
                    color: exit_codes_$element_id.map(function(code) { return code == 0 ? 'blue' : 'red'; })
                },
                name: 'Runtime vs Exit Code'
            };

            Plotly.newPlot('$element_id-exit-codes', [exitCodePlot], {title: '$title - Exit Codes'});
            Plotly.newPlot('$element_id-runs', [userPlot], {title: '$title - Runs per User'});
            Plotly.newPlot('$element_id-violins', [violinPlot], {title: '$title - Exit Codes per User'});
            Plotly.newPlot('$element_id-runtimes', [runtimePlot], {title: '$title - Runtimes'});
            Plotly.newPlot('$element_id-runtime-vs-exit-code', [runtimeVsExitCodePlot], {title: '$title - Runtime vs Exit Code'});

            if ($show_sbatch_plot) {
                var sbatchPlot = {
                    x: has_sbatch_$element_id,
                    type: 'bar',
                    name: 'SBatch Usage'
                };
                Plotly.newPlot('$element_id-sbatch', [sbatchPlot], {title: '$title - SBatch Usage'});
            }
        </script>";
    }

    function generate_html_table($data, $headers) {
        echo "<table>";
        echo "<tr>";
        foreach ($headers as $header) {
            echo "<th>$header</th>";
        }
        echo "</tr>";

        foreach ($data as $row) {
            echo "<tr>";
            foreach ($row as $cell) {
                echo "<td>$cell</td>";
            }
            echo "</tr>";
        }

        echo "</table>";
    }

    $params = $_GET;
    $stats_dir = 'stats';
    $csv_path = $stats_dir . '/usage_statistics.csv';

    if (validate_parameters($params)) {
        if (!file_exists($stats_dir)) {
            mkdir($stats_dir, 0777, true);
        }

        if (is_writable($stats_dir)) {
            append_to_csv($params, $csv_path);
            echo "<p>Data successfully written to CSV.</p>";
        } else {
            log_error("Stats directory is not writable.");
        }
    }

    if (validate_csv($csv_path)) {
        $data = array_map('str_getcsv', file($csv_path));
        $headers = array_shift($data);

        [$developer_ids, $test_ids, $regular_data] = filter_data($data);

        echo "<h2>Regular Users</h2>";
        generate_html_table($regular_data, $headers);
        display_plots($regular_data, "Regular Users Statistics", "regular_plots");

        echo "<h2>Developer Machines</h2>";
        generate_html_table($developer_ids, $headers);
        display_plots($developer_ids, "Developer Machines Statistics", "developer_plots");

        echo "<h2>Automated Tests</h2>";
        generate_html_table($test_ids, $headers);
        display_plots($test_ids, "Automated Tests Statistics", "test_plots");
    } else {
        log_error("No valid data available to display.");
    }
    ?>
</body>
</html>
