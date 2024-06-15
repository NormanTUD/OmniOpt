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
    // Helper function to log and display errors
    function log_error($error_message) {
        error_log($error_message);
        echo "<p>Error: $error_message</p>";
    }

    // Validate input parameters
    function validate_parameters($params) {
        assert(is_array($params), "Parameters should be an array");

        $required_params = ['anon_user', 'has_sbatch', 'run_uuid', 'git_hash', 'exit_code'];
        $patterns = [
            'anon_user' => '/^[a-f0-9]{32}$/',  // MD5 hash
            'has_sbatch' => '/^[01]$/',  // 0 or 1
            'run_uuid' => '/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/',  // UUID
            'git_hash' => '/^[0-9a-f]{40}$/',  // Git hash
            'exit_code' => '/^\d{1,3}$/'  // 0-255
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

        return true;
    }

    // Append data to CSV file
    function append_to_csv($params, $filepath) {
        assert(is_array($params), "Parameters should be an array");
        assert(is_string($filepath), "Filepath should be a string");

        $headers = ['anon_user', 'has_sbatch', 'run_uuid', 'git_hash', 'exit_code'];
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

    // Create plots using Plotly.js
    function display_plots($filepath) {
        assert(is_string($filepath), "Filepath should be a string");

        echo '<div id="plotly-chart"></div>';

        try {
            $data = array_map('str_getcsv', file($filepath));
        } catch (Exception $e) {
            log_error("Failed to read CSV file: " . $e->getMessage());
            return;
        }

        $headers = array_shift($data);

        $anon_users = array_column($data, 0);
        $has_sbatch = array_column($data, 1);
        $run_uuids = array_column($data, 2);
        $git_hashes = array_column($data, 3);
        $exit_codes = array_map('intval', array_column($data, 4));

        $non_zero_exit_codes = array_filter($exit_codes, function($code) { return $code != 0; });

        echo "<script>
            const anon_users = " . json_encode($anon_users) . ";
            const has_sbatch = " . json_encode($has_sbatch) . ";
            const git_hashes = " . json_encode($git_hashes) . ";
            const exit_codes = " . json_encode($exit_codes) . ";
            const non_zero_exit_codes = " . json_encode($non_zero_exit_codes) . ";

            // Histogram of Exit Codes
            const exitCodePlot = {
                x: exit_codes,
                type: 'histogram',
                marker: {
                    color: exit_codes.map(code => code == 0 ? 'blue' : 'red')
                },
                name: 'Exit Codes'
            };

            // Histogram of Runs per User
            const userPlot = {
                x: anon_users,
                type: 'histogram',
                name: 'Runs per User'
            };

            // Violin Plot of Exit Codes per User
            const violinPlot = {
                y: exit_codes,
                x: anon_users,
                type: 'violin',
                points: 'all',
                jitter: 0.3,
                name: 'Exit Codes per User'
            };

            // Scatter Plot for Exit Codes and Git Hashes
            const scatterPlot = {
                x: git_hashes,
                y: exit_codes,
                mode: 'markers',
                marker: {
                    color: exit_codes.map(code => code == 0 ? 'blue' : 'red')
                },
                name: 'Exit Code vs Git Hash'
            };

            // Bar Chart for SBatch Usage
            const sbatchPlot = {
                x: has_sbatch,
                type: 'bar',
                name: 'SBatch Usage'
            };

            const layout = {
                title: 'Usage Statistics',
                grid: {rows: 3, columns: 2, pattern: 'independent'},
                annotations: [{
                    text: 'Exit Codes',
                    x: 0.25,
                    xref: 'paper',
                    y: 1.05,
                    yref: 'paper',
                    showarrow: false
                }, {
                    text: 'Runs per User',
                    x: 0.75,
                    xref: 'paper',
                    y: 1.05,
                    yref: 'paper',
                    showarrow: false
                }, {
                    text: 'Exit Codes per User',
                    x: 0.25,
                    xref: 'paper',
                    y: 0.65,
                    yref: 'paper',
                    showarrow: false
                }, {
                    text: 'Exit Code vs Git Hash',
                    x: 0.75,
                    xref: 'paper',
                    y: 0.65,
                    yref: 'paper',
                    showarrow: false
                }, {
                    text: 'SBatch Usage',
                    x: 0.25,
                    xref: 'paper',
                    y: 0.25,
                    yref: 'paper',
                    showarrow: false
                }]
            };

            Plotly.newPlot('plotly-chart', [exitCodePlot, userPlot, violinPlot, scatterPlot, sbatchPlot], layout);
        </script>";

        // Generate HTML table
        echo "<h2>Data Table</h2>";
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

    // Main script execution
    $params = $_GET;

    if (validate_parameters($params)) {
        $stats_dir = 'stats';
        if (!file_exists($stats_dir)) {
            mkdir($stats_dir, 0777, true);
        }

        if (is_writable($stats_dir)) {
            $filepath = $stats_dir . '/usage_statistics.csv';
            append_to_csv($params, $filepath);
            echo "<p>Data successfully written to CSV.</p>";
        } else {
            log_error("Stats directory is not writable.");
        }
    } else {
        $filepath = 'stats/usage_statistics.csv';
        if (file_exists($filepath)) {
            display_plots($filepath);
        } else {
            log_error("No data available to display.");
        }
    }
    ?>
</body>
</html>
