<?php
// Helper function to log and display errors
function log_error($error_message) {
    error_log($error_message);
    echo "<p>Error: $error_message</p>";
}

// Validate input parameters
function validate_parameters($params) {
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
    $headers = array('anon_user', 'has_sbatch', 'run_uuid', 'git_hash', 'exit_code');
    $file_exists = file_exists($filepath);

    $file = fopen($filepath, 'a');
    if (!$file_exists) {
        fputcsv($file, $headers);
    }
    fputcsv($file, $params);
    fclose($file);
}

// Create plots using Plotly.js
function display_plots($filepath) {
    echo '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>';
    echo '<div id="plotly-chart"></div>';

    $data = array_map('str_getcsv', file($filepath));
    $headers = array_shift($data);
    $data_json = json_encode($data);

    echo "<script>
        const headers = " . json_encode($headers) . ";
        const data = $data_json;

        const users = data.map(row => row[0]);
        const exitCodes = data.map(row => parseInt(row[4]));

        const userPlot = {
            x: users,
            type: 'histogram',
            name: 'Runs per User'
        };

        const exitCodePlot = {
            y: exitCodes,
            type: 'violin',
            name: 'Exit Codes'
        };

        const tablePlot = {
            type: 'table',
            header: {
                values: headers
            },
            cells: {
                values: headers.map((_, i) => data.map(row => row[i]))
            }
        };

        const layout = {
            title: 'Usage Statistics',
            grid: {rows: 2, columns: 2, pattern: 'independent'}
        };

        Plotly.newPlot('plotly-chart', [userPlot, exitCodePlot, tablePlot], layout);
    </script>";
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
        echo "Data successfully written to CSV.";
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
