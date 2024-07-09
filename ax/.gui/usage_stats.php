<?php
    error_reporting(E_ALL);
    set_error_handler(function ($severity, $message, $file, $line) {
        throw new \ErrorException($message, $severity, $severity, $file, $line);
    });

    ini_set('display_errors', 1);

    function dier($msg) {
        print("<pre>" . print_r($msg, true) . "</pre>");
        exit(1);
    }

    include("_header_base.php");
?>
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
  <link rel="stylesheet" href="jquery-ui.css">
  <script src="jquery-ui.js"></script>
  <script>
  $( function() {
    $( "#tabs" ).tabs();
  } );
  </script>
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
            'runtime' => '/^\d+(\.\d+)?$/'
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
        if ($exit_code < -1 || $exit_code > 255) {
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

        $headers = ['anon_user', 'has_sbatch', 'run_uuid', 'git_hash', 'exit_code', 'runtime', 'time'];
        $file_exists = file_exists($filepath);
        $params["time"] = time();

        try {
            $file = fopen($filepath, 'a');
            if (!$file_exists) {
                fputcsv($file, $headers);
            }
            fputcsv($file, $params);
            fclose($file);
        } catch (Exception $e) {
            log_error("Failed to write to CSV: " . $e->getMessage(). ". Make sure <tt>$filepath</tt> is owned by the www-data group and do <tt>chmod g+w $filepath</tt>");
            exit(1);
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
        $statistics = calculate_statistics($data);
        display_statistics($statistics);

        $anon_users = array_column($data, 0);
        $has_sbatch = array_column($data, 1);
        $exit_codes = array_map('intval', array_column($data, 4));
        $runtimes = array_map('floatval', array_column($data, 5));

        $unique_sbatch = array_unique($has_sbatch);
        $show_sbatch_plot = count($unique_sbatch) > 1 ? '1' : 0;

        echo "<div id='$element_id-exit-codes' style='height: 400px;'></div>";
        echo "<div id='$element_id-runs' style='height: 400px;'></div>";
        echo "<div id='$element_id-runtimes' style='height: 400px;'></div>";
        echo "<div id='$element_id-runtime-vs-exit-code' style='height: 400px;'></div>";
        echo "<div id='$element_id-exit-code-pie' style='height: 400px;'></div>";
        echo "<div id='$element_id-avg-runtime-bar' style='height: 400px;'></div>";
        echo "<div id='$element_id-runtime-box' style='height: 400px;'></div>";
        echo "<div id='$element_id-top-users' style='height: 400px;'></div>";

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

		var exitCodeCounts = {};
			exit_codes_$element_id.forEach(function(code) {
			if (!exitCodeCounts[code]) {
			    exitCodeCounts[code] = 0;
			}
			exitCodeCounts[code]++;
		});

            var exitCodePie = {
		values: Object.values(exitCodeCounts),
		labels: Object.keys(exitCodeCounts),
                type: 'pie',
                name: 'Exit Code Distribution'
            };

            var avgRuntimes = {};
            var runtimeSum = {};
            var runtimeCount = {};
            for (var i = 0; i < exit_codes_$element_id.length; i++) {
                var code = exit_codes_".$element_id."[i];
                var runtime = runtimes_".$element_id."[i];
                if (!(code in runtimeSum)) {
                    runtimeSum[code] = 0;
                    runtimeCount[code] = 0;
                }
                runtimeSum[code] += runtime;
                runtimeCount[code] += 1;
            }

            for (var code in runtimeSum) {
                avgRuntimes[code] = runtimeSum[code] / runtimeCount[code];
            }

            var avgRuntimeBar = {
                x: Object.keys(avgRuntimes),
                y: Object.values(avgRuntimes),
                type: 'bar',
                name: 'Average Runtime per Exit Code'
            };

            var runtimeBox = {
                y: runtimes_$element_id,
                type: 'box',
                name: 'Runtime Distribution'
            };

    // Top N users by number of jobs
    var userJobCounts = {};
    anon_users_$element_id.forEach(function(user) {
        if (!userJobCounts[user]) {
            userJobCounts[user] = 0;
        }
        userJobCounts[user]++;
    });

    var topNUsers = Object.entries(userJobCounts).sort((a, b) => b[1] - a[1]).slice(0, 10);
    var topUserBar = {
        x: topNUsers.map(item => item[0]),
        y: topNUsers.map(item => item[1]),
        type: 'bar',
        name: 'Top Users by Number of Jobs'
    };

            Plotly.newPlot('$element_id-exit-codes', [exitCodePlot], {title: 'Exit Codes'});
            Plotly.newPlot('$element_id-runs', [userPlot], {title: 'Runs per User'});
            Plotly.newPlot('$element_id-runtimes', [runtimePlot], {title: 'Runtimes'});
            Plotly.newPlot('$element_id-runtime-vs-exit-code', [runtimeVsExitCodePlot], {title: 'Runtime vs Exit Code'});
            Plotly.newPlot('$element_id-exit-code-pie', [exitCodePie], {title: 'Exit Code Distribution'});
            Plotly.newPlot('$element_id-avg-runtime-bar', [avgRuntimeBar], {title: 'Average Runtime per Exit Code'});
            Plotly.newPlot('$element_id-runtime-box', [runtimeBox], {title: 'Runtime Distribution'});
    Plotly.newPlot('$element_id-top-users', [topUserBar], {title: 'Top Users by Number of Jobs'});

            if ($show_sbatch_plot) {
                var sbatchPlot = {
                    x: has_sbatch_$element_id,
                    type: 'histogram',
                    name: 'Runs with and without sbatch'
                };
                Plotly.newPlot('$element_id-sbatch', [sbatchPlot], {title: 'Runs with and without sbatch'});
            }
        </script>";
    }

function calculate_statistics($data) {
    $total_jobs = count($data);
    $failed_jobs = count(array_filter($data, function($row) { return intval($row[4]) != 0; }));
    $successful_jobs = $total_jobs - $failed_jobs;
    $failure_rate = $total_jobs > 0 ? ($failed_jobs / $total_jobs) * 100 : 0;

    $runtimes = array_map('floatval', array_column($data, 5));
    $total_runtime = array_sum($runtimes);
    $average_runtime = $total_jobs > 0 ? $total_runtime / $total_jobs : 0;

    sort($runtimes);
    $median_runtime = $total_jobs > 0 ? (count($runtimes) % 2 == 0 ? ($runtimes[count($runtimes) / 2 - 1] + $runtimes[count($runtimes) / 2]) / 2 : $runtimes[floor(count($runtimes) / 2)]) : 0;

    $successful_runtimes = array_filter($data, function($row) { return intval($row[4]) == 0; });
    $successful_runtimes = array_map('floatval', array_column($successful_runtimes, 5));
    $avg_success_runtime = !empty($successful_runtimes) ? array_sum($successful_runtimes) / count($successful_runtimes) : 0;
    sort($successful_runtimes);
    $median_success_runtime = !empty($successful_runtimes) ? (count($successful_runtimes) % 2 == 0 ? ($successful_runtimes[count($successful_runtimes) / 2 - 1] + $successful_runtimes[count($successful_runtimes) / 2]) / 2 : $successful_runtimes[floor(count($successful_runtimes) / 2)]) : 0;

    $failed_runtimes = array_filter($data, function($row) { return intval($row[4]) != 0; });
    $failed_runtimes = array_map('floatval', array_column($failed_runtimes, 5));
    $avg_failed_runtime = !empty($failed_runtimes) ? array_sum($failed_runtimes) / count($failed_runtimes) : 0;
    sort($failed_runtimes);
    $median_failed_runtime = !empty($failed_runtimes) ? (count($failed_runtimes) % 2 == 0 ? ($failed_runtimes[count($failed_runtimes) / 2 - 1] + $failed_runtimes[count($failed_runtimes) / 2]) / 2 : $failed_runtimes[floor(count($failed_runtimes) / 2)]) : 0;

    if (count($runtimes)) {
        return [
            'total_jobs' => $total_jobs,
            'failed_jobs' => $failed_jobs,
            'successful_jobs' => $successful_jobs,
            'failure_rate' => $failure_rate,
            'average_runtime' => $average_runtime,
            'median_runtime' => $median_runtime,
            'max_runtime' => max($runtimes),
            'min_runtime' => min($runtimes),
            'avg_success_runtime' => $avg_success_runtime,
            'median_success_runtime' => $median_success_runtime,
            'avg_failed_runtime' => $avg_failed_runtime,
            'median_failed_runtime' => $median_failed_runtime
        ];
    } else {
        return [
            'total_jobs' => $total_jobs,
            'failed_jobs' => $failed_jobs,
            'successful_jobs' => $successful_jobs,
            'failure_rate' => $failure_rate
        ];
    }
}

function display_statistics($stats) {
    echo "<div class='statistics'>";
    echo "<h3>Statistics</h3>";
    echo "<p>Total jobs: " . $stats['total_jobs'] . "</p>";
    echo "<p>Failed jobs: " . $stats['failed_jobs'] . " (" . number_format($stats['failure_rate'], 2) . "%)</p>";
    echo "<p>Successful jobs: " . $stats['successful_jobs'] . "</p>";
    if (isset($stats["average_runtime"])) {
        echo "<p>Average runtime: " . gmdate("H:i:s", intval($stats['average_runtime'])) . "</p>";
        echo "<p>Median runtime: " . gmdate("H:i:s", intval($stats['median_runtime'])) . "</p>";
        echo "<p>Max runtime: " . gmdate("H:i:s", intval($stats['max_runtime'])) . "</p>";
        echo "<p>Min runtime: " . gmdate("H:i:s", intval($stats['min_runtime'])) . "</p>";
        echo "<p>Average success runtime: " . gmdate("H:i:s", intval($stats['avg_success_runtime'])) . "</p>";
        echo "<p>Median success runtime: " . gmdate("H:i:s", intval($stats['median_success_runtime'])) . "</p>";
        echo "<p>Average failed runtime: " . gmdate("H:i:s", intval($stats['avg_failed_runtime'])) . "</p>";
        echo "<p>Median failed runtime: " . gmdate("H:i:s", intval($stats['median_failed_runtime'])) . "</p>";
    }
    echo "</div>";
}

    // Main code execution

    $data_filepath = 'stats/usage_statistics.csv';

    if ($_SERVER['REQUEST_METHOD'] === 'POST') {
        $params = $_POST;
        if (validate_parameters($params)) {
            append_to_csv($params, $data_filepath);
        }
    }
    if (validate_csv($data_filepath)) {
        $data = array_map('str_getcsv', file($data_filepath));
        array_shift($data); // Remove header row

        list($developer_ids, $test_ids, $regular_data) = filter_data($data);

?>
<br>
	<div id="tabs">
	  <ul>
<?php
		if(count($regular_data)) {
?>
			<li><a href="#regular_data">Regular Users</a></li>
<?php
		}
?>
<?php
		if(count($test_ids)) {
?>
			<li><a href="#test_ids">Tests</a></li>
<?php
		}
?>
<?php
		if(count($developer_ids)) {
?>
			<li><a href="#developer_ids">Developer</a></li>
<?php
		}
?>
		<li><a href="#exit_codes">Exit-Codes</a></li>
	  </ul>
<?php
		if(count($regular_data)) {
?>
	  <div id="regular_data">
<?php
		echo "<h2>Regular Users</h2>";
		display_plots($regular_data, 'Regular Users', 'regular');
?>
	  </div>
<?php
		}

		if(count($test_ids)) {
?>
	  <div id="test_ids">
<?php
		echo "<h2>Test Users</h2>";
		display_plots($test_ids, 'Test Users', 'test');
?>
	  </div>
<?php
		}
		if(count($developer_ids)) {

?>
	  <div id="developer_ids">
<?php
		echo "<h2>Developer Users</h2>";
		display_plots($developer_ids, 'Developer Users', 'developer');
?>
	  </div>
<?php
		}
?>
	  <div id="exit_codes">
<?php
		include("exit_code_table.php");
?>
	</div>
<?php
    } else {
	echo "No valid CSV file found";
    }
?>
