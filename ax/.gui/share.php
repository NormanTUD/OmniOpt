<?php
	include_once("share2_functions.php");

        require "_header_base.php";

	$GLOBALS["json_data"] = [];

	$SPECIAL_COL_NAMES = [
		"trial_index",
		"arm_name",
		"trial_status",
		"generation_method",
		"generation_node"
	];

	$errors = [];
	if(!is_dir($GLOBALS["sharesPath"])) {
		$errors[] = "Folder <tt>$".$GLOBALS["sharesPath"]."</tt> not found.";
	}

	$user_id = get_get("user_id");
	$experiment_name = get_get("experiment_name");
	$run_nr = get_get("run_nr", -1);

	if($user_id) {
		if(!is_valid_user_or_experiment_name($user_id)) {
			$errors[] = "<tt>".htmlentities($user_id)."</tt> is not a valid username";
		}
	}

	if($experiment_name) {
		if(!is_valid_user_or_experiment_name($experiment_name)) {
			$errors[] = "<tt>".htmlentities($experiment_name)."</tt> is not a valid experiment name";
		}
	}

	if($run_nr != -1) {
		if(!string_is_numeric($run_nr)) {
			$errors[] = "<tt>".htmlentities($run_nr)."</tt> is not a valid run nr";
		}
	}

	if($user_id) {
		$run_dir = $GLOBALS["sharesPath"]."/$user_id";

		if(!is_dir($run_dir)) {
			$errors[] = "<tt>".htmlentities($run_dir)."</tt> cannot be found!";
		}
	}

	if($run_nr == -1) {
		$run_nr = null;
	} else {
		if(!count($errors)) {
			$run_dir = $GLOBALS["sharesPath"]."/$user_id/$experiment_name/$run_nr";

			if(!is_dir($run_dir)) {
				$errors[] = "<tt>".htmlentities($run_dir)."</tt> cannot be found!";
			}
		}
	}

	$run_dir = $GLOBALS["sharesPath"]."/$user_id/$experiment_name/$run_nr";

	if(!count($errors) && $user_id && $experiment_name && $run_nr != -1 && $run_nr !== null && is_dir($run_dir)) {
		$result_names_file = "$run_dir/result_names.txt";
		$result_min_max_file = "$run_dir/result_min_max.txt";

		$result_names = ["RESULT"];
		$result_min_max = ["min"];

		if(is_file($result_names_file)) {
			$result_names = read_file_as_array($result_names_file);
		}
		if(is_file($result_min_max_file)) {
			$result_min_max = read_file_as_array($result_min_max_file);
		}

		$GLOBALS["json_data"]["result_names"] = $result_names;
		$GLOBALS["json_data"]["result_min_max"] = $result_min_max;

		$best_results_txt = "$run_dir/best_result.txt";
		$overview_html = "";

		if(is_file("$run_dir/ui_url.txt")) {
			$filePath = "$run_dir/ui_url.txt";
			$firstLine = fgets(fopen($filePath, 'r'));

			if (filter_var($firstLine, FILTER_VALIDATE_URL) && (strpos($firstLine, 'http://') === 0 || strpos($firstLine, 'https://') === 0)) {
				$overview_html .= "<button onclick=\"window.open('".htmlspecialchars($firstLine)."', '_blank')\">GUI page with all the settings of this job</button><br><br>";
			}
		}

		if(is_file($best_results_txt)) {
			$overview_html .= "<pre>\n".htmlentities(remove_ansi_colors(file_get_contents($best_results_txt)))."</pre>";
		}

		if(is_file("$run_dir/parameters.txt")) {
			$overview_html .= "<pre>\n".htmlentities(remove_ansi_colors(file_get_contents("$run_dir/parameters.txt")))."</pre>";
		}

		$status_data = null;

		if(is_file("$run_dir/results.csv")) {
			$status_data = getStatusForResultsCsv("$run_dir/results.csv");

			$overview_table = '<table border="1">';
			$overview_table .= '<tbody>';
			$overview_table .= '<tr>';

			foreach ($status_data as $key => $value) {
				$capitalizedKey = ucfirst($key);
				$overview_table .= '<th style="border: 1px solid black">' . $capitalizedKey . '</th>';
			}
			$overview_table .= '</tr>';

			$overview_table .= '<tr>';

			foreach ($status_data as $value) {
				$overview_table .= '<td style="border: 1px solid black">' . $value . '</td>';
			}
			$overview_table .= '</tr>';

			$overview_table .= '</tbody>';
			$overview_table .= '</table>';

			$overview_html .= "<br>$overview_table";
		}

		if(count($result_names)) {
			$result_names_table = '<br><table border="1">';
			$result_names_table .= '<tr><th style="border: 1px solid black">name</th><th style="border: 1px solid black">min/max</th></tr>';
			for ($i = 0; $i < count($result_names); $i++) {
				$min_or_max = "min";

				if(isset($result_min_max[$i])) {
					$min_or_max = $result_min_max[$i];
				}

				$result_names_table .= '<tr>';
				$result_names_table .= '<td style="border: 1px solid black">' . htmlspecialchars($result_names[$i]) . '</td>';
				$result_names_table .= '<td style="border: 1px solid black">' . htmlspecialchars($min_or_max) . '</td>';
				$result_names_table .= '</tr>';
			}
			$result_names_table .= '</table><br>';

			$overview_html .= $result_names_table;
		}

		if(file_exists("$run_dir/progressbar") && filesize("$run_dir/progressbar")) {
			$lastLine = trim(array_slice(file("$run_dir/progressbar"), -1)[0]);

			$overview_html .= "Last progressbar status: <pre>".htmlentities($lastLine)."</pre>";
		}

		if($overview_html != "") {
			$tabs['Overview'] = [
				'id' => 'tab_overview',
				'content' => $overview_html
			];
		}

		if(file_exists("$run_dir/pareto_front_data.json") && file_exists("$run_dir/pareto_front_table.txt")) {
			$pareto_front_html = "";

			$pareto_front_txt_file = "$run_dir/pareto_front_table.txt";

			$pareto_front_txt = remove_ansi_colors(file_get_contents($pareto_front_txt_file));

			if($pareto_front_txt) {
				$pareto_front_html .= "<pre>$pareto_front_txt</pre><br>\n";
			}

			$GLOBALS["json_data"]["pareto_front_data"] = json_decode(file_get_contents("$run_dir/pareto_front_data.json"));

			if($pareto_front_html) {
				$pareto_front_html = "<div id='pareto_front_graphs_container'></div>\n$pareto_front_html";

				$tabs['Pareto-Fronts'] = [
					'id' => 'tab_pareto_fronts',
					'content' => $pareto_front_html,
					'onclick' => "load_pareto_graph();"
				];
			}
		}

		$tabs = add_simple_csv_tab_from_file($tabs, "$run_dir/results.csv", "Results", "tab_results");
		$tabs = add_simple_csv_tab_from_file($tabs, "$run_dir/job_infos.csv", "Job-Infos", "tab_job_infos");
		$tabs = add_simple_csv_tab_from_file($tabs, "$run_dir/get_next_trials.csv", "Get-Next-Trials", "tab_get_next_trials", ["time", "got", "requested"]);
		$tabs = add_simple_pre_tab_from_file($tabs, "$run_dir/oo_errors.txt", "Errors", "tab_errors", true);
		$tabs = add_simple_pre_tab_from_file($tabs, "$run_dir/outfile", "Main-Log", "tab_main_log", true);
		$tabs = add_simple_pre_tab_from_file($tabs, "$run_dir/trial_index_to_params", "Trial-Index-to-Param", "tab_trial_index_to_param");
		$tabs = add_simple_pre_tab_from_file($tabs, "$run_dir/experiment_overview.txt", "Experiment Overview", "tab_experiment_overview");
		$tabs = add_simple_pre_tab_from_file($tabs, "$run_dir/progressbar", "Progressbar log", "tab_progressbar_log");
		$tabs = add_simple_pre_tab_from_file($tabs, "$run_dir/args_overview.txt", "Args Overview", "tab_args_overview");
		$tabs = add_simple_pre_tab_from_file($tabs, "$run_dir/verbose_log.txt", "Verbose log", "tab_verbose_log");
		$tabs = add_worker_usage_plot_from_file($tabs, "$run_dir/worker_usage.csv", "Worker-Usage", "tab_worker_usage");
		$tabs = add_cpu_ram_usage_main_worker_from_file($tabs, "$run_dir/cpu_ram_usage.csv", "CPU/RAM-Usage (main worker)", "tab_main_worker_cpu_ram");
		$tabs = add_debug_log_from_file($tabs, "$run_dir/log", "Debug-Logs", "tab_debug_logs");
		$tabs = add_worker_cpu_ram_from_file($tabs, "$run_dir/eval_nodes_cpu_ram_logs.txt", "Worker-CPU-RAM-Graphs", "tab_worker_cpu_ram_graphs");

		$out_files = get_log_files($run_dir);

		if($status_data && isset($status_data["succeeded"]) && $status_data["succeeded"] > 0) {
			$tabs = add_parallel_plot_tab($tabs);

			if (count($result_names) == 1) {
				/* Calculating the difference of the sets of columns to find how many parameters have been used, except the special column names and result column names. */
				$non_special_columns = array_diff($GLOBALS["json_data"]["tab_results_headers_json"], $SPECIAL_COL_NAMES);
				$non_special_columns_without_result_columns = array_diff($non_special_columns, $result_names);

				if(count($non_special_columns_without_result_columns) >= 2) {
					$tabs = add_scatter_2d_plots($tabs, "$run_dir/results.csv", "Scatter-2D", "tab_scatter_2d");
				}

				if(count($non_special_columns_without_result_columns) >= 3) {
					$tabs = add_scatter_3d_plots($tabs, "$run_dir/results.csv", "Scatter-3D", "tab_scatter_3d");
				}
			}

			$tabs = add_box_plot_tab($tabs);
			$tabs = add_heatmap_plot_tab($tabs);
			$tabs = add_histogram_plot($tabs);
			$tabs = add_violin_plot($tabs);
		}

		if($status_data && isset($status_data["total"]) && $status_data["total"] > 0) {
			$tabs = add_exit_codes_pie_plot($tabs);
		}

		if(count($out_files)) {
			$tabs['Single Logs'] = [
				'id' => 'tab_logs',
				'content' => generate_log_tabs($run_dir, $out_files, $result_names)
			];
		}
	}

	if(!count($tabs) && $run_dir != "" && count($errors)) {
		$errors[] = "Cannot plot any data in <tt>".htmlentities($run_dir)."</tt>";
	}
?>
	<script>
		function close_main_window() {
			const url = new URL(window.location.href);

			if (url.searchParams.has('run_nr')) {
				url.searchParams.delete('run_nr');
			} else if (url.searchParams.has('experiment_name')) {
				url.searchParams.delete('experiment_name');
			} else if (url.searchParams.has('user_id')) {
				url.searchParams.delete('user_id');
			}

			window.location.assign(url.toString());
		}

		function show_main_window() {
			document.getElementById('spinner').style.display = 'none';
			document.getElementById('main_window').style.display = 'block';
		}

		function initialize_tabs () {
			function setupTabs(container) {
				const tabs = container.querySelectorAll('[role="tab"]');
				const tabPanels = container.querySelectorAll('[role="tabpanel"]');

				if (tabs.length === 0 || tabPanels.length === 0) {
					return;
				}

				tabs.forEach(tab => tab.setAttribute("aria-selected", "false"));
				tabPanels.forEach(panel => panel.hidden = true);

				const firstTab = tabs[0];
				const firstPanel = tabPanels[0];

				if (firstTab && firstPanel) {
					firstTab.setAttribute("aria-selected", "true");
					firstPanel.hidden = false;
				}

				tabs.forEach(tab => {
					tab.addEventListener("click", function () {
						const parentContainer = tab.closest(".tabs");

						const parentTabs = parentContainer.querySelectorAll('[role="tab"]');
						const parentPanels = parentContainer.querySelectorAll('[role="tabpanel"]');

						parentTabs.forEach(t => t.setAttribute("aria-selected", "false"));
						parentPanels.forEach(panel => panel.hidden = true);

						this.setAttribute("aria-selected", "true");
						const targetPanel = document.getElementById(this.getAttribute("aria-controls"));
						if (targetPanel) {
							targetPanel.hidden = false;

							const nestedTabs = targetPanel.querySelector(".tabs");
							if (nestedTabs) {
								setupTabs(nestedTabs);
							}
						}
					});
				});
			}

			document.querySelectorAll(".tabs").forEach(setupTabs);
		}

		var special_col_names = <?php print json_encode($SPECIAL_COL_NAMES); ?>;
<?php
		if(count($GLOBALS["json_data"])) {
			foreach ($GLOBALS["json_data"] as $json_name => $json_data) {
				print "\tvar $json_name = ".json_encode($json_data).";\n";
			}
		}
?>

		document.addEventListener("DOMContentLoaded", initialize_tabs);

		if($("#spinner").length) {
			document.getElementById('spinner').style.display = 'block';
		}
	</script>
	<div class="page window" style='font-family: sans-serif'>
		<div class="title-bar invert_in_dark_mode" style="height: fit-content;">
			<div class="title-bar-text">OmniOpt2-Share
<?php
			if(get_get("user_id") || get_get("experiment_name") || get_get("run_nr")) {
				$user_id_link = get_get("user_id");
				$experiment_name_link = get_get("experiment_name");
				$run_nr_link = get_get("run_nr");

				if (!is_valid_user_id($user_id_link)) {
					$user_id_link = '';
				}

				if (!is_valid_experiment_name($experiment_name_link)) {
					$experiment_name_link = '';
				}

				if (!is_valid_run_nr($run_nr_link)) {
					$run_nr_link = '';
				}

				$base_url = "?";

				$links = [];

				if (!empty($user_id_link)) {
					$links[] = '<button onclick="window.location.href=\'' . $base_url . 'user_id=' . urlencode($user_id_link) . '\'">' . $user_id_link . '</button>';
				}

				if (!empty($experiment_name_link)) {
					$links[] = '<button onclick="window.location.href=\'' . $base_url . 'user_id=' . urlencode($user_id_link) . '&experiment_name=' . urlencode($experiment_name_link) . '\'">' . $experiment_name_link . '</button>';
				}

				if ($run_nr_link != "") {
					$links[] = '<button onclick="window.location.href=\'' . $base_url . 'user_id=' . urlencode($user_id_link) . '&experiment_name=' . urlencode($experiment_name_link) . '&run_nr=' . urlencode($run_nr_link) . '\'">' . $run_nr_link . '</button>';
				}

				if(count($links)) {
					$home = $_SERVER["PHP_SELF"];
					$home = preg_replace("/.*\//", "", $home);
					$home = preg_replace("/\.php$/", "", $home);

					array_unshift($links, "<button onclick=\"window.location.href='$home'\">Home</button>");
				}

				$path_with_links = implode(" / ", $links);

				if(count($links)) {
					echo " ($path_with_links)";
				}
			}
?>
			</div>
		</div>
		<div id="spinner" class="spinner"></div>

		<div id="main_window" style="display: none" class="container py-4 has-space">
<?php
			if(count($errors)) {
				if (count($errors) > 1) {
					print "<h2>Errors:</h2>\n";
					print "<ul>\n";
					foreach ($errors as $error) {
						print "<li>$error</li>";
					}
					print "</ul>\n";
				} else {
					print "<h2>Error:</h2>\n";
					print $errors[0];
				}

			} else {
				if($user_id && $experiment_name && !is_null($run_nr)) {
?>
					<section class="tabs" style="width: 100%">
						<menu role="tablist" aria-label="OmniOpt2-Run">
<?php
							$first_tab = true;
							foreach ($tabs as $tab_name => $tab_data) {
								echo '<button role="tab" '.(isset($tab_data["onclick"]) ? " onclick='" . $tab_data["onclick"] . "'" : "" ).' aria-controls="' . $tab_data['id'] . '" ' . ($first_tab ? 'aria-selected="true"' : '') . '>' . $tab_name . '</button>';
								$first_tab = false;
							}
?>
						</menu>

<?php
						foreach ($tabs as $tab_name => $tab_data) {
							echo '<article role="tabpanel" id="' . $tab_data['id'] . '" ' . ($tab_name === 'General Info' ? '' : 'hidden') . ">\n";
							echo $tab_data['content'];
							echo "</article>\n";
						}
?>
					</section>
<?php
				} else {
					if(!$user_id && !$experiment_name && !$run_nr) {
						generateFolderButtons($GLOBALS["sharesPath"], "user_id");
					} else if($user_id && !$experiment_name && !$run_nr) {
						generateFolderButtons($GLOBALS["sharesPath"]."/$user_id", "experiment_name");
					} else if($user_id && $experiment_name && !$run_nr) {
						generateFolderButtons($GLOBALS["sharesPath"]."/$user_id/$experiment_name", "run_nr");
					} else {
						print "DONT KNOW!!! >>$run_nr<<";
					}
				}
			}
?>
		</div>
	</div>
	<script>
		show_main_window();
	</script>
<?php
	include("footer.php");
?>
