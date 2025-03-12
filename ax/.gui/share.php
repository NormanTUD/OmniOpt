<?php
	include_once("share_functions.php");

        require "_header_base.php";

	$GLOBALS["json_data"] = [];

	$SPECIAL_COL_NAMES = [
		"trial_index",
		"arm_name",
		"trial_status",
		"generation_method",
		"generation_node",
		"hostname",
		"run_time",
		"start_time",
		"exit_code",
		"signal",
		"end_time",
		"program_string",
	];

	$errors = [];
	$warnings = [];

	if(!is_dir($GLOBALS["sharesPath"])) {
		$errors[] = "Folder <tt>$".$GLOBALS["sharesPath"]."</tt> not found.";
	}

	$expected_user_and_group = "www-data";
	$alternative_user_and_group = get_current_user();
	if(checkFolderPermissions($GLOBALS["sharesPath"], $expected_user_and_group, $expected_user_and_group, $alternative_user_and_group, $alternative_user_and_group, 0755)) {
		exit(1);
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
		$result_min_max_file = "$run_dir/result_min_max";

		$result_names = [];
		$result_min_max = [];

		if(is_file($result_names_file)) {
			$result_names = read_file_as_array($result_names_file);
		} else {
			$warnings[] = "$result_names_file not found";
		}

		if(is_file($result_min_max_file)) {
			$result_min_max = read_file_as_array($result_min_max_file);
		} else {
			$warnings[] = "$result_min_max_file not found";
		}

		$GLOBALS["json_data"]["result_names"] = $result_names;
		$GLOBALS["json_data"]["result_min_max"] = $result_min_max;

		$best_results_txt = "$run_dir/best_result.txt";
		$overview_html = "";

		$experiment_overview = "$run_dir/experiment_overview.txt";

		if(is_file("$run_dir/ui_url.txt")) {
			$filePath = "$run_dir/ui_url.txt";
			$firstLine = fgets(fopen($filePath, 'r'));

			if (filter_var($firstLine, FILTER_VALIDATE_URL) && (strpos($firstLine, 'http://') === 0 || strpos($firstLine, 'https://') === 0)) {
				$overview_html .= "<button onclick=\"window.open('".htmlspecialchars($firstLine)."', '_blank')\">GUI page with all the settings of this job</button><br><br>";
			}
		} else {
			$warnings[] = "$run_dir/ui_url.txt not found";
		}

		if(file_exists($experiment_overview) && filesize($experiment_overview)) {
			$experiment_overview_table = asciiTableToHtml(remove_ansi_colors(htmlentities(file_get_contents($experiment_overview))));
			if($experiment_overview_table) {
				$experiment_overview .= $experiment_overview_table;

				$overview_html .= $experiment_overview_table;
			} else {
				$warnings[] = "Could not create \$experiment_overview_table";
			}
		} else {
			if(!file_exists($experiment_overview)) {
				$warnings[] = "$experiment_overview not found";
			} else if(!filesize($experiment_overview)) {
				$warnings[] = "$experiment_overview is empty";
			}
		}

		if(is_file($best_results_txt)) {
			$overview_html .= asciiTableToHtml(remove_ansi_colors(htmlentities(file_get_contents($best_results_txt))));
		} else {
			$warnings[] = "$best_results_txt not found";
		}

		if(is_file("$run_dir/parameters.txt")) {
			$overview_html .= asciiTableToHtml(remove_ansi_colors(htmlentities(file_get_contents("$run_dir/parameters.txt"))));
		} else {
			$warnings[] = "$run_dir/parameters.txt not found";
		}

		$status_data = null;

		if(is_file("$run_dir/results.csv")) {
			$status_data = getStatusForResultsCsv("$run_dir/results.csv");

			$overview_table = '<h2>Number of evaluations:</h2>'."\n";
			$overview_table .= '<table border="1">'."\n";
			$overview_table .= '<tbody>'."\n";
			$overview_table .= '<tr>'."\n";

			foreach ($status_data as $key => $value) {
				$capitalizedKey = ucfirst($key);
				$overview_table .= '<th style="border: 1px solid black">' . $capitalizedKey . '</th>'."\n";
			}
			$overview_table .= '</tr>'."\n";

			$overview_table .= '<tr>'."\n";

			foreach ($status_data as $value) {
				$overview_table .= '<td style="border: 1px solid black">' . $value . '</td>'."\n";
			}
			$overview_table .= '</tr>'."\n";

			$overview_table .= '</tbody>'."\n";
			$overview_table .= '</table>'."\n";

			$overview_html .= "<br>$overview_table";
		} else {
			$warnings[] = "$run_dir/results.csv not found";
		}

		if(count($result_names)) {
			$result_names_table = '<h2>Result names and types:</h2>'."\n";
			$result_names_table .= '<br><table border="1">'."\n";
			$result_names_table .= '<tr><th style="border: 1px solid black">name</th><th style="border: 1px solid black">min/max</th></tr>'."\n";
			for ($i = 0; $i < count($result_names); $i++) {
				$min_or_max = "min";

				if(isset($result_min_max[$i])) {
					$min_or_max = $result_min_max[$i];
				}

				$result_names_table .= '<tr>'."\n";
				$result_names_table .= '<td style="border: 1px solid black">' . htmlspecialchars($result_names[$i]) . '</td>'."\n";
				$result_names_table .= '<td style="border: 1px solid black">' . htmlspecialchars($min_or_max) . '</td>'."\n";
				$result_names_table .= '</tr>'."\n";
			}
			$result_names_table .= '</table><br>'."\n";

			$overview_html .= $result_names_table;
		} else {
			$warnings[] = "No result-names could be found";
		}

		if(file_exists("$run_dir/progressbar") && filesize("$run_dir/progressbar")) {
			$lastLine = trim(array_slice(file("$run_dir/progressbar"), -1)[0]);

			$overview_html .= "<h2>Last progressbar status:</h2>\n";
			$overview_html .= "<tt>".htmlentities($lastLine)."</tt>";
		} else {
			$warnings[] = "$run_dir/progressbar not found or empty";
		}

		if(file_exists("$run_dir/git_version") && filesize("$run_dir/git_version")) {
			$lastLine = htmlentities(file_get_contents("$run_dir/git_version"));

			$overview_html .= "<h2>Git-Version:</h2>\n";
			$overview_html .= "<tt>".htmlentities($lastLine)."</tt>";
		} else {
			$warnings[] = "$run_dir/git_version not found or empty";
		}


		if($overview_html != "") {
			$tabs['Overview'] = [
				'id' => 'tab_overview',
				'content' => $overview_html
			];
		} else {
			$warnings[] = "\$overview_html was empty";
		}

		$pareto_front_txt_file = "$run_dir/pareto_front_table.txt";
		$pareto_front_json_file = "$run_dir/pareto_front_data.json";

		[$tabs, $warnings] = add_pareto_from_from_file($tabs, $warnings, $pareto_front_txt_file, $pareto_front_json_file);
		[$tabs, $warnings] = add_simple_csv_tab_from_file($tabs, $warnings, "$run_dir/results.csv", "Results", "tab_results");
		[$tabs, $warnings] = add_simple_csv_tab_from_file($tabs, $warnings, "$run_dir/job_infos.csv", "Job-Infos", "tab_job_infos");
		[$tabs, $warnings] = add_simple_csv_tab_from_file($tabs, $warnings, "$run_dir/get_next_trials.csv", "Get-Next-Trials", "tab_get_next_trials", ["time", "got", "requested"]);
		[$tabs, $warnings] = add_simple_pre_tab_from_file($tabs, $warnings, "$run_dir/oo_errors.txt", "Errors", "tab_errors", true);
		[$tabs, $warnings] = add_simple_pre_tab_from_file($tabs, $warnings, "$run_dir/outfile", "Main-Log", "tab_main_log", true);
		[$tabs, $warnings] = add_simple_pre_tab_from_file($tabs, $warnings, "$run_dir/trial_index_to_params", "Trial-Index-to-Param", "tab_trial_index_to_param");
		[$tabs, $warnings] = add_simple_pre_tab_from_file($tabs, $warnings, "$run_dir/progressbar", "Progressbar log", "tab_progressbar_log");
		[$tabs, $warnings] = add_simple_table_from_ascii_table_file($tabs, $warnings, "$run_dir/args_overview.txt", "Args Overview", "tab_args_overview");
		[$tabs, $warnings] = add_simple_pre_tab_from_file($tabs, $warnings, "$run_dir/verbose_log.txt", "Verbose log", "tab_verbose_log");
		[$tabs, $warnings] = add_worker_usage_plot_from_file($tabs, $warnings, "$run_dir/worker_usage.csv", "Worker-Usage", "tab_worker_usage");
		[$tabs, $warnings] = add_debug_log_from_file($tabs, $warnings, "$run_dir/log", "Debug-Logs", "tab_debug_logs");
		[$tabs, $warnings] = add_cpu_ram_usage_main_worker_from_file($tabs, $warnings, "$run_dir/cpu_ram_usage.csv", "CPU/RAM-Usage (main)", "tab_main_worker_cpu_ram");
		[$tabs, $warnings] = add_worker_cpu_ram_from_file($tabs, $warnings, "$run_dir/eval_nodes_cpu_ram_logs.txt", "CPU/RAM-Usage (worker)", "tab_worker_cpu_ram_graphs");


		if($status_data && isset($status_data["succeeded"]) && $status_data["succeeded"] > 0) {
			$tabs = add_parallel_plot_tab($tabs);

			#if (count($result_names) >= 1) { // TODO: This works theoretically, but when there are too many plots, it may crash the browser.
			if (count($result_names) == 1) {
				/* Calculating the difference of the sets of columns to find how many parameters have been used, except the special column names and result column names. */
				$non_special_columns = array_diff($GLOBALS["json_data"]["tab_results_headers_json"], $SPECIAL_COL_NAMES);
				$non_special_columns_without_result_columns = array_diff($non_special_columns, $result_names);

				$nr_of_numerical_and_non_numerical_columns = analyze_column_types($GLOBALS["json_data"]["tab_results_csv_json"], $non_special_columns_without_result_columns);

				list($nr_numerical_cols, $nr_string_cols) = count_column_types($nr_of_numerical_and_non_numerical_columns);

				if(count($non_special_columns_without_result_columns) >= 2) {
					if($nr_numerical_cols >= 2) {
						$tabs = add_scatter_2d_plots($tabs, "$run_dir/results.csv", "Scatter-2D", "tab_scatter_2d");
					} else {
						$warnings[] = "Has enough columns for 2d scatter plot, but at not enough if you discard non-numerical columns (numerical: $nr_numerical_cols, non-numerical: $nr_string_cols)";
					}
				}

				if(count($non_special_columns_without_result_columns) >= 3) {
					if($nr_numerical_cols >= 3) {
						$tabs = add_scatter_3d_plots($tabs, "$run_dir/results.csv", "Scatter-3D", "tab_scatter_3d");
					} else {
						$warnings[] = "Has enough columns for 3d scatter plot, but at not enough if you discard non-numerical columns (numerical: $nr_numerical_cols, non-numerical: $nr_string_cols)";
					}
				}
			}

			if($status_data["succeeded"] > 1) {
				$tabs = add_box_plot_tab($tabs);
				$non_special_columns = array_diff($GLOBALS["json_data"]["tab_results_headers_json"], $SPECIAL_COL_NAMES);
				$non_special_columns_without_result_columns = array_diff($non_special_columns, $result_names);
				if(count($non_special_columns_without_result_columns) > 2) {
					$tabs = add_heatmap_plot_tab($tabs);
				}
				$tabs = add_violin_plot($tabs);
				$tabs = add_histogram_plot($tabs);
				if (count($result_names) > 1) {
					$tabs = add_plot_result_pairs($tabs);
				}

				if(count($result_names)) {
					if (isset($GLOBALS["json_data"]["tab_job_infos_headers_json"])) {
						$tabs = add_result_evolution_tab($tabs);
					} else {
						$warnings[] = "tab_job_infos_headers_json not found in global json_data";
					}
				} else {
					$warnings[] = "Not adding evolution tab because no result names could be found";
				}
			} else {
				$warnings[] = "No succeeded jobs found";
			}
		} else {
			$warnings[] = "No successful jobs were found";
		}

		if($status_data && ((isset($status_data["succeeded"]) && $status_data["succeeded"] > 0) || (isset($status_data["failed"]) && $status_data["failed"] > 0))) {
			if(isset($GLOBALS["json_data"]["tab_job_infos_headers_json"]) && isset($GLOBALS["json_data"]["tab_job_infos_csv_json"])) {
				$headers = $GLOBALS["json_data"]["tab_job_infos_headers_json"];

				$exitCodeIndex = array_search("exit_code", $headers, true);

				if ($exitCodeIndex !== false) {
					$rows = $GLOBALS["json_data"]["tab_job_infos_csv_json"];

					$exitCodes = array_column($rows, $exitCodeIndex);

					$uniqueExitCodes = array_unique($exitCodes);

					$uniqueExitCodes = array_filter($uniqueExitCodes, function ($value) {
						return $value !== "None";
					});

					$uniqueExitCodes = array_values($uniqueExitCodes);

					if (count($uniqueExitCodes) > 1) {
						$tabs = add_exit_codes_pie_plot($tabs);
					} else {
						$warnings[] = "No exit-codes found or all are none";
					}
				} else {
					$warnings[] = "exit_code not found in the headers.";
				}
			} else {
				$warnings[] = "Global variables tab_job_infos_headers_json or tab_job_infos_csv_json is not set.";
			}
		} else {
			$warnings[] = "No successful or failed jobs found, cannot show plot for exit-codes";
		}


		[$tabs, $warnings] = get_outfiles_tab_from_run_dir($run_dir, $tabs, $warnings, $result_names);

		$tabs = get_share_debug_tab($tabs, $warnings);
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
			document.getElementById('main_window').style.display = 'contents';
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
				print "\tvar $json_name = ".json_encode($json_data, JSON_PRETTY_PRINT).";\n";
			}
		}
?>

		document.addEventListener("DOMContentLoaded", initialize_tabs);

		if($("#spinner").length) {
			document.getElementById('spinner').style.display = 'block';
		}
	</script>
	<div class="page" style='font-family: sans-serif'>
		<div class="invert_in_dark_mode" style="height: fit-content;">
			<div id="share_path" class="invert_in_dark_mode title-bar-text">
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
					#if (isset($_GET["sort"]) && preg_match("/^[a-zA-Z0-9_]+$/", $_GET["sort"])) {
					#	$base_url = "?sort=" . $_GET["sort"];
					#}

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
						echo $path_with_links;
					}
				}
?>
			</div>
		</div>
		<div id="spinner" class="spinner"></div>

		<div id="main_window" style="display: none; border: 1px solid black;" class="container py-4 has-space">
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
								$onclick = isset($tab_data["onclick"]) ? " onclick='" . $tab_data["onclick"] . "'" : "" ;
								$first_tab = $first_tab ? 'aria-selected="true"' : '';
								$tab_id = $tab_data['id'];

								echo "<button role='tab' $onclick aria-controls='$tab_id' $first_tab>$tab_name</button>";

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
