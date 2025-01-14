<?php
	require_once "power_on_self_test.php";
	require_once "_functions.php";

	function get_dir_path () {
		$dir_path = realpath(dirname(__FILE__));
		if (preg_match("/\/tutorials\/?$/", dirname($_SERVER["PHP_SELF"]))) {
			$dir_path = $dir_path . "/../";
		}

		return $dir_path;
	}

	function js ($names) {
		if(is_array($names)) {
			foreach ($names as $name) {
				js($name);
			}
		} else {
			$_p = $names;
			if (!file_exists($_p)) {
				dier("$_p not found");
			}
?>
			<script src="<?php print $_p; ?>"></script>
<?php
		}
	}
?>
<!DOCTYPE html>
<html lang="en">
	<head>
		<meta charset="UTF-8">
		<title>OmniOpt2</title>
		<link href="prism.css" rel="stylesheet">
		<link rel="icon" type="image/x-icon" href="favicon.ico">
		<?php js("jquery-3.7.1.js"); ?>
		<?php js("jquery-ui.min.js"); ?>
		<?php js("prism.js"); ?>
		<?php js("tooltipster.bundle.min.js"); ?>
		<?php js("darkmode.js"); ?>
		<?php js("ansi_up.js"); ?>
		<?php js("jquery.dataTables.min.js"); ?>
		<?php js("crypto-core.js"); ?>
		<?php js("md5.js"); ?>
		<?php js("main.js"); ?>
		<?php js("search.js"); ?>
		<?php js("initialization.js"); ?>

		<script>
			apply_theme_based_on_system_preferences();
		</script>

		<link href="style.css" rel="stylesheet">
<?php
		if (!preg_match("/gui\.php$/", $_SERVER["SCRIPT_FILENAME"])) {
?>
			    <link href="_tutorial.css" rel="stylesheet">
<?php
		}
?>
		<link href="jquery-ui.css" rel="stylesheet">
		<script>
			document.onkeypress = function (e) {
				e = e || window.event;

				if (document.activeElement == $("body")[0]) {
					var keycode = e.keyCode;
					if (keycode >= 97 && keycode <= 122 || keycode == 45) {
						e.preventDefault();
						$("#search").val("");
						$("#search").val(String.fromCharCode(e.keyCode));
						$("#search").focus().trigger("change");
					}
				} else if (keycode === 8) { // Backspace key
					delete_search();
				}
			};

			function openURLInNewTab() {
				var url = window.location.protocol + "//" + window.location.host + window.location.pathname +
					'?partition=alpha&experiment_name=small_test_experiment&reservation=&account=&mem_gb=1&time=60&worker_timeout=60' +
					'&max_eval=500&num_parallel_jobs=20&gpus=1&num_random_steps=20&follow=1&send_anonymized_usage_stats=1&show_sixel_graphics=1&' +
					'run_program=echo "RESULT%3A %25(x)%25(y)"&cpus_per_task=1&tasks_per_node=1&seed=&verbose=0&debug=0&maximize=0&gridsearch=0' +
					'&model=BOTORCH_MODULAR&run_mode=local&constraints=&parameter_0_name=x&parameter_0_type=range&parameter_0_min=123' +
					'&parameter_0_max=100000000&parameter_0_number_type=int&parameter_1_name=y&parameter_1_type=range&parameter_1_min=5431' +
					'&parameter_1_max=1234&parameter_1_number_type=float&partition=alpha&num_parameters=2';
				window.open(url, '_blank');
			}

			function handleKeyDown(event) {
				// Check if 'Control' key and '*' key are pressed
				var isControlPressed = event.ctrlKey;
				var isAsteriskPressed = event.key === '*';

				if (isControlPressed && isAsteriskPressed) {
					openURLInNewTab();
				}
			}

			document.addEventListener('keydown', handleKeyDown);
		</script>
	</head>
	<body>
		<div id="scads_bar">
			<table border=0 class="header_table" style='display: inline !important;'>
				<tr class="header_table">
					<td class='header_table'>
						<a style="text-decoration: none; margin-right: 20px;" target="_blank" href="https://scads.ai/">
							<img height=90 class="img_auto_width invert_in_dark_mode" src="scads_logo.svg" alt="ScaDS.ai-Logo">
						</a>
					</td>

					<td class='header_table'>
						<a style="text-decoration: none;" href="index.php">
							<img style="margin-right: 10px" class="img_auto_width invert_in_dark_mode" height=73 src="logo.png" alt="OmniOpt2-Logo">
						</a>
					</td>

					<td class='header_table'>
						<table border=0 class="header_table" style='display: inline !important;'>
							<tr class="header_table">
								<td class='header_table'>
									<a target="_blank" href="https://github.com/NormanTUD/OmniOpt/actions">
										<img class="img_auto_width" style="width: 100% !important;" src="https://github.com/NormanTUD/OmniOpt/actions/workflows/main.yml/badge.svg?event=push" alt="Current CI-Pipeline Badge">
									</a>
								</td>
							</tr>
							<tr class="header_table" style='background-color: revert !important;'>
								<td class="header_table">
									<img class="img_auto_width" style="width: 100% !important;" src="https://img.shields.io/github/last-commit/NormanTUD/OmniOpt" alt="Time since last commit">
								</td>
							</tr>

							<tr class="header_table" style='background-color: revert !important;'>
								<td class="header_table">
									<a target="_blank" href="https://coveralls.io/github/NormanTUD/OmniOpt">
										<img class="img_auto_width" width=116 height=24 src="https://coveralls.io/repos/github/NormanTUD/OmniOpt/badge.svg?branch=main" alt="Test coverage">
									</a>
								</td>
							</tr>
						</table>
					</td>
	<?php
					require "searchable_php_files.php";

					$current_file = basename($_SERVER["PHP_SELF"]);

					foreach ($GLOBALS["files"] as $fn => $n) {
						if (is_array($n)) {
							$n = $n["name"];
						}

						$tab_is_active = preg_match("/^$fn.php/", $current_file);
						$tab_class = $tab_is_active ? 'active_tab' : 'inactive_tab';
						$_link = "$fn.php";

						if (!file_exists($_link)) {
							dier("Coult not find $_link");
						}
						echo "<td class='header_table invert_in_dark_mode' style='border: 0'>";
						echo "\t<a href='$_link' class='tab $tab_class'>$n</a>\n";
						echo "</td>";
					}

					$current_tag = get_current_tag();

					if ($current_tag) {
						echo "<td class='header_table'>";
						echo " $current_tag";
						echo "</td>";
					}
?>
					<td class="header_table">
						<span style="display: inline-grid;">
							<select id="themeSelect" name="theme">
								<option value="light">Light Mode</option>
								<option value="dark">Dark Mode</option>
							</select>
						</span>
					</td>
				</tr>
			</table>
		<br>
		<span style="display: inline-flex;">
			<img src="images/search.svg" height=32 alt="Search">
			<input class="invert_in_dark_mode" onkeyup="start_search()" onfocus="start_search()" onblur="start_search()" onchange='start_search()' style="width: 500px;" type="text" placeholder="Search help topics and shares (Regex without delimiter by default)..." id="search">
			<button id="del_search_button" class="invert_in_dark_mode" style="display: none;" onclick="delete_search()">&#10060;</button>
		</span>
	</div>

	<script>
		apply_theme_based_on_system_preferences();
	</script>

	<div id="searchResults"></div>

	<div id="mainContent">
