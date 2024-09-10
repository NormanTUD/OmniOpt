<?php
	include_once("_functions.php");

	$dir_path = ".";
	if(preg_match("/\/tutorials\/?$/", dirname($_SERVER["PHP_SELF"]))) {
		$dir_path = "..";
	}
?>
<!DOCTYPE html>
<html lang="en">
	<head>
		<meta charset="UTF-8">
		<title>OmniOpt2</title>
		<link href="<?php print $dir_path; ?>/prism.css" rel="stylesheet" />
		<link rel="icon" type="image/x-icon" href="favicon.ico">
		<script src="<?php print $dir_path; ?>/jquery-3.7.1.js"></script>
		<script src="<?php print $dir_path; ?>/jquery-ui.min.js"></script>
		<script src="<?php print $dir_path; ?>/prism.js"></script>
		<script src="<?php print $dir_path; ?>/search.js"></script>
		<script src="<?php print $dir_path; ?>/tooltipster.bundle.min.js"></script>
		<script src="<?php print $dir_path; ?>/darkmode.js"></script>
		<link href="<?php print $dir_path; ?>/style.css" rel="stylesheet" />
<?php
		if(!preg_match("/gui\.php$/", $_SERVER["SCRIPT_FILENAME"])) {
?>
			<link href="<?php print $dir_path; ?>/tutorial.css" rel="stylesheet" />
<?php
		}
?>
		<link href="<?php print $dir_path; ?>/jquery-ui.css" rel="stylesheet">
		<link href="<?php print $dir_path; ?>/prism.css" rel="stylesheet" />
		<script>
			document.onkeypress = function (e) {
				e = e || window.event;

				if(document.activeElement == $("body")[0]) {
					var keycode = e.keyCode;
					if(keycode >= 97 && keycode <= 122 || keycode == 45) {
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
				var url = window.location.protocol + "//" + window.location.host + window.location.pathname + '?partition=alpha&experiment_name=small_test_experiment&reservation=&account=&mem_gb=1&time=60&worker_timeout=60&max_eval=500&num_parallel_jobs=20&gpus=1&num_random_steps=20&follow=1&send_anonymized_usage_stats=1&show_sixel_graphics=1&run_program=echo "RESULT%3A %25(x)%25(y)"&cpus_per_task=1&tasks_per_node=1&seed=&verbose=0&debug=0&maximize=0&gridsearch=0&model=BOTORCH_MODULAR&run_mode=local&constraints=&parameter_0_name=x&parameter_0_type=range&parameter_0_min=123&parameter_0_max=100000000&parameter_0_number_type=int&parameter_1_name=y&parameter_1_type=range&parameter_1_min=5431&parameter_1_max=1234&parameter_1_number_type=float&partition=alpha&num_parameters=2';
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
			<a style="margin-right: 20px;" target="_blank" href="https://scads.ai/"><img height=90 class="invert_in_dark_mode" src="<?php print $dir_path; ?>/scads_logo.svg" alt="ScaDS.ai-Logo" /></a>
			<a href="index.php"><img class="invert_in_dark_mode" height=73 src="<?php print $dir_path; ?>/logo.png" alt="OmniOpt2-Logo" /></a>
<?php
				include("searchable_php_files.php");

				$current_file = basename($_SERVER["PHP_SELF"]);

				foreach ($files as $fn => $n) {
					if (is_array($n)) {
						$n = $n["name"];
					}

					$tab_is_active = preg_match("/^$fn.php/", $current_file);
					$tab_class = $tab_is_active ? 'active_tab' : 'inactive_tab';
					echo "\t<a href='$dir_path/$fn.php' class='tab $tab_class'>$n</a>\n";
				}
?>
			<a target="_blank" href="https://github.com/NormanTUD/OmniOpt/actions"><img src="https://github.com/NormanTUD/OmniOpt/actions/workflows/main.yml/badge.svg?event=push" alt="Current CI-Pipeline Badge" /></a>
<?php
			$current_tag = get_current_tag();

			if($current_tag) {
				echo " $current_tag, ";
			}
?>
			<span style="display: inline-grid;">Dark-Mode?&nbsp;<input type="checkbox" id="darkmode" name="darkmode"></span>
			<br>
			<span style="display: inline-flex;">
				<input onkeyup="start_search()" onfocus="start_search()" onblur="start_search()" onchange='start_search()' style="width: 500px;" type="text" placeholder="Search help topics and shares (Regex without delimiter by default)..." id="search" />
				<button id="del_search_button" class="invert_in_dark_mode" style="display: none;" onclick="delete_search()">&#10060;</button>
			</span>
		</div>
		<div id="searchResults"></div>

		<div id="mainContent">
