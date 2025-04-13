<?php
	if (function_exists('apache_get_modules')) {
		$modules = apache_get_modules();
		if (!in_array('mod_rewrite', $modules)) {
			echo "!!! mod_rewrite is NOT activated !!!<br>\n";
			echo "Run <tt>sudo a2enmod rewrite</tt> to activate!<br>\n";
		}
	}

	require_once "power_on_self_test.php";
	require_once "_functions.php";
	require "searchable_php_files.php";

	function get_main_script_dir() {
		$script_name = $_SERVER["SCRIPT_NAME"];
		$main_script_dir = preg_replace("/(\/.*)\/.*/", "\\1/", $script_name);
		$main_script_dir = preg_replace("/\/+/", "/", $main_script_dir);
		return $main_script_dir;
	}

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
			<script src="<?php print get_main_script_dir()."/$_p"; ?>"></script>
<?php
		}
	}
?>
<!DOCTYPE html>
<html lang="en">
	<head>
		<meta charset="UTF-8">
		<meta name="viewport" content="width=device-width, initial-scale=1.0">
		<title>OmniOpt2</title>
		<link href="<?php print get_main_script_dir(); ?>/prism.css" rel="stylesheet">
		<link rel="icon" type="image/x-icon" href="<?php print get_main_script_dir(); ?>/favicon.ico">
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

<?php
		if (isset($_SERVER["SCRIPT_FILENAME"]) && strpos($_SERVER['SCRIPT_FILENAME'], 'share.php') !== false) {
?>
			<link rel="stylesheet" href="css/share.css">
			<meta name="robots" content="noindex, nofollow">
			<?php js("plotly-latest.min.js"); ?>
			<?php js("js/gridjs.umd.js"); ?>

			<?php js("js/share_functions.js"); ?>

			<link href="<?php print get_main_script_dir(); ?>/css/mermaid.min.css" rel="stylesheet" />
<?php
		}
?>

		<link rel="stylesheet" href="css/xp.css">
		<script>
			apply_theme_based_on_system_preferences();
		</script>

		<link href="<?php print get_main_script_dir(); ?>/style.css" rel="stylesheet">
		<link href="<?php print get_main_script_dir(); ?>/jquery-ui.css" rel="stylesheet">
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
					'&max_eval=5&num_parallel_jobs=20&gpus=1&num_random_steps=2&follow=1&send_anonymized_usage_stats=1&show_sixel_graphics=1&' +
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
<?php
		if(preg_match("/tutorials.php/", $_SERVER["PHP_SELF"])) {
?>
			<script id="MathJax-script" async src="tex-mml-chtml.js"></script>

<?php
		}
?>
	</head>
	<body>
		<div id="scads_bar" class="header-container">
			<div class="header-logo-group">
				<a href="index">
					<img class="logo-img" src="<?php print get_main_script_dir(); ?>/logo.png" alt="OmniOpt2-Logo">
				</a>
				<a href="https://scads.ai/" target="_blank">
					<img class="logo-img" src="<?php print get_main_script_dir(); ?>/scads_logo.svg" alt="ScaDS.ai-Logo">
				</a>
			</div>

			<div class="header-badges">
				<a href="https://github.com/NormanTUD/OmniOpt/actions" target="_blank">
					<img class="badge-img" src="https://github.com/NormanTUD/OmniOpt/actions/workflows/main.yml/badge.svg?event=push" alt="CI Badge">
				</a>
				<a href="https://pypi.org/project/omniopt2/" target="_blank">
					<img class="badge-img" src="https://img.shields.io/pypi/v/omniopt2" alt="PyPI Version">
				</a>
				<a href="https://scads.ai/imprint/" target="_blank">
					<button>Imprint/Impressum</button>
				</a>
			</div>

			<div class="header-tabs">
<?php
				$current_file = basename($_SERVER["PHP_SELF"]);
				foreach ($GLOBALS["files"] as $fn => $n) {
					if (is_array($n)) {
						$n = $n["name"];
					}

					$tab_is_active = preg_match("/^$fn.php/", $current_file);
					$_link = "$fn.php";

					if (!file_exists($_link)) {
						dier("Could not find $_link");
					}

					$link_no_php = preg_replace("/\.php$/", "", $_link);
					if ($tab_is_active) {
						$n = "<i><b>$n</b></i>";
					}

					$script_link = preg_replace("/\/\/*/", "/", get_main_script_dir() . "/$link_no_php");
					echo "<a href='$script_link'><button class='header_button'>$n</button></a>";
				}
?>
			</div>
			<div class="header-theme-toggle">
				<label class="switch"> 
					<input type="checkbox" id="themeSelect">
						<span class="slider">
						<span class="mode-text"></span>
					</span>
				</label>
			</div>
		</div>
		<br>
		<span style="display: inline-flex;">
			<input class="invert_in_dark_mode" onkeyup="start_search()" onfocus="start_search()" onblur="start_search()" onchange='start_search()' type="text" placeholder="Search..." id="search">
			<button id="del_search_button" class="invert_in_dark_mode" style="display: none;" onclick="delete_search()"><img src='i/red_x.svg' style='height: 1em' /></button>
		</span>
	</div>

	<script>
		apply_theme_based_on_system_preferences();

		function adapt_header_button_width () {
			var buttons = document.querySelectorAll('.header_button');
			var maxWidth = 0;

			buttons.forEach(function (btn) {
				btn.style.width = 'auto';
				var width = btn.offsetWidth;
				if (width > maxWidth) maxWidth = width;
			});

			buttons.forEach(function (btn) {
				btn.style.width = maxWidth + 'px';
			});
		}

		window.addEventListener('load', adapt_header_button_width);
		window.addEventListener('resize', adapt_header_button_width);
	</script>

	<div id="searchResults"></div>

	<div id="mainContent">
