<?php
	include_once("_functions.php");

	$dir_path = ".";
	if(preg_match("/\/tutorials\/?$/", dirname($_SERVER["PHP_SELF"]))) {
		$dir_path = "..";
	}
?>
<!DOCTYPE html>
<html>
	<head>
		<meta charset="UTF-8">
		<title>OmniOpt2</title>
		<link href="<?php print $dir_path; ?>/prism.css" rel="stylesheet" />
		<link rel="icon" type="image/x-icon" href="favicon.ico">
		<style>
			.marked_text {
				background-color: yellow;
			}

			.time_picker_container {
				font-variant: small-caps;
				width: 100%;
			}

			.time_picker_container > input {
				width: 50px;
			}

			#loader {
				display: grid;
				justify-content: center; /* Horizontal zentriert */
				align-items: center; /* Vertikal zentriert */
				height: 100%;
			}

			.no_linebreak {
				line-break: auto;
			}

			h2 {
				margin: 0px;
				padding: 0px;
			}

			body {
				font-family: Verdana, sans-serif;
			}

			.dark_code_bg {
				background-color: #363636;
				color: white;
			}

			.code_bg {
				background-color: #C0C0C0;
			}

			#commands {
				line-break: anywhere;
			}

			.color_red {
				color: red;
			}

			.color_orange {
				color: orange;
			}

			#hidden_config_table > tbody > tr:nth-child(odd) {
				background-color: #fafafa;
			}

			#hidden_config_table > tbody > tr:nth-child(even) {
				background-color: #ddd;
			}

			#hidden_config_table {
				border-collapse: collapse;
				margin: 25px 0;
				font-size: 0.9em;
				min-width: 200px;
				box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
			}

			#hidden_config_table thead tr {
				background-color: #36A9AE;
				color: #ffffff;
				text-align: left;
			}

			#hidden_config_table tbody tr:last-of-type {
				border-bottom: 1px solid #36A9AE;
			}

			#hidden_config_table tbody tr {
				border-bottom: 1px solid #dddddd;
			}

			#config_table {
				width: 100%;
			}

			#config_table > tbody > tr:nth-child(odd) {
				background-color: #fafafa;
			}

			#config_table > tbody > tr:nth-child(even) {
				background-color: #ddd;
			}

			#config_table {
				border-collapse: collapse;
				margin: 25px 0;
				font-size: 0.9em;
				min-width: 200px;
				box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
			}

			#config_table thead tr {
				background-color: #36A9AE;
				color: #ffffff;
				text-align: left;
			}

			#config_table tbody tr:last-of-type {
				border-bottom: 1px solid #36A9AE;
			}

			#config_table tbody tr {
				border-bottom: 1px solid #dddddd;
			}

			.error_element {
				background-color: #e57373;
			}

			button {
				margin: 5px;
				background-color: #36A9AE;
				border: 1px solid #2A8387;
				border-radius: 4px;
				box-shadow: rgba(0, 0, 0, 0.12) 0 1px 1px;
				color: #FFFFFF;
				cursor: pointer;
				display: block;
				font-family: -apple-system, ".SFNSDisplay-Regular", "Helvetica Neue", Helvetica, Arial, sans-serif;
				font-size: 17px;
				line-height: 100%;
				outline: 0;
				padding: 11px 15px 12px;
				text-align: center;
				transition: box-shadow .05s ease-in-out, opacity .05s ease-in-out;
				user-select: none;
				-webkit-user-select: none;
				touch-action: manipulation;
			}

			button:hover {
				box-shadow: rgba(255, 255, 255, 0.3) 0 0 2px inset, rgba(0, 0, 0, 0.4) 0 1px 2px;
				text-decoration: none;
				transition-duration: .15s, .15s;
			}

			button:active {
				box-shadow: rgba(0, 0, 0, 0.15) 0 2px 4px inset, rgba(0, 0, 0, 0.4) 0 1px 1px;
			}

			button:disabled {
				cursor: not-allowed;
				opacity: .6;
			}

			button:disabled:active {
				pointer-events: none;
			}

			button:disabled:hover {
				box-shadow: none;
			}

			select {
				width: 98%;
			}

			input {
				width: 95%;
			}

			.remove_parameter {
				background-color: #f1807e !important;
				background-image: linear-gradient(#db4a44, #f24f49);
				border: 1px solid #db4a44;
			}

			.half_width_td {
				vertical-align: baseline;
				width: 50%;
			}

			#scads_bar {
				width: 100%;
				height: 80px;
				margin: 0;
				padding: 0;
			}

			/* Allgemeine Stile für alle Tabs */
			.tab {
				display: inline-block;
				padding: 10px 20px;
				margin: 5px;
				font-size: 16px;
				font-weight: bold;
				text-align: center;
				border-radius: 25px;
				text-decoration: none;
				transition: background-color 0.3s, color 0.3s;
			}

			/* Inaktiver Tab */
			.inactive_tab {
				background-color: #f0f0f0;
				color: #555;
				border: 2px solid #ddd;
			}

			.inactive_tab:hover {
				background-color: #e0e0e0;
				color: #444;
			}

			/* Aktiver Tab */
			.active_tab {
				background-color: #4CAF50;
				color: white;
				border: 2px solid #4CAF50;
			}

			.active_tab:hover {
				background-color: #45a049;
			}

			.dropdown {
				display: inline-grid;
			}

			.dropdown-content {
				display: none;
				position: absolute;
				background-color: #f9f9f9;
				min-width: 160px;
				box-shadow: 0px 8px 16px 0px rgba(0, 0, 0, 0.2);
				z-index: 1;
				top: 100%; /* Ensure the dropdown opens below */
			}

			.dropdown-content a {
				color: black;
				padding: 12px 16px;
				text-decoration: none;
				display: block;
			}

			.dropdown-content a:hover {
				background-color: #f1f1f1;
			}

			.dropdown:hover .dropdown-content {
				display: block;
			}

			.dropdown:hover .dropbtn {
				background-color: #3e8e41;
			}

			.tooltipster-base {
				border: 1px solid black;
				position: absolute;
				border-radius: 8px;
				padding: 2px;
				color: white;
				background-color: #61686f;
				width: 70%;
				min-width: 200px;
			}

			td {
				padding-top: 3px;
				padding-bottom: 3px;
			}

			.left_side {
				text-align: right;
			}

			.right_side {
				text-align: left;
			}
		</style>
		<script src="<?php print $dir_path; ?>/jquery-3.7.1.js"></script>
		<script src="<?php print $dir_path; ?>/jquery-ui.min.js"></script>
		<script src="<?php print $dir_path; ?>/prism.js"></script>
		<script src="<?php print $dir_path; ?>/search.js"></script>
		<script src="<?php print $dir_path; ?>/tooltipster.bundle.min.js"></script>
		<link href="<?php print $dir_path; ?>/tutorial.css" rel="stylesheet" />
		<link href="<?php print $dir_path; ?>/jquery-ui.css" rel="stylesheet">
		<link href="<?php print $dir_path; ?>/prism.css" rel="stylesheet" />
		<script>
			document.onkeypress = function (e) {
				e = e || window.event;

				if(document.activeElement == $("body")[0]) {
					var keycode = e.keyCode;
					if(keycode >= 97 && keycode <= 122) {
						e.preventDefault();
						$("#search").val("");
						$("#search").val(String.fromCharCode(e.keyCode));
						$("#search").focus().trigger("change");
					}
				} else if (keycode === 8) { // Backspace key
					delete_search();
				}
			};
		</script>
	</head>
	<body>
		<div id="scads_bar">
			<a target="_blank" href="https://scads.ai/"><img src="<?php print $dir_path; ?>/scads_logo.svg" /></a>
			<?php
				include("searchable_php_files.php");

				$current_file = basename($_SERVER["PHP_SELF"]);

				foreach ($files as $fn => $n) {
					if (is_array($n)) {
						$n = $n["name"];
					}

					$tab_is_active = preg_match("/^$fn.php/", $current_file);
					$tab_class = $tab_is_active ? 'active_tab' : 'inactive_tab';
					echo "\t<a href='$dir_path/$fn.php' class='tab $tab_class'>OmniOpt $n</a>\n";
				}
			?>
			<span style="display: inline-flex;">
				<input onkeyup="start_search()" onfocus="start_search()" onblur="start_search()" onchange='start_search()' style="width: 180px;" type="text" placeholder="Search (allows Regex)..." id="search"></input>
				<button id="del_search_button" style="display: none;" onclick="delete_search()">&#10060;</button>
			</span>
		</div>
		<div id="searchResults"></div>

		<div id="mainContent">
