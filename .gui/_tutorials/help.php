<?php
	if(file_exists("_functions.php")) {
		include_once("_functions.php");
	} else {
		include_once("../_functions.php");
	}
?>
<div id="toc"></div>

<!-- The <tt>--help</tt> of the main script of OmniOpt2 -->

<!-- Category: Preparations, Basics and Setup -->

<h1 id="available_parameters_help"><img class='emoji_nav' src='emojis/blue_book.svg' /> Available Parameters (--help)</h1>

<p>An overview table of all arguments that OmniOpt2 accepts via the Command line. Same data as in <tt>omniopt --help</tt>.</p>

<?php
	$file_path = "../omniopt";

	// Backwards-compat: if the new Python file (omniopt) exists but the
	// old .omniopt.py doesn't, make a temporary symlink with a .py
	// extension so the PHP parser (which only handles .py files) still
	// finds the argparse definitions.  The symlink lives in PHP's temp
	// dir because the web server user often can't write into .gui/, and
	// is removed again at the end of the request via a shutdown
	// handler.  A uniqid() suffix avoids clashes between concurrent
	// requests that share the same temp dir.
	if (!file_exists($file_path . ".py") && file_exists($file_path)) {
		$link_path = sys_get_temp_dir() . "/omniopt_" . uniqid("", true) . ".py";
		if (!@symlink(realpath($file_path), $link_path)) {
			echo "<p><strong>ERROR:</strong> Could not create symlink $link_path</p>";
		} else {
			register_shutdown_function(function () use ($link_path) {
				if (is_link($link_path)) {
					@unlink($link_path);
				}
			});
			$file_path = $link_path;
		}
	}
	parse_arguments_and_print_html_table($file_path);
?>
