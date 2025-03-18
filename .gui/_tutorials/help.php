<?php
	if(file_exists("_functions.php")) {
		include_once("_functions.php");
	} else {
		include_once("../_functions.php");
	}
?>
<div id="toc"></div>

<!-- The <tt>--help</tt> of the main script of OmniOpt2 -->

<h2 id="available_parameters_help">Available Parameters (--help)</h2>

<p>An overview table of all arguments that OmniOpt2 accepts via the Command line. Same data as in <tt>./omniopt --help</tt>.</p>

<?php
	$file_path = "../.omniopt.py";
	parse_arguments_and_print_html_table($file_path);
?>
