<div id="toc"></div>

<!-- The <tt>--help</tt> of the main script of OmniOpt2 -->

<h2 id="available_parameters_help">Available Parameters (--help)</h2>

<p>An overview table of all arguments that OmniOpt2 accepts via the Command line. Same data as in <tt>./omniopt --help</tt>.</p>

<?php
	include("../_functions.php");
	$file_path = "../.omniopt.py";
	$arguments = parse_arguments($file_path);
	echo generate_argparse_html_table($arguments);
?>
