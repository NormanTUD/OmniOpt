<?php
	if(file_exists("_functions.php")) {
		include_once("_functions.php");
	} else {
		include_once("../_functions.php");
	}
?>
<h1>Plot your results</h1>

<!-- What kinds of plots are available and how to use them -->

<!-- Category: Plotting and Sharing Results -->

<div id="toc"></div>

<h2 id="plot-over-x11">Plot over X11</h2>
<h3 id="plot-overview">Plot from overview</h3>

To plot over X11, make sure you are connected with <samp>ssh -X user@login2.barnard.hpc.tu-dresden.de</samp> (of course, use the HPC system you wish instead of barnard, if applicable, and change it to your user).

Then, <samp>cd</samp> into your OmniOpt2 directory. Assuming you have already ran an OmniOpt2-run and the results are in <samp>runs/my_experiment/0</samp> (adapt this to your experiment folder!), run this:

<pre class="invert_in_dark_mode"><code class="language-bash">./omniopt_plot runs/my_experiment/0</code></pre>

You will be presented by a menu like this:<br>

<img alt="Plot-Overview" src="imgs/plot_overview.png" /><br>

Use your arrow keys to navigate to the plot type you like, and then press enter.

<h3 id="plot-directly">Plot directly</h3>
If you know what plot you want, you can directly plot it by using:
<pre class="invert_in_dark_mode"><code class="language-bash">./omniopt_plot runs/my_experiment/0 scatter # change plot_type accordingly</code></pre>

<h3 id="plot_to_file">Plot to file</h3>
All plot scripts support to export your plot to a file.
<pre class="invert_in_dark_mode"><code class="language-bash">./omniopt_plot runs/my_experiment/0 scatter --save_to_file filename.svg # change plot_type and file name accordingly. Allowed are svg and png.</code></pre>

<h2 id="plot-types">Plot types</h2>
<p>There are many different plot types, some of which can only be shown on jobs that ran on Taurus, or jobs with more than a specific number of results or parameters. If you run the <samp>omniopt_plot</samp>-script, it will automatically show you plots that are readily available.</p>

<?php
	$pattern = "../.omniopt_plot_*.py";
	$files = glob($pattern);

	foreach ($files as $file) {
		if (preg_match('/\.omniopt_plot_(.*?)\.py$/', $file, $matches)) {
			$plot_type = $matches[1];
			$file_path = "../.omniopt_plot_$plot_type.py";

			$title = extract_magic_comment($file_path, "TITLE");
			if(!$title) {
				dier("Cannot find TITLE for $file_path");
			}

			$desc = extract_magic_comment($file_path, "FULL_DESCRIPTION");
			if (!$desc) {
				$desc = extract_magic_comment($file_path, "DESCRIPTION");
			}

			if (!$desc) {
				dier("Cannot find either FULL_DESCRIPTION or DESCRIPTION for $file_path");
			}

			echo "<h3 id='$plot_type'>$title</h3>\n";

			echo "<pre class='invert_in_dark_mode'><code class='language-bash'>./omniopt_plot runs/my_experiment/0 $plot_type</code></pre>\n";

			echo "<p>$desc</p>\n";

			$plot_type_img_path = "imgs/$plot_type.png";

			if(file_exists($plot_type_img_path)) {
				echo "<img style='max-width: 80%' alt='$plot_type plot example' src='$plot_type_img_path' /><br>\n";
			} else {
				dier("Error: $plot_type_img_path cannot be found!");
			}

			parse_arguments_and_print_html_table($file_path);
		}
	}
?>
