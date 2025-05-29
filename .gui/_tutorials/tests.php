<h1><span class="tutorial_icon invert_in_dark_mode">✅</span> Run automated tests</h1>

<!-- How to run automated tests and what options are available -->

<!-- Category: Developing -->

<div id="toc"></div>

<h2 id="what_are_automated_tests">What are automated tests?</h2>

<p>A large part of the source code of OmniOpt2 is to make sure that everything works as expected. This code executes real test cases and looks at the results to check if they are as expected. Many things in OmniOpt2 get tested automatically to see if they work properly. All test related files are in the folder <samp>.tests</samp>.</p>

<h2 id="why_run_these">Why would I want to run those?</h2>

<p>OmniOpt2 is supposed to be run on a wide variety of Linux systems. Not every system specific thing can be caught, though, since I cannot test it manually on all the available Linux-distributions. If you encounter problems in OmniOpt2, I may ask you to run those tests and submit the output to me, so that I can debug it thoroughly.</p>

<p>You may have made a change to OmniOpt2 and want to see if it still runs and you haven't broken anything.</p>

<h2 id="how_to_run_tests">How to run tests?</h2>

<p>To run all tests, which takes a lot of time, run:</p>

<pre class="invert_in_dark_mode"><code class="language-bash">./.tests/main</code></pre>

<p>Possible options:</p>

<table>
	<tr class="invert_in_dark_mode">
		<th>Option</th>
		<th>Meaning</th>
	</tr>
	<tr>
		<td><pre class="invert_in_dark_mode"><code class="language-bash">--max_eval=(INT)</code></pre></td>
		<td>How many evaluations should be tried for each test (the lower, the faster)</td>
	</tr>
	<tr>
		<td><pre class="invert_in_dark_mode"><code class="language-bash">--num_random_steps=(INT)</code></pre></td>
		<td>Number of random steps that should be tried (the lower, the faster)</td>
	</tr>
	<tr>
		<td><pre class="invert_in_dark_mode"><code class="language-bash">--num_parallel_jobs=(INT)</code></pre></td>
		<td>How many parallel jobs should be started (ignored on non-sbatch-systems)</td>
	</tr>
	<tr>
		<td><pre class="invert_in_dark_mode"><code class="language-bash">--run_with_coverage</code></pre></td>
		<td>Allows to use coverage instead of python3 for coverage testing unit test coverage</td>
	</tr>
	<tr>
		<td><pre class="invert_in_dark_mode"><code class="language-bash">--exit_on_first_error</code></pre></td>
		<td>Exit on first error</td>
	</tr>
	<tr>
		<td><pre class="invert_in_dark_mode"><code class="language-bash">--gpus=(INT)</code></pre></td>
		<td>How many GPUs you want for each worker/need to allocate an sbatch job</td>
	</tr>
	<tr>
		<td><pre class="invert_in_dark_mode"><code class="language-bash">--debug</code></pre></td>
		<td>Enables debug mode</td>
	</tr>
	<tr>
		<td><pre class="invert_in_dark_mode"><code class="language-bash">--no_plots</code></pre></td>
		<td>Disables plot tests</td>
	</tr>
	<tr>
		<td><pre class="invert_in_dark_mode"><code class="language-bash">--quick</code></pre></td>
		<td>Only runs quick tests (faster)</td>
	</tr>
	<tr>
		<td><pre class="invert_in_dark_mode"><code class="language-bash">--reallyquick</code></pre></td>
		<td>Only runs really quick tests (fastest)</td>
	</tr>
</table>

<h3 id="example_run_quick">Example on the quickest useful test</h3>

<p>When this succeeds without any errors, you can be reasonably sure that OmniOpt2 will properly do the following things under normal circumstances:</p>

<ul>
	<li>Run a simple run (one random step and 2 steps in total, so both model, <samp>SOBOL</samp> and <samp>BOTORCH_MODULAR</samp> get tested)</li>
	<li>Continue a run</li>
	<li>Continue an already continued run</li>
	<li>Test the of the number of results for all these jobs</li>
	<li>Plot scripts create svg files that contain strings that are to be expected</li>
	<li>Basic documentation tests are done</li>
</ul>

<pre class="invert_in_dark_mode"><code class="language-bash">./.tests/main --num_random_steps=1 --max_eval=2 --reallyquick</code></pre>

<h2 id="All test scripts">All test scripts</h2>

<?php

$directory = realpath('../.tests');

foreach (new DirectoryIterator($directory) as $file) {
	if ($file->isDot() || !$file->isFile()) continue;

	$handle = fopen($file->getPathname(), 'r');
	if ($handle === false) continue;

	$firstLine = fgets($handle);
	fclose($handle);

	if (strpos($firstLine, '#!/usr/bin/env bash') === 0) {
		print "<h3>" . basename($file->getPathname()) . "</h3>\n";

		$contents = file($file->getPathname());
		$helpTextFound = false;

		foreach ($contents as $line) {
			if (preg_match('/^#\s*HELPPAGE:\s*(.+)$/', $line, $matches)) {
				print "<p>" . htmlspecialchars($matches[1]) . "</p>\n";
				$helpTextFound = true;
				break;
			}
		}

		if (!$helpTextFound) {
			print '<div class="caveat error">No help text could be found</div>' . "\n";
		}


		parse_arguments_and_print_html_table($file->getPathname(), 1);
	}
}
?>
