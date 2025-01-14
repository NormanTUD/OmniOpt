<h1>Run automated tests</h1>

<!-- How to run automated tests and what options are available -->

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
		<td><samp>--max_eval=(INT)</samp></td>
		<td>How many evaluations should be tried for each test (the lower, the faster)</td>
	</tr>
	<tr>
		<td><samp>--num_random_steps=(INT)</samp></td>
		<td>Number of random steps that should be tried (the lower, the faster)</td>
	</tr>
	<tr>
		<td><samp>--num_parallel_jobs=(INT)</samp></td>
		<td>How many parallel jobs should be started (ignored on non-sbatch-systems)</td>
	</tr>
	<tr>
		<td><samp>--run_with_coverage</samp></td>
		<td>Allows to use coverage instead of python3 for coverage testing unit test coverage</td>
	</tr>
	<tr>
		<td><samp>--exit_on_first_error</samp></td>
		<td>Exit on first error</td>
	</tr>
	<tr>
		<td><samp>--gpus=(INT)</samp></td>
		<td>How many GPUs you want for each worker/need to allocate an sbatch job</td>
	</tr>
	<tr>
		<td><samp>--debug</samp></td>
		<td>Enables debug mode</td>
	</tr>
	<tr>
		<td><samp>--no_plots</samp></td>
		<td>Disables plot tests</td>
	</tr>
	<tr>
		<td><samp>--quick</samp></td>
		<td>Only runs quick tests (faster)</td>
	</tr>
	<tr>
		<td><samp>--reallyquick</samp></td>
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
