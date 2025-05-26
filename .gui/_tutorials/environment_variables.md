# <span class="tutorial_icon invert_in_dark_mode">🌿</span> Environment Variables

<!-- List of all environment variables that change how OmniOpt2 works -->

<!-- Category: Advanced Usage -->

<div id="toc"></div>

## What are environment variables?

Every program on Linux has an environment, like a bash- or zsh-shell, that it runs in. These shells can contain variables that can change how OmniOpt2 works. Here is a list of all shell variables that change how OmniOpt2 works.

It is important that you run these commands before you run OmniOpt2, and also that you write <samp>export</samp> in front of them. Unexported variables are not passed to programs started by the shell.

## Table of environment variables

<table>
	<tr class="invert_in_dark_mode">
		<th>Name</th>
		<th>What does it do?</th>
	</tr>
	<tr>
		<th class="section-header invert_in_dark_mode" colspan=2>OmniOpt2</th>
	</tr>
	<tr>
		<td><pre class="invert_in_dark_mode"><code class="language-bash">export HIDE_PARETO_FRONT_TABLE_DATA=1</code></pre></td>
		<td>Hide pareto front table</td>
	</tr>
	<tr>
	<tr>
		<td><pre class="invert_in_dark_mode"><code class="language-bash">export RUN_WITH_PYSPY=1</code></pre></td>
		<td>Run the OmniOpt2 main script with py-spy to create flame graphs</td>
	</tr>
	<tr>
		<td><pre class="invert_in_dark_mode"><code class="language-bash">export DIE_AFTER_THIS_NR_OF_DONE_JOBS=1</code></pre></td>
		<td>Dies after DIE_AFTER_THIS_NR_OF_DONE_JOBS jobs, only useful for debugging</td>
	</tr>
	<tr>
		<td><pre class="invert_in_dark_mode"><code class="language-bash">export install_tests=1</code></pre></td>
		<td>Install test modules</td>
	</tr>
	<tr>
		<td><pre class="invert_in_dark_mode"><code class="language-bash">export RUN_UUID=$(uuidgen)</code></pre></td>
		<td>Sets the UUID for the run. Default is a new one via <samp>uuidgen</code></pre></td>
	</tr>
	<tr>
		<td><pre class="invert_in_dark_mode"><code class="language-bash">export OO_NO_LOGO=1</code></pre></td>
		<td>Disables showing the logo</td>
	</tr>
	<tr>
		<td><pre class="invert_in_dark_mode"><code class="language-bash">export OO_MAIN_TESTS=1</code></pre></td>
		<td>Sets the user-id to <samp>affed00faffed00faffed00faffed00f</samp>, so the statistics can determine whether you are a real user or a test-user</td>
	</tr>
	<tr>
		<td><pre class="invert_in_dark_mode"><code class="language-bash">export ITWORKSONMYMACHINE=1</code></pre></td>
		<td>Sets the user-id to <samp>affeaffeaffeaffeaffeaffeaffeaffe</samp>, so the statistics can determine whether you are a real user or a the main developer (only I should set this variable)</td>
	</tr>
	<tr>
		<td><pre class="invert_in_dark_mode"><code class="language-bash">export root_venv_dir=/path/to/venv</code></pre></td>
		<td>Path to where virtualenv should be installed. Default is <samp>$HOME</code></pre></td>
	</tr>
	<tr>
		<td><pre class="invert_in_dark_mode"><code class="language-bash">export DISABLE_SIXEL_GRAPHICS=1</code></pre></td>
		<td>Disables <a href="https://en.wikipedia.org/wiki/Sixel">sixel</a>-graphics, no matter what other parameters are set</td>
	</tr>
	<tr>
		<td><pre class="invert_in_dark_mode"><code class="language-bash">export DONT_INSTALL_MODULES=1</code></pre></td>
		<td>Disables installing modules</td>
	</tr>
	<tr>
		<td><pre class="invert_in_dark_mode"><code class="language-bash">export DONT_SHOW_DONT_INSTALL_MESSAGE=1</code></pre></td>
		<td>Don't show messages regarding the installation of modules</td>
	</tr>
	<tr>
		<td><pre class="invert_in_dark_mode"><code class="language-bash">export DONTSTARTFIREFOX=1</code></pre></td>
		<td>Don't start firefox when <samp>RUN_WITH_COVERAGE</samp> is defined.</td>
	</tr>
	<tr>
		<td><pre class="invert_in_dark_mode"><code class="language-bash">export RUN_WITH_COVERAGE=1</code></pre></td>
		<td>Runs omniopt- and plot-script with coverage to find out test code coverage</td>
	</tr>
	<tr>
		<td><pre class="invert_in_dark_mode"><code class="language-bash">export CI=1</code></pre></td>
		<td>Disables certain tests in a CI environment</td>
	</tr>
	<tr>
		<td><pre class="invert_in_dark_mode"><code class="language-bash">export PRINT_SEPERATOR=1</code></pre></td>
		<td>Prints a seperator line after OmniOpt2 runs (useful for automated tests)</td>
	</tr>
	<tr>
		<th class="section-header invert_in_dark_mode" colspan=2>Plot-Script</th>
	</tr>
	<tr>
		<td><pre class="invert_in_dark_mode"><code class="language-bash">export BUBBLESIZEINPX=1</code></pre></td>
		<td>Size of bubbles in plot scripts in px</td>
	</tr>
	<tr>
		<th class="section-header invert_in_dark_mode" colspan=2>Test-Scripts</th>
	</tr>
	<tr>
		<td><pre class="invert_in_dark_mode"><code class="language-bash">export CREATE_PLOT_HELPS=1</code></pre></td>
		<td>Creates the help files for the tutorials page for each <samp>omniopt_plot</code></pre></td>
	</tr>
	<tr>
		<td><pre class="invert_in_dark_mode"><code class="language-bash">export NO_RUNTIME=1</code></pre></td>
		<td>Don't show <samp>omniopt_plot</samp> runtime at the end</td>
	</tr>
	<tr>
		<td><pre class="invert_in_dark_mode"><code class="language-bash">export NO_TESTS=1</code></pre></td>
		<td>Disable tests for <samp>.tests/pre-commit</samp>-hook</td>
	</tr>
	<tr>
		<td><pre class="invert_in_dark_mode"><code class="language-bash">export NO_NO_RESULT_ERROR=1</code></pre></td>
		<td>Disable errors to stdout for plots when no results are found</td>
	</tr>
	<tr>
		<td><pre class="invert_in_dark_mode"><code class="language-bash">export SHOW_COMMAND_BEFORE_EXECUTION=1</code></pre></td>
		<td>Show <samp>omniopt_plot</samp>-commands before executing them</td>
	</tr>
	<tr>
		<td><pre class="invert_in_dark_mode"><code class="language-bash">export OMNIOPT_CALL="./omniopt"</code></pre></td>
		<td>How to call OmniOpt2. Is useful to differentiate between pip installs and installs via git.</td>
	</tr>
</table>
