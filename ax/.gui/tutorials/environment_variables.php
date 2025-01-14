<h1>Environment Variables</h1>

<!-- List of all environment variables that change how OmniOpt2 works -->

<div id="toc"></div>

<h2 id="what_are_environment_variables">What are environment variables?</h2>

<p>Every program on Linux has an environment, like a bash- or zsh-shell, that it runs in. These shells can contain variables that can change how OmniOpt2 works. Here is a list of all shell variables that change how OmniOpt2 works.</p>

<p>It is important that you run these commands before you run OmniOpt2, and also that you write <samp>export</samp> in front of them. Unexported variables are not passed to programs started by the shell.</p>

<table>
	<tr class="invert_in_dark_mode">
		<th>Name</th>
		<th>What does it do?</th>
	</tr>
	<tr>
		<th class="section-header invert_in_dark_mode" colspan=2>OmniOpt2</th>
	</tr>
	<tr>
		<td><samp>export RUN_UUID=$(uuidgen)</samp></td>
		<td>Sets the UUID for the run. Default is a new one via <samp>uuidgen</samp></td>
	</tr>
	<tr>
		<td><samp>export OO_NO_LOGO=1</samp></td>
		<td>Disables showing the logo</td>
	</tr>
	<tr>
		<td><samp>export OO_MAIN_TESTS=1</samp></td>
		<td>Sets the user-id to <samp>affed00faffed00faffed00faffed00f</samp>, so the statistics can determine whether you are a real user or a test-user</td>
	</tr>
	<tr>
		<td><samp>export ITWORKSONMYMACHINE=1</samp></td>
		<td>Sets the user-id to <samp>affeaffeaffeaffeaffeaffeaffeaffe</samp>, so the statistics can determine whether you are a real user or a the main developer (only I should set this variable)</td>
	</tr>

	<tr>
		<td><samp>export root_venv_dir=/path/to/venv</samp></td>
		<td>Path to where virtualenv should be installed. Default is <samp>$HOME</samp></td>
	</tr>
	<tr>
		<td><samp>export DISABLE_SIXEL_GRAPHICS=1</samp></td>
		<td>Disables <a href="https://en.wikipedia.org/wiki/Sixel">sixel</a>-graphics, no matter what other parameters are set</td>
	</tr>
	<tr>
		<td><samp>export DONT_INSTALL_MODULES=1</samp></td>
		<td>Disables installing modules</td>
	</tr>
	<tr>
		<td><samp>export DONT_SHOW_DONT_INSTALL_MESSAGE=1</samp></td>
		<td>Don't show messages regarding the installation of modules</td>
	</tr>
	<tr>
		<td><samp>export DONTSTARTFIREFOX=1</samp></td>
		<td>Don't start firefox when <samp>RUN_WITH_COVERAGE</samp> is defined.</td>
	</tr>
	<tr>
		<td><samp>export RUN_WITH_COVERAGE=1</samp></td>
		<td>Runs omniopt- and plot-script with coverage to find out test code coverage</td>
	</tr>
	<tr>
		<td><samp>export CI=1</samp></td>
		<td>Disables certain tests in a CI environment</td>
	</tr>
	<tr>
		<td><samp>export PRINT_SEPERATOR=1</samp></td>
		<td>Prints a seperator line after OmniOpt2 runs (useful for automated tests)</td>
	</tr>
	<tr>
		<th class="section-header invert_in_dark_mode" colspan=2>Plot-Script</th>
	</tr>
	<tr>
		<td><samp>export BUBBLESIZEINPX=1</samp></td>
		<td>Size of bubbles in plot scripts in px</td>
	</tr>

	<tr>
		<th class="section-header invert_in_dark_mode" colspan=2>Test-Scripts</th>
	</tr>
	<tr>
		<td><samp>export CREATE_PLOT_HELPS=1</samp></td>
		<td>Creates the help files for the tutorials page for each <samp>omniopt_plot</samp></td>
	</tr>
	<tr>
		<td><samp>export NO_RUNTIME=1</samp></td>
		<td>Don't show <samp>omniopt_plot</samp> runtime at the end</td>
	</tr>
	<tr>
		<td><samp>export NO_TESTS=1</samp></td>
		<td>Disable tests for <samp>.tests/pre-commit</samp>-hook</td>
	</tr>
	<tr>
		<td><samp>export NO_NO_RESULT_ERROR=1</samp></td>
		<td>Disable errors to stdout for plots when no results are found</td>
	</tr>
	<tr>
		<td><samp>export SHOW_COMMAND_BEFORE_EXECUTION=1</samp></td>
		<td>Show <samp>omniopt_plot</samp>-commands before executing them</td>
	</tr>
</table>
