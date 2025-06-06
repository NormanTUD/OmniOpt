<?php
	function get_all_get_parameters_as_query_string() {
		$query_string = '';
		if (!isset($_GET)) {
			return;
		}
		$parameters = array();
		foreach ($_GET as $key => $value) {
			$parameters[] = urlencode($key) . '=' . urlencode($value);
		}
		$query_string = implode('&', $parameters);
		return $query_string;
	}

	if (array_key_exists('partition', $_GET)) {
		$all_parameters_query_string = get_all_get_parameters_as_query_string();
		if ($all_parameters_query_string && !empty($all_parameters_query_string)) {
			$redirect_url = 'gui.php?' . $all_parameters_query_string;
			header("Location: " . $redirect_url);
			exit(0);
		}
	}

	require "_header_base.php";
?>
			<p><i>OmniOpt2</i> allows you to easily optimize complex hyperparameter configurations in situations where gradient information is not available or too complex. It is based on <a target="_blank" href="https://ax.dev">Ax</a> and <a target="_blank" href="https://botorch.org">BoTorch</a>, but allows any method of generation of new hyperparameter points by using <a href="tutorials?tutorial=external_generator">External Generators</a>.</p>

			<p>You can run <i>OmniOpt2</i> on any linux that has <samp>python3</samp> and some basic necessities. If something is missing that cannot be installed, it will tell you how to install it. If the system you run it on has Slurm installed, it will use it to parallelize as you set the settings. If you run it locally without slurm, they will simply run sequentially.</p>

			<p><i>OmniOpt2</i> installs all of it's python-dependencies automatically in a virtual environment once at first run. This may take up to 20 minutes, but has to be done once. This is done to isolate it from other dependencies and to make you focus on your task of hyperparameter-minimization. For even more isolation, you can also use <a href="tutorials?tutorial=basics">Docker</a>.</p>

			<p>In short: It will try out your program with different hyperparameter settings and tries to find new ones to minimize the result using an algorithm called <a target="_blank" href="https://web.archive.org/web/20240715080430/https://proceedings.neurips.cc/paper/2020/file/f5b1b89d98b7286673128a5fb112cb9a-Paper.pdf">BoTorch-Modular.</a></p>

			<p><a target="_blank" href="tutorials.php?tutorial=run_sh">Read the documentation of how you need to change your program</a> and then go to the <a href="gui.php">OmniOpt2-GUI</a> and start optimizing.</p>

			<script src="prism.js"></script>
			<script src="footer.js"></script>
		</div>
<?php
	include("footer.php");
?>
