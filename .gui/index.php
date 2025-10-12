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
			<p><strong>OmniOpt2</strong> helps you effortlessly optimize complex hyperparameter configurations — even when gradients are unavailable or too complicated. It’s built on <a target="_blank" href="https://ax.dev">Ax</a> and <a target="_blank" href="https://botorch.org">BoTorch</a>, and also supports any method of generating new hyperparameter points using <a href="tutorials?tutorial=external_generator">External Generators</a>.</p>

			<p>You can run <strong>OmniOpt2</strong> on any Linux system with <code>python3</code> installed. Missing dependencies? Don’t worry — it will guide you step by step. If your system has Slurm installed, OmniOpt2 will automatically parallelize jobs; without Slurm, jobs run sequentially, hassle-free.</p>

			<p>All Python dependencies are installed automatically in a dedicated virtual environment on the first run. This may take a few minutes, but it ensures your system stays clean and you can focus on optimization. For complete isolation, you can also use <a href="tutorials?tutorial=basics">Docker</a>.</p>

			<p>In short, OmniOpt2 tests different hyperparameter combinations and uses the <a target="_blank" href="https://web.archive.org/web/20240715080430/https://proceedings.neurips.cc/paper/2020/file/f5b1b89d98b7286673128a5fb112cb9a-Paper.pdf">BoTorch-Modular</a> algorithm to find settings that minimize your target metric.</p>

			<ul>
				<li>Works on any Linux + Python3</li>
				<li>Automatic dependency installation</li>
				<li>Supports Slurm for parallel jobs</li>
				<li>Docker-ready for full isolation</li>
			</ul>

			<p>Check the <a target="_blank" href="tutorials.php?tutorial=run_sh">documentation</a> to adjust your program, then launch the <a href="gui.php">OmniOpt2 GUI</a> and start optimizing — faster, smarter, simpler.</p>


			<script src="prism.js"></script>
			<script src="footer.js"></script>
		</div>
<?php
	include("footer.php");
?>
