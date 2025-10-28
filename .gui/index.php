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
<style>
  .code { background-color: #f0f0f0; padding: 2px 6px; border-radius: 4px; font-family: monospace; }
  .bullet-list li { margin: 8px 0; }
  .cta-box { 
    background: #fef9e7; 
    border: 2px solid #f1c40f; 
    padding: 12px; 
    margin: 16px 0; 
    border-radius: 8px; 
    font-weight: bold; 
  }
</style>
<p><i>OmniOpt2</i> helps you effortlessly optimize complex hyperparameter configurations — even when gradients are unavailable or too complicated. Built on <a target="_blank" href="https://ax.dev">Ax</a> and <a target="_blank" href="https://botorch.org">BoTorch</a>, it also supports any method of generating new hyperparameter points using <a href="tutorials?tutorial=external_generator">External Generators</a>.</p>

<p>You can run <i>OmniOpt2</i> on any Linux system with <span class="code">python3</span>. Missing dependencies will be installed automatically.</p>

<p>Have <i>Slurm</i>? Jobs will be automatically parallelized. No Slurm? Jobs run sequentially, hassle-free.</p>

<p>All Python dependencies are installed automatically in a dedicated virtual environment on the first run. This may take a few minutes, but ensures your system stays clean and you can focus on optimization. For complete isolation, use <a href="tutorials?tutorial=basics">Docker</a>.</p>

<p>In short, OmniOpt2 tests different hyperparameter combinations and uses the <a target="_blank" href="https://web.archive.org/web/20240715080430/https://proceedings.neurips.cc/paper/2020/file/f5b1b89d98b7286673128a5fb112cb9a-Paper.pdf"><i>BoTorch-Modular</i></a> (<a href="tutorials?tutorial=models">among others</a>) algorithm to find settings that minimize your target metric.</p>

<ul class="bullet-list">
  <li>Works on any Linux + Python3</li>
  <li>Automatic dependency installation</li>
  <li>Supports <i>Slurm</i> for parallel jobs</li>
  <li>Docker-ready for full isolation</li>
</ul>

<div class="cta-box">
Check the <a target="_blank" href="tutorials.php?tutorial=run_sh">documentation</a> to adjust your program, then launch the <a href="gui.php">OmniOpt2 GUI</a> and start optimizing — faster, smarter, simpler!
</div>
			<script src="footer.js"></script>
		</div>
<?php
	include("footer.php");
?>
