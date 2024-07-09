<?php
	include("_header_base.php");
?>
	<p><i>OmniOpt2</i> allows you to easily optimize complex hyperparameter configurations. It is based on <a target="_blank" href="https://ax.dev">Ax</a> and <a target="_blank" href="https://botorch.dev">BoTorch</a></p>

	<p>You can run <i>OmniOpt2</i> on any linux that has <tt>python3</tt> and some basic neccessities. If something is missing that cannot be installed, it will tell you how to install it.</p>

	<p>If the system you run it on has Slurm installed, it will use it to parallelize as you set the settings. If you run it locally without slurm, they will simply run sequentially.</p>

	<p><i>OmniOpt2</i> installs all of it's python-dependencies automatically in a virtual environment once at first run. This may take up to 20 minutes, but has to be done once. This is done to isolate it from other dependencies and to make you focus on your task of hyperparameter-minimization.</p>

	<p>In short: It will try out your program with different hyperparameter settings and tries to find new ones to minimize the result using an algorithm called BoTorch-Modular. TODO: Paper verlinken.</p> 

	<p><a target="_blank" href="tutorials.php?tutorial=run_sh">Read the documentation of how you need to change your program</a> and then go to the <a href="gui.php">OmniOpt2-GUI</a> and start optimizing.</p>

	<script src="<?php print $dir_path; ?>/prism.js"></script>
	<script src="<?php print $dir_path; ?>/footer.js"></script>
</body>
</html>
