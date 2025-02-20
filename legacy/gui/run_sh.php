<html>
	<head>
		<title>Create a run.sh-file for OmniOpt</title>
		<link href="jquery-ui.css" rel="stylesheet">
		<style>
			body {
				font-family: Verdana, Geneva, sans-serif
			}

			th {
				background-color: blue;
				color: white;
			}

			.parameter {
				border-style: ridge;
			}

			.parameter_input {
				min-width: 400px;
				width: 95%;
			}

			#bashcommand {
				max-width: 600px;
				max-height: 100px;
				overflow: scroll;
				white-space: pre-wrap;
			}
		</style>
		<link href="prism.css" rel="stylesheet" />
	</head>
	<body>
		<p>To make your script robust enough for the environment of OmniOpt on Taurus,
		it is recommeded that you do not run your script directly in the objective program
		string. Rather, it is recommeded that you create a <tt>run.sh</tt>-file from which
		your program gets run.</p>

		<p>It may look like this:</p>

		<pre><code class="language-bash">#!/bin/bash -l
# ^ Shebang-Line, so that it is known that this is a bash file
# -l means 'load this as login shell', so that /etc/profile gets loaded and you can use 'module load' or 'ml' as usual

# If you use this script not via `./run.sh' or just `srun run.sh', but like `srun bash run.sh', please add the '-l' there too.
# Like this:
# srun bash -l run.sh

# Load modules your program needs, always specify versions!
ml TensorFlow/2.3.1-fosscuda-2019b-Python-3.7.4 # Or whatever modules you need

# Load specific virtual environment (if applicable)
source /path/to/environment/bin/activate

# Load your script. $@ is all the parameters that are given to this run.sh file.
python3 /absolute/path/to_script.py $@
</code></pre>

		<p>Even though <tt>sbatch</tt> may inherit shell variables like loaded modules, 
		it is not recommended to rely on that heavily, because, especially when
		copying the <tt>curl</tt>-command from this website, you may forget loading
		the correct modules. This makes your script much more robust to changes.</p>

		<p>Also, always load specific module-versions and never let <tt>lmod</tt> guess
		the versions you want. Once these change, you'll almost certainly have problems
		otherwise.</p>

		<script src="prism.js"></script>

		<script>
			Prism.highlightAll();
		</script>
	</body>
</html>
