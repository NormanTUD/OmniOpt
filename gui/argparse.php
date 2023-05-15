<html>
	<head>
		<title>Parse command-line-arguments in Python</title>
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
		<p>To get parameters in your python-scripts from the command line, you can use the module 'argparse'.</p>

		<p>It may look like this:</p>

		<pre><code class="language-python">import argparse

# Initialize parser-object, description is for automatically generated --help parameter
parser = argparse.ArgumentParser(description='My neural network.')

# Add parameter --epochs=integer. This parameter is required (there is an error when it
# is not set). The help-option is for the --help-menu.
parser.add_argument("--epochs", help="Number of epochs", required=True, type=int)

# Adds a --learning_rate=float parameter. Defaults to 0.001 when not specified.
parser.add_argument("--learning_rate", help="Learning rate to be used", type=float, default=0.001)

# Add the parameter --use_database. Default is false, when used sets it to true.
parser.add_argument("--use_database", help="Should I use the Database?", action="store_true")

args = parser.parse_args()

print("Epochs: %d" % args.epochs)
print("Learning-Rate: %0.5f" % args.learning_rate)
print("Use database? %s" % args.use_database)</code></pre>

		<p>Example call:</p>
		<pre><code class="language-bash">python3 script.py --epochs=10 --use_database</code></pre>

		<script src="prism.js"></script>

		<script>
			Prism.highlightAll();
		</script>
	</body>
</html>
