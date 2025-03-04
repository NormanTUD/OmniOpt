<!DOCTYPE html>
<html lang="en">
	<head>
		<meta charset="UTF-8">
		<meta name="viewport" content="width=device-width, initial-scale=1.0">
		<title>OmniOpt2-Share</title>
		<script src="../plotly-latest.min.js"></script>
		<script src="gridjs.umd.js"></script>
		<link href="mermaid.min.css" rel="stylesheet" />
		<link href="tabler.min.css" rel="stylesheet">
		<?php include("css.php"); ?>
	</head>
	<body>
		<div class="page window" style='font-family: sans-serif'>
			<div class="title-bar">
				<div class="title-bar-text">OmniOpt2-Share</div>
			</div>
			<div id="spinner" class="spinner"></div>

			<div id="main_window" style="display: none" class="container py-4 window-body has-space">
				<section class="tabs" style="width: 100%">
					<menu role="tablist" aria-label="OmniOpt2-Run">
						<button role="tab" aria-selected="true" aria-controls="tab_scatter_2d">2D-Scatter</button>
						<button role="tab" aria-controls="tab_scatter_3d">3D-Scatter</button>
						<button role="tab" aria-controls="tab_parallel">Parallel Plot</button>
						<button role="tab" aria-controls="tab_table">Results-Table</button>
						<button role="tab" aria-controls="tab_logs">Single Logs</button>
					</menu>

					<article role="tabpanel" id="tab_scatter_2d">
						<div id="scatter2d"></div>
					</article>

					<article role="tabpanel" hidden id="tab_scatter_3d">
						<div id="scatter3d"></div>
					</article>

					<article role="tabpanel" hidden id="tab_parallel">
						<div id="parallel"></div>
					</article>

					<article role="tabpanel" hidden id="tab_table">
						<div id="table"></div>
					</article>

					<article role="tabpanel" hidden id="tab_logs">

						<section class="tabs" style="width: 100%">
							<menu role="tablist" aria-label="Single-Runs">
								<button role="tab" aria-selected="true" aria-controls="single_run_1">Single-Run-1</button>
								<button role="tab" aria-controls="single_run_2">Single-Run-2</button>
							</menu>

							<article role="tabpanel" id="single_run_1">
								<pre>C:&#92;WINDOWS&#92;SYSTEM32> Single-Run 1</pre>
							</article>

							<article role="tabpanel" hidden id="single_run_2">
								<pre>C:&#92;WINDOWS&#92;SYSTEM32> Single-Run 2</pre>
							</article>
						</section>
					</article>
				</section>
			</div>
		</div>

		<script src="functions.js"></script>
		<script src="main.js"></script>
	</body>
</html>
