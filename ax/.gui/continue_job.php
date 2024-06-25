<?php
	include("_header_base.php");
?>
	<link href="tutorial.css" rel="stylesheet" />
	<link href="jquery-ui.css" rel="stylesheet">
	<link href="prism.css" rel="stylesheet" />

	<h1>Continue jobs</h1>
    
	<div id="toc"></div>

	<h2 id="continue_with_same_options">Continue an old job with the same options</h2>
	<p>Continueing an old job with the same options as previously, but with awareness of the hyperparameter-constellations
	that have already been tested, is as simple as this, assuming your job is in <tt>runs/my_experiment/0</tt>:

	<pre><code class="language-bash">#!/bin/bash -l
./omniopt --continue runs/my_experiment/0
</code></pre>

	This will start a new run with the same settings as the old one, but load all already tried out data points, and 
	continue the search from there.

	<script src="prism.js"></script>
	<script>
		Prism.highlightAll();
	</script>
	<script src="footer.js"></script>
</body>
</html>

