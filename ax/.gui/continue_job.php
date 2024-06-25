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

	<p>This will start a new run with the same settings as the old one, but load all already tried out data points, and 
	continue the search from there.</p>

	<h2 id="continue_with_changed_options">Continue an old job with the changed options</h2>

	<p>In continued runs, some options can be changed. For example,</p>

	<pre><code class="language-bash">#!/bin/bash -l
./omniopt --continue runs/my_experiment/0 --time=360
</code></pre>

	<p>Will run the continuation for 6 hours ( = 360 minutes), indepenent of how long the old job ran.</p>

	<p>It is also possible to change parameter borders, though narrowing is not currently supported, i.e.
	the parameters need to be equal or wider than in the previous run. You cannot remove parameters here or add
	names of parameters that have previously not existed, though.</p>

	<p>Also, parameter names and types must stay the same. This code, for example, runs a continued run but changes
	the <tt>epochs</tt> parameter that was previously defined and could have been, for example, between 0 and 10. In
	this new run, all old values will be considered, but new values of epochs can be between 0 and 1000.</p>

	<pre><code class="language-bash">#!/bin/bash -l
./omniopt --continue runs/my_experiment/0 --parameter epochs range 0 1000 int
</code></pre>

	<p>All parameters that are not specified here are taken out of the old run, and thus stay in the same borders.</p>

	<script src="prism.js"></script>
	<script>
		Prism.highlightAll();
	</script>
	<script src="footer.js"></script>
</body>
</html>

