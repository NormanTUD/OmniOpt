<?php
	include("_header_base.php");
?>
	<link href="tutorial.css" rel="stylesheet" />
	<link href="jquery-ui.css" rel="stylesheet">
	<link href="prism.css" rel="stylesheet" />

	<h1>OmniOpt-Share</h1>
    
	<div id="toc"></div>

	<h2 id="what_is_oo_share">What is OmniOpt-Share?</h2>

	<p>OmniOpt-Share allows you to Share your results with others, online. You can simply submit a job by
	<code class="language-bash">./omniopt_share runs/my_experiment/0</code>. The program will upload the
	job to our server, and allow give you a link to it which is valid for 30 days.</p>

	<h2 class="privacy">Notes on Privacy</h2>
	<p>You can chose a random name to which OmniOpt-Share should call you. But remember: the data you upload
	is publically available for 30 days.</p>


	<script src="prism.js"></script>
	<script src="footer.js"></script>
</body>
</html>
