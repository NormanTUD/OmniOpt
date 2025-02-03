<?php
        require "_header_base.php";
?>
	<script src='plotly-latest.min.js'></script>
	<script src='share.js'></script>
	<script src='parallel.js'></script>
	<script src='share_graphs.js'></script>
	<link href="share.css" rel="stylesheet" />
        <div id="share_main" style="display: none"></div>

</div>
<script>
	alert = console.error;

	$(document).ready(function() {
		if($("#main_tabbed").length) {
			$("#main_tabbed").tabs();
		}

		load_content("Loading OmniOpt2-Share...");

		var share_internal_url = window.location.toString();
		share_internal_url = share_internal_url.replace(/share\.php/, "share_internal.php");
		var end_sign = "&";
		if(share_internal_url.endsWith("share_internal.php")) {
			end_sign = "?";
		}
		var hashUrl = share_internal_url + end_sign + 'get_hash_only=1';

		var urlParams = new URLSearchParams(window.location.search);

		if(urlParams.get("user_id") && urlParams.get("experiment_name") && !isNaN(parseInt(urlParams.get("run_nr")))) {
			last_hash = getHashUrlContent(hashUrl);

			var auto_update = getParameterByName('update');

			var interval = 2000;
			setInterval(function() {
				fetchHashAndUpdateContent();
			}, interval);
		}
	});

	showSpinnerOverlay("Initializing OmniOpt2-Share...");
</script>
