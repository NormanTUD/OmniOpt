<?php
	include("_header_base.php");
?>
	<div style="visibility: hidden" id="countdown"></div>
	<div id="share_main" style="display: none">
	</div>
</div>
<script>
	var last_load_content = "";
	var countdownInterval;

	function getParameterByName(name) {
		var regex = new RegExp('[?&]' + encodeURIComponent(name) + '=([^&]*)');
		var results = regex.exec(window.location.search);
		return results === null ? '' : decodeURIComponent(results[1]);
	}

	function load_content(msg) {
		var queryString = window.location.search;
		var requestUrl = 'share_internal.php' + queryString;

		showSpinnerOverlay(msg);

		$.ajax({
			url: requestUrl,
			method: 'GET',
			success: function(response) {
				if (response != last_load_content) {
					$('#share_main').html(response).show();
					last_load_content = response;
				}
				removeSpinnerOverlay();
			},
			error: function() {
				showSpinnerOverlay(msg);
				console.error('Error loading the content.');
				$('#share_main').html('Error loading the requested content!').show();
				removeSpinnerOverlay();
			}
		});
	}

	function updateCountdown(interval) {
		var countdown = interval / 1000; // Interval in Sekunden
		$('#countdown').text('Next update in ' + countdown + ' seconds').show();

		countdownInterval = setInterval(function() {
			countdown--;
			if (countdown <= 0) {
				$('#countdown').css("visibility", "hidden");
				clearInterval(countdownInterval);
			} else {
				$('#countdown').text('Next update in ' + countdown + ' seconds').css("visibility", "visible");
			}
		}, 1000);
	}

	$(document).ready(function() {
		load_content("Loading OmniOpt-Share...");

		var updateInterval = getParameterByName('update_interval');

		if (updateInterval) {
			var interval = parseInt(updateInterval, 10) * 1000; // Umwandlung in Millisekunden
			updateCountdown(interval);
			setInterval(function() {
				load_content("Reloading content...");
				updateCountdown(interval);
			}, interval);
		}
	});
</script>
