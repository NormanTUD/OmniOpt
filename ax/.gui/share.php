<?php
	include("_header_base.php");
?>
	<div style="visibility: none" id="countdown"></div>
	<div id="loading_screen">
		<center>
			<br>
			Loading OmniOpt-Share...<br>
			<br>
			<div class="spinner"></div>
		</center>
	</div>
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

	function load_content() {
		var queryString = window.location.search;
		var requestUrl = 'share_internal.php' + queryString;

		$.ajax({
		url: requestUrl,
			method: 'GET',
			success: function(response) {
				if (response != last_load_content) {
					$('#share_main').html(response);
					$('#loading_screen').hide();
					$('#share_main').show();
					last_load_content = response;
				}
			},
			error: function() {
				console.error('Error loading the content.');
				$('#share_main').html('Error loading the requested content!').show();
			}
		});
	}

	function updateCountdown(interval) {
		var countdown = interval / 1000; // Interval in Sekunden
		$('#countdown').text('Next update in ' + countdown + ' seconds').show();

		countdownInterval = setInterval(function() {
			countdown--;
			if (countdown <= 0) {
				$('#countdown').hide();
				clearInterval(countdownInterval);
			} else {
				$('#countdown').text('Next update in ' + countdown + ' seconds');
			}
		}, 1000);
	}

	$(document).ready(function() {
		load_content();

		var updateInterval = getParameterByName('update_interval');

		if (updateInterval) {
			var interval = parseInt(updateInterval, 10) * 1000; // Umwandlung in Millisekunden
			updateCountdown(interval); // Startet den Countdown
			setInterval(function() {
				$('.spinner').show(); // Spinner anzeigen
				load_content(); // Ruft den Inhalt ab
				updateCountdown(interval); // Aktualisiert den Countdown
			}, interval);
		}
	});
</script>
