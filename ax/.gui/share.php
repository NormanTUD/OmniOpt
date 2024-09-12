<?php
	include("_header_base.php");
?>
		<div id="loading_screen">
			<center>
				<br>
				Loading OmniOpt-Share...<br>
				<br>
				<div class="spinner"></div>
			</center>
		</div>
		<div id="share_main" style="display: none">
			hidden
		</div>
	</div>
	<script>
		function load_content() {
			var queryString = window.location.search;

			var requestUrl = 'share_internal.php' + queryString;

			$.ajax({
			url: requestUrl,
				method: 'GET',
				success: function(response) {
					$('#share_main').html(response);
					$('#loading_screen').hide();
					$('#share_main').show();
				},
				error: function() {
					console.error('Error loading the content.');
					$('#loading_screen').hide();
					$('#share_main').html('Error loading the requested content!').show();
				}
			});
		}		

		$(document).ready(function() {
			load_content();
		});
	</script>
</body>
</html>
