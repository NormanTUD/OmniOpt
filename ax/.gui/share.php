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
		$(document).ready(function() {
			// Die aktuellen URL-Parameter holen
			var queryString = window.location.search;

			// Die URL für die AJAX-Anfrage erstellen, indem die aktuellen Parameter angehängt werden
			var requestUrl = 'share_internal.php' + queryString;

			// AJAX-Anfrage starten, um den Inhalt von 'share_internal.php' zu laden
			$.ajax({
			url: requestUrl,
				method: 'GET',
				success: function(response) {
					// Den Inhalt von 'share_internal.php' in das div #share_main schreiben
					$('#share_main').html(response);

					// Das div loading_screen verstecken
					$('#loading_screen').hide();

					// Das div #share_main anzeigen
					$('#share_main').show();
				},
				error: function() {
					// Fehlerbehandlung, falls die Anfrage fehlschlägt
					console.error('Fehler beim Laden des Inhalts.');
				}
			});
		});
	</script>
</body>
</html>
