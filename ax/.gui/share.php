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

	var already_initialized_tables = [];
	var last_load_content = "";
	var last_hash = "";
	var countdownInterval;
	var currently_switching = false;

	var tab_ids = ["out_files_tabs", "main_tabbed"];

	var activeTabIndices = {};

	function initialize_tabs () {
		for (var i = 0; i < tab_ids.length; i++) {
			try {
				var tab_id = tab_ids[i];
				$("#" + tab_id).tabs();
			} catch (e) {
				warn(e);
			}
		}
	}

	function saveActiveTab() {
		for (var i = 0; i < tab_ids.length; i++) {
			try {
				var tab_id = tab_ids[i];
				var _active_tab = $("#" + tab_id).tabs().tabs("option", "active");

				if(typeof(_active_tab) == "number") {
					activeTabIndices[tab_id] = _active_tab;
				}
			} catch (e) {
				warn(e);
			}
		}
	}

	function restoreActiveTab() {
		for (var i = 0; i < tab_ids.length; i++) {
			var tab_id = tab_ids[i];

			if (Object.keys(activeTabIndices).includes(tab_id)) {
				var _saved_active_tab = activeTabIndices[tab_id];

				if(typeof(_saved_active_tab) == "number") {
					try {
						var _tab_id = "#" + tab_id;
						log(`Trying to set ${_tab_id} to ${_saved_active_tab}`);

						$(_tab_id).tabs("option", "active", _saved_active_tab);

						log(`Set ${_tab_id} to ${_saved_active_tab}`);
					} catch (e) {
						if(!("" + e).includes("cannot call methods on tabs prior to initialization")) {
							error(e);
						}
					}
				} else {
					log(`Error: _saved_active_tab is not an integer, but ${typeof(_saved_active_tab)}:`, _saved_active_tab);
				}
			}
		}
	}

	function getParameterByName(name) {
		var regex = new RegExp('[?&]' + encodeURIComponent(name) + '=([^&]*)');
		var results = regex.exec(window.location.search);
		return results === null ? '' : decodeURIComponent(results[1]);
	}

	async function load_content(msg) {
		while (currently_switching) {
			await sleep(10_000);
		}

		currently_switching = true;
		var queryString = window.location.search;
		var requestUrl = 'share_internal.php' + queryString;

		showSpinnerOverlay(msg);

		$.ajax({
		url: requestUrl,
			method: 'GET',
			success: async function(response) {
				saveActiveTab();
				if (response != last_load_content) {
					$('#share_main').html(response);
					last_load_content = response;
				}

				already_initialized_tables = [];
				$("[id*='autotable_']").remove();
				$(".toggle_raw_data").remove();

				initialize_autotables();
				restoreActiveTab();

				var urlParams = new URLSearchParams(window.location.search);

				if(urlParams.get("user_id") && urlParams.get("experiment_name") && !isNaN(parseInt(urlParams.get("run_nr")))) {
					await load_all_data();
				}

				initialize_tabs();

				$("#share_main").show();

				removeSpinnerOverlay();
				currently_switching = false;
			},
			error: function() {
				showSpinnerOverlay(msg);
				error('Error loading the content.');
				$('#share_main').html('Error loading the requested content!').show();
				removeSpinnerOverlay();
				currently_switching = false;
			}
		});
	}

	function getHashUrlContent(url) {
		var xhr = new XMLHttpRequest();
		xhr.open("GET", url, false);
		xhr.send();

		if (xhr.status === 200) {
			return xhr.responseText;
		} else {
			throw new Error("Error fetching URL: " + xhr.status);
		}
	}

	function fetchHashAndUpdateContent() {
		var urlParams = new URLSearchParams(window.location.search);

		if(urlParams.get("user_id") && urlParams.get("experiment_name") && !isNaN(parseInt(urlParams.get("run_nr")))) {
			if(currently_switching) {
				return;
			}

			var share_internal_url = window.location.toString();
			share_internal_url = share_internal_url.replace(/share\.php/, "share_internal.php");
			var end_sign = "&";
			if(share_internal_url.endsWith("share_internal.php")) {
				end_sign = "?";
			}
			var hashUrl = share_internal_url + end_sign + 'get_hash_only=1';

			var newHash = getHashUrlContent(hashUrl);

			if (newHash !== last_hash) {
				$("#refresh_button").text("Refresh (new data available)");
			}
		}
	}

	$(document).ready(function() {
		if($("#main_tabbed").length) {
			$("#main_tabbed").tabs();
		}

		load_content("Loading OmniOpt-Share...");

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

			if (auto_update) {
				var interval = 10000;
				setInterval(function() {
					fetchHashAndUpdateContent();
				}, interval);
			}
		}
	});

	showSpinnerOverlay("Initializing OmniOpt-Share...");
</script>
