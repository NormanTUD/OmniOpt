"use strict";

var current_folder = "";

function parsePathAndGenerateLink(path) {
	// Define the regular expression to capture the different parts of the path
	var regex = /\/([^\/]+)\/?([^\/]*)\/?(\d+)?\/?$/;
	var match = path.match(regex);

	// Check if the path matches the expected format
	if (match) {
		var user = match[1] || "";
		var experiment = match[2] || "";
		var runNr = match[3] || "";


		// Construct the query string
		var queryString = "share?user_id=" + encodeURIComponent(user);
		if (experiment) {
			queryString += "&experiment_name=" + encodeURIComponent(experiment);
		}
		if (runNr) {
			queryString += "&run_nr=" + encodeURIComponent(runNr);
		}

		return queryString;
	} else {
		console.error(`Invalid path format: ${path}, regex: {regex}`);
	}
}

function createBreadcrumb(currentFolderPath) {
	var breadcrumb = document.getElementById("breadcrumb");
	breadcrumb.innerHTML = "";

	var pathArray = currentFolderPath.split("/");
	var fullPath = "";

	var currentPath = "/Start/";

	pathArray.forEach(function(folderName, index) {
		if (folderName == ".") {
			folderName = "Start";
		}
		if (folderName !== "") {
			var originalFolderName = folderName;
			fullPath += originalFolderName + "/";

			var link = document.createElement("a");
			link.classList.add("breadcrumb_nav");
			link.classList.add("box-shadow");
			link.textContent = decodeURI(folderName);

			var parsedPath = "";

			if (folderName == "Start") {
				eval(`$(link).on("click", async function () {
						window.location.href = "share";
					});
				`);
			} else {
				currentPath += `/${folderName}`;
				parsedPath = parsePathAndGenerateLink(currentPath);

				eval(`$(link).on("click", async function () {
						window.location.href = parsedPath;
					});
				`);
			}

			breadcrumb.appendChild(link);

			breadcrumb.appendChild(document.createTextNode(" / "));
		}
	});
}

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
	if (!loaded_share) {
		return;
	}
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

		try {
			var newHash = getHashUrlContent(hashUrl);

			if (newHash !== last_hash) {
				$("#refresh_button").text("Refresh (new data available)");
			}
		} catch (e) {
			if (!("" + e).includes("A network error occured")) {
				error("" + e);
			}
		} 
	}
}
