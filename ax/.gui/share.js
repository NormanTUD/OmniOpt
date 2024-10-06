"use strict";
var current_folder = ""

function parsePathAndGenerateLink(path) {
	// Define the regular expression to capture the different parts of the path
	var regex = /\/([^\/]+)\/?([^\/]*)\/?(\d+)?\/?$/;
	var match = path.match(regex);

	// Check if the path matches the expected format
	if (match) {
		var user = match[1] || '';
		var experiment = match[2] || '';
		var runNr = match[3] || '';


		// Construct the query string
		var queryString = 'share.php?user_id=' + encodeURIComponent(user);
		if (experiment) {
			queryString += '&experiment_name=' + encodeURIComponent(experiment);
		}
		if (runNr) {
			queryString += '&run_nr=' + encodeURIComponent(runNr);
		}

		return queryString;
	} else {
		console.error(`Invalid path format: ${path}, regex: {regex}`);
	}
}

function createBreadcrumb(currentFolderPath) {
	var breadcrumb = document.getElementById('breadcrumb');
	breadcrumb.innerHTML = '';

	var pathArray = currentFolderPath.split('/');
	var fullPath = '';

	var currentPath = "/Start/"

	pathArray.forEach(function(folderName, index) {
		if (folderName == ".") {
			folderName = "Start";
		}
		if (folderName !== '') {
			var originalFolderName = folderName;
			fullPath += originalFolderName + '/';

			var link = document.createElement('a');
			link.classList.add("breadcrumb_nav");
			link.classList.add("box-shadow");
			link.textContent = decodeURI(folderName);

			var parsedPath = "";

			if (folderName == "Start") {
				eval(`$(link).on("click", async function () {
						window.location.href = "share.php";
					});
				`);
			} else {
				currentPath += `/${folderName}`;
				parsedPath = parsePathAndGenerateLink(currentPath)

				eval(`$(link).on("click", async function () {
						window.location.href = parsedPath;
					});
				`);
			}

			breadcrumb.appendChild(link);

			breadcrumb.appendChild(document.createTextNode(' / '));
		}
	});
}
