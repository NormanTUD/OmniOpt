var log = console.log;

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
		var queryString = 'share.php?user=' + encodeURIComponent(user);
		if (experiment) {
			queryString += '&experiment=' + encodeURIComponent(experiment);
		}
		if (runNr) {
			queryString += '&run_nr=' + encodeURIComponent(runNr);
		}

		return queryString;
	} else {
		console.error(`Invalid path format: ${path}, regex: {regex}`);
	}
}

function adjustTextareaHeight(textarea) {
	try {
		textarea.style.height = 'auto'; // Setzt die Höhe zurück, damit die Scroll-Höhe korrekt ermittelt wird
		textarea.style.height = textarea.scrollHeight + 'px'; // Passt die Höhe basierend auf der Scroll-Höhe an
	} catch (error) {
		console.error("Fehler beim Anpassen der Textarea-Höhe:", error);
	}
}

/**
 * Findet alle Textareas auf der Seite und passt ihre Höhe basierend auf dem Inhalt an.
 */
function initializeTextareas() {
	const textareas = document.querySelectorAll('textarea');

	if (textareas.length === 0) {
		console.warn("Keine Textareas auf der Seite gefunden.");
		return;
	}

	textareas.forEach(textarea => {
		adjustTextareaHeight(textarea); // Initiale Anpassung

		// Setzt ein Event-Listener, um die Höhe bei der Eingabe dynamisch anzupassen
		textarea.addEventListener('input', () => adjustTextareaHeight(textarea));
	});
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

			// Füge ein Trennzeichen hinzu, außer beim letzten Element
			breadcrumb.appendChild(document.createTextNode(' / '));
		}
	});
}
