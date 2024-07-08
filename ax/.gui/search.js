var log = console.log;
var l = log;

var searchTimer; // Globale Variable für den Timer
var lastSearch = "";

async function start_search() {
	var searchTerm = $('#search').val();

	if(searchTerm == lastSearch) {
		return;
	}

	lastSearch = searchTerm;

	// Funktion zum Abbrechen der vorherigen Suchanfrage
	function abortPreviousRequest() {
		if (searchTimer) {
			clearTimeout(searchTimer);
			searchTimer = null;
		}
	}

	abortPreviousRequest();

	// Funktion zum Durchführen der Suchanfrage
	async function performSearch() {
		// Abbrechen der vorherigen Anfrage, falls vorhanden
		abortPreviousRequest();

		if (!/^\s*$/.test(searchTerm)) {
			$("#delete_search").show();
			$("#searchResults").show();
			$("#mainContent").hide();
			$.ajax({
			url: 'search.php',
				type: 'GET',
				data: { regex: searchTerm },
				success: async function (response) {
					await displaySearchResults(searchTerm, response);
				},
				error: function (xhr, status, error) {
					console.error(error);
				}
			});
		} else {
			$("#delete_search").hide();
			$("#searchResults").hide();
			$("#mainContent").show();
		}
	}

	// Starten der Suche nach 10 ms Verzögerung
	searchTimer = setTimeout(performSearch, 10);

	if(searchTerm.length) {
		$("#del_search_button").show();
	} else {
		$("#del_search_button").hide();
	}
}

function delete_search () {
	$("#search").val("").trigger("change");
}

function mark_search_result_yellow(content, search) {
	try {
		// Escape the search term to safely use it in a regular expression
		var escapedSearch = search.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
		var regex = new RegExp("(" + escapedSearch + ")", 'gi');
		return content.replace(regex, "<span class='marked_text'>$1</span>");
	} catch (error) {
		console.error("Error while marking search results: ", error);
		return content; // return original content if there's an error
	}
}

// Funktion zur Anzeige der Suchergebnisse
async function displaySearchResults(searchTerm, results) {
	var $searchResults = $('#searchResults');
	$searchResults.empty();


	if (results.length > 0) {
		$searchResults.append('<h2>Search results:</h2>\n<p>To get back back to the original page, clear the search.</p>');

		var result_lis = [];

		results.forEach(function(result) {
			var result_line = `<li><a onclick='delete_search()' href="${result.link}">${mark_search_result_yellow(result.content, searchTerm)}</a></li>`;
			result_lis.push(result_line);

		});

		if(result_lis.length) {
			$searchResults.append("<ul>\n" + result_lis.join("\n") + "</ul>");
		}
	} else {
		$searchResults.append('<p>No results found.</p>');
	}
}
