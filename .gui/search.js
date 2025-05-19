"use strict";
var searchTimer;
var lastSearch = "";

async function start_search() {
	var searchTerm = $("#search").val();

	if(searchTerm == lastSearch) {
		return;
	}

	lastSearch = searchTerm;

	function abortPreviousRequest() {
		if (searchTimer) {
			clearTimeout(searchTimer);
			searchTimer = null;
		}
	}

	abortPreviousRequest();

	async function performSearch() {
		abortPreviousRequest();

		if (!/^\s*$/.test(searchTerm)) {
			showSpinnerOverlay("Searching...");
			$("#delete_search").show();
			$("#searchResults").show();
			$("#mainContent").hide();
			$.ajax({
				url: "search.php",
				type: "GET",
				data: {
					regex: searchTerm
				},
				success: async function (response) {
					await displaySearchResults(searchTerm, response);
					removeSpinnerOverlay();
				},
				error: function (xhr, status, error) {
					console.error(error);
					removeSpinnerOverlay();
				}
			});
		} else {
			$("#delete_search").hide();
			$("#searchResults").hide();
			$("#mainContent").show();
		}
	}

	searchTimer = setTimeout(performSearch, 10);

	if(searchTerm.length) {
		$("#del_search_button").show();
	} else {
		$("#del_search_button").hide();
	}
}

function delete_search () {
	$("#search").val("").trigger("change").blur();
}

function mark_search_result_yellow(content, search) {
	try {
		// Escape the search term to safely use it in a regular expression
		var escapedSearch = search.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
		var regex = new RegExp("(" + escapedSearch + ")", "gi");
		return content.replace(regex, "<span class='marked_text'>$1</span>");
	} catch (error) {
		console.error("Error while marking search results: ", error);
		return content; // return original content if there's an error
	}
}

function get_category_icon(category) {
	var icons = {
		"Tutorials": "📚",
		"Shares": "🌏",
		"Default": "📄"
	};
	return icons[category] || icons["Default"];
}

async function displaySearchResults(searchTerm, results) {
	var $searchResults = $("#searchResults");
	$searchResults.empty();

	if (Object.keys(results).length > 0) {
		$searchResults.append("<h2>Search results:</h2>\n<p>To get back to the original page, clear the search or press Escape.</p>");

		Object.keys(results).forEach(function (category) {
			var entries = results[category];
			if (entries.length > 0) {
				var result_lis = [];

				entries.forEach(function (result) {
					try {
						var markedContent = mark_search_result_yellow(result.content, searchTerm);
						var result_line = `<li><a onclick='delete_search()' href="${result.link}">${markedContent}</a></li>`;
						result_lis.push(result_line);
					} catch (err) {
						console.error("Error creating result line: ", err);
					}
				});

				if (result_lis.length) {
					var icon = get_category_icon(category);
					var heading = `<h3>${icon} ${category} (${entries.length})</h3>`;
					var list = `<ul>\n${result_lis.join("\n")}</ul>`;
					$searchResults.append(heading + "\n" + list);
				}
			}
		});
	} else {
		$searchResults.append("<p>No results found.</p>");
	}
}
