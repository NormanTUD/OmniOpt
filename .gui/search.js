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
                $searchResults.append(
                        "<h2>Search results:</h2>\n<p>To get back to the original page, clear the search or press Escape.</p>"
                );

                Object.keys(results).forEach(function (category) {
                        var entries = results[category];
                        if (entries.length > 0) {
                                // Gruppieren nach headline
                                var groupedByHeadline = {};

                                entries.forEach(function (entry) {
                                        var headlineKey = (entry.headline && typeof entry.headline === "string" && entry.headline.trim() !== "")
                                                ? entry.headline.trim()
                                                : "_no_headline_";

                                        if (!(headlineKey in groupedByHeadline)) {
                                                groupedByHeadline[headlineKey] = [];
                                        }

                                        groupedByHeadline[headlineKey].push(entry);
                                });

                                var blocks = [];

                                Object.keys(groupedByHeadline).forEach(function (headline) {
                                        var group = groupedByHeadline[headline];
                                        var itemLines = [];

                                        group.forEach(function (result) {
                                                try {
							if(category.match(/Shares/)) {
								var itemLine = `<li>${result.content}</li>`;
								itemLines.push(itemLine);
							} else {
								var markedContent = mark_search_result_yellow(result.content, searchTerm);
								var itemLine = `<li><a href="${result.link}">${markedContent}</a></li>`;
								itemLines.push(itemLine);
							}
                                                } catch (err) {
                                                        console.error("Error creating result line: ", err);
                                                        console.trace();
                                                }
                                        });

                                        if (itemLines.length > 0) {
                                                var headlineHtml = "";
                                                if (headline !== "_no_headline_") {
                                                        headlineHtml = `<div class="search_headline">${headline}</div>\n`;
                                                }

                                                var listHtml = `<ul>\n${itemLines.join("\n")}\n</ul>`;
                                                blocks.push(headlineHtml + listHtml);
                                        }
                                });

                                if (blocks.length > 0) {
                                        var icon = get_category_icon(category);
                                        var heading = `<h3>${icon} ${category} (${entries.length})</h3>`;
                                        $searchResults.append(heading + "\n" + blocks.join("\n"));
                                }
                        }
                });
        } else {
                $searchResults.append("<p>No results found.</p>");
        }

        MathJax.typeset();
}
