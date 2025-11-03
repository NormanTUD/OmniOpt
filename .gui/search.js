"use strict";
var searchTimer;
var lastSearch = "";
var lastResultsHash = "";

async function start_search() {
	try {
		var searchTerm = $("#search").val();

		if (searchTerm === lastSearch) {
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
			try {
				abortPreviousRequest();

				if (!/^\s*$/.test(searchTerm)) {
					showSpinnerOverlay("Searching...");
					$("#delete_search").show();
					$("#searchResults").show();
					$("#mainContent").hide();

					var response = await new Promise(function(resolve, reject) {
						$.ajax({
							url: "search.php",
							type: "GET",
							data: { regex: searchTerm },
							dataType: "json",
							success: function(data) {
								resolve(data);
							},
							error: function(xhr, status, error) {
								reject(new Error(error));
							}
						});
					});

					var jsonString = JSON.stringify(response);
					var currentHash = md5(jsonString);

					if (currentHash !== lastResultsHash) {
						lastResultsHash = currentHash;
						await displaySearchResults(searchTerm, response);
					}

					removeSpinnerOverlay();
				} else {
					$("#delete_search").hide();
					$("#searchResults").hide();
					$("#mainContent").show();
					lastResultsHash = "";
				}
			} catch (error) {
				console.error("Error during search:", error);
				removeSpinnerOverlay();
			}
		}

		searchTimer = setTimeout(performSearch, 10);

		if (searchTerm.length) {
			$("#del_search_button").show();
		} else {
			$("#del_search_button").hide();
		}
	} catch (error) {
		console.error("Error in start_search:", error);
	}
}

function delete_search () {
	$("#search").val("").trigger("change").blur();
}

function mark_search_result_yellow(content, search) {
	try {
		var escapedSearch = search.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
		var regex = new RegExp("(" + escapedSearch + ")", "gi");
		return content.replace(regex, "<span class='marked_text invert_in_dark_mode'>$1</span>");
	} catch (error) {
		console.error("Error while marking search results: ", error);
		return content;
	}
}

function get_category_icon(category) {
	var icons = {
		"Tutorials": "<img class='emoji_nav' src='emojis/books.svg' />",
		"Shares": "<img class='emoji_nav' src='emojis/world.svg' />",
		"Default": "<img class='emoji_nav' src='emojis/page.svg' />"
	};
	return icons[category] || icons["Default"];
}

function replace_backticks_with_tt(str) {
	let result = '';
	let i = 0;
	while (i < str.length) {
		if (str[i] === '`') {
			let end = str.indexOf('`', i + 1);
			if (end !== -1) {
				result += `<tt>${str.slice(i + 1, end)}</tt>`;
				i = end + 1;
			} else {
				result += str[i];
				i++;
			}
		} else {
			result += str[i];
			i++;
		}
	}

	return result;
}

async function displaySearchResults(searchTerm, results) {
        var $searchResults = $("#searchResults");
        $searchResults.empty();

	if (
		Object.keys(results).some(function (category) {
			return Array.isArray(results[category]) && results[category].length > 0;
		})
	) {
		$searchResults.append(
			"<h2>Search results</h2>\n<p>To get back to the original page, clear the search or press Escape.</p>"
		);

		Object.keys(results).forEach(function (category) {
			var entries = results[category];
			if (entries.length > 0) {
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
							headlineHtml = `<div class="search_headline">${replace_backticks_with_tt(headline)}</div>\n`;
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

	apply_theme_based_on_system_preferences();
}
