"use strict";
$(document).ready(function() {
	apply_theme_based_on_system_preferences();

	document.addEventListener("keydown", function(event) {
		if (event.key === "Escape") {
			var deleteButton = document.getElementById("del_search_button");
			if (deleteButton && getComputedStyle(deleteButton).display !== "none") {
				delete_search();
			}
		}
	});

	var themeSelect = document.getElementById("themeSelect");

	var currentTheme = get_cookie("theme");
	if (currentTheme) {
		themeSelect.value = currentTheme;
		if (currentTheme === "dark") {
			enable_dark_mode();
		} else {
			enable_light_mode();
		}
	}

	themeSelect.addEventListener("change", function() {
		if (themeSelect.value === "dark") {
			enable_dark_mode();
		} else {
			enable_light_mode();
		}
	});

	// Listen for changes to the system color scheme preferences
	window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function(e) {
		if(!get_cookie("theme")) {
			if (e.matches) {
				enable_dark_mode();
			} else {
				enable_light_mode();
			}
		}
	});

	Prism.highlightAll();
	generateTOC();
});
