var theme = "light";

function set_cookie(name, value, days = 365) {
	var expires = "";
	if (days) {
		var date = new Date();
		date.setTime(date.getTime() + days * 24 * 60 * 60 * 1000);
		expires = "; expires=" + date.toUTCString();
	}

	// Set SameSite and secure attributes
	var cookieOptions = "; SameSite=None; secure";

	document.cookie = name + "=" + (value || "") + expires + "; path=/" + cookieOptions;
}

function get_cookie(name) {
	var nameEQ = name + "=";
	var ca = document.cookie.split(";");
	for(var i=0;i < ca.length;i++) {
		var c = ca[i];
		while (c.charAt(0)==" ") c = c.substring(1,c.length);
		if (c.indexOf(nameEQ) == 0) return c.substring(nameEQ.length,c.length);
	}
	return null;
}

function enable_dark_mode() {
	$("#themeSelect").val("dark");

	$('html').css('filter', 'invert(1)');
	$('.invert_in_dark_mode').css('filter', 'invert(1)');
	$('img').css('filter', 'invert(1)');
	$('.share_graph').css('filter', 'invert(1)');
	$('.usage_plot').css('filter', 'invert(1)');

	set_cookie("theme", "dark");

	theme = "dark";
}

function enable_light_mode() {
	$("#themeSelect").val("light");

	$('html').css('filter', '');
	$('.invert_in_dark_mode').css('filter', '');
	$('img').css('filter', 'invert(0)');
	$('.share_graph').css('filter', 'invert(0)');
	$('.usage_plot').css('filter', 'invert(0)');

	set_cookie("theme", "light");

	theme = "light";
}

function apply_theme_based_on_system_preferences() {
	if(get_cookie("theme")) {
		if(get_cookie("theme") == "dark") {
			enable_dark_mode();
		} else {
			enable_light_mode();
		}
	} else {
		if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
			enable_dark_mode();
		} else {
			enable_light_mode();
		}
	}
}

document.addEventListener('DOMContentLoaded', function() {
	var themeSelect = document.getElementById('themeSelect');

	// Setze das aktuelle Thema beim Laden der Seite
	var currentTheme = get_cookie("theme");
	if (currentTheme) {
		themeSelect.value = currentTheme;
		if (currentTheme === "dark") {
			enable_dark_mode();
		} else {
			enable_light_mode();
		}
	}

	themeSelect.addEventListener('change', function() {
		if (themeSelect.value === 'dark') {
			enable_dark_mode();
		} else {
			enable_light_mode();
		}
	});
});

// Listen for changes to the system color scheme preferences
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function(e) {
	if(!get_cookie("theme")) {
		if (e.matches) {
			enable_dark_mode();
		} else {
			enable_light_mode();
		}
	}
});

$(document).ready(function() {
	apply_theme_based_on_system_preferences();
});
