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
	$('html').css('filter', 'invert(1)');
	$('.invert_in_dark_mode').css('filter', 'invert(1)');
	$('img').css('filter', 'invert(1)');

	set_cookie("theme", "dark");

	$("#darkmode").prop("checked", true);
}

function enable_light_mode() {
	$('html').css('filter', '');
	$('.invert_in_dark_mode').css('filter', '');
	$('img').css('filter', 'invert(0)');

	set_cookie("theme", "light");

	$("#darkmode").prop("checked", false);
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
	var darkModeCheckbox = document.getElementById('darkmode');

	darkModeCheckbox.addEventListener('change', function() {
		if (darkModeCheckbox.checked) {
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
