function enable_dark_mode() {
	$('html').css('filter', 'invert(1)');
	$('.invert_in_dark_mode').css('filter', 'invert(1)');
}

function enable_light_mode() {
	$('html').css('filter', '');
	$('.invert_in_dark_mode').css('filter', '');
}

function apply_theme_based_on_system_preferences() {
	if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
		enable_dark_mode();
	} else {
		enable_light_mode();
	}
}

// Apply the theme when the page loads
$(document).ready(function() {
});

// Listen for changes to the system color scheme preferences
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function(e) {
	if (e.matches) {
		enable_dark_mode();
	} else {
		enable_light_mode();
	}
});

$(document).ready(function() {
	apply_theme_based_on_system_preferences();
});
