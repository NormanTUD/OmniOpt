"use strict";
var theme = "light";

function invertColor(color) {
	var rgb = color.match(/\d+/g);
	if (rgb) {
		var r = 255 - parseInt(rgb[0]);
		var g = 255 - parseInt(rgb[1]);
		var b = 255 - parseInt(rgb[2]);
		return 'rgb(' + r + ',' + g + ',' + b + ')';
	}
	return color;
}

function set_table_headers_to_dark_mode () {
	$('th').each(function() {
		var currentColor = $(this).css('background-color');
		var invertedColor = invertColor(currentColor);
		$(this)[0].style.setProperty('background-color', invertedColor, 'important');
	});
}

function set_table_headers_to_light_mode () {
	$('th').each(function() {
		$(this)[0].style.setProperty('background-color', '#00015a', 'important');
	});
}

function set_cookie(name, value, days = 365) {
	var expires = "";
	if (days) {
		var date = new Date();
		date.setTime(date.getTime() + days * 24 * 60 * 60 * 1000);
		expires = "; expires=" + date.toUTCString();
	}

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

function updateTextColorIfUsageStatsInURL() {
	var color = "black";

	if (theme == "dark") {
		color = "white";
	}

	var url = window.location.href;

	if (typeof url === "string" && url.indexOf("usage_stats") !== -1) {
		$("text").css("fill", color);
	}
}

function enable_dark_mode(set_theme_cookie=true) {
	$("#themeSelect").val("dark");

	$('#oologo').attr('src','logo_black.png');

	$('#main_window pre:not(.invert_in_dark_mode)').addClass('invert_in_dark_mode');
	$('#main_window th:not(.invert_in_dark_mode)').addClass('invert_in_dark_mode');
	$('#main_window .gridjs-th:not(.invert_in_dark_mode)').addClass('invert_in_dark_mode');

	$("html").css("filter", "invert(1)");
	$(".invert_in_dark_mode").css("filter", "invert(1)");
	$(".caveat").css("filter", "invert(1)");
	$("img").not($(".invert_in_dark_mode img")).css("filter", "invert(1)");
	$(".invert_icon").css("filter", "invert(0)");
	$(".share_graph").css("filter", "invert(1)");
	$(".usage_plot").css("filter", "invert(1)");

	if(set_theme_cookie) {
		set_cookie("theme", "dark");
	}

	theme = "dark";

	$("body").css("color", "#6F116F");
	$("body").css("background-color", "white");
	$(".header_table").css("color", "green");
	$(".window").css("background", "unset");

	$(".mode-text").text("Dark-Mode");

	if (typeof make_text_in_parallel_plot_nicer === "function") {
		make_text_in_parallel_plot_nicer();
	}

	$(".mode-text").css("color", "black")

	updateImagesForTheme();

	$("text").css("fill", "white")

	updateTextColorIfUsageStatsInURL();
}

function enable_light_mode(set_theme_cookie=true) {
	$("#themeSelect").val("light");

	$('#oologo').attr('src','logo_white.png');

	$("html").css("filter", "");
	$(".invert_in_dark_mode").css("filter", "");
	$(".caveat").css("filter", "");
	$("img").css("filter", "invert(0)");
	$(".invert_icon").css("filter", "invert(0)");
	$(".share_graph").css("filter", "invert(0)");
	$(".usage_plot").css("filter", "invert(0)");

	if(set_theme_cookie) {
		set_cookie("theme", "light");
	}

	theme = "light";

	$("body").css("color", "unset");
	$("body").css("background-color", "#fafafa");
	$(".header_table").css("color", "unset");
	$(".window").css("background", "#ece9d8");

	$(".mode-text").text("Light-Mode");

	if (typeof make_text_in_parallel_plot_nicer === "function") {
		make_text_in_parallel_plot_nicer();
	}

	$(".mode-text").css("color", "black")

	updateImagesForTheme();

	updateTextColorIfUsageStatsInURL();
}

function apply_theme_based_on_system_preferences() {
	wrapEmojisInSpans();

	if(get_cookie("theme")) {
		if(get_cookie("theme") == "dark") {
			enable_dark_mode();
		} else {
			enable_light_mode();
		}
	} else {
		if (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) {
			enable_dark_mode();
		} else {
			enable_light_mode();
		}
	}
}

function updateImagesForTheme() {
	var imgs = document.querySelectorAll('img[data-lightsrc][data-darksrc]');
	imgs.forEach(function(img) {
		var src = theme === 'dark' ? img.getAttribute('data-darksrc') : img.getAttribute('data-lightsrc');
		if (img.src !== src) img.src = src;
	});
}

document.addEventListener("DOMContentLoaded", function() {
	var themeSelect = document.getElementById("themeSelect");

	themeSelect.checked = (theme === "dark");

	themeSelect.addEventListener("change", function() {
		if (themeSelect.checked) {
			enable_dark_mode();
		} else {
			enable_light_mode();
		}
	});
});

window.addEventListener('beforeprint', () => {
	enable_light_mode(false);
});

window.addEventListener('afterprint', () => {
	apply_theme_based_on_system_preferences();
});
