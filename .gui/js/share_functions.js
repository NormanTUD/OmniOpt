"use strict";

var last_resize_width = 0;

function add_default_layout_data (layout, no_height = 0) {
	layout["width"] = get_graph_width();
	if (!no_height) {
		layout["height"] = get_graph_height();
	}
	layout["paper_bgcolor"] = 'rgba(0,0,0,0)';
	layout["plot_bgcolor"] = 'rgba(0,0,0,0)';

	return layout;
}

function get_marker_size() {
	return 12;
}

function get_text_color() {
	return theme == "dark" ? "white" : "black";
}

function get_font_size() {
	return 14;
}

function get_graph_height() {
	return 800;
}

function get_font_data() {
	return {
		size: get_font_size(),
		color: get_text_color()
	}
}

function get_axis_title_data(name, axis_type = "") {
	if(axis_type) {
		return {
			text: name,
			type: axis_type,
			font: get_font_data()
		};
	}

	return {
		text: name,
		font: get_font_data()
	};
}

function get_graph_width() {
	var width = document.body.clientWidth || window.innerWidth || document.documentElement.clientWidth;
	return Math.max(800, Math.floor(width * 0.9));
}

function createTable(data, headers, table_name) {
	if (!$("#" + table_name).length) {
		console.error("#" + table_name + " not found");
		return;
	}

	new gridjs.Grid({
		columns: headers,
		data: data,
		search: true,
		sort: true,
		ellipsis: false
	}).render(document.getElementById(table_name));

	if (typeof apply_theme_based_on_system_preferences === 'function') {
		apply_theme_based_on_system_preferences();
	}

	colorize_table_entries();

	add_colorize_to_gridjs_table();
}

function download_as_file(id, filename) {
	var text = $("#" + id).text();
	var blob = new Blob([text], {
		type: "text/plain"
	});
	var link = document.createElement("a");
	link.href = URL.createObjectURL(blob);
	link.download = filename;
	document.body.appendChild(link);
	link.click();
	document.body.removeChild(link);
}

function copy_to_clipboard_from_id (id) {
	var text = $("#" + id).text();

	copy_to_clipboard(text);
}

function copy_to_clipboard(text) {
	if (!navigator.clipboard) {
		let textarea = document.createElement("textarea");
		textarea.value = text;
		document.body.appendChild(textarea);
		textarea.select();
		try {
			document.execCommand("copy");
		} catch (err) {
			console.error("Copy failed:", err);
		}
		document.body.removeChild(textarea);
		return;
	}

	navigator.clipboard.writeText(text).then(() => {
		console.log("Text copied to clipboard");
	}).catch(err => {
		console.error("Failed to copy text:", err);
	});
}

function filterNonEmptyRows(data) {
	var new_data = [];
	for (var row_idx = 0; row_idx <  data.length; row_idx++) {
		var line = data[row_idx];

		var line_has_empty_data = false;

		for (var col_idx = 0; col_idx < line.length; col_idx++) {
			var col_header_name = tab_results_headers_json[col_idx];
			var single_data_point = line[col_idx];

			if(single_data_point === "" && !special_col_names.includes(col_header_name)) {
				line_has_empty_data = true;
				continue;
			}
		}

		if(!line_has_empty_data) {
			new_data.push(line);
		}
	}

	return new_data;
}

function make_text_in_parallel_plot_nicer() {
	$(".parcoords g > g > text").each(function() {
		if (theme == "dark") {
			$(this)
				.css("text-shadow", "unset")
				.css("font-size", "0.9em")
				.css("fill", "white")
				.css("stroke", "black")
				.css("stroke-width", "2px")
				.css("paint-order", "stroke fill");
		} else {
			$(this)
				.css("text-shadow", "unset")
				.css("font-size", "0.9em")
				.css("fill", "black")
				.css("stroke", "unset")
				.css("stroke-width", "unset")
				.css("paint-order", "stroke fill");
		}
	});
}

function load_log_file(log_nr, filename) {
	var pre_id = `single_run_${log_nr}_pre`;

	if (!$("#" + pre_id).data("loaded")) {
		const params = new URLSearchParams(window.location.search);

		const user_id = params.get('user_id');
		const experiment_name = params.get('experiment_name');
		const run_nr = params.get('run_nr');

		var url = `get_log?user_id=${user_id}&experiment_name=${experiment_name}&run_nr=${run_nr}&filename=${filename}`;

		fetch(url)
			.then(response => response.json())
			.then(data => {
				if (data.data) {
					$("#" + pre_id).html(data.data);
					$("#" + pre_id).data("loaded", true);
				} else {
					log(`No 'data' key found in response.`);
				}

				$("#spinner_log_" + log_nr).remove();
			})
			.catch(error => {
				log(`Error loading log: ${error}`);
				$("#spinner_log_" + log_nr).remove();
			});
	}
}

function load_debug_log () {
	var pre_id = `here_debuglogs_go`;

	if (!$("#" + pre_id).data("loaded")) {
		const params = new URLSearchParams(window.location.search);

		const user_id = params.get('user_id');
		const experiment_name = params.get('experiment_name');
		const run_nr = params.get('run_nr');

		var url = `get_debug_log?user_id=${user_id}&experiment_name=${experiment_name}&run_nr=${run_nr}`;

		fetch(url)
			.then(response => response.json())
			.then(data => {
				$("#debug_log_spinner").remove();

				if (data.data) {
					try {
						$("#" + pre_id).html(data.data);
					} catch (err) {
						$("#" + pre_id).text(`Error loading data: ${err}`);
					}

					$("#" + pre_id).data("loaded", true);

					if (typeof apply_theme_based_on_system_preferences === 'function') {
						apply_theme_based_on_system_preferences();
					}
				} else {
					log(`No 'data' key found in response.`);
				}
			})
			.catch(error => {
				log(`Error loading log: ${error}`);
				$("#debug_log_spinner").remove();
			});
	}
}


function _colorize_table_entries_by_generation_method () {
	document.querySelectorAll('[data-column-id="generation_node"]').forEach(el => {
		let text = el.textContent.toLowerCase();
		let color = text.includes("manual") ? "green" :
			text.includes("sobol") ? "orange" :
			text.includes("saasbo") ? "pink" :
			text.includes("uniform") ? "lightblue" :
			text.includes("legacy_gpei") ? "sienna" :
			text.includes("bo_mixed") ? "aqua" :
			text.includes("randomforest") ? "darkseagreen" :
			text.includes("external_generator") ? "purple" :
			text.includes("botorch") ? "yellow" : "";
		if (color !== "") {
			el.style.backgroundColor = color;
		}
		el.classList.add("invert_in_dark_mode");
	});
}

function _colorize_table_entries_by_trial_status () {
	document.querySelectorAll('[data-column-id="trial_status"]').forEach(el => {
		let color = el.textContent.includes("COMPLETED") ? "lightgreen" :
			el.textContent.includes("RUNNING") ? "orange" :
			el.textContent.includes("FAILED") ? "red" :
			el.textContent.includes("CANDIDATE") ? "lightblue" :
			el.textContent.includes("ABANDONED") ? "yellow" : "";
		if (color) el.style.backgroundColor = color;
		el.classList.add("invert_in_dark_mode");
	});
}

function _colorize_table_entries(configs) {
	configs.forEach(cfg => {
		let cells = [...document.querySelectorAll(cfg.selector)];
		if (cells.length === 0) return;

		let rawValues = cells.map(el => parseFloat(el.textContent));
		if (cfg.filter) rawValues = rawValues.filter(cfg.filter);
		else rawValues = rawValues.filter(v => !isNaN(v));

		if (rawValues.length === 0) return;

		if (cfg.type === "categorical") {
			let unique = [...new Set(cells.map(el => el.textContent.trim()))];
			let colorMap = {};
			unique.forEach((v, i) => {
				let hue = Math.round((360 / unique.length) * i);
				colorMap[v] = `hsl(${hue}, 70%, 60%)`;
			});
			cells.forEach(el => {
				let v = el.textContent.trim();
				if (colorMap[v]) {
					el.style.backgroundColor = colorMap[v];
					el.classList.add("invert_in_dark_mode");
				}
			});
			return;
		}

		let values = rawValues;
		if (cfg.type === "log") values = values.map(v => Math.log(v));

		let min = Math.min(...values);
		let max = Math.max(...values);
		let range = max - min || 1;

		cells.forEach(el => {
			let value = parseFloat(el.textContent);
			if (isNaN(value)) return;
			if (cfg.type === "log" && value <= 0) return;

			let val = cfg.type === "log" ? Math.log(value) : value;
			let ratio = (val - min) / range;
			if (cfg.invert) ratio = 1 - ratio;

			let red = Math.round(255 * ratio);
			let green = Math.round(255 * (1 - ratio));

			el.style.backgroundColor = `rgb(${red}, ${green}, 0)`;
			el.classList.add("invert_in_dark_mode");
		});
	});
}

function _apply_colorization() {
	_colorize_table_entries([
		{ selector: '[data-column-id="queue_time"]', type: "linear" },
		{ selector: '[data-column-id="run_time"]', type: "linear" },
		...result_names.map((name, i) => ({
			selector: `[data-column-id="${name}"]`,
			type: "log",
			filter: v => v > 0 && !isNaN(v),
			invert: result_min_max[i] === "max"
		})),
		{ selector: '[data-column-id="hostname"]:not(.gridjs-th)', type: "categorical" },
		{ selector: '[data-column-id="generation_node"]:not(.gridjs-th)', type: "categorical" }
	]);
}

function colorize_table_entries() {
	setTimeout(() => {
		if (typeof result_names !== "undefined" && Array.isArray(result_names) && result_names.length > 0) {
			_colorize_table_entries_by_trial_status();

			_colorize_table_entries([
				{ selector: '[data-column-id="queue_time"]', type: "linear" },
				{ selector: '[data-column-id="run_time"]', type: "linear" },
				...result_names.map((name, i) => ({
					selector: `[data-column-id="${name}"]`,
					type: "log",
					filter: v => v > 0 && !isNaN(v),
					invert: result_min_max[i] === "max"
				})),
				{ selector: '[data-column-id="hostname"]:not(.gridjs-th)', type: "categorical" },
				{ selector: '[data-column-id="generation_node"]:not(.gridjs-th)', type: "categorical" }
			]);

			_colorize_table_entries_by_generation_method();

			if (typeof apply_theme_based_on_system_preferences === 'function') {
				apply_theme_based_on_system_preferences();
			}
		}
	}, 300);
}

function add_colorize_to_gridjs_table () {
	let searchInput = document.querySelector(".gridjs-search-input");
	if (searchInput) {
		searchInput.addEventListener("input", colorize_table_entries);
	}
}

function updatePreWidths() {
	var width = window.innerWidth * 0.95;
	var pres = document.getElementsByTagName('pre');
	for (var i = 0; i < pres.length; i++) {
		pres[i].style.width = width + 'px';
	}
}

function resizePlotlyCharts() {
	const windowWidth = window.innerWidth;

	if(last_resize_width == windowWidth) {
		return;
	}

	const plotlyElements = document.querySelectorAll('.js-plotly-plot');

	if (!plotlyElements.length) {
		return;
	}

	const windowHeight = window.innerHeight;

	const newWidth = windowWidth * 0.9;

	plotlyElements.forEach(function(element, index) {
		const layout = {
			width: newWidth,
			plot_bgcolor: 'rgba(0, 0, 0, 0)',
			paper_bgcolor: 'rgba(0, 0, 0, 0)',
		};

		Plotly.relayout(element, layout)
	});

	make_text_in_parallel_plot_nicer();

	if (typeof apply_theme_based_on_system_preferences === 'function') {
		apply_theme_based_on_system_preferences();
	}

	last_resize_width = windowWidth;
}

function createResultParameterCanvases(this_res_name) {
	if (
		typeof special_col_names === "undefined" ||
		typeof result_names === "undefined" ||
		typeof result_min_max === "undefined" ||
		typeof tab_results_headers_json === "undefined" ||
		typeof tab_results_csv_json === "undefined"
	) {
		console.error("Missing one or more required global variables.");
		return null;
	}

	if (
		!Array.isArray(special_col_names) ||
		!Array.isArray(result_names) ||
		!Array.isArray(result_min_max) ||
		!Array.isArray(tab_results_headers_json) ||
		!Array.isArray(tab_results_csv_json)
	) {
		console.error("All inputs must be arrays.");
		return null;
	}

	function getColumnIndexMap(headers) {
		var map = {};
		for (var i = 0; i < headers.length; i++) {
			map[headers[i]] = i;
		}
		return map;
	}

	function getColumnData(data, index) {
		var result = [];
		for (var i = 0; i < data.length; i++) {
			result.push(data[i][index]);
		}
		return result;
	}

	function normalize(value, min, max) {
		if (max === min) {
			return 0.5;
		}
		return (value - min) / (max - min);
	}

	function interpolateColor(ratio, reverse) {
		var r = reverse ? ratio : 1 - ratio;
		var g = reverse ? 1 - ratio : ratio;
		var b = 0;
		r = Math.floor(r * 255);
		g = Math.floor(g * 255);
		return "rgb(" + r + "," + g + "," + b + ")";
	}

	function createCanvas(width, height) {
		var canvas = document.createElement("canvas");
		canvas.width = width;
		canvas.height = height;
		return canvas;
	}

	function isNumericArray(arr) {
		for (var i = 0; i < arr.length; i++) {
			var val = arr[i];
			if (typeof val !== "number" || isNaN(val)) {
				return false;
			}
		}
		return true;
	}

	function findBestRowIndex() {
		var bestIndex = 0;

		for (var i = 1; i < tab_results_csv_json.length; i++) {
			var better = false;

			for (var r = 0; r < result_names.length; r++) {
				var col = result_names[r];
				var colIdx = header_map[col];
				var goal = result_min_max[r]; // "min" or "max"

				var valCurrent = tab_results_csv_json[i][colIdx];
				var valBest = tab_results_csv_json[bestIndex][colIdx];

				if (goal === "min" && valCurrent < valBest) {
					better = true;
					break;
				}

				if (goal === "max" && valCurrent > valBest) {
					better = true;
					break;
				}
			}

			if (better) {
				bestIndex = i;
			}
		}

		return bestIndex;
	}

	var canvas_width = 1000;
	var canvas_height = 100;

	var header_map = getColumnIndexMap(tab_results_headers_json);

	var parameter_columns = tab_results_headers_json.filter(function (name) {
		return (
			!special_col_names.includes(name) &&
			!result_names.includes(name) &&
			!name.startsWith("OO_Info_")
		);
	});

	var container = document.createElement("div");

	for (var r = 0; r < result_names.length; r++) {
		var result_name = result_names[r];
		if (this_res_name == result_name) {
			var result_index = header_map[result_name];
			var result_goal = result_min_max[r]; // "min" or "max"

			var result_values = getColumnData(tab_results_csv_json, result_index);
			var result_min = Math.min.apply(null, result_values);
			var result_max = Math.max.apply(null, result_values);

			var table = document.createElement("table");
			table.style.borderCollapse = "collapse";
			table.style.marginBottom = "32px";

			var thead = document.createElement("thead");
			var headRow = document.createElement("tr");

			var th1 = document.createElement("th");
			th1.textContent = "Parameter";
			th1.style.textAlign = "left";
			th1.style.padding = "6px 12px";
			var th2 = document.createElement("th");
			th2.textContent = "Distribution";
			th2.style.textAlign = "left";
			th2.style.padding = "6px 12px";

			headRow.appendChild(th1);
			headRow.appendChild(th2);
			thead.appendChild(headRow);
			table.appendChild(thead);

			var tbody = document.createElement("tbody");

			for (var p = 0; p < parameter_columns.length; p++) {
				var param_name = parameter_columns[p];
				var param_index = header_map[param_name];
				var param_values = getColumnData(tab_results_csv_json, param_index);

				if (!isNumericArray(param_values)) {
					continue;
				}

				var param_min = Math.min.apply(null, param_values);
				var param_max = Math.max.apply(null, param_values);

				var canvas = createCanvas(canvas_width, canvas_height);

				canvas.classList.add("invert_in_dark_mode");

				var ctx = canvas.getContext("2d");

				ctx.fillStyle = "white";
				ctx.fillRect(0, 0, canvas.width, canvas.height);

				var x_groups = {};

				for (var i = 0; i < tab_results_csv_json.length; i++) {
					var raw_param = tab_results_csv_json[i][param_index];
					var raw_result = tab_results_csv_json[i][result_index];

					var x_ratio = normalize(raw_param, param_min, param_max);
					var x = Math.floor(x_ratio * (canvas_width - 1));

					if (!x_groups[x]) {
						x_groups[x] = [];
					}

					x_groups[x].push(raw_result);
				}

				for (var x in x_groups) {
					var values = x_groups[x];
					values.sort(function (a, b) {
						return a - b;
					});

					var stripe_height = canvas_height / values.length;
					for (var i = 0; i < values.length; i++) {
						var y_start = i * stripe_height;
						var y_end = (i + 1) * stripe_height;

						var value = values[i];
						var result_ratio = normalize(value, result_min, result_max);
						var color = interpolateColor(result_ratio, result_goal === "min");

						ctx.beginPath();
						ctx.strokeStyle = color;
						ctx.lineWidth = 1;
						ctx.moveTo(Number(x) + 0.5, y_start);
						ctx.lineTo(Number(x) + 0.5, y_end);
						ctx.stroke();
					}
				}

				var row = document.createElement("tr");

				var cell_param = document.createElement("td");
				cell_param.textContent = param_name;
				cell_param.style.padding = "4px 12px";
				cell_param.style.verticalAlign = "top";
				cell_param.style.fontFamily = "monospace";
				cell_param.style.whiteSpace = "nowrap";

				var cell_canvas = document.createElement("td");
				cell_canvas.appendChild(canvas);
				cell_canvas.style.padding = "4px 12px";

				row.appendChild(cell_param);
				row.appendChild(cell_canvas);
				tbody.appendChild(row);
			}

			table.appendChild(tbody);
			container.appendChild(table);
		}
	}

	// === Summary: Best result ===
	var bestIndex = findBestRowIndex();
	var bestRow = tab_results_csv_json[bestIndex];

	var ul = document.createElement("ul");
	ul.style.margin = "0";
	ul.style.paddingLeft = "24px";

	// Alle Result-Spalten
	for (var i = 0; i < result_names.length; i++) {
		var name = result_names[i];
		var val = bestRow[header_map[name]];
		var li = document.createElement("li");
		li.textContent = name + " = " + val;
		ul.appendChild(li);
	}

	// Alle Parameter-Spalten (außer special_col_names)
	for (var i = 0; i < tab_results_headers_json.length; i++) {
		var name = tab_results_headers_json[i];
		if (special_col_names.includes(name) || name.startsWith("OO_Info_") || result_names.includes(name)) {
			continue;
		}

		var val = bestRow[header_map[name]];
		var li = document.createElement("li");
		li.textContent = name + " = " + val;
		ul.appendChild(li);
	}

	return container;
}

function initializeResultParameterVisualizations() {
        try {
                var elements = $('.result_parameter_visualization');

                if (!elements || elements.length === 0) {
                        console.warn('No .result_parameter_visualization elements found.');
                        return;
                }

                elements.each(function () {
                        var element = $(this);

                        if (element.data('initialized')) {
                                return; // Already initialized, skip
                        }

                        var resname = element.attr('data-resname');

                        if (!resname) {
                                console.error('Missing data-resname attribute for element:', this);
                                return;
                        }

                        try {
                                var html = createResultParameterCanvases(resname);

                                element.html(html);
                                element.data('initialized', true);

                        } catch (err) {
                                console.error('Error while calling createResultParameterCanvases for resname:', resname, err);
                        }
                });
        } catch (outerErr) {
                console.error('Failed to initialize result parameter visualizations:', outerErr);
        }

	if (typeof apply_theme_based_on_system_preferences === 'function') {
		apply_theme_based_on_system_preferences();
	}
}

function insertSortableSelectForSingleLogsTabs() {
	if (typeof $ === "undefined") {
		console.error("jQuery is not loaded.");
		return;
	}

	var $tab_logs = $("#tab_logs");
	if ($tab_logs.length === 0) {
		return;
	}

	var $menu = $tab_logs.find("menu[role='tablist']");
	if ($menu.length === 0) {
		console.error("No <menu> with role='tablist' found inside #tab_logs.");
		return;
	}

	var $buttons = $menu.find("button[data-trial_index]");
	if ($buttons.length === 0) {
		console.warn("No buttons with data-trial_index found inside the menu.");
		return;
	}

	// Alle data-Attribute sammeln
	var dataAttrs = {};
	$buttons.each(function() {
		var attrs = this.attributes;
		for (var i = 0; i < attrs.length; i++) {
			var name = attrs[i].name;
			if (name.startsWith("data-")) {
				dataAttrs[name] = true;
			}
		}
	});

	var attrList = Object.keys(dataAttrs);
	if (attrList.length === 0) {
		console.warn("No data attributes found on buttons.");
		return;
	}

	// <select> erstellen
	var $select = $("<select></select>").css({marginBottom: "10px"});
	$select.append($("<option disabled selected>Select attribute to sort</option>"));

	attrList.forEach(function(attr) {
		var cleanName = attr.replace("data-", "");
		$select.append($("<option></option>").attr("value", attr + "|asc").text(cleanName + " (ascending)"));
		$select.append($("<option></option>").attr("value", attr + "|desc").text(cleanName + " (descending)"));
	});

	// Select ganz oben in #tab_logs einfügen
	$tab_logs.prepend($select);

	// Sortierfunktion
	$select.on("change", function() {
		var val = $(this).val();
		if (!val) return;

		var parts = val.split("|");
		if (parts.length !== 2) return;

		var attr = parts[0];
		var order = parts[1];

		var $btnsArray = $buttons.toArray();

		$btnsArray.sort(function(a, b) {
			var va = $(a).attr(attr);
			var vb = $(b).attr(attr);

			// Fehlertolerant: wenn Wert fehlt, auf null setzen
			if (va === undefined || va === null) va = "";
			if (vb === undefined || vb === null) vb = "";

			// Zahlen vergleichen, sonst Strings
			var na = parseFloat(va);
			var nb = parseFloat(vb);
			if (!isNaN(na) && !isNaN(nb)) {
				return order === "asc" ? na - nb : nb - na;
			} else {
				if (va < vb) return order === "asc" ? -1 : 1;
				if (va > vb) return order === "asc" ? 1 : -1;
				return 0;
			}
		});

		// Buttons neu anordnen
		$menu.empty().append($btnsArray);
	});
}

window.addEventListener('load', updatePreWidths);
window.addEventListener('resize', updatePreWidths);

$(document).ready(function() {
	colorize_table_entries();

	add_colorize_to_gridjs_table();
});

window.addEventListener('resize', function() {
	resizePlotlyCharts();
});
