"use strict";

var last_resize_width = 0;

function add_default_layout_data(layout, no_height = 0) {
	layout["width"] = get_graph_width();
	if (!no_height) {
		layout["height"] = get_graph_height();
	}
	layout["paper_bgcolor"] = 'rgba(0,0,0,0)';
	layout["plot_bgcolor"] = 'rgba(0,0,0,0)';
	return layout;
}

function get_marker_size() {
	return 4;
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
	};
}

function get_axis_title_data(name, axis_type = "") {
	if (axis_type) {
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
	var el = document.getElementById(table_name);
	if (!el) {
		console.error("#" + table_name + " not found");
		return;
	}

	new gridjs.Grid({
		columns: headers,
		data: data,
		search: true,
		sort: true,
		ellipsis: false
	}).render(el);

	if (typeof apply_theme_based_on_system_preferences === 'function') {
		apply_theme_based_on_system_preferences();
	}

	colorize_table_entries();
	add_colorize_to_gridjs_table();
}

function download_as_file(id, filename) {
	var text = document.getElementById(id).textContent;
	var blob = new Blob([text], { type: "text/plain" });
	var link = document.createElement("a");
	link.href = URL.createObjectURL(blob);
	link.download = filename;
	document.body.appendChild(link);
	link.click();
	document.body.removeChild(link);
}

function copy_to_clipboard_from_id(id) {
	var text = document.getElementById(id).textContent;
	copy_to_clipboard(text);
}

function copy_to_clipboard(text) {
	if (!navigator.clipboard) {
		var textarea = document.createElement("textarea");
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
	var specialSet = new Set(special_col_names);
	var new_data = [];
	for (var row_idx = 0; row_idx < data.length; row_idx++) {
		var line = data[row_idx];
		var line_has_empty_data = false;

		for (var col_idx = 0; col_idx < line.length; col_idx++) {
			if (line[col_idx] === "" && !specialSet.has(tab_results_headers_json[col_idx])) {
				line_has_empty_data = true;
				break;
			}
		}

		if (!line_has_empty_data) {
			new_data.push(line);
		}
	}
	return new_data;
}

function load_log_file(log_nr, filename) {
	var pre_id = `single_run_${log_nr}_pre`;
	var preEl = document.getElementById(pre_id);

	if (preEl && !preEl.dataset.loaded) {
		var params = new URLSearchParams(window.location.search);
		var user_id = params.get('user_id');
		var experiment_name = params.get('experiment_name');
		var run_nr = params.get('run_nr');

		var url = `get_log?user_id=${user_id}&experiment_name=${experiment_name}&run_nr=${run_nr}&filename=${filename}`;

		fetch(url)
			.then(response => response.json())
			.then(data => {
				if (data.data) {
					preEl.innerHTML = data.data;
					preEl.dataset.loaded = "true";
				} else {
					log(`No 'data' key found in response.`);
				}
				var spinner = document.getElementById("spinner_log_" + log_nr);
				if (spinner) spinner.remove();
			})
			.catch(error => {
				log(`Error loading log: ${error}`);
				var spinner = document.getElementById("spinner_log_" + log_nr);
				if (spinner) spinner.remove();
			});
	}
}

function load_debug_log() {
	var preEl = document.getElementById("here_debuglogs_go");

	if (preEl && !preEl.dataset.loaded) {
		var params = new URLSearchParams(window.location.search);
		var user_id = params.get('user_id');
		var experiment_name = params.get('experiment_name');
		var run_nr = params.get('run_nr');

		var url = `get_debug_log?user_id=${user_id}&experiment_name=${experiment_name}&run_nr=${run_nr}`;

		fetch(url)
			.then(response => response.json())
			.then(data => {
				var spinner = document.getElementById("debug_log_spinner");
				if (spinner) spinner.remove();

				if (data.data) {
					try {
						preEl.innerHTML = data.data;
					} catch (err) {
						preEl.textContent = `Error loading data: ${err}`;
					}
					preEl.dataset.loaded = "true";

					if (typeof apply_theme_based_on_system_preferences === 'function') {
						apply_theme_based_on_system_preferences();
					}
				} else {
					log(`No 'data' key found in response.`);
				}
			})
			.catch(error => {
				log(`Error loading log: ${error}`);
				var spinner = document.getElementById("debug_log_spinner");
				if (spinner) spinner.remove();
			});
	}
}

function _colorize_table_entries_by_generation_method() {
	var colorMap = {
		"manual": "green",
		"sobol": "orange",
		"saasbo": "pink",
		"uniform": "lightblue",
		"legacy_gpei": "sienna",
		"bo_mixed": "aqua",
		"randomforest": "darkseagreen",
		"external_generator": "purple",
		"botorch": "yellow"
	};

	var els = document.querySelectorAll('[data-column-id="generation_node"]');
	for (var i = 0; i < els.length; i++) {
		var el = els[i];
		var text = el.textContent.toLowerCase();
		var color = "";
		for (var key in colorMap) {
			if (text.includes(key)) {
				color = colorMap[key];
				break;
			}
		}
		if (color) {
			el.style.backgroundColor = color;
		}
		el.classList.add("invert_in_dark_mode");
	}
}

function _colorize_table_entries_by_trial_status() {
	var els = document.querySelectorAll('[data-column-id="trial_status"]');
	for (var i = 0; i < els.length; i++) {
		var el = els[i];
		var text = el.textContent;
		var color = text.includes("COMPLETED") ? "lightgreen" :
			text.includes("RUNNING") ? "orange" :
			text.includes("FAILED") ? "red" :
			text.includes("CANDIDATE") ? "lightblue" :
			text.includes("ABANDONED") ? "yellow" : "";
		if (color) el.style.backgroundColor = color;
		el.classList.add("invert_in_dark_mode");
	}
}

function _colorize_table_entries(configs) {
	for (var c = 0; c < configs.length; c++) {
		var cfg = configs[c];
		var cells = document.querySelectorAll(cfg.selector);
		if (cells.length === 0) continue;

		if (cfg.type === "categorical") {
			var uniqueMap = {};
			var uniqueCount = 0;
			for (var i = 0; i < cells.length; i++) {
				var v = cells[i].textContent.trim();
				if (!(v in uniqueMap)) {
					uniqueMap[v] = uniqueCount++;
				}
			}
			for (var i = 0; i < cells.length; i++) {
				var v = cells[i].textContent.trim();
				var hue = Math.round((360 / uniqueCount) * uniqueMap[v]);
				cells[i].style.backgroundColor = `hsl(${hue}, 70%, 60%)`;
				cells[i].classList.add("invert_in_dark_mode");
			}
			continue;
		}

		// Numeric: compute min/max in one pass
		var min = Infinity, max = -Infinity;
		var parsedValues = new Float64Array(cells.length);
		var validMask = new Uint8Array(cells.length);

		for (var i = 0; i < cells.length; i++) {
			var val = parseFloat(cells[i].textContent);
			parsedValues[i] = val;
			if (isNaN(val)) continue;
			if (cfg.type === "log" && val <= 0) continue;
			if (cfg.filter && !cfg.filter(val)) continue;
			validMask[i] = 1;
			var transformed = cfg.type === "log" ? Math.log(val) : val;
			if (transformed < min) min = transformed;
			if (transformed > max) max = transformed;
		}

		if (min === Infinity) continue;
		var range = max - min || 1;

		for (var i = 0; i < cells.length; i++) {
			if (!validMask[i]) continue;
			var val = parsedValues[i];
			var transformed = cfg.type === "log" ? Math.log(val) : val;
			var ratio = (transformed - min) / range;
			if (cfg.invert) ratio = 1 - ratio;

			var red = (255 * ratio) | 0;
			var green = (255 * (1 - ratio)) | 0;

			cells[i].style.backgroundColor = `rgb(${red},${green},0)`;
			cells[i].classList.add("invert_in_dark_mode");
		}
	}
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

var _colorize_timer = null;

function colorize_table_entries() {
	if (_colorize_timer) {
		clearTimeout(_colorize_timer);
	}
	_colorize_timer = setTimeout(function() {
		_colorize_timer = null;
		if (typeof result_names === "undefined" || !Array.isArray(result_names) || result_names.length === 0) {
			return;
		}

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
	}, 100);
}

function add_colorize_to_gridjs_table() {
	var searchInput = document.querySelector(".gridjs-search-input");
	if (searchInput) {
		var debounceTimer = null;
		searchInput.addEventListener("input", function() {
			if (debounceTimer) clearTimeout(debounceTimer);
			debounceTimer = setTimeout(colorize_table_entries, 200);
		});
	}
}

function updatePreWidths() {
	var width = (window.innerWidth * 0.95) + 'px';
	var pres = document.getElementsByTagName('pre');
	for (var i = 0; i < pres.length; i++) {
		pres[i].style.width = width;
	}
}

var _resizeRAF = null;

function resizePlotlyCharts() {
	if (_resizeRAF) return;
	_resizeRAF = requestAnimationFrame(function() {
		_resizeRAF = null;
		var windowWidth = window.innerWidth;

		if (last_resize_width === windowWidth) return;

		var plotlyElements = document.querySelectorAll('.js-plotly-plot');
		if (!plotlyElements.length) return;

		var newWidth = windowWidth * 0.9;
		var layout = {
			width: newWidth,
			plot_bgcolor: 'rgba(0, 0, 0, 0)',
			paper_bgcolor: 'rgba(0, 0, 0, 0)',
		};

		for (var i = 0; i < plotlyElements.length; i++) {
			Plotly.relayout(plotlyElements[i], layout);
		}

		make_text_in_parallel_plot_nicer();

		if (typeof apply_theme_based_on_system_preferences === 'function') {
			apply_theme_based_on_system_preferences();
		}

		last_resize_width = windowWidth;
	});
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

	var header_map = {};
	for (var i = 0; i < tab_results_headers_json.length; i++) {
		header_map[tab_results_headers_json[i]] = i;
	}

	var specialSet = new Set(special_col_names);
	var resultSet = new Set(result_names);

	var parameter_columns = [];
	for (var i = 0; i < tab_results_headers_json.length; i++) {
		var name = tab_results_headers_json[i];
		if (!specialSet.has(name) && !resultSet.has(name) && !name.startsWith("OO_Info_")) {
			parameter_columns.push(name);
		}
	}

	var canvas_width = 1000;
	var canvas_height = 100;
	var dataLen = tab_results_csv_json.length;

	var container = document.createElement("div");

	for (var r = 0; r < result_names.length; r++) {
		var result_name = result_names[r];
		if (this_res_name !== result_name) continue;

		var result_index = header_map[result_name];
		var result_goal = result_min_max[r];

		// Pre-extract result column and compute min/max
		var result_values = new Float64Array(dataLen);
		var result_min = Infinity, result_max = -Infinity;
		for (var i = 0; i < dataLen; i++) {
			var v = tab_results_csv_json[i][result_index];
			result_values[i] = v;
			if (v < result_min) result_min = v;
			if (v > result_max) result_max = v;
		}
		var result_range = result_max - result_min || 1;

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

			// Check if numeric and compute min/max in one pass
			var param_min = Infinity, param_max = -Infinity;
			var isNumeric = true;
			for (var i = 0; i < dataLen; i++) {
				var val = tab_results_csv_json[i][param_index];
				if (typeof val !== "number" || isNaN(val)) {
					isNumeric = false;
					break;
				}
				if (val < param_min) param_min = val;
				if (val > param_max) param_max = val;
			}

			if (!isNumeric) continue;

			var param_range = param_max - param_min || 1;

			var canvas = document.createElement("canvas");
			canvas.width = canvas_width;
			canvas.height = canvas_height;
			canvas.classList.add("invert_in_dark_mode");

			var ctx = canvas.getContext("2d");
			ctx.fillStyle = "white";
			ctx.fillRect(0, 0, canvas_width, canvas_height);

			// Use a typed array for grouping by x pixel
			// Each x can have multiple results; collect them
			var x_groups = new Array(canvas_width);

			for (var i = 0; i < dataLen; i++) {
				var raw_param = tab_results_csv_json[i][param_index];
				var x = ((raw_param - param_min) / param_range * (canvas_width - 1)) | 0;

				if (!x_groups[x]) {
					x_groups[x] = [result_values[i]];
				} else {
					x_groups[x].push(result_values[i]);
				}
			}

			// Use ImageData for pixel-level drawing (much faster than individual strokes)
			var imageData = ctx.getImageData(0, 0, canvas_width, canvas_height);
			var pixels = imageData.data;

			for (var x = 0; x < canvas_width; x++) {
				var values = x_groups[x];
				if (!values) continue;

				values.sort(function(a, b) { return a - b; });

				var count = values.length;
				for (var vi = 0; vi < count; vi++) {
					var y_start = (vi / count * canvas_height) | 0;
					var y_end = ((vi + 1) / count * canvas_height) | 0;

					var result_ratio = (values[vi] - result_min) / result_range;
					var red, green;
					if (result_goal === "min") {
						red = (255 * result_ratio) | 0;
						green = (255 * (1 - result_ratio)) | 0;
					} else {
						red = (255 * (1 - result_ratio)) | 0;
						green = (255 * result_ratio) | 0;
					}

					for (var y = y_start; y < y_end; y++) {
						var idx = (y * canvas_width + x) * 4;
						pixels[idx] = red;
						pixels[idx + 1] = green;
						pixels[idx + 2] = 0;
						pixels[idx + 3] = 255;
					}
				}
			}

			ctx.putImageData(imageData, 0, 0);

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

	// Find best row
	var bestIndex = 0;
	for (var i = 1; i < dataLen; i++) {
		for (var ri = 0; ri < result_names.length; ri++) {
			var colIdx = header_map[result_names[ri]];
			var goal = result_min_max[ri];
			var valCurrent = tab_results_csv_json[i][colIdx];
			var valBest = tab_results_csv_json[bestIndex][colIdx];

			if ((goal === "min" && valCurrent < valBest) || (goal === "max" && valCurrent > valBest)) {
				bestIndex = i;
				break;
			}
		}
	}

	var bestRow = tab_results_csv_json[bestIndex];

	var ul = document.createElement("ul");
	ul.style.margin = "0";
	ul.style.paddingLeft = "24px";

	for (var i = 0; i < result_names.length; i++) {
		var li = document.createElement("li");
		li.textContent = result_names[i] + " = " + bestRow[header_map[result_names[i]]];
		ul.appendChild(li);
	}

	for (var i = 0; i < tab_results_headers_json.length; i++) {
		var name = tab_results_headers_json[i];
		if (specialSet.has(name) || name.startsWith("OO_Info_") || resultSet.has(name)) continue;
		var li = document.createElement("li");
		li.textContent = name + " = " + bestRow[header_map[name]];
		ul.appendChild(li);
	}

	return container;
}

function initializeResultParameterVisualizations() {
	try {
		var elements = document.querySelectorAll('.result_parameter_visualization');

		if (!elements || elements.length === 0) {
			console.warn('No .result_parameter_visualization elements found.');
			return;
		}

		for (var i = 0; i < elements.length; i++) {
			var element = elements[i];

			if (element.dataset.initialized) continue;

			var resname = element.getAttribute('data-resname');

			if (!resname) {
				console.error('Missing data-resname attribute for element:', element);
				continue;
			}

			try {
				var html = createResultParameterCanvases(resname);
				element.innerHTML = '';
				element.appendChild(html);
				element.dataset.initialized = "true";
			} catch (err) {
				console.error('Error while calling createResultParameterCanvases for resname:', resname, err);
			}
		}
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
	if ($tab_logs.length === 0) return;

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

	var $select = $("<select></select>").css({ marginBottom: "10px" });
	$select.append($("<option disabled selected>Select attribute to sort</option>"));

	for (var i = 0; i < attrList.length; i++) {
		var attr = attrList[i];
		var cleanName = attr.replace("data-", "");
		$select.append($("<option></option>").attr("value", attr + "|asc").text(cleanName + " (ascending)"));
		$select.append($("<option></option>").attr("value", attr + "|desc").text(cleanName + " (descending)"));
	}

	$tab_logs.prepend($select);

	$select.on("change", function() {
		var val = $(this).val();
		if (!val) return;

		var parts = val.split("|");
		if (parts.length !== 2) return;

		var attr = parts[0];
		var order = parts[1];

		var $btnsArray = $buttons.toArray();

		$btnsArray.sort(function(a, b) {
			var va = a.getAttribute(attr) || "";
			var vb = b.getAttribute(attr) || "";

			var na = parseFloat(va);
			var nb = parseFloat(vb);
			if (!isNaN(na) && !isNaN(nb)) {
				return order === "asc" ? na - nb : nb - na;
			}
			if (va < vb) return order === "asc" ? -1 : 1;
			if (va > vb) return order === "asc" ? 1 : -1;
			return 0;
		});

		$menu.empty().append($btnsArray);
	});
}

window.addEventListener('load', updatePreWidths);
window.addEventListener('resize', updatePreWidths);

$(document).ready(function() {
	colorize_table_entries();
	add_colorize_to_gridjs_table();
});

window.addEventListener('resize', resizePlotlyCharts);
