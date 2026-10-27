"use strict";

// Lightweight results table for the share page.
//
// GridJS rendered every row at once, which blocked the main thread for
// seconds on large runs and made switching tabs feel frozen.  This table only
// keeps only the visible window of rows in the DOM, while search and sorting
// still operate on the complete data set.  Cell colors are computed once from
// the full data so they stay stable while scrolling.

function _resultsColumnKinds(data, headers) {
	return headers.map(function (_header, columnIndex) {
		var hasNumber = false;
		var allNumeric = true;

		for (var r = 0; r < data.length; r++) {
			var raw = data[r][columnIndex];
			if (raw === null || raw === undefined || String(raw).trim() === "") {
				continue;
			}
			var value = parseFloat(raw);
			if (isNaN(value)) {
				allNumeric = false;
				break;
			}
			hasNumber = true;
		}

		return allNumeric && hasNumber ? "number" : "string";
	});
}

function _resultsCompareValues(a, b, kind, direction) {
	var aEmpty = a === null || a === undefined || String(a).trim() === "";
	var bEmpty = b === null || b === undefined || String(b).trim() === "";

	// Empty cells always go to the end, regardless of sort direction.
	if (aEmpty && bEmpty) return 0;
	if (aEmpty) return 1;
	if (bEmpty) return -1;

	if (kind === "number") {
		return (parseFloat(a) - parseFloat(b)) * direction;
	}

	var left = String(a);
	var right = String(b);
	if (left === right) return 0;
	return (left < right ? -1 : 1) * direction;
}

function _resultsCsvEscape(value) {
	var text = value === null || value === undefined ? "" : String(value);
	if (/[",\n\r]/.test(text)) {
		return '"' + text.replace(/"/g, '""') + '"';
	}
	return text;
}

function _resultsCsvString(headers, rows) {
	var lines = [headers.map(_resultsCsvEscape).join(",")];
	for (var r = 0; r < rows.length; r++) {
		lines.push(rows[r].map(_resultsCsvEscape).join(","));
	}
	return lines.join("\n");
}

function _resultsBuildColorizer(data, headers) {
	var headerIndex = new Map();
	for (var i = 0; i < headers.length; i++) {
		headerIndex.set(headers[i], i);
	}

	var numericRanges = {};
	var categoricalMaps = {};

	function numericRange(columnName, useLog) {
		if (Object.prototype.hasOwnProperty.call(numericRanges, columnName)) {
			return numericRanges[columnName];
		}
		var index = headerIndex.get(columnName);
		if (index === undefined) {
			numericRanges[columnName] = null;
			return null;
		}

		var min = Infinity;
		var max = -Infinity;
		for (var r = 0; r < data.length; r++) {
			var value = parseFloat(data[r][index]);
			if (isNaN(value)) continue;
			if (useLog && value <= 0) continue;
			var transformed = useLog ? Math.log(value) : value;
			if (transformed < min) min = transformed;
			if (transformed > max) max = transformed;
		}

		var range = min === Infinity ? null : {
			min: min,
			max: max,
			span: max - min || 1,
			useLog: useLog
		};
		numericRanges[columnName] = range;
		return range;
	}

	function categoricalMap(columnName) {
		if (Object.prototype.hasOwnProperty.call(categoricalMaps, columnName)) {
			return categoricalMaps[columnName];
		}
		var index = headerIndex.get(columnName);
		if (index === undefined) {
			categoricalMaps[columnName] = null;
			return null;
		}

		var values = {};
		var count = 0;
		for (var r = 0; r < data.length; r++) {
			var text = String(data[r][index] === null || data[r][index] === undefined ? "" : data[r][index]);
			if (!Object.prototype.hasOwnProperty.call(values, text)) {
				values[text] = count++;
			}
		}

		var map = count === 0 ? null : { values: values, count: count };
		categoricalMaps[columnName] = map;
		return map;
	}

	var resultColumns = [];
	if (typeof result_names !== "undefined" && Array.isArray(result_names)) {
		for (var rn = 0; rn < result_names.length; rn++) {
			resultColumns.push({
				name: result_names[rn],
				invert: typeof result_min_max !== "undefined" && result_min_max[rn] === "max"
			});
		}
	}

	var generationColors = {
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

	return function colorize(table) {
		var cells;

		cells = table.querySelectorAll('[data-column-id="queue_time"]');
		var range = numericRange("queue_time", false);
		if (range) {
			for (var i = 0; i < cells.length; i++) {
				var value = parseFloat(cells[i].textContent);
				if (isNaN(value)) continue;
				var ratio = (value - range.min) / range.span;
				cells[i].style.backgroundColor = "rgb(" + ((255 * ratio) | 0) + "," + ((255 * (1 - ratio)) | 0) + ",0)";
			}
		}

		cells = table.querySelectorAll('[data-column-id="run_time"]');
		range = numericRange("run_time", false);
		if (range) {
			for (var i = 0; i < cells.length; i++) {
				var value = parseFloat(cells[i].textContent);
				if (isNaN(value)) continue;
				var ratio = (value - range.min) / range.span;
				cells[i].style.backgroundColor = "rgb(" + ((255 * ratio) | 0) + "," + ((255 * (1 - ratio)) | 0) + ",0)";
			}
		}

		for (var rc = 0; rc < resultColumns.length; rc++) {
			var column = resultColumns[rc];
			cells = table.querySelectorAll('[data-column-id="' + column.name + '"]');
			range = numericRange(column.name, true);
			if (!range) continue;

			for (var i = 0; i < cells.length; i++) {
				var value = parseFloat(cells[i].textContent);
				if (isNaN(value) || value <= 0) continue;
				var ratio = (Math.log(value) - range.min) / range.span;
				if (column.invert) ratio = 1 - ratio;
				cells[i].style.backgroundColor = "rgb(" + ((255 * ratio) | 0) + "," + ((255 * (1 - ratio)) | 0) + ",0)";
			}
		}

		var categoricalColumns = ["hostname", "generation_node"];
		for (var cc = 0; cc < categoricalColumns.length; cc++) {
			var name = categoricalColumns[cc];
			cells = table.querySelectorAll('[data-column-id="' + name + '"]');
			var map = categoricalMap(name);
			if (!map) continue;

			for (var i = 0; i < cells.length; i++) {
				var text = cells[i].textContent.trim();
				var hue = Math.round((360 / map.count) * (map.values[text] || 0));
				cells[i].style.backgroundColor = "hsl(" + hue + ", 70%, 60%)";
			}
		}

		cells = table.querySelectorAll('[data-column-id="generation_node"]');
		for (var i = 0; i < cells.length; i++) {
			var text = cells[i].textContent.toLowerCase();
			for (var key in generationColors) {
				if (text.includes(key)) {
					cells[i].style.backgroundColor = generationColors[key];
					break;
				}
			}
		}

		cells = table.querySelectorAll('[data-column-id="trial_status"]');
		for (var i = 0; i < cells.length; i++) {
			var text = cells[i].textContent;
			var color = text.includes("COMPLETED") ? "lightgreen" :
				text.includes("RUNNING") ? "orange" :
				text.includes("FAILED") ? "red" :
				text.includes("CANDIDATE") ? "lightblue" :
				text.includes("ABANDONED") ? "yellow" : "";
			if (color) {
				cells[i].style.backgroundColor = color;
			}
		}

		// One filter on the table is much cheaper than one per cell, and the
		// theme toggle updates it automatically through the CSS class.
		table.classList.add("invert_in_dark_mode");
		table.style.filter = (typeof theme !== "undefined" && theme === "dark") ? "invert(1)" : "";
	};
}

function createResultsTable(data, headers, table_name) {
	var container = document.getElementById(table_name);
	if (!container) {
		console.error("#" + table_name + " not found");
		return;
	}

	container.innerHTML = "";
	container.classList.add("results-table-root");

	var state = {
		sortColumn: -1,
		sortDirection: 1,
		search: "",
		view: []
	};

	var ROW_HEIGHT = 32;
	var OVERSCAN = 10;
	var lastFirstRow = -1;
	var lastLastRow = -1;

	var columnKinds = _resultsColumnKinds(data, headers);
	var colorize = _resultsBuildColorizer(data, headers);

	var toolbar = document.createElement("div");
	toolbar.className = "results-toolbar";

	var searchInput = document.createElement("input");
	searchInput.type = "search";
	searchInput.placeholder = "Search results...";
	searchInput.className = "results-search";
	toolbar.appendChild(searchInput);

	var info = document.createElement("span");
	info.className = "results-info";
	toolbar.appendChild(info);

	var copyButton = document.createElement("button");
	copyButton.type = "button";
	copyButton.textContent = "Copy CSV";
	copyButton.className = "results-button";
	toolbar.appendChild(copyButton);

	var downloadButton = document.createElement("button");
	downloadButton.type = "button";
	downloadButton.textContent = "Download CSV";
	downloadButton.className = "results-button";
	toolbar.appendChild(downloadButton);

	container.appendChild(toolbar);

	var scroll = document.createElement("div");
	scroll.className = "results-table-scroll";

	var table = document.createElement("table");
	table.className = "results-table";
	var thead = document.createElement("thead");
	var headRow = document.createElement("tr");
	var arrows = [];

	for (var c = 0; c < headers.length; c++) {
		var th = document.createElement("th");
		var label = document.createElement("span");
		label.className = "results-column-label";
		label.textContent = headers[c];
		var arrow = document.createElement("span");
		arrow.className = "results-sort-arrow";
		th.appendChild(label);
		th.appendChild(arrow);
		arrows.push(arrow);

		th.addEventListener("click", (function (columnIndex) {
			return function () {
				if (state.sortColumn === columnIndex) {
					state.sortDirection = -state.sortDirection;
				} else {
					state.sortColumn = columnIndex;
					state.sortDirection = 1;
				}
				updateView();
				scroll.scrollTop = 0;
				renderWindow(true);
			};
		})(c));

		headRow.appendChild(th);
	}
	thead.appendChild(headRow);
	table.appendChild(thead);

	var tbody = document.createElement("tbody");
	table.appendChild(tbody);
	scroll.appendChild(table);
	container.appendChild(scroll);

	function createSpacer(height) {
		var tr = document.createElement("tr");
		tr.className = "results-spacer";
		var td = document.createElement("td");
		td.colSpan = headers.length;
		td.style.height = height + "px";
		tr.appendChild(td);
		return tr;
	}

	function updateView() {
		var query = state.search.trim().toLowerCase();
		var view = [];

		if (query === "") {
			for (var r = 0; r < data.length; r++) {
				view.push(r);
			}
		} else {
			for (var r = 0; r < data.length; r++) {
				var row = data[r];
				for (var c = 0; c < row.length; c++) {
					if (String(row[c] === null || row[c] === undefined ? "" : row[c]).toLowerCase().indexOf(query) !== -1) {
						view.push(r);
						break;
					}
				}
			}
		}

		if (state.sortColumn >= 0) {
			var columnIndex = state.sortColumn;
			var kind = columnKinds[columnIndex];
			var direction = state.sortDirection;
			view.sort(function (a, b) {
				var comparison = _resultsCompareValues(
					data[a][columnIndex],
					data[b][columnIndex],
					kind,
					direction
				);
				return comparison === 0 ? a - b : comparison;
			});
		}

		state.view = view;
	}

	function updateHeaderArrows() {
		for (var c = 0; c < arrows.length; c++) {
			if (c === state.sortColumn) {
				arrows[c].textContent = state.sortDirection > 0 ? " \u25B2" : " \u25BC";
			} else {
				arrows[c].textContent = "";
			}
		}
	}

	/**
	 * Render only the rows that are currently visible (plus a small
	 * overscan).  There is no pagination: the full result set is in one
	 * scrolling table, but the DOM only contains the window around the
	 * viewport.
	 */
	function renderWindow(force) {
		updateHeaderArrows();

		var total = state.view.length;
		if (total === 0) {
			(lastFirstRow = -1);
			lastLastRow = -1;
			tbody.innerHTML = "";
			info.textContent = "No matching results";
			return;
		}

		var headerHeight = thead.offsetHeight || ROW_HEIGHT;
		var scrollTop = Math.max(0, scroll.scrollTop - headerHeight);
		var viewportHeight = scroll.clientHeight || 700;
		var visibleCount = Math.ceil(viewportHeight / ROW_HEIGHT) + OVERSCAN * 2;

		var first = Math.max(0, Math.floor(scrollTop / ROW_HEIGHT) - OVERSCAN);
		var last = Math.min(total, first + visibleCount);
		first = Math.max(0, Math.min(first, total - visibleCount));
		last = Math.min(total, first + visibleCount);

		if (!force && first === lastFirstRow && last === lastLastRow) {
			return;
		}
		lastFirstRow = first;
		lastLastRow = last;

		var fragment = document.createDocumentFragment();
		if (first > 0) {
			fragment.appendChild(createSpacer(first * ROW_HEIGHT));
		}

		for (var i = first; i < last; i++) {
			var source = data[state.view[i]];
			var tr = document.createElement("tr");
			tr.className = "results-row";
			for (var c = 0; c < headers.length; c++) {
				var td = document.createElement("td");
				td.dataset.columnId = headers[c];
				var value = source[c];
				td.textContent = value === null || value === undefined ? "" : value;
				tr.appendChild(td);
			}
			fragment.appendChild(tr);
		}

		if (last < total) {
			fragment.appendChild(createSpacer((total - last) * ROW_HEIGHT));
		}

		tbody.innerHTML = "";
		tbody.appendChild(fragment);
		info.textContent = total + " results";
		colorize(table);
	}

	var scrollRaf = null;
	scroll.addEventListener("scroll", function () {
		if (scrollRaf !== null) {
			return;
		}
		scrollRaf = requestAnimationFrame(function () {
			scrollRaf = null;
			renderWindow(false);
		});
	});

	var searchTimer = null;
	searchInput.addEventListener("input", function () {
		if (searchTimer !== null) {
			clearTimeout(searchTimer);
		}
		searchTimer = setTimeout(function () {
			searchTimer = null;
			state.search = searchInput.value;
			updateView();
			scroll.scrollTop = 0;
			renderWindow(true);
		}, 120);
	});

	copyButton.addEventListener("click", function () {
		var csv = _resultsCsvString(headers, data);
		copy_to_clipboard(csv);
	});

	downloadButton.addEventListener("click", function () {
		var csv = _resultsCsvString(headers, data);
		var blob = new Blob([csv], { type: "text/csv" });
		var link = document.createElement("a");
		link.href = URL.createObjectURL(blob);
		link.download = table_name + ".csv";
		document.body.appendChild(link);
		link.click();
		document.body.removeChild(link);
		URL.revokeObjectURL(link.href);
	});

	updateView();
	renderWindow(true);

	// The panel is only just becoming visible, so the client height may still be
	// tiny on the first synchronous render.  Re-render once layout is ready
	// and whenever the viewport height changes (window resize, browser zoom).
	requestAnimationFrame(function () {
		renderWindow(true);
	});
	if (typeof ResizeObserver !== "undefined") {
		new ResizeObserver(function () {
			renderWindow(true);
		}).observe(scroll);
	}
}
