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

function _resultsCsvEscape(value, delimiter, quoteAll) {
	var text = value === null || value === undefined ? "" : String(value);
	var mustQuote = quoteAll === true || /["\n\r]/.test(text);

	if (!mustQuote) {
		mustQuote = text.indexOf(delimiter || ",") !== -1;
	}

	if (mustQuote) {
		return '"' + text.replace(/"/g, '""') + '"';
	}

	return text;
}

function _resultsCsvString(headers, rows, options) {
	options = options || {};
	var delimiter = typeof options.delimiter === "string" && options.delimiter !== "" ? options.delimiter : ",";
	var includeHeader = options.includeHeader !== false;
	var quoteAll = options.quoteAll === true;
	var lines = [];

	function escape(value) {
		return _resultsCsvEscape(value, delimiter, quoteAll);
	}

	if (includeHeader) {
		lines.push(headers.map(escape).join(delimiter));
	}

	for (var r = 0; r < rows.length; r++) {
		lines.push(rows[r].map(escape).join(delimiter));
	}

	return lines.join("\n");
}

var _RESULTS_TECHNICAL_COLUMNS = [
	"submit_time",
	"queue_time",
	"worker_generator_uuid",
	"start_time",
	"end_time",
	"run_time",
	"program_string",
	"exit_code",
	"signal",
	"hostname",
	"arm_name",
	"trial_status",
	"generation_node",
	"generation_method",
	"idxs",
	"OO_Info_SLURM_JOB_ID"
];

function _resultsTechnicalColumnLookup(specialColumns) {
	var lookup = Object.create(null);

	for (var i = 0; i < _RESULTS_TECHNICAL_COLUMNS.length; i++) {
		lookup[_RESULTS_TECHNICAL_COLUMNS[i]] = true;
	}

	if (Array.isArray(specialColumns)) {
		for (var j = 0; j < specialColumns.length; j++) {
			lookup[specialColumns[j]] = true;
		}
	}

	return lookup;
}

function _resultsIsInterestingColumn(header, resultNames, technicalLookup) {
	if (header === "OO_Info_SLURM_JOB_ID") {
		return false;
	}

	if (Array.isArray(resultNames) && resultNames.indexOf(header) !== -1) {
		return true;
	}

	if (header === "trial_index") {
		return true;
	}

	if (String(header).indexOf("OO_Info_") === 0) {
		return true;
	}

	return !technicalLookup[header];
}

function _resultsIsMinimalColumn(header, resultNames, technicalLookup) {
	if (Array.isArray(resultNames) && resultNames.indexOf(header) !== -1) {
		return true;
	}

	if (header === "trial_index") {
		return true;
	}

	if (String(header).indexOf("OO_Info_") === 0) {
		return false;
	}

	return !technicalLookup[header];
}

function _resultsProjectRows(rows, indices) {
	var projected = [];

	for (var r = 0; r < rows.length; r++) {
		var row = rows[r];
		var projectedRow = [];

		for (var c = 0; c < indices.length; c++) {
			projectedRow.push(row[indices[c]]);
		}

		projected.push(projectedRow);
	}

	return projected;
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

function _resultsRowMatchesSearch(row, query) {
	if (query === "") {
		return true;
	}
	for (var c = 0; c < row.length; c++) {
		if (String(row[c] === null || row[c] === undefined ? "" : row[c]).toLowerCase().indexOf(query) !== -1) {
			return true;
		}
	}
	return false;
}

function _resultsFilterRows(data, search, externalFilter) {
	var query = String(search === null || search === undefined ? "" : search).trim().toLowerCase();
	var view = [];

	for (var r = 0; r < data.length; r++) {
		if (externalFilter && !externalFilter(data[r], r)) {
			continue;
		}
		if (!_resultsRowMatchesSearch(data[r], query)) {
			continue;
		}
		view.push(r);
	}

	return view;
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
		externalFilter: null,
		hiddenColumns: {},
		interestingOnly: false,
		view: []
	};

	var ROW_HEIGHT = 32;
	var OVERSCAN = 10;
	var lastFirstRow = -1;
	var lastLastRow = -1;
	var columnStorageKey = "omniopt-results-columns:" + table_name + ":" +
		(typeof window !== "undefined" ? window.location.pathname + window.location.search : "");
	var exportOptionsStorageKey = "omniopt-results-export-options";

	function readStoredJson(key) {
		try {
			var raw = window.localStorage.getItem(key);
			return raw === null ? null : JSON.parse(raw);
		} catch (error) {
			return null;
		}
	}

	function writeStoredJson(key, value) {
		try {
			window.localStorage.setItem(key, JSON.stringify(value));
		} catch (error) {
			// Storage can be unavailable in private or restricted contexts.
		}
	}

	function columnIsEmpty(columnIndex) {
		for (var r = 0; r < data.length; r++) {
			var value = data[r][columnIndex];

			if (value !== null && value !== undefined && String(value).trim() !== "") {
				return false;
			}
		}

		return true;
	}

	var columnKinds = _resultsColumnKinds(data, headers);
	var colorize = _resultsBuildColorizer(data, headers);
	var resultNames = typeof result_names !== "undefined" && Array.isArray(result_names) ? result_names : [];
	var specialColumns = typeof special_col_names !== "undefined" && Array.isArray(special_col_names) ? special_col_names : [];
	var technicalLookup = _resultsTechnicalColumnLookup(specialColumns);
	var exportOptions = { delimiter: ",", includeHeader: true, quoteAll: false };
	var storedExportOptions = readStoredJson(exportOptionsStorageKey);

	if (storedExportOptions && typeof storedExportOptions === "object") {
		if (typeof storedExportOptions.delimiter === "string" && storedExportOptions.delimiter !== "") {
			exportOptions.delimiter = storedExportOptions.delimiter;
		}

		exportOptions.includeHeader = storedExportOptions.includeHeader !== false;
		exportOptions.quoteAll = storedExportOptions.quoteAll === true;
	}

	var storedColumns = readStoredJson(columnStorageKey);

	if (Array.isArray(storedColumns)) {
		for (var sc = 0; sc < headers.length; sc++) {
			if (storedColumns.indexOf(headers[sc]) !== -1) {
				state.hiddenColumns[sc] = true;
			}
		}
	} else {
		for (var ec = 0; ec < headers.length; ec++) {
			if (!_resultsIsInterestingColumn(headers[ec], resultNames, technicalLookup) && columnIsEmpty(ec)) {
				state.hiddenColumns[ec] = true;
			}
		}
	}

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

	var interestingToggleButton = document.createElement("button");
	interestingToggleButton.type = "button";
	interestingToggleButton.className = "results-button results-interesting-toggle";
	interestingToggleButton.textContent = "Show only interesting columns";
	toolbar.appendChild(interestingToggleButton);

	var columnsControl = document.createElement("details");
	columnsControl.className = "results-columns";
	var columnsToggle = document.createElement("summary");
	columnsToggle.className = "results-button results-columns-toggle";
	columnsToggle.textContent = "Columns";
	columnsControl.appendChild(columnsToggle);

	var columnsPanel = document.createElement("div");
	columnsPanel.className = "results-columns-panel";
	columnsControl.appendChild(columnsPanel);

	var columnsActions = document.createElement("div");
	columnsActions.className = "results-columns-actions";
	columnsPanel.appendChild(columnsActions);

	var minimalButton = document.createElement("button");
	minimalButton.type = "button";
	minimalButton.textContent = "Minimal";
	minimalButton.className = "results-button";
	columnsActions.appendChild(minimalButton);

	var showAllButton = document.createElement("button");
	showAllButton.type = "button";
	showAllButton.textContent = "All columns";
	showAllButton.className = "results-button";
	columnsActions.appendChild(showAllButton);

	var columnsList = document.createElement("div");
	columnsList.className = "results-columns-list";
	columnsPanel.appendChild(columnsList);

	var columnCheckboxes = [];

	for (var hc = 0; hc < headers.length; hc++) {
		var columnLabel = document.createElement("label");
		columnLabel.className = "results-column-option";
		var columnCheckbox = document.createElement("input");
		columnCheckbox.type = "checkbox";
		columnCheckbox.checked = true;
		columnLabel.appendChild(columnCheckbox);
		columnLabel.appendChild(document.createTextNode(" " + headers[hc]));
		columnsList.appendChild(columnLabel);
		columnCheckboxes.push(columnCheckbox);

		(function (columnIndex, checkbox) {
			checkbox.addEventListener("change", function () {
				setColumnVisible(columnIndex, checkbox.checked);
			});
		})(hc, columnCheckbox);
	}

	toolbar.appendChild(columnsControl);

	var exportOptionsControl = document.createElement("details");
	exportOptionsControl.className = "results-export-options";
	var exportOptionsToggle = document.createElement("summary");
	exportOptionsToggle.className = "results-button results-export-options-toggle";
	exportOptionsToggle.textContent = "Export options";
	exportOptionsControl.appendChild(exportOptionsToggle);

	var exportOptionsPanel = document.createElement("div");
	exportOptionsPanel.className = "results-export-options-panel";
	exportOptionsControl.appendChild(exportOptionsPanel);

	var exportOptionsList = document.createElement("div");
	exportOptionsList.className = "results-export-options-list";
	exportOptionsPanel.appendChild(exportOptionsList);

	var delimiterLabel = document.createElement("label");
	delimiterLabel.className = "results-export-option";
	delimiterLabel.appendChild(document.createTextNode("Delimiter "));
	var delimiterSelect = document.createElement("select");
	delimiterSelect.className = "results-export-select";

	[[",", "Comma"], [";", "Semicolon"], ["\t", "Tab"], ["|", "Pipe"]].forEach(function (choice) {
		var option = document.createElement("option");
		option.value = choice[0];
		option.textContent = choice[1];
		option.selected = exportOptions.delimiter === choice[0];
		delimiterSelect.appendChild(option);
	});

	delimiterSelect.addEventListener("change", function () {
		exportOptions.delimiter = delimiterSelect.value;
		writeStoredJson(exportOptionsStorageKey, exportOptions);
	});

	delimiterLabel.appendChild(delimiterSelect);
	exportOptionsList.appendChild(delimiterLabel);

	var headerOptionLabel = document.createElement("label");
	headerOptionLabel.className = "results-export-option";
	var headerOptionCheckbox = document.createElement("input");
	headerOptionCheckbox.type = "checkbox";
	headerOptionCheckbox.checked = exportOptions.includeHeader;
	headerOptionCheckbox.addEventListener("change", function () {
		exportOptions.includeHeader = headerOptionCheckbox.checked;
		writeStoredJson(exportOptionsStorageKey, exportOptions);
	});
	headerOptionLabel.appendChild(headerOptionCheckbox);
	headerOptionLabel.appendChild(document.createTextNode(" Include header row"));
	exportOptionsList.appendChild(headerOptionLabel);

	var quoteAllOptionLabel = document.createElement("label");
	quoteAllOptionLabel.className = "results-export-option";
	var quoteAllCheckbox = document.createElement("input");
	quoteAllCheckbox.type = "checkbox";
	quoteAllCheckbox.checked = exportOptions.quoteAll;
	quoteAllCheckbox.addEventListener("change", function () {
		exportOptions.quoteAll = quoteAllCheckbox.checked;
		writeStoredJson(exportOptionsStorageKey, exportOptions);
	});
	quoteAllOptionLabel.appendChild(quoteAllCheckbox);
	quoteAllOptionLabel.appendChild(document.createTextNode(" Quote all fields"));
	exportOptionsList.appendChild(quoteAllOptionLabel);

	toolbar.appendChild(exportOptionsControl);

	var actions = document.createElement("div");
	actions.className = "results-toolbar-actions";
	toolbar.appendChild(actions);

	var copyButton = document.createElement("button");
	copyButton.type = "button";
	copyButton.textContent = "Copy CSV";
	copyButton.className = "results-button";
	actions.appendChild(copyButton);

	var copyVisibleButton = document.createElement("button");
	copyVisibleButton.type = "button";
	copyVisibleButton.textContent = "Copy visible CSV";
	copyVisibleButton.className = "results-button";
	actions.appendChild(copyVisibleButton);

	var downloadButton = document.createElement("button");
	downloadButton.type = "button";
	downloadButton.textContent = "Download CSV";
	downloadButton.className = "results-button";
	actions.appendChild(downloadButton);

	var downloadVisibleButton = document.createElement("button");
	downloadVisibleButton.type = "button";
	downloadVisibleButton.textContent = "Download visible CSV";
	downloadVisibleButton.className = "results-button";
	actions.appendChild(downloadVisibleButton);

	container.appendChild(toolbar);

	var scroll = document.createElement("div");
	scroll.className = "results-table-scroll";

	var table = document.createElement("table");
	table.className = "results-table";
	var thead = document.createElement("thead");
	var headRow = document.createElement("tr");
	var arrows = [];
	var headerCells = [];

	for (var c = 0; c < headers.length; c++) {
		var th = document.createElement("th");
		th.dataset.columnId = headers[c];
		headerCells.push(th);
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

	function visibleColumnIndices() {
		var indices = [];

		for (var c = 0; c < headers.length; c++) {
			if (!state.hiddenColumns[c]) {
				indices.push(c);
			}
		}

		return indices;
	}

	function createSpacer(height) {
		var tr = document.createElement("tr");
		tr.className = "results-spacer";
		var td = document.createElement("td");
		td.colSpan = Math.max(1, visibleColumnIndices().length);
		td.style.height = height + "px";
		tr.appendChild(td);
		return tr;
	}

	function saveColumnSelection() {
		var hidden = [];

		for (var c = 0; c < headers.length; c++) {
			if (state.hiddenColumns[c]) {
				hidden.push(headers[c]);
			}
		}

		writeStoredJson(columnStorageKey, hidden);
	}

	function selectionMatchesInteresting() {
		for (var c = 0; c < headers.length; c++) {
			var interesting = _resultsIsInterestingColumn(headers[c], resultNames, technicalLookup);
			var hidden = !!state.hiddenColumns[c];

			if (interesting === hidden) {
				return false;
			}
		}

		return true;
	}

	function updateColumnControls() {
		var visibleCount = visibleColumnIndices().length;
		columnsToggle.textContent = "Columns (" + visibleCount + "/" + headers.length + ")";
		interestingToggleButton.textContent = state.interestingOnly ? "Show all columns" : "Show only interesting columns";
		interestingToggleButton.setAttribute("aria-pressed", state.interestingOnly ? "true" : "false");
		interestingToggleButton.classList.toggle("active", state.interestingOnly);
	}

	function applyColumnVisibility() {
		for (var c = 0; c < headers.length; c++) {
			var hidden = !!state.hiddenColumns[c];

			if (headerCells[c]) {
				headerCells[c].style.display = hidden ? "none" : "";
			}

			if (columnCheckboxes[c]) {
				columnCheckboxes[c].checked = !hidden;
			}
		}

		updateColumnControls();
	}

	function setColumnVisible(columnIndex, visible) {
		if (visible) {
			delete state.hiddenColumns[columnIndex];
		} else {
			state.hiddenColumns[columnIndex] = true;
		}

		state.interestingOnly = false;
		applyColumnVisibility();
		saveColumnSelection();
		renderWindow(true);
	}

	function setInterestingColumnsOnly() {
		for (var c = 0; c < headers.length; c++) {
			if (_resultsIsInterestingColumn(headers[c], resultNames, technicalLookup)) {
				delete state.hiddenColumns[c];
			} else {
				state.hiddenColumns[c] = true;
			}
		}

		state.interestingOnly = true;
		applyColumnVisibility();
		saveColumnSelection();
		renderWindow(true);
	}

	function setMinimalColumnsOnly() {
		for (var c = 0; c < headers.length; c++) {
			if (_resultsIsMinimalColumn(headers[c], resultNames, technicalLookup)) {
				delete state.hiddenColumns[c];
			} else {
				state.hiddenColumns[c] = true;
			}
		}

		state.interestingOnly = false;
		applyColumnVisibility();
		saveColumnSelection();
		renderWindow(true);
	}

	function showAllColumns() {
		state.hiddenColumns = {};
		state.interestingOnly = false;
		applyColumnVisibility();
		saveColumnSelection();
		renderWindow(true);
	}

	function toggleInterestingColumns() {
		if (state.interestingOnly) {
			showAllColumns();
		} else {
			setInterestingColumnsOnly();
		}
	}

	function exportCsvRows(rows) {
		var indices = visibleColumnIndices();
		var visibleHeaders = indices.map(function (index) {
			return headers[index];
		});

		return _resultsCsvString(visibleHeaders, _resultsProjectRows(rows, indices), exportOptions);
	}

	function updateView() {
		var view = _resultsFilterRows(data, state.search, state.externalFilter);

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
		var hasFilter = state.externalFilter !== null || state.search.trim() !== "";
		if (total === 0) {
			(lastFirstRow = -1);
			lastLastRow = -1;
			tbody.innerHTML = "";
			info.textContent = hasFilter ? "No matching results (0 of " + data.length + ")" : "No matching results";
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
				if (state.hiddenColumns[c]) {
					continue;
				}

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
		info.textContent = hasFilter ? total + " of " + data.length + " results" : total + " results";
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

	interestingToggleButton.addEventListener("click", toggleInterestingColumns);
	minimalButton.addEventListener("click", setMinimalColumnsOnly);
	showAllButton.addEventListener("click", showAllColumns);

	copyButton.addEventListener("click", function () {
		copy_to_clipboard(exportCsvRows(data));
	});

	copyVisibleButton.addEventListener("click", function () {
		var rows = state.view.map(function (index) {
			return data[index];
		});
		copy_to_clipboard(exportCsvRows(rows));
	});

	function downloadCsv(rows) {
		var csv = exportCsvRows(rows);
		var blob = new Blob([csv], { type: "text/csv" });
		var link = document.createElement("a");
		link.href = URL.createObjectURL(blob);
		link.download = table_name + ".csv";
		document.body.appendChild(link);
		link.click();
		document.body.removeChild(link);
		URL.revokeObjectURL(link.href);
	}

	downloadButton.addEventListener("click", function () {
		downloadCsv(data);
	});

	downloadVisibleButton.addEventListener("click", function () {
		var rows = state.view.map(function (index) {
			return data[index];
		});
		downloadCsv(rows);
	});

	state.interestingOnly = selectionMatchesInteresting();
	applyColumnVisibility();
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

	function refreshFilteredView() {
		updateView();
		scroll.scrollTop = 0;
		renderWindow(true);
	}

	return {
		setExternalFilter: function (filter) {
			state.externalFilter = typeof filter === "function" ? filter : null;
			refreshFilteredView();
		},
		clearExternalFilter: function () {
			state.externalFilter = null;
			refreshFilteredView();
		},
		setSearch: function (text) {
			searchInput.value = text || "";
			state.search = searchInput.value;
			refreshFilteredView();
		},
		getVisibleCount: function () {
			return state.view.length;
		},
		getTotalCount: function () {
			return data.length;
		},
		getData: function () {
			return data;
		},
		getHeaders: function () {
			return headers;
		}
	};
}
