function plotScatter3d() {
	if ($("#plotScatter3d").data("loaded") == "true") {
		return;
	}

	var plotDiv = document.getElementById("plotScatter3d");
	if (!plotDiv) {
		console.error("Div element with id 'plotScatter3d' not found");
		return;
	}
	var previousState = captureScatter3dState(plotDiv);
	plotDiv.innerHTML = "";

	var controlsDiv = document.createElement("div");
	controlsDiv.style.display = "flex";
	controlsDiv.style.flexWrap = "wrap";
	controlsDiv.style.alignItems = "center";
	controlsDiv.style.gap = "10px";
	controlsDiv.style.marginBottom = "10px";

	var minInput3d = document.createElement("input");
	minInput3d.id = "minValue3d";
	minInput3d.type = "number";
	minInput3d.placeholder = "Min Value";
	minInput3d.step = "any";

	var maxInput3d = document.createElement("input");
	maxInput3d.id = "maxValue3d";
	maxInput3d.type = "number";
	maxInput3d.placeholder = "Max Value";
	maxInput3d.step = "any";

	controlsDiv.appendChild(minInput3d);
	controlsDiv.appendChild(maxInput3d);

	var select3d = null;

	if (result_names.length > 1) {
		select3d = document.createElement("select");
		select3d.id = "select3dScatter";
		select3d.innerHTML = result_names.map(name => `<option value="${name}">${name}</option>`).join("");
		controlsDiv.appendChild(select3d);
	}

	var comboDetails = document.createElement("details");
	comboDetails.id = "scatter3d_comboDetails";
	comboDetails.open = true;
	comboDetails.style.marginBottom = "10px";

	var summary = document.createElement("summary");
	summary.style.cursor = "pointer";
	comboDetails.appendChild(summary);

	var comboButtons = document.createElement("div");
	comboButtons.style.display = "flex";
	comboButtons.style.flexWrap = "wrap";
	comboButtons.style.gap = "6px";
	comboButtons.style.margin = "8px 0";

	var filterInput = document.createElement("input");
	filterInput.id = "scatter3d_filterInput";
	filterInput.type = "search";
	filterInput.placeholder = "Filter combinations...";
	filterInput.style.width = "100%";
	filterInput.style.boxSizing = "border-box";
	filterInput.style.marginBottom = "8px";

	var comboList = document.createElement("div");
	comboList.id = "scatter3d_comboList";
	comboList.style.maxHeight = "260px";
	comboList.style.overflowY = "auto";
	comboList.style.border = "1px solid #bbb";
	comboList.style.borderRadius = "6px";
	comboList.style.padding = "8px";

	comboDetails.appendChild(comboButtons);
	comboDetails.appendChild(filterInput);
	comboDetails.appendChild(comboList);

	var plotsContainer = document.createElement("div");
	plotsContainer.id = "scatter3d_plots";

	plotDiv.appendChild(controlsDiv);
	plotDiv.appendChild(comboDetails);
	plotDiv.appendChild(plotsContainer);

	var numericColumns = [];
	var combos = [];
	var defaultKeys = [];
	var selectedKeys = new Set();
	var checkboxes = {};
	var updateTimer = null;
	var updateToken = 0;

	function captureScatter3dState(container) {
		var boxes = container.querySelectorAll("#scatter3d_comboList input[type='checkbox'][data-combo-key]");

		if (boxes.length === 0) {
			return null;
		}

		var state = {
			selectedKeys: new Set(),
			minValue: (container.querySelector("#minValue3d") || {}).value || "",
			maxValue: (container.querySelector("#maxValue3d") || {}).value || "",
			result: (container.querySelector("#select3dScatter") || {}).value || "",
			filter: (container.querySelector("#scatter3d_filterInput") || {}).value || "",
			open: (container.querySelector("#scatter3d_comboDetails") || {}).open !== false
		};

		boxes.forEach(function (checkbox) {
			if (checkbox.checked) {
				state.selectedKeys.add(checkbox.dataset.comboKey);
			}
		});

		return state;
	}

	function getNumericColumns() {
		var headers = tab_results_headers_json;
		var rows = tab_results_csv_json;
		var headerIndex = new Map();
		var specialSet = new Set(special_col_names);
		var resultSet = new Set(result_names);

		for (var i = 0; i < headers.length; i++) {
			headerIndex.set(headers[i], i);
		}

		var columns = [];

		for (var c = 0; c < headers.length; c++) {
			var col = headers[c];

			if (specialSet.has(col) || resultSet.has(col) || col.startsWith("OO_Info")) {
				continue;
			}

			var colIndex = headerIndex.get(col);
			var isNumeric = true;

			for (var r = 0; r < rows.length; r++) {
				if (isNaN(parseFloat(rows[r][colIndex]))) {
					isNumeric = false;
					break;
				}
			}

			if (isNumeric) {
				columns.push(col);
			}
		}

		return columns;
	}

	function computeCombos() {
		numericColumns = getNumericColumns();
		combos = [];

		for (var i = 0; i < numericColumns.length; i++) {
			for (var j = i + 1; j < numericColumns.length; j++) {
				for (var k = j + 1; k < numericColumns.length; k++) {
					combos.push({
						a: numericColumns[i],
						b: numericColumns[j],
						c: numericColumns[k],
						key: `triple:${numericColumns[i]}:${numericColumns[j]}:${numericColumns[k]}`
					});
				}
			}
		}

		defaultKeys = combos.length ? [combos[0].key] : [];
	}

	function comboLabel(combo) {
		return `${combo.a} vs ${combo.b} vs ${combo.c}`;
	}

	function updateSummary() {
		summary.textContent = `Plot combinations (${selectedKeys.size} selected)`;
	}

	function syncCheckboxes() {
		Object.keys(checkboxes).forEach(key => {
			checkboxes[key].checked = selectedKeys.has(key);
		});
		updateSummary();
	}

	function renderComboList() {
		comboList.innerHTML = "";
		checkboxes = {};
		combos.forEach(combo => {
			var label = document.createElement("label");
			label.style.display = "flex";
			label.style.alignItems = "center";
			label.style.gap = "6px";
			label.style.padding = "2px 0";
			label.dataset.search = comboLabel(combo).toLowerCase();

			var checkbox = document.createElement("input");
			checkbox.type = "checkbox";
			checkbox.dataset.comboKey = combo.key;
			checkbox.checked = selectedKeys.has(combo.key);
			checkbox.addEventListener("change", () => {
				if (checkbox.checked) {
					selectedKeys.add(combo.key);
				} else {
					selectedKeys.delete(combo.key);
				}
				updateSummary();
				scheduleUpdate();
			});

			checkboxes[combo.key] = checkbox;
			label.appendChild(checkbox);
			label.appendChild(document.createTextNode(comboLabel(combo)));
			comboList.appendChild(label);
		});
		updateSummary();
	}

	function scheduleUpdate() {
		clearTimeout(updateTimer);
		updateTimer = setTimeout(updatePlots3d, 150);
	}

	function makeButton(text, handler) {
		var button = document.createElement("button");
		button.type = "button";
		button.textContent = text;
		button.addEventListener("click", handler);
		return button;
	}

	function clearPlotsContainer() {
		Array.from(plotsContainer.children).forEach(child => {
			if (typeof Plotly !== "undefined" && typeof Plotly.purge === "function" &&
				child.classList && child.classList.contains("js-plotly-plot")) {
				try {
					Plotly.purge(child);
				} catch (err) {
					//
				}
			}
		});
		plotsContainer.innerHTML = "";
	}

	function showMessage(message) {
		clearPlotsContainer();
		plotsContainer.textContent = message;
	}

	comboButtons.appendChild(makeButton("Select all", () => {
		selectedKeys = new Set(combos.map(combo => combo.key));
		syncCheckboxes();
		scheduleUpdate();
	}));
	comboButtons.appendChild(makeButton("Select none", () => {
		selectedKeys = new Set();
		syncCheckboxes();
		scheduleUpdate();
	}));

	filterInput.addEventListener("input", () => {
		var query = filterInput.value.toLowerCase();
		Array.from(comboList.children).forEach(label => {
			label.style.display = label.dataset.search.includes(query) ? "flex" : "none";
		});
	});

	minInput3d.addEventListener("input", scheduleUpdate);
	maxInput3d.addEventListener("input", scheduleUpdate);
	if (select3d) {
		select3d.addEventListener("change", scheduleUpdate);
	}

	computeCombos();

	if (previousState) {
		var validKeys = new Set(combos.map(combo => combo.key));
		selectedKeys = new Set(Array.from(previousState.selectedKeys).filter(key => validKeys.has(key)));

		if (selectedKeys.size === 0) {
			selectedKeys = new Set(defaultKeys);
		}

		minInput3d.value = previousState.minValue;
		maxInput3d.value = previousState.maxValue;
		filterInput.value = previousState.filter;
		comboDetails.open = previousState.open;

		if (select3d && previousState.result) {
			select3d.value = previousState.result;
		}
	} else {
		selectedKeys = new Set(defaultKeys);
	}

	renderComboList();
	updatePlots3d();

	async function updatePlots3d() {
		var token = ++updateToken;
		var selectedResult = select3d ? select3d.value : result_names[0];
		var minValue3d = parseFloat(minInput3d.value);
		var maxValue3d = parseFloat(maxInput3d.value);

		if (isNaN(minValue3d)) minValue3d = -Infinity;
		if (isNaN(maxValue3d)) maxValue3d = Infinity;

		var resultIndex = tab_results_headers_json.findIndex(header =>
			header.toLowerCase() === String(selectedResult).toLowerCase()
		);

		if (resultIndex < 0) {
			showMessage("No result column selected.");
			return;
		}

		var resultValues = tab_results_csv_json
			.map(row => parseFloat(row[resultIndex]))
			.filter(value => !isNaN(value));

		if (resultValues.length === 0) {
			showMessage("No numeric result values available.");
			return;
		}

		var minResult = Math.min(...resultValues);
		var maxResult = Math.max(...resultValues);

		if (minValue3d !== -Infinity) minResult = Math.max(minResult, minValue3d);
		if (maxValue3d !== Infinity) maxResult = Math.min(maxResult, maxValue3d);

		var invertColor = typeof result_min_max !== "undefined" &&
			result_min_max[result_names.indexOf(selectedResult)] === "max";
		var combosToPlot = combos.filter(combo => selectedKeys.has(combo.key));

		if (combosToPlot.length === 0) {
			showMessage("Select at least one combination to display.");
			return;
		}

		var comboPlotIndex = 0;

		for (const combo of combosToPlot) {
			if (token !== updateToken) {
				return;
			}

			var xCol = combo.a;
			var yCol = combo.b;
			var zCol = combo.c;
			var xIndex = tab_results_headers_json.indexOf(xCol);
			var yIndex = tab_results_headers_json.indexOf(yCol);
			var zIndex = tab_results_headers_json.indexOf(zCol);

			if (xIndex < 0 || yIndex < 0 || zIndex < 0) {
				continue;
			}

			var data = tab_results_csv_json.map(row => ({
				x: parseFloat(row[xIndex]),
				y: parseFloat(row[yIndex]),
				z: parseFloat(row[zIndex]),
				result: row[resultIndex] !== "" ? parseFloat(row[resultIndex]) : null
			}));

			data = data.filter(d =>
				d.result !== null && !isNaN(d.result) && d.result >= minResult && d.result <= maxResult
			);

			var layoutTitle = `${xCol} (x) vs ${yCol} (y) vs ${zCol} (z), result: ${selectedResult}`;
			var layout = {
				title: layoutTitle,
				scene: {
					xaxis: {
						title: get_axis_title_data(xCol)
					},
					yaxis: {
						title: get_axis_title_data(yCol)
					},
					zaxis: {
						title: get_axis_title_data(zCol)
					}
				},
				showlegend: false
			};

			var trace = {
				x: data.map(d => d.x),
				y: data.map(d => d.y),
				z: data.map(d => d.z),
				mode: 'markers',
				marker: {
					size: get_marker_size(),
					color: data.map(d => d.result !== null ? d.result : null),
					colorscale: invertColor ? [
						[0, 'red'],
						[1, 'green']
					] : [
						[0, 'green'],
						[1, 'red']
					],
					colorbar: {
						title: 'Result',
						tickvals: [minResult, maxResult],
						ticktext: [`${minResult}`, `${maxResult}`]
					},
				},
				text: data.map(d => d.result !== null ? `Result: ${d.result}` : 'No result'),
				type: 'scatter3d',
				showlegend: false
			};

			var subDiv = plotsContainer.children[comboPlotIndex];

			if (!subDiv) {
				subDiv = document.createElement("div");
				plotsContainer.appendChild(subDiv);
			}

			comboPlotIndex++;

			if (typeof Plotly.react === "function") {
				await Plotly.react(subDiv, [trace], add_default_layout_data(layout));
			} else {
				await Plotly.newPlot(subDiv, [trace], add_default_layout_data(layout));
			}
		}

		while (plotsContainer.children.length > comboPlotIndex) {
			plotsContainer.removeChild(plotsContainer.lastChild);
		}
	}

	window.__scatter3d_refresh = function () {
		var oldSelectedKeys = selectedKeys;
		computeCombos();

		var validKeys = new Set(combos.map(combo => combo.key));
		selectedKeys = new Set(Array.from(oldSelectedKeys).filter(key => validKeys.has(key)));

		if (selectedKeys.size === 0) {
			selectedKeys = new Set(defaultKeys);
		}

		renderComboList();
		updatePlots3d();
	};

	$("#plotScatter3d").data("loaded", "true");
}
