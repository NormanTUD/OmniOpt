function plotScatter2d() {
	if ($("#plotScatter2d").data("loaded") == "true") {
		return;
	}

	var plotDiv = document.getElementById("plotScatter2d");
	plotDiv.innerHTML = "";

	var controlsDiv = document.createElement("div");
	controlsDiv.style.display = "flex";
	controlsDiv.style.flexWrap = "wrap";
	controlsDiv.style.alignItems = "center";
	controlsDiv.style.gap = "10px";
	controlsDiv.style.marginBottom = "10px";

	var minInput = document.createElement("input");
	minInput.id = "scatter2d_minValue";
	minInput.type = "number";
	minInput.placeholder = "Min result";
	minInput.step = "any";

	var maxInput = document.createElement("input");
	maxInput.id = "scatter2d_maxValue";
	maxInput.type = "number";
	maxInput.placeholder = "Max result";
	maxInput.step = "any";

	controlsDiv.appendChild(minInput);
	controlsDiv.appendChild(maxInput);

	var resultSelect = null;
	if (result_names.length > 1) {
		resultSelect = document.createElement("select");
		resultSelect.id = "scatter2d_resultSelect";

		var sortedResults = [...result_names].sort();
		sortedResults.forEach(result => {
			var option = document.createElement("option");
			option.value = result;
			option.textContent = result;
			resultSelect.appendChild(option);
		});
		controlsDiv.appendChild(resultSelect);
	}

	var comboDetails = document.createElement("details");
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
	filterInput.type = "search";
	filterInput.placeholder = "Filter combinations...";
	filterInput.style.width = "100%";
	filterInput.style.boxSizing = "border-box";
	filterInput.style.marginBottom = "8px";

	var comboList = document.createElement("div");
	comboList.style.maxHeight = "260px";
	comboList.style.overflowY = "auto";
	comboList.style.border = "1px solid #bbb";
	comboList.style.borderRadius = "6px";
	comboList.style.padding = "8px";

	comboDetails.appendChild(comboButtons);
	comboDetails.appendChild(filterInput);
	comboDetails.appendChild(comboList);

	var plotsContainer = document.createElement("div");
	plotsContainer.id = "scatter2d_plots";

	plotDiv.appendChild(controlsDiv);
	plotDiv.appendChild(comboDetails);
	plotDiv.appendChild(plotsContainer);

	function makeButton(text, handler) {
		var button = document.createElement("button");
		button.type = "button";
		button.textContent = text;
		button.addEventListener("click", handler);
		return button;
	}

	function getSelectedResult() {
		return resultSelect ? resultSelect.value : (result_names[0] || null);
	}

	function getResultIndex(selectedResult) {
		if (!selectedResult) return -1;
		return tab_results_headers_json.findIndex(header =>
			header.toLowerCase() === selectedResult.toLowerCase()
		);
	}

	function getParamColumns() {
		return tab_results_headers_json.filter(col =>
			!special_col_names.includes(col) && !result_names.includes(col) &&
			!col.startsWith("OO_Info") &&
			tab_results_csv_json.every(row => !isNaN(parseFloat(row[tab_results_headers_json.indexOf(col)])))
		);
	}

	var paramColumns = getParamColumns();
	var combos = [];
	paramColumns.forEach(col => {
		combos.push({ a: col, b: null, includesResult: true, key: `result:${col}` });
	});
	for (let i = 0; i < paramColumns.length; i++) {
		for (let j = i + 1; j < paramColumns.length; j++) {
			combos.push({
				a: paramColumns[i],
				b: paramColumns[j],
				includesResult: false,
				key: `pair:${paramColumns[i]}:${paramColumns[j]}`
			});
		}
	}

	var defaultKeys = combos.filter(combo => combo.includesResult).map(combo => combo.key);
	var selectedKeys = new Set(defaultKeys);
	var checkboxes = {};
	var updateTimer = null;
	var updateToken = 0;

	function comboLabel(combo) {
		if (combo.includesResult) {
			return `${combo.a} vs result`;
		}
		return `${combo.a} vs ${combo.b}`;
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
		updateTimer = setTimeout(updatePlots, 150);
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
	comboButtons.appendChild(makeButton("Only result combinations", () => {
		selectedKeys = new Set(defaultKeys);
		syncCheckboxes();
		scheduleUpdate();
	}));

	filterInput.addEventListener("input", () => {
		var query = filterInput.value.toLowerCase();
		Array.from(comboList.children).forEach(label => {
			label.style.display = label.dataset.search.includes(query) ? "flex" : "none";
		});
	});

	minInput.addEventListener("input", scheduleUpdate);
	maxInput.addEventListener("input", scheduleUpdate);
	if (resultSelect) {
		resultSelect.addEventListener("change", scheduleUpdate);
	}

	renderComboList();
	updatePlots();

	async function updatePlots() {
		var token = ++updateToken;
		var minValue = parseFloat(minInput.value);
		var maxValue = parseFloat(maxInput.value);
		if (isNaN(minValue)) minValue = -Infinity;
		if (isNaN(maxValue)) maxValue = Infinity;

		clearPlotsContainer();

		var selectedResult = getSelectedResult();
		var resultIndex = getResultIndex(selectedResult);
		if (resultIndex < 0) {
			plotsContainer.textContent = "No result column selected.";
			return;
		}

		var resultValues = tab_results_csv_json
			.map(row => parseFloat(row[resultIndex]))
			.filter(value => !isNaN(value));
		if (resultValues.length === 0) {
			plotsContainer.textContent = "No numeric result values available.";
			return;
		}

		var minResult = Math.min(...resultValues);
		var maxResult = Math.max(...resultValues);
		if (minValue !== -Infinity) minResult = Math.max(minResult, minValue);
		if (maxValue !== Infinity) maxResult = Math.min(maxResult, maxValue);

		var invertColor = typeof result_min_max !== "undefined" &&
			result_min_max[result_names.indexOf(selectedResult)] === "max";
		var combosToPlot = combos.filter(combo => selectedKeys.has(combo.key));

		if (combosToPlot.length === 0) {
			plotsContainer.textContent = "Select at least one combination to display.";
			return;
		}

		for (const combo of combosToPlot) {
			if (token !== updateToken) {
				return;
			}

			var xCol = combo.a;
			var yCol = combo.includesResult ? selectedResult : combo.b;
			var xIndex = tab_results_headers_json.indexOf(xCol);
			var yIndex = combo.includesResult ? resultIndex : tab_results_headers_json.indexOf(yCol);
			if (xIndex < 0 || yIndex < 0) {
				continue;
			}

			var data = tab_results_csv_json.map(row => ({
				x: parseFloat(row[xIndex]),
				y: parseFloat(row[yIndex]),
				result: row[resultIndex] !== "" ? parseFloat(row[resultIndex]) : null
			}));
			data = data.filter(d =>
				d.result !== null && !isNaN(d.result) && d.result >= minResult && d.result <= maxResult
			);

			var layoutTitle = `${xCol} (x) vs ${yCol} (y), result: ${selectedResult}`;
			var layout = {
				title: layoutTitle,
				xaxis: {
					title: get_axis_title_data(xCol)
				},
				yaxis: {
					title: get_axis_title_data(yCol)
				},
				showlegend: false
			};

			var trace = {
				x: data.map(d => d.x),
				y: data.map(d => d.y),
				mode: 'markers',
				marker: {
					size: get_marker_size(),
					color: data.map(d => d.result),
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
					symbol: 'circle'
				},
				text: data.map(d => `Result: ${d.result}`),
				type: 'scatter',
				showlegend: false
			};

			var subDiv = document.createElement("div");
			plotsContainer.appendChild(subDiv);
			await Plotly.newPlot(subDiv, [trace], add_default_layout_data(layout));
		}
	}

	$("#plotScatter2d").data("loaded", "true");
}
