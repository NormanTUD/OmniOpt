function plotScatter2d() {
	if ($("#plotScatter2d").data("loaded") === "true") {
		return;
	}

	var plotDiv = document.getElementById("plotScatter2d");

	var minInput = document.getElementById("minValue");
	var maxInput = document.getElementById("maxValue");

	if (!minInput || !maxInput) {
		minInput = document.createElement("input");
		minInput.id = "minValue";
		minInput.type = "number";
		minInput.placeholder = "Min Value";
		minInput.step = "any";

		maxInput = document.createElement("input");
		maxInput.id = "maxValue";
		maxInput.type = "number";
		maxInput.placeholder = "Max Value";
		maxInput.step = "any";

		var inputContainer = document.createElement("div");
		inputContainer.style.marginBottom = "10px";
		inputContainer.appendChild(minInput);
		inputContainer.appendChild(maxInput);
		plotDiv.appendChild(inputContainer);
	}

	var resultSelect = document.getElementById("resultSelect");
	if (result_names.length > 1 && !resultSelect) {
		resultSelect = document.createElement("select");
		resultSelect.id = "resultSelect";
		resultSelect.style.marginBottom = "10px";

		var sortedResults = [...result_names].sort();
		for (var i = 0; i < sortedResults.length; i++) {
			var option = document.createElement("option");
			option.value = sortedResults[i];
			option.textContent = sortedResults[i];
			resultSelect.appendChild(option);
		}

		var selectContainer = document.createElement("div");
		selectContainer.style.marginBottom = "10px";
		selectContainer.appendChild(resultSelect);
		plotDiv.appendChild(selectContainer);
	}

	// Debounce to avoid re-rendering on every keystroke
	var debounceTimer = null;
	function debouncedUpdate() {
		if (debounceTimer) clearTimeout(debounceTimer);
		debounceTimer = setTimeout(updatePlots, 250);
	}

	minInput.addEventListener("input", debouncedUpdate);
	maxInput.addEventListener("input", debouncedUpdate);
	if (resultSelect) {
		resultSelect.addEventListener("change", updatePlots);
	}

	// Cache for plot sub-divs so we can reuse them
	var plotSubDivs = [];

	// Pre-compute numeric columns once (they don't change)
	var numericColumns = [];
	var numericColumnIndices = [];
	for (var ci = 0; ci < tab_results_headers_json.length; ci++) {
		var col = tab_results_headers_json[ci];
		if (special_col_names.includes(col) || result_names.includes(col) || col.startsWith("OO_Info")) {
			continue;
		}
		var isNumeric = true;
		for (var ri = 0; ri < tab_results_csv_json.length; ri++) {
			if (isNaN(parseFloat(tab_results_csv_json[ri][ci]))) {
				isNumeric = false;
				break;
			}
		}
		if (isNumeric) {
			numericColumns.push(col);
			numericColumnIndices.push(ci);
		}
	}

	// Pre-parse all numeric data into typed arrays for speed
	var numRows = tab_results_csv_json.length;
	var parsedNumericData = new Array(numericColumnIndices.length);
	for (var c = 0; c < numericColumnIndices.length; c++) {
		parsedNumericData[c] = new Float64Array(numRows);
		var colIdx = numericColumnIndices[c];
		for (var r = 0; r < numRows; r++) {
			parsedNumericData[c][r] = parseFloat(tab_results_csv_json[r][colIdx]);
		}
	}

	if (numericColumns.length < 2) {
		console.error("Not enough columns for Scatter-Plots");
		$("#plotScatter2d").data("loaded", "true");
		return;
	}

	// Calculate total number of plot pairs
	var totalPairs = (numericColumns.length * (numericColumns.length - 1)) / 2;

	updatePlots();

	function updatePlots() {
		var minValue = parseFloat(minInput.value);
		var maxValue = parseFloat(maxInput.value);
		if (isNaN(minValue)) minValue = -Infinity;
		if (isNaN(maxValue)) maxValue = Infinity;

		var selectedResult = resultSelect ? resultSelect.value : result_names[0];

		var resultIndex = -1;
		var selectedLower = selectedResult.toLowerCase();
		for (var h = 0; h < tab_results_headers_json.length; h++) {
			if (tab_results_headers_json[h].toLowerCase() === selectedLower) {
				resultIndex = h;
				break;
			}
		}

		// Pre-parse result values once
		var parsedResults = new Float64Array(numRows);
		var resultValid = new Uint8Array(numRows);
		var minResult = Infinity;
		var maxResult = -Infinity;

		for (var r = 0; r < numRows; r++) {
			var raw = tab_results_csv_json[r][resultIndex];
			if (raw !== null && raw !== "") {
				var val = parseFloat(raw);
				parsedResults[r] = val;
				resultValid[r] = 1;
				if (val < minResult) minResult = val;
				if (val > maxResult) maxResult = val;
			} else {
				parsedResults[r] = NaN;
				resultValid[r] = 0;
			}
		}

		if (minValue !== -Infinity) minResult = Math.max(minResult, minValue);
		if (maxValue !== Infinity) maxResult = Math.min(maxResult, maxValue);

		var invertColor = result_min_max[result_names.indexOf(selectedResult)] === "max";

		var colorscale = invertColor
			? [[0, 'red'], [1, 'green']]
			: [[0, 'green'], [1, 'red']];

		// Pre-filter rows that pass the result range
		var filteredIndices = [];
		for (var r = 0; r < numRows; r++) {
			if (resultValid[r] && parsedResults[r] >= minResult && parsedResults[r] <= maxResult) {
				filteredIndices.push(r);
			}
		}
		var filteredCount = filteredIndices.length;

		// Pre-extract filtered result values and symbols
		var filteredResults = new Float64Array(filteredCount);
		var filteredTexts = new Array(filteredCount);
		for (var f = 0; f < filteredCount; f++) {
			var rv = parsedResults[filteredIndices[f]];
			filteredResults[f] = rv;
			filteredTexts[f] = "Result: " + rv;
		}

		var markerSize = get_marker_size();

		// Remove old plot divs beyond the control elements
		while (plotDiv.children.length > 2) {
			plotDiv.removeChild(plotDiv.lastChild);
		}

		// Use DocumentFragment for batch DOM insertion
		var fragment = document.createDocumentFragment();
		plotSubDivs = [];

		for (var p = 0; p < totalPairs; p++) {
			var subDiv = document.createElement("div");
			plotSubDivs.push(subDiv);
			fragment.appendChild(subDiv);
		}
		plotDiv.appendChild(fragment);

		// Now render all plots without awaiting — use requestAnimationFrame batching
		var pairIndex = 0;
		var batchSize = 4; // render 4 plots per animation frame

		function renderBatch() {
			var rendered = 0;
			while (rendered < batchSize && pairIndex < totalPairs) {
				// Decode pair index back to i, j
				var i, j;
				var idx = pairIndex;
				// Find i, j from linear index
				i = 0;
				var remaining = idx;
				while (remaining >= (numericColumns.length - 1 - i)) {
					remaining -= (numericColumns.length - 1 - i);
					i++;
				}
				j = i + 1 + remaining;

				var xData = parsedNumericData[i];
				var yData = parsedNumericData[j];

				// Build trace arrays from pre-filtered indices
				var xArr = new Array(filteredCount);
				var yArr = new Array(filteredCount);
				for (var f = 0; f < filteredCount; f++) {
					var ri = filteredIndices[f];
					xArr[f] = xData[ri];
					yArr[f] = yData[ri];
				}

				var xCol = numericColumns[i];
				var yCol = numericColumns[j];

				var trace = {
					x: xArr,
					y: yArr,
					mode: 'markers',
					marker: {
						size: markerSize,
						color: filteredResults,
						colorscale: colorscale,
						colorbar: {
							title: 'Result',
							tickvals: [minResult, maxResult],
							ticktext: [String(minResult), String(maxResult)]
						},
						symbol: 'circle'
					},
					text: filteredTexts,
					type: 'scattergl',  // Use WebGL renderer for speed
					showlegend: false
				};

				var layout = {
					title: xCol + " (x) vs " + yCol + " (y), result: " + selectedResult,
					xaxis: { title: get_axis_title_data(xCol) },
					yaxis: { title: get_axis_title_data(yCol) },
					showlegend: false
				};

				Plotly.newPlot(plotSubDivs[pairIndex], [trace], add_default_layout_data(layout));

				pairIndex++;
				rendered++;
			}

			if (pairIndex < totalPairs) {
				requestAnimationFrame(renderBatch);
			}
		}

		requestAnimationFrame(renderBatch);
	}

	$("#plotScatter2d").data("loaded", "true");
}
