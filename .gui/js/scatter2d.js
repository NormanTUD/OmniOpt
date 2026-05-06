function plotScatter2d() {
	if ($("#plotScatter2d").data("loaded") == "true") {
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
		sortedResults.forEach(result => {
			var option = document.createElement("option");
			option.value = result;
			option.textContent = result;
			resultSelect.appendChild(option);
		});

		var selectContainer = document.createElement("div");
		selectContainer.style.marginBottom = "10px";
		selectContainer.appendChild(resultSelect);
		plotDiv.appendChild(selectContainer);
	}

	minInput.addEventListener("input", updatePlots);
	maxInput.addEventListener("input", updatePlots);
	if (resultSelect) {
		resultSelect.addEventListener("change", updatePlots);
	}

	updatePlots();

	async function updatePlots() {
		var minValue = parseFloat(minInput.value);
		var maxValue = parseFloat(maxInput.value);
		if (isNaN(minValue)) minValue = -Infinity;
		if (isNaN(maxValue)) maxValue = Infinity;

		while (plotDiv.children.length > 2) {
			plotDiv.removeChild(plotDiv.lastChild);
		}

		var selectedResult = resultSelect ? resultSelect.value : result_names[0];

		var resultIndex = tab_results_headers_json.findIndex(header =>
			header.toLowerCase() === selectedResult.toLowerCase()
		);
		var resultValues = tab_results_csv_json.map(row => row[resultIndex]);

		var minResult = Math.min(...resultValues.filter(value => value !== null && value !== ""));
		var maxResult = Math.max(...resultValues.filter(value => value !== null && value !== ""));

		if (minValue !== -Infinity) minResult = Math.max(minResult, minValue);
		if (maxValue !== Infinity) maxResult = Math.min(maxResult, maxValue);

		var invertColor = result_min_max[result_names.indexOf(selectedResult)] === "max";

		var numericColumns = tab_results_headers_json.filter(col =>
			!special_col_names.includes(col) && !result_names.includes(col) &&
			!col.startsWith("OO_Info") &&
			tab_results_csv_json.every(row => !isNaN(parseFloat(row[tab_results_headers_json.indexOf(col)])))
		);

		if (numericColumns.length < 2) {
			console.error("Not enough columns for Scatter-Plots");
			return;
		}

		for (let i = 0; i < numericColumns.length; i++) {
			for (let j = i + 1; j < numericColumns.length; j++) {
				let xCol = numericColumns[i];
				let yCol = numericColumns[j];

				let xIndex = tab_results_headers_json.indexOf(xCol);
				let yIndex = tab_results_headers_json.indexOf(yCol);

				let data = tab_results_csv_json.map(row => ({
					x: parseFloat(row[xIndex]),
					y: parseFloat(row[yIndex]),
					result: row[resultIndex] !== "" ? parseFloat(row[resultIndex]) : null
				}));

				data = data.filter(d => d.result >= minResult && d.result <= maxResult);

				let layoutTitle = `${xCol} (x) vs ${yCol} (y), result: ${selectedResult}`;
				let layout = {
					title: layoutTitle,
					xaxis: {
						title: get_axis_title_data(xCol)
					},
					yaxis: {
						title: get_axis_title_data(yCol)
					},
					showlegend: false
				};
				let subDiv = document.createElement("div");

				let spinnerContainer = document.createElement("div");
				spinnerContainer.style.display = "flex";
				spinnerContainer.style.alignItems = "center";
				spinnerContainer.style.justifyContent = "center";
				spinnerContainer.style.width = layout.width + "px";
				spinnerContainer.style.height = layout.height + "px";
				spinnerContainer.style.position = "relative";

				let spinner = document.createElement("div");
				spinner.className = "spinner";
				spinner.style.width = "40px";
				spinner.style.height = "40px";

				let loadingText = document.createElement("span");
				loadingText.innerText = `Loading ${layoutTitle}`;
				loadingText.style.marginLeft = "10px";

				spinnerContainer.appendChild(spinner);
				spinnerContainer.appendChild(loadingText);

				plotDiv.appendChild(spinnerContainer);

				await new Promise(resolve => setTimeout(resolve, 50));

				let colors = data.map(d => {
					if (d.result === null) {
						return 'rgb(0, 0, 0)';
					} else {
						let norm = (d.result - minResult) / (maxResult - minResult);
						if (invertColor) {
							norm = 1 - norm;
						}
						return `rgb(${Math.round(255 * norm)}, ${Math.round(255 * (1 - norm))}, 0)`;
					}
				});

				let trace = {
					x: data.map(d => d.x),
					y: data.map(d => d.y),
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
						symbol: data.map(d => d.result === null ? 'x' : 'circle'),
					},
					text: data.map(d => d.result !== null ? `Result: ${d.result}` : 'No result'),
					type: 'scatter',
					showlegend: false
				};

				try {
					plotDiv.replaceChild(subDiv, spinnerContainer);
				} catch (err) {
					//
				}
				Plotly.newPlot(subDiv, [trace], add_default_layout_data(layout));
			}
		}
	}

	$("#plotScatter2d").data("loaded", "true");
}
