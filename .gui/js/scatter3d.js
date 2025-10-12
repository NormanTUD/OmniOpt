function plotScatter3d() {
	if ($("#plotScatter3d").data("loaded") == "true") {
		return;
	}

	var plotDiv = document.getElementById("plotScatter3d");
	if (!plotDiv) {
		console.error("Div element with id 'plotScatter3d' not found");
		return;
	}
	plotDiv.innerHTML = "";

	var minInput3d = document.getElementById("minValue3d");
	var maxInput3d = document.getElementById("maxValue3d");

	if (!minInput3d || !maxInput3d) {
		minInput3d = document.createElement("input");
		minInput3d.id = "minValue3d";
		minInput3d.type = "number";
		minInput3d.placeholder = "Min Value";
		minInput3d.step = "any";

		maxInput3d = document.createElement("input");
		maxInput3d.id = "maxValue3d";
		maxInput3d.type = "number";
		maxInput3d.placeholder = "Max Value";
		maxInput3d.step = "any";

		var inputContainer3d = document.createElement("div");
		inputContainer3d.style.marginBottom = "10px";
		inputContainer3d.appendChild(minInput3d);
		inputContainer3d.appendChild(maxInput3d);
		plotDiv.appendChild(inputContainer3d);
	}

	var select3d = document.getElementById("select3dScatter");
	if (result_names.length > 1 && !select3d) {
		if (!select3d) {
			select3d = document.createElement("select");
			select3d.id = "select3dScatter";
			select3d.style.marginBottom = "10px";
			select3d.innerHTML = result_names.map(name => `<option value="${name}">${name}</option>`).join("");

			select3d.addEventListener("change", updatePlots3d);

			plotDiv.appendChild(select3d);
		}
	}

	minInput3d.addEventListener("input", updatePlots3d);
	maxInput3d.addEventListener("input", updatePlots3d);

	updatePlots3d();

	async function updatePlots3d() {
		var selectedResult = select3d ? select3d.value : result_names[0];
		var minValue3d = parseFloat(minInput3d.value);
		var maxValue3d = parseFloat(maxInput3d.value);

		if (isNaN(minValue3d)) minValue3d = -Infinity;
		if (isNaN(maxValue3d)) maxValue3d = Infinity;

		while (plotDiv.children.length > 2) {
			plotDiv.removeChild(plotDiv.lastChild);
		}

		var resultIndex = tab_results_headers_json.findIndex(header =>
			header.toLowerCase() === selectedResult.toLowerCase()
		);
		var resultValues = tab_results_csv_json.map(row => row[resultIndex]);

		var minResult = Math.min(...resultValues.filter(value => value !== null && value !== ""));
		var maxResult = Math.max(...resultValues.filter(value => value !== null && value !== ""));

		if (minValue3d !== -Infinity) minResult = Math.max(minResult, minValue3d);
		if (maxValue3d !== Infinity) maxResult = Math.min(maxResult, maxValue3d);

		var invertColor = result_min_max[result_names.indexOf(selectedResult)] === "max";

		var numericColumns = tab_results_headers_json.filter(col =>
			!special_col_names.includes(col) && !result_names.includes(col) &&
			!col.startsWith("OO_Info") &&
			tab_results_csv_json.every(row => !isNaN(parseFloat(row[tab_results_headers_json.indexOf(col)])))
		);

		if (numericColumns.length < 3) {
			console.error("Not enough columns for 3D scatter plots");
			return;
		}

		for (let i = 0; i < numericColumns.length; i++) {
			for (let j = i + 1; j < numericColumns.length; j++) {
				for (let k = j + 1; k < numericColumns.length; k++) {
					let xCol = numericColumns[i];
					let yCol = numericColumns[j];
					let zCol = numericColumns[k];

					let xIndex = tab_results_headers_json.indexOf(xCol);
					let yIndex = tab_results_headers_json.indexOf(yCol);
					let zIndex = tab_results_headers_json.indexOf(zCol);

					let data = tab_results_csv_json.map(row => ({
						x: parseFloat(row[xIndex]),
						y: parseFloat(row[yIndex]),
						z: parseFloat(row[zIndex]),
						result: row[resultIndex] !== "" ? parseFloat(row[resultIndex]) : null
					}));

					data = data.filter(d => d.result >= minResult && d.result <= maxResult);

					let layoutTitle = `${xCol} (x) vs ${yCol} (y) vs ${zCol} (z), result: ${selectedResult}`;
					let layout = {
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

					let subDiv = document.createElement("div");
					try {
						plotDiv.replaceChild(subDiv, spinnerContainer);
					} catch (err) {
						//
					}
					Plotly.newPlot(subDiv, [trace], add_default_layout_data(layout));
				}
			}
		}
	}

	$("#plotScatter3d").data("loaded", "true");
}


