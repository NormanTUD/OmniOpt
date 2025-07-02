"use strict";

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

function createParallelPlot(dataArray, headers, resultNames, ignoreColumns = [], reload = false) {
	try {
		if ($("#parallel-plot").data("loaded") === "true" && !reload) {
			return;
		}

		// Filter rows ohne leere Werte (wie in deinem Originalcode)
		dataArray = filterNonEmptyRows(dataArray);

		const ignoreSet = new Set(ignoreColumns);
		const numericalCols = [];
		const categoricalCols = [];
		const categoryMappings = {};
		const enable_slurm_id_if_exists = $("#enable_slurm_id_if_exists").is(":checked");

		// Spalten einteilen in numerisch oder kategorisch + category mappings aufbauen
		headers.forEach((header, colIndex) => {
			if (ignoreSet.has(header)) return;
			if (!enable_slurm_id_if_exists && header === "OO_Info_SLURM_JOB_ID") return;

			const values = dataArray.map(row => row[colIndex]);
			if (values.every(val => !isNaN(parseFloat(val)))) {
				numericalCols.push({ name: header, index: colIndex });
			} else {
				categoricalCols.push({ name: header, index: colIndex });
				const uniqueValues = [...new Set(values)];
				categoryMappings[header] = Object.fromEntries(uniqueValues.map((val, i) => [val, i]));
			}
		});

		// Erzeuge UI für Checkboxen und Min/Max Inputs für numerische Spalten
		const controlContainerId = "parallel-plot-controls";
		let controlContainer = $("#" + controlContainerId);
		if (controlContainer.length === 0) {
			controlContainer = $('<div id="' + controlContainerId + '" style="margin-bottom:10px; display: flex;"></div>');
			$("#parallel-plot").before(controlContainer);
		} else {
			controlContainer.empty();
		}

		// Map um Checkbox-Zustände und Min/Max-Werte zu speichern
		const columnVisibility = {};
		const minMaxLimits = {};

		// Checkboxen + Min/Max Felder generieren mit Boxen, max-Breite, Umbruch und Zeilenumbruch nach jeder Box
		headers.forEach((header) => {
			try {
				if (ignoreSet.has(header)) return;
				if (!enable_slurm_id_if_exists && header === "OO_Info_SLURM_JOB_ID") return;

				const isNumerical = numericalCols.some(col => col.name === header);

				const checkboxId = `chk_${header}`;
				const minInputId = `min_${header}`;
				const maxInputId = `max_${header}`;

				columnVisibility[header] = true;
				minMaxLimits[header] = { min: null, max: null };

				// Wrapper Box mit max-Breite, Umbruch, Block-Level-Element für newline nach jeder Box
				const boxWrapper = $('<div></div>').css({
					border: "1px solid #ddd",
					borderRadius: "8px",
					padding: "12px 16px",
					marginBottom: "12px",
					boxShadow: "0 2px 6px rgba(0,0,0,0.1)",
					backgroundColor: "#fff",
					display: "flex",
					flexWrap: "wrap",
					alignItems: "center",
					gap: "15px",
					maxWidth: "350px",
					width: "100%", // damit bei kleinen Screens die Box maximal voll breit ist
					boxSizing: "border-box"
				});

				// Innerer Container mit Flexbox für Ausrichtung der Elemente, flex-grow damit Inputs genug Platz bekommen
				const container = $('<div></div>').css({
					display: "flex",
					alignItems: "center",
					gap: "10px",
					flexWrap: "wrap",
					flexGrow: 1,
					minWidth: "0" // wichtig für flexbox Overflow Handling
				});

				// Checkbox mit Label
				const checkbox = $(`<input type="checkbox" id="${checkboxId}" checked />`);
				const label = $(`<label for="${checkboxId}" style="font-weight: 600; min-width: 140px; cursor: pointer; white-space: nowrap;">${header}</label>`);

				container.append(checkbox).append(label);

				if (isNumerical) {
					// Werte ermitteln (nur gültige Zahlen)
					const numericValues = dataArray
						.map(row => parseFloat(row[headers.indexOf(header)]))
						.filter(val => !isNaN(val));

					const minVal = numericValues.length > 0 ? Math.min(...numericValues) : 0;
					const maxVal = numericValues.length > 0 ? Math.max(...numericValues) : 100;

					// Min Input mit Label
					const minWrapper = $('<div></div>').css({
						display: "flex",
						flexDirection: "column",
						alignItems: "flex-start",
						minWidth: "90px"
					});
					const minLabel = $('<label></label>').attr("for", minInputId).text("Min").css({
						fontSize: "0.75rem",
						color: "#555",
						marginBottom: "2px"
					});
					const minInput = $(`<input type="number" id="${minInputId}" placeholder="min" />`).css({
						width: "80px",
						padding: "5px 8px",
						borderRadius: "5px",
						border: "1px solid #ccc",
						boxShadow: "inset 0 1px 3px rgba(0,0,0,0.1)",
						transition: "border-color 0.3s ease"
					});
					minInput.attr("min", minVal);
					minInput.attr("max", maxVal);

					minInput.on("focus", function () {
						$(this).css("border-color", "#007BFF");
					});
					minInput.on("blur", function () {
						$(this).css("border-color", "#ccc");
					});

					minWrapper.append(minLabel).append(minInput);

					// Max Input mit Label
					const maxWrapper = $('<div></div>').css({
						display: "flex",
						flexDirection: "column",
						alignItems: "flex-start",
						minWidth: "90px"
					});
					const maxLabel = $('<label></label>').attr("for", maxInputId).text("Max").css({
						fontSize: "0.75rem",
						color: "#555",
						marginBottom: "2px"
					});
					const maxInput = $(`<input type="number" id="${maxInputId}" placeholder="max" />`).css({
						width: "80px",
						padding: "5px 8px",
						borderRadius: "5px",
						border: "1px solid #ccc",
						boxShadow: "inset 0 1px 3px rgba(0,0,0,0.1)",
						transition: "border-color 0.3s ease"
					});
					maxInput.attr("min", minVal);
					maxInput.attr("max", maxVal);

					maxInput.on("focus", function () {
						$(this).css("border-color", "#007BFF");
					});
					maxInput.on("blur", function () {
						$(this).css("border-color", "#ccc");
					});

					maxWrapper.append(maxLabel).append(maxInput);

					// Events für min/max Eingaben
					minInput.on("input", function () {
						const val = parseFloat($(this).val());
						minMaxLimits[header].min = isNaN(val) ? null : val;
						updatePlot();
					});

					maxInput.on("input", function () {
						const val = parseFloat($(this).val());
						minMaxLimits[header].max = isNaN(val) ? null : val;
						updatePlot();
					});

					container.append(minWrapper).append(maxWrapper);
				}

				// Checkbox Change Event
				checkbox.on("change", function () {
					columnVisibility[header] = $(this).is(":checked");
					updatePlot();
				});

				boxWrapper.append(container);

				// Jede Box bekommt ihren eigenen Block (also newline)
				controlContainer.append(boxWrapper);
			} catch (error) {
				console.error(`Fehler bei Header '${header}':`, error);
			}
		});



		// Erzeuge Ergebnis-Auswahl für Farbskala (color by result)
		const resultSelectId = "result-select";
		let resultSelect = $(`#${resultSelectId}`);
		if (resultSelect.length === 0) {
			resultSelect = $(`<select id="${resultSelectId}"></select>`);
			controlContainer.before(resultSelect);
		} else {
			resultSelect.empty();
		}

		resultSelect.append('<option value="none">No color</option>');

		for (let i = 0; i < resultNames.length; i++) {
			let minMaxInfo = "min [auto]";
			if (typeof result_min_max !== "undefined" && result_min_max[i] !== undefined) {
				minMaxInfo = result_min_max[i];
			}
			resultSelect.append(`<option value="${resultNames[i]}">${resultNames[i]} (${minMaxInfo})</option>`);
		}

		let colorValues = null;
		let colorScale = null;

		resultSelect.off("change").on("change", function () {
			const selectedResult = $(this).val();
			if (selectedResult === "none") {
				colorValues = null;
				colorScale = null;
			} else {
				const col = numericalCols.find(c => c.name.toLowerCase() === selectedResult.toLowerCase());
				if (!col) {
					colorValues = null;
					colorScale = null;
					updatePlot();
					return;
				}
				colorValues = dataArray.map(row => parseFloat(row[col.index]));

				let invertColor = false;
				if (typeof result_min_max !== "undefined") {
					const idx = resultNames.indexOf(selectedResult);
					if (idx !== -1) {
						invertColor = result_min_max[idx] === "max";
					}
				}

				colorScale = invertColor
					? [[0, 'red'], [1, 'green']]
					: [[0, 'green'], [1, 'red']];
			}
			updatePlot();
		});

		// Initial Auswahl: kein Farbwert, oder erstes Ergebnis falls nur eins
		if (resultNames.length === 1) {
			resultSelect.val(resultNames[0]).trigger("change");
		} else {
			resultSelect.val("none").trigger("change");
		}

		function updatePlot() {
			try {
				// Filter Spalten nach Checkboxen
				const filteredNumericalCols = numericalCols.filter(col => columnVisibility[col.name]);
				const filteredCategoricalCols = categoricalCols.filter(col => columnVisibility[col.name]);

				// Filtere die Datenzeilen, um nur die zu behalten, die innerhalb aller gesetzten Min/Max Limits liegen
				const filteredData = dataArray.filter(row => {
					for (let col of filteredNumericalCols) {
						const val = parseFloat(row[col.index]);
						if (isNaN(val)) return false; // ungültiger Wert raus

						const limits = minMaxLimits[col.name];
						if (limits.min !== null && val < limits.min) return false;
						if (limits.max !== null && val > limits.max) return false;
					}
					// Kategorische Werte ignorieren Filter (könntest hier evtl. erweitern)
					return true;
				});

				const dimensions = [];

				// Füge numerische Dimensionen hinzu mit Min/Max Limits (Range anhand gefilterter Daten)
				filteredNumericalCols.forEach(col => {
					let vals = filteredData.map(row => parseFloat(row[col.index]));

					// Fallback falls alle Werte NaN (sollte eigentlich nicht vorkommen)
					const realMin = vals.length > 0 ? Math.min(...vals) : 0;
					const realMax = vals.length > 0 ? Math.max(...vals) : 100;

					dimensions.push({
						label: col.name,
						values: vals,
						range: [realMin, realMax]
					});
				});

				// Kategorische Dimensionen (aus gefilterten Daten)
				filteredCategoricalCols.forEach(col => {
					const vals = filteredData.map(row => categoryMappings[col.name][row[col.index]]);
					dimensions.push({
						label: col.name,
						values: vals,
						tickvals: Object.values(categoryMappings[col.name]),
						ticktext: Object.keys(categoryMappings[col.name])
					});
				});

				// Linienfarbe bestimmen, falls Farbskala gesetzt ist
				let filteredColorValues = null;
				if (colorValues) {
					// Da colorValues für alle Daten sind, filtere sie auch entsprechend
					filteredColorValues = filteredData.map(row => {
						const col = numericalCols.find(c => c.name.toLowerCase() === resultSelect.val().toLowerCase());
						return col ? parseFloat(row[col.index]) : null;
					});
				}

				const trace = {
					type: 'parcoords',
					dimensions: dimensions,
					line: filteredColorValues ? { color: filteredColorValues, colorscale: colorScale } : {},
					unselected: {
						line: {
							color: get_text_color(),
							opacity: 0
						}
					},
				};

				dimensions.forEach(dim => {
					if (!dim.line) {
						dim.line = {};
					}
					if (!dim.line.color) {
						dim.line.color = 'rgba(169,169,169, 0.01)';
					}
				});

				Plotly.newPlot('parallel-plot', [trace], add_default_layout_data({}));

				make_text_in_parallel_plot_nicer();
			} catch (error) {
				console.error("Fehler in updatePlot():", error);
			}
		}


		updatePlot();

		$("#parallel-plot").data("loaded", "true");

		make_text_in_parallel_plot_nicer();

	} catch (err) {
		console.error("Error in createParallelPlot:", err);
	}
}

function plotWorkerUsage() {
	if($("#workerUsagePlot").data("loaded") == "true") {
		return;
	}
	var data = tab_worker_usage_csv_json;
	if (!Array.isArray(data) || data.length === 0) {
		console.error("Invalid or empty data provided.");
		return;
	}

	let timestamps = [];
	let desiredWorkers = [];
	let realWorkers = [];

	for (let i = 0; i < data.length; i++) {
		let entry = data[i];

		if (!Array.isArray(entry) || entry.length < 3) {
			console.warn("Skipping invalid entry:", entry);
			continue;
		}

		let unixTime = parseFloat(entry[0]);
		let desired = parseInt(entry[1], 10);
		let real = parseInt(entry[2], 10);

		if (isNaN(unixTime) || isNaN(desired) || isNaN(real)) {
			console.warn("Skipping invalid numerical values:", entry);
			continue;
		}

		timestamps.push(new Date(unixTime * 1000).toISOString());
		desiredWorkers.push(desired);
		realWorkers.push(real);
	}

	let trace1 = {
		x: timestamps,
		y: desiredWorkers,
		mode: 'lines+markers',
		name: 'Desired Workers',
		line: {
			color: 'blue'
		}
	};

	let trace2 = {
		x: timestamps,
		y: realWorkers,
		mode: 'lines+markers',
		name: 'Real Workers',
		line: {
			color: 'red'
		}
	};

	let layout = {
		title: "Worker Usage Over Time",
		xaxis: {
			title: get_axis_title_data("Time", "date")
		},
		yaxis: {
			title: get_axis_title_data("Number of Workers")
		},
		legend: {
			x: 0,
			y: 1
		}
	};

	Plotly.newPlot('workerUsagePlot', [trace1, trace2], add_default_layout_data(layout));
	$("#workerUsagePlot").data("loaded", "true");
}

function plotCPUAndRAMUsage() {
	if($("#mainWorkerCPURAM").data("loaded") == "true") {
		return;
	}
	var timestamps = tab_main_worker_cpu_ram_csv_json.map(row => new Date(row[0] * 1000));
	var ramUsage = tab_main_worker_cpu_ram_csv_json.map(row => row[1]);
	var cpuUsage = tab_main_worker_cpu_ram_csv_json.map(row => row[2]);

	var trace1 = {
		x: timestamps,
		y: cpuUsage,
		mode: 'lines+markers',
		marker: {
			size: get_marker_size(),
		},
		name: 'CPU Usage (%)',
		type: 'scatter',
		yaxis: 'y1'
	};

	var trace2 = {
		x: timestamps,
		y: ramUsage,
		mode: 'lines+markers',
		marker: {
			size: get_marker_size(),
		},
		name: 'RAM Usage (MB)',
		type: 'scatter',
		yaxis: 'y2'
	};

	var layout = {
		title: 'CPU and RAM Usage Over Time',
		xaxis: {
			title: get_axis_title_data("Timestamp", "date"),
			tickmode: 'array',
			tickvals: timestamps.filter((_, index) => index % Math.max(Math.floor(timestamps.length / 10), 1) === 0),
			ticktext: timestamps.filter((_, index) => index % Math.max(Math.floor(timestamps.length / 10), 1) === 0).map(t => t.toLocaleString()),
			tickangle: -45
		},
		yaxis: {
			title: get_axis_title_data("CPU Usage (%)"),
			rangemode: 'tozero'
		},
		yaxis2: {
			title: get_axis_title_data("RAM Usage (MB)"),
			overlaying: 'y',
			side: 'right',
			rangemode: 'tozero'
		},
		legend: {
			x: 0.1,
			y: 0.9
		}
	};

	var data = [trace1, trace2];
	Plotly.newPlot('mainWorkerCPURAM', data, add_default_layout_data(layout));
	$("#mainWorkerCPURAM").data("loaded", "true");
}

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

async function plot_worker_cpu_ram() {
	if($("#worker_cpu_ram_pre").data("loaded") == "true") {
		return;
	}

	const logData = $("#worker_cpu_ram_pre").text();
	const regex = /^Unix-Timestamp: (\d+), Hostname: ([\w-]+), CPU: ([\d.]+)%, RAM: ([\d.]+) MB \/ ([\d.]+) MB$/;

	const hostData = {};

	logData.split("\n").forEach(line => {
		line = line.trim();
		const match = line.match(regex);
		if (match) {
			const timestamp = new Date(parseInt(match[1]) * 1000);
			const hostname = match[2];
			const cpu = parseFloat(match[3]);
			const ram = parseFloat(match[4]);

			if (!hostData[hostname]) {
				hostData[hostname] = { timestamps: [], cpuUsage: [], ramUsage: [] };
			}

			hostData[hostname].timestamps.push(timestamp);
			hostData[hostname].cpuUsage.push(cpu);
			hostData[hostname].ramUsage.push(ram);
		}
	});

	if (!Object.keys(hostData).length) {
		console.log("No valid data found");
		return;
	}

	const container = document.getElementById("cpuRamWorkerChartContainer");
	container.innerHTML = "";

	var i = 1;

	Object.entries(hostData).forEach(([hostname, { timestamps, cpuUsage, ramUsage }], index) => {
		const chartId = `workerChart_${index}`;
		const chartDiv = document.createElement("div");
		chartDiv.id = chartId;
		chartDiv.style.marginBottom = "40px";
		container.appendChild(chartDiv);

		const cpuTrace = {
			x: timestamps,
			y: cpuUsage,
			mode: "lines+markers",
			name: "CPU Usage (%)",
			yaxis: "y1",
			line: {
				color: "red"
			}
		};

		const ramTrace = {
			x: timestamps,
			y: ramUsage,
			mode: "lines+markers",
			name: "RAM Usage (MB)",
			yaxis: "y2",
			line: {
				color: "blue"
			}
		};

		const layout = {
			title: `Worker CPU and RAM Usage - ${hostname}`,
			xaxis: {
				title: get_axis_title_data("Timestamp", "date")
			},
			yaxis: {
				title: get_axis_title_data("CPU Usage (%)"),
				side: "left",
				color: "red"
			},
			yaxis2: {
				title: get_axis_title_data("RAM Usage (MB)"),
				side: "right",
				overlaying: "y",
				color: "blue"
			},
			showlegend: true
		};

		Plotly.newPlot(chartId, [cpuTrace, ramTrace], add_default_layout_data(layout));
		i++;
	});

	$("#plot_worker_cpu_ram_button").remove();
	$("#worker_cpu_ram_pre").data("loaded", "true");
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

function plotBoxplot() {
	if ($("#plotBoxplot").data("loaded") == "true") {
		return;
	}
	var numericColumns = tab_results_headers_json.filter(col =>
		!special_col_names.includes(col) && !result_names.includes(col) &&
		!col.startsWith("OO_Info") &&
		tab_results_csv_json.every(row => !isNaN(parseFloat(row[tab_results_headers_json.indexOf(col)])))
	);

	if (numericColumns.length < 1) {
		console.error("Not enough numeric columns for Boxplot");
		return;
	}

	var resultIndex = tab_results_headers_json.findIndex(function(header) {
		return result_names.includes(header.toLowerCase());
	});
	var resultValues = tab_results_csv_json.map(row => row[resultIndex]);
	var minResult = Math.min(...resultValues.filter(value => value !== null && value !== ""));
	var maxResult = Math.max(...resultValues.filter(value => value !== null && value !== ""));

	var plotDiv = document.getElementById("plotBoxplot");
	plotDiv.innerHTML = "";

	let traces = numericColumns.map(col => {
		let index = tab_results_headers_json.indexOf(col);
		let data = tab_results_csv_json.map(row => parseFloat(row[index]));

		return {
			y: data,
			type: 'box',
			name: col,
			boxmean: 'sd',
			marker: {
				color: 'rgb(0, 255, 0)'
			},
		};
	});

	let layout = {
		title: 'Boxplot of Numerical Columns',
		xaxis: {
			title: get_axis_title_data("Columns")
		},
		yaxis: {
			title: get_axis_title_data("Value")
		},
		showlegend: false
	};

	Plotly.newPlot(plotDiv, traces, add_default_layout_data(layout));
	$("#plotBoxplot").data("loaded", "true");
}

function plotHeatmap() {
	if ($("#plotHeatmap").data("loaded") === "true") {
		return;
	}

	var numericColumns = tab_results_headers_json.filter(col => {
		if (special_col_names.includes(col) || result_names.includes(col)) {
			return false;
		}
		if (!col.startsWith("OO_Info")) {
			return true;
		}
		let index = tab_results_headers_json.indexOf(col);
		return tab_results_csv_json.every(row => {
			let value = parseFloat(row[index]);
			return !isNaN(value) && isFinite(value);
		});
	});

	if (numericColumns.length < 2) {
		console.error("Not enough valid numeric columns for Heatmap");
		return;
	}

	var columnData = numericColumns.map(col => {
		let index = tab_results_headers_json.indexOf(col);
		return tab_results_csv_json.map(row => parseFloat(row[index]));
	});

	var dataMatrix = numericColumns.map((_, i) =>
		numericColumns.map((_, j) => {
			let values = columnData[i].map((val, index) => (val + columnData[j][index]) / 2);
			return values.reduce((a, b) => a + b, 0) / values.length;
		})
	);

	var trace = {
		z: dataMatrix,
		x: numericColumns,
		y: numericColumns,
		colorscale: 'Viridis',
		type: 'heatmap'
	};

	var layout = {
		xaxis: {
			title: get_axis_title_data("Columns")
		},
		yaxis: {
			title: get_axis_title_data("Columns")
		},
		showlegend: false
	};

	var plotDiv = document.getElementById("plotHeatmap");
	plotDiv.innerHTML = "";

	Plotly.newPlot(plotDiv, [trace], add_default_layout_data(layout));
	$("#plotHeatmap").data("loaded", "true");
}

function plotHistogram() {
	if ($("#plotHistogram").data("loaded") == "true") {
		return;
	}

	var numericColumns = tab_results_headers_json.filter(col =>
		!special_col_names.includes(col) && !result_names.includes(col) &&
		!col.startsWith("OO_Info") &&
		tab_results_csv_json.every(row => !isNaN(parseFloat(row[tab_results_headers_json.indexOf(col)])))
	);

	if (numericColumns.length < 1) {
		console.error("Not enough columns for Histogram");
		return;
	}

	var plotDiv = document.getElementById("plotHistogram");
	plotDiv.innerHTML = "";

	const colorPalette = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6'];

	let traces = numericColumns.map((col, index) => {
		let data = tab_results_csv_json.map(row => parseFloat(row[tab_results_headers_json.indexOf(col)]));

		return {
			x: data,
			type: 'histogram',
			name: col,
			opacity: 0.7,
			marker: {
				color: colorPalette[index % colorPalette.length]
			},
			autobinx: true
		};
	});

	let layout = {
		title: 'Histogram of Numerical Columns',
		xaxis: {
			title: get_axis_title_data("Value")
		},
		yaxis: {
			title: get_axis_title_data("Frequency")
		},
		showlegend: true,
		barmode: 'overlay'
	};

	Plotly.newPlot(plotDiv, traces, add_default_layout_data(layout));
	$("#plotHistogram").data("loaded", "true");
}

function plotViolin() {
	if ($("#plotViolin").data("loaded") == "true") {
		return;
	}
	var numericColumns = tab_results_headers_json.filter(col =>
		!special_col_names.includes(col) && !result_names.includes(col) &&
		!col.startsWith("OO_Info") &&
		tab_results_csv_json.every(row => !isNaN(parseFloat(row[tab_results_headers_json.indexOf(col)])))
	);

	if (numericColumns.length < 1) {
		console.error("Not enough columns for Violin Plot");
		return;
	}

	var plotDiv = document.getElementById("plotViolin");
	plotDiv.innerHTML = "";

	let traces = numericColumns.map(col => {
		let index = tab_results_headers_json.indexOf(col);
		let data = tab_results_csv_json.map(row => parseFloat(row[index]));

		return {
			y: data,
			type: 'violin',
			name: col,
			box: {
				visible: true
			},
			line: {
				color: 'rgb(0, 255, 0)'
			},
			marker: {
				color: 'rgb(0, 255, 0)'
			},
			meanline: {
				visible: true
			},
		};
	});

	let layout = {
		title: 'Violin Plot of Numerical Columns',
		yaxis: {
			title: get_axis_title_data("Value")
		},
		xaxis: {
			title: get_axis_title_data("Columns")
		},
		showlegend: false
	};

	Plotly.newPlot(plotDiv, traces, add_default_layout_data(layout));
	$("#plotViolin").data("loaded", "true");
}

function plotExitCodesPieChart() {
	if ($("#plotExitCodesPieChart").data("loaded") == "true") {
		return;
	}

	var exitCodes = tab_results_csv_json.map(row => row[tab_results_headers_json.indexOf("exit_code")]);

	var exitCodeCounts = exitCodes.reduce(function(counts, exitCode) {
		counts[exitCode] = (counts[exitCode] || 0) + 1;
		return counts;
	}, {});

	var labels = Object.keys(exitCodeCounts);
	var values = Object.values(exitCodeCounts);

	var plotDiv = document.getElementById("plotExitCodesPieChart");
	plotDiv.innerHTML = "";

	var trace = {
		labels: labels,
		values: values,
		type: 'pie',
		hoverinfo: 'label+percent',
		textinfo: 'label+value',
		marker: {
			colors: ['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0']
		}
	};

	var layout = {
		title: 'Exit Code Distribution',
		showlegend: true
	};

	Plotly.newPlot(plotDiv, [trace], add_default_layout_data(layout));
	$("#plotExitCodesPieChart").data("loaded", "true");
}

function plotResultEvolution() {
	if ($("#plotResultEvolution").data("loaded") == "true") {
		return;
	}

	result_names.forEach(resultName => {
		var relevantColumns = tab_results_headers_json.filter(col =>
			!special_col_names.includes(col) && !col.startsWith("OO_Info") && col.toLowerCase() !== resultName.toLowerCase()
		);

		var xColumnIndex = tab_results_headers_json.indexOf("trial_index");
		var resultIndex = tab_results_headers_json.indexOf(resultName);

		let data = tab_results_csv_json.map(row => ({
			x: row[xColumnIndex],
			y: parseFloat(row[resultIndex])
		}));

		data.sort((a, b) => a.x - b.x);

		let xData = data.map(item => item.x);
		let yData = data.map(item => item.y);

		let trace = {
			x: xData,
			y: yData,
			mode: 'lines+markers',
			name: resultName,
			line: {
				shape: 'linear'
			},
			marker: {
				size: get_marker_size()
			}
		};

		let layout = {
			title: `Evolution of ${resultName} over time`,
			xaxis: {
				title: get_axis_title_data("Trial-Index")
			},
			yaxis: {
				title: get_axis_title_data(resultName)
			},
			showlegend: true
		};

		let subDiv = document.createElement("div");
		document.getElementById("plotResultEvolution").appendChild(subDiv);

		Plotly.newPlot(subDiv, [trace], add_default_layout_data(layout));
	});

	$("#plotResultEvolution").data("loaded", "true");
}

function plotResultPairs() {
	if ($("#plotResultPairs").data("loaded") == "true") {
		return;
	}

	var plotDiv = document.getElementById("plotResultPairs");
	plotDiv.innerHTML = "";

	for (let i = 0; i < result_names.length; i++) {
		for (let j = i + 1; j < result_names.length; j++) {
			let xName = result_names[i];
			let yName = result_names[j];

			let xIndex = tab_results_headers_json.indexOf(xName);
			let yIndex = tab_results_headers_json.indexOf(yName);

			let data = tab_results_csv_json
				.filter(row => row[xIndex] !== "" && row[yIndex] !== "")
				.map(row => ({
					x: parseFloat(row[xIndex]),
					y: parseFloat(row[yIndex]),
					status: row[tab_results_headers_json.indexOf("trial_status")]
				}));

			let colors = data.map(d => d.status === "COMPLETED" ? 'green' : (d.status === "FAILED" ? 'red' : 'gray'));

			let trace = {
				x: data.map(d => d.x),
				y: data.map(d => d.y),
				mode: 'markers',
				marker: {
					size: get_marker_size(),
					color: colors
				},
				text: data.map(d => `Status: ${d.status}`),
				type: 'scatter',
				showlegend: false
			};

			let layout = {
				xaxis: {
					title: get_axis_title_data(xName)
				},
				yaxis: {
					title: get_axis_title_data(yName)
				},
				showlegend: false
			};

			let subDiv = document.createElement("div");
			plotDiv.appendChild(subDiv);

			Plotly.newPlot(subDiv, [trace], add_default_layout_data(layout));
		}
	}

	$("#plotResultPairs").data("loaded", "true");
}

function add_up_down_arrows_for_scrolling () {
	const upArrow = document.createElement('div');
	const downArrow = document.createElement('div');

	const style = document.createElement('style');
	style.innerHTML = `
		.scroll-arrow {
			position: fixed;
			right: 10px;
			z-index: 100;
			cursor: pointer;
			font-size: 25px;
			display: none;
			background-color: green;
			color: white;
			padding: 5px;
			outline: 2px solid white;
			box-shadow: 0 0 10px rgba(0, 0, 0, 0.5);
			transition: background-color 0.3s, transform 0.3s;
		}
		.scroll-arrow:hover {
			background-color: darkgreen;
			transform: scale(1.1);
		}
		#up-arrow {
			top: 10px;
		}
		#down-arrow {
			bottom: 10px;
		}
	`;
	document.head.appendChild(style);

	upArrow.id = "up-arrow";
	upArrow.classList.add("scroll-arrow");
	upArrow.classList.add("invert_in_dark_mode");
	upArrow.innerHTML = "&#8593;";

	downArrow.id = "down-arrow";
	downArrow.classList.add("scroll-arrow");
	downArrow.classList.add("invert_in_dark_mode");
	downArrow.innerHTML = "&#8595;";

	document.body.appendChild(upArrow);
	document.body.appendChild(downArrow);

	function checkScrollPosition() {
		const scrollPosition = window.scrollY;
		const pageHeight = document.documentElement.scrollHeight;
		const windowHeight = window.innerHeight;

		if (scrollPosition > 0) {
			upArrow.style.display = "block";
		} else {
			upArrow.style.display = "none";
		}

		if (scrollPosition + windowHeight < pageHeight) {
			downArrow.style.display = "block";
		} else {
			downArrow.style.display = "none";
		}
	}

	window.addEventListener("scroll", checkScrollPosition);

	upArrow.addEventListener("click", function () {
		window.scrollTo({ top: 0, behavior: 'smooth' });
	});

	downArrow.addEventListener("click", function () {
		window.scrollTo({ top: document.documentElement.scrollHeight, behavior: 'smooth' });
	});

	checkScrollPosition();

	if (typeof apply_theme_based_on_system_preferences === 'function') {
		apply_theme_based_on_system_preferences();
	}
}

function plotGPUUsage() {
	if ($("#tab_gpu_usage").data("loaded") === "true") {
		return;
	}

	Object.keys(gpu_usage).forEach(node => {
		const nodeData = gpu_usage[node];

		var timestamps = [];
		var gpuUtilizations = [];
		var temperatures = [];

		nodeData.forEach(entry => {
			try {
				var timestamp = new Date(entry[0]* 1000);
				var utilization = parseFloat(entry[1]);
				var temperature = parseFloat(entry[2]);

				if (!isNaN(timestamp) && !isNaN(utilization) && !isNaN(temperature)) {
					timestamps.push(timestamp);
					gpuUtilizations.push(utilization);
					temperatures.push(temperature);
				} else {
					console.warn("Invalid data point:", entry);
				}
			} catch (error) {
				console.error("Error processing GPU data entry:", error, entry);
			}
		});

		var trace1 = {
			x: timestamps,
			y: gpuUtilizations,
			mode: 'lines+markers',
			marker: {
				size: get_marker_size(),
			},
			name: 'GPU Utilization (%)',
			type: 'scatter',
			yaxis: 'y1'
		};

		var trace2 = {
			x: timestamps,
			y: temperatures,
			mode: 'lines+markers',
			marker: {
				size: get_marker_size(),
			},
			name: 'GPU Temperature (°C)',
			type: 'scatter',
			yaxis: 'y2'
		};

		var layout = {
			title: 'GPU Usage Over Time - ' + node,
			xaxis: {
				title: get_axis_title_data("Timestamp", "date"),
				tickmode: 'array',
				tickvals: timestamps.filter((_, index) => index % Math.max(Math.floor(timestamps.length / 10), 1) === 0),
				ticktext: timestamps.filter((_, index) => index % Math.max(Math.floor(timestamps.length / 10), 1) === 0).map(t => t.toLocaleString()),
				tickangle: -45
			},
			yaxis: {
				title: get_axis_title_data("GPU Utilization (%)"),
				overlaying: 'y',
				rangemode: 'tozero'
			},
			yaxis2: {
				title: get_axis_title_data("GPU Temperature (°C)"),
				overlaying: 'y',
				side: 'right',
				position: 0.85,
				rangemode: 'tozero'
			},
			legend: {
				x: 0.1,
				y: 0.9
			}
		};

		var divId = 'gpu_usage_plot_' + node;

		if (!document.getElementById(divId)) {
			var div = document.createElement('div');
			div.id = divId;
			div.className = 'gpu-usage-plot';
			document.getElementById('tab_gpu_usage').appendChild(div);
		}

		var plotData = [trace1, trace2];

		Plotly.newPlot(divId, plotData, add_default_layout_data(layout));
	});

	$("#tab_gpu_usage").data("loaded", "true");
}

function plotResultsDistributionByGenerationMethod() {
	if ("true" === $("#plotResultsDistributionByGenerationMethod").data("loaded")) {
		return;
	}

	var res_col = result_names[0];
	var gen_method_col = "generation_node";

	var data = {};

	tab_results_csv_json.forEach(row => {
		var gen_method = row[tab_results_headers_json.indexOf(gen_method_col)];
		var result = row[tab_results_headers_json.indexOf(res_col)];

		if (!data[gen_method]) {
			data[gen_method] = [];
		}
		data[gen_method].push(result);
	});

	var traces = Object.keys(data).map(method => {
		return {
			y: data[method],
			type: 'box',
			name: method,
			boxpoints: 'outliers',
			jitter: 0.5,
			pointpos: 0
		};
	});

	var layout = {
		title: 'Distribution of Results by Generation Method',
		yaxis: {
			title: get_axis_title_data(res_col)
		},
		xaxis: {
			title: get_axis_title_data("Generation Method")
		},
		boxmode: 'group'
	};

	Plotly.newPlot("plotResultsDistributionByGenerationMethod", traces, add_default_layout_data(layout));
	$("#plotResultsDistributionByGenerationMethod").data("loaded", "true");
}

function plotJobStatusDistribution() {
	if ($("#plotJobStatusDistribution").data("loaded") === "true") {
		return;
	}

	var status_col = "trial_status";
	var status_counts = {};

	tab_results_csv_json.forEach(row => {
		var status = row[tab_results_headers_json.indexOf(status_col)];
		if (status) {
			status_counts[status] = (status_counts[status] || 0) + 1;
		}
	});

	var statuses = Object.keys(status_counts);
	var counts = Object.values(status_counts);

	var colors = statuses.map((status, i) =>
		status === "FAILED" ? "#FF0000" : `hsl(${30 + ((i * 137) % 330)}, 70%, 50%)`
	);

	var trace = {
		x: statuses,
		y: counts,
		type: 'bar',
		marker: { color: colors }
	};

	var layout = {
		title: 'Distribution of Job Status',
		xaxis: { title: 'Trial Status' },
		yaxis: { title: 'Nr. of jobs' }
	};

	Plotly.newPlot("plotJobStatusDistribution", [trace], add_default_layout_data(layout));
	$("#plotJobStatusDistribution").data("loaded", "true");
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
			el.textContent.includes("ABANDONED") ? "yellow" : "";
		if (color) el.style.backgroundColor = color;
		el.classList.add("invert_in_dark_mode");
	});
}

function _colorize_table_entries_by_run_time() {
	let cells = [...document.querySelectorAll('[data-column-id="run_time"]')];
	if (cells.length === 0) return;

	let values = cells.map(el => parseFloat(el.textContent)).filter(v => !isNaN(v));
	if (values.length === 0) return;

	let min = Math.min(...values);
	let max = Math.max(...values);
	let range = max - min || 1;

	cells.forEach(el => {
		let value = parseFloat(el.textContent);
		if (isNaN(value)) return;

		let ratio = (value - min) / range;
		let red = Math.round(255 * ratio);
		let green = Math.round(255 * (1 - ratio));

		el.style.backgroundColor = `rgb(${red}, ${green}, 0)`;
		el.classList.add("invert_in_dark_mode");
	});
}

function _colorize_table_entries_by_results() {
	result_names.forEach((name, index) => {
		let minMax = result_min_max[index];
		let selector_query = `[data-column-id="${name}"]`;
		let cells = [...document.querySelectorAll(selector_query)];
		if (cells.length === 0) return;

		let values = cells.map(el => parseFloat(el.textContent)).filter(v => v > 0 && !isNaN(v));
		if (values.length === 0) return;

		let logValues = values.map(v => Math.log(v));
		let logMin = Math.min(...logValues);
		let logMax = Math.max(...logValues);
		let logRange = logMax - logMin || 1;

		cells.forEach(el => {
			let value = parseFloat(el.textContent);
			if (isNaN(value) || value <= 0) return;

			let logValue = Math.log(value);
			let ratio = (logValue - logMin) / logRange;
			if (minMax === "max") ratio = 1 - ratio;

			let red = Math.round(255 * ratio);
			let green = Math.round(255 * (1 - ratio));

			el.style.backgroundColor = `rgb(${red}, ${green}, 0)`;
			el.classList.add("invert_in_dark_mode");
		});
	});
}

function _colorize_table_entries_by_generation_node_or_hostname() {
	["hostname", "generation_node"].forEach(element => {
		let selector_query = '[data-column-id="' + element + '"]:not(.gridjs-th)';
		let cells = [...document.querySelectorAll(selector_query)];
		if (cells.length === 0) return;

		let uniqueValues = [...new Set(cells.map(el => el.textContent.trim()))];
		let colorMap = {};

		uniqueValues.forEach((value, index) => {
			let hue = Math.round((360 / uniqueValues.length) * index);
			colorMap[value] = `hsl(${hue}, 70%, 60%)`;
		});

		cells.forEach(el => {
			let value = el.textContent.trim();
			if (colorMap[value]) {
				el.style.backgroundColor = colorMap[value];
				el.classList.add("invert_in_dark_mode");
			}
		});
	});
}

function colorize_table_entries () {
	setTimeout(() => {
		if (typeof result_names !== "undefined" && Array.isArray(result_names) && result_names.length > 0) {
			_colorize_table_entries_by_trial_status();
			_colorize_table_entries_by_results();
			_colorize_table_entries_by_run_time();
			_colorize_table_entries_by_generation_method();
			_colorize_table_entries_by_generation_node_or_hostname();
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

function demo_mode(nr_sec = 3) {
	let i = 0;
	let tabs = $('menu[role="tablist"] > button');

	setInterval(() => {
		tabs.attr('aria-selected', 'false').removeClass('active');

		let tab = tabs.eq(i % tabs.length);
		tab.attr('aria-selected', 'true').addClass('active');

		tab.trigger('click');

		i++;
	}, nr_sec * 1000);
}

function resizePlotlyCharts() {
	const plotlyElements = document.querySelectorAll('.js-plotly-plot');

	if (plotlyElements.length) {
		const windowWidth = window.innerWidth;
		const windowHeight = window.innerHeight;

		const newWidth = windowWidth * 0.9;
		const newHeight = windowHeight * 0.9;

		plotlyElements.forEach(function(element, index) {
			const layout = {
				width: newWidth,
				height: newHeight,
				plot_bgcolor: 'rgba(0, 0, 0, 0)',
				paper_bgcolor: 'rgba(0, 0, 0, 0)',
			};

			Plotly.relayout(element, layout)
		});
	}

	make_text_in_parallel_plot_nicer();
	apply_theme_based_on_system_preferences();
}

function plotTimelineFromGlobals() {
	if (
		typeof tab_results_headers_json === "undefined" ||
		typeof tab_results_csv_json === "undefined" ||
		!Array.isArray(tab_results_headers_json) ||
		!Array.isArray(tab_results_csv_json)
	) {
		console.warn("Global variables 'tab_results_headers_json' or 'tab_results_csv_json' missing or invalid.");
		return null;
	}

	const headers = tab_results_headers_json;
	const data = tab_results_csv_json;

	const col = name => headers.indexOf(name);
	const ix_trial_index = col("trial_index");
	const ix_start_time = col("start_time");
	const ix_end_time = col("end_time");
	const ix_status = col("trial_status");

	if ([ix_trial_index, ix_start_time, ix_end_time, ix_status].some(ix => ix === -1)) {
		console.warn("One or more needed columns missing");
		return null;
	}

	const traces = [];

	// Add dummy traces for legend
	traces.push({
		type: "scatter",
		mode: "lines",
		x: [null, null],
		y: [null, null],
		line: { color: "green", width: 4 },
		name: "COMPLETED",
		showlegend: true,
		hoverinfo: "none"
	});
	traces.push({
		type: "scatter",
		mode: "lines",
		x: [null, null],
		y: [null, null],
		line: { color: "yellow", width: 4 },
		name: "RUNNING",
		showlegend: true,
		hoverinfo: "none"
	});
	traces.push({
		type: "scatter",
		mode: "lines",
		x: [null, null],
		y: [null, null],
		line: { color: "red", width: 4 },
		name: "FAILED/OTHER",
		showlegend: true,
		hoverinfo: "none"
	});

	for (const row of data) {
		const trial_index = row[ix_trial_index];
		const start = row[ix_start_time];
		const end = row[ix_end_time];
		const status = row[ix_status];

		if (
			trial_index === "" || start === "" || end === "" ||
			isNaN(start) || isNaN(end)
		) continue;

		let color = "red"; // default
		if (status === "COMPLETED") color = "green";
		else if (status === "RUNNING") color = "yellow";

		traces.push({
			type: "scatter",
			mode: "lines",
			x: [new Date(start * 1000), new Date(end * 1000)],
			y: [trial_index, trial_index],
			line: { color: color, width: 4 },
			name: `Trial ${trial_index} (${status})`,
			showlegend: false,
			hoverinfo: "x+y+name"
		});
	}

	if (traces.length <= 3) { // only dummy traces added
		console.warn("No valid data for plotting found.");
		return null;
	}

	const layout = {
		title: "Trial Timeline",
		xaxis: {
			title: "Time",
			type: "date"
		},
		yaxis: {
			title: "Trial Index",
			autorange: "reversed"
		},
		margin: { t: 50 }
	};

	Plotly.newPlot('plot_timeline', traces, add_default_layout_data(layout));
	return true;
}

window.addEventListener('load', updatePreWidths);
window.addEventListener('resize', updatePreWidths);

$(document).ready(function() {
	colorize_table_entries();

	add_up_down_arrows_for_scrolling();

	add_colorize_to_gridjs_table();
});

window.addEventListener('resize', function() {
	resizePlotlyCharts();
});
