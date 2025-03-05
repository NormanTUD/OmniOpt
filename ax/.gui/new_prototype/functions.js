var log = console.log;

function enable_dark_mode() {
	document.body.classList.add('dark-mode');
}

function disable_dark_mode() {
	document.body.classList.remove('dark-mode');
}

function createTable(data, headers, table_name) {
	if (!$("#" + table_name).length) {
		console.warn("#" + table_name + " not found");
		return;
	}

	new gridjs.Grid({
		columns: headers,
		data: data,
		search: true,
		sort: true
	}).render(document.getElementById(table_name));
}

function filterTableOnZoom(eventData, data, keyX, keyY) {
	if(!$("#table").length) {
		console.warn("#table not found");
		return;
	}
	const xRange = eventData['xaxis.range'];
	const yRange = eventData['yaxis.range'];
	if (!xRange || !yRange) return;

	const filtered = data.filter(row =>
		row[keyX] >= xRange[0] && row[keyX] <= xRange[1] &&
		row[keyY] >= yRange[0] && row[keyY] <= yRange[1]
	);

	document.getElementById('table').innerHTML = '';
	createTable(filtered);
}

function copy_to_clipboard_base64(text) {
	copy_to_clipboard(atob(text));
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

function createParallelPlot(dataArray, headers, resultNames, ignoreColumns = []) {
	const ignoreSet = new Set(ignoreColumns);
	const numericalCols = [];
	const categoricalCols = [];
	const categoryMappings = {};

	headers.forEach((header, colIndex) => {
		if (ignoreSet.has(header)) return;

		const values = dataArray.map(row => row[colIndex]);
		log(header);
		log(values);
		if (values.every(val => !isNaN(parseFloat(val)))) {
			numericalCols.push({ name: header, index: colIndex });
		} else {
			categoricalCols.push({ name: header, index: colIndex });
			const uniqueValues = [...new Set(values)];
			categoryMappings[header] = Object.fromEntries(uniqueValues.map((val, i) => [val, i]));
		}
	});

	const dimensions = [];

	numericalCols.forEach(col => {
		dimensions.push({
			label: col.name,
			values: dataArray.map(row => parseFloat(row[col.index])),
			range: [
				Math.min(...dataArray.map(row => parseFloat(row[col.index]))),
				Math.max(...dataArray.map(row => parseFloat(row[col.index])))
			]
		});
	});

	categoricalCols.forEach(col => {
		dimensions.push({
			label: col.name,
			values: dataArray.map(row => categoryMappings[col.name][row[col.index]]),
			tickvals: Object.values(categoryMappings[col.name]),
			ticktext: Object.keys(categoryMappings[col.name])
		});
	});

	let colorScale = null;
	let colorValues = null;

	if (resultNames.length === 1 && numericalCols.some(col => col.name === resultNames[0])) {
		const resultCol = numericalCols.find(col => col.name === resultNames[0]);
		colorValues = dataArray.map(row => parseFloat(row[resultCol.index]));
		colorScale = [[0, 'green'], [1, 'red']];
	} else {
		console.log(numericalCols);
	}

	const trace = {
		type: 'parcoords',
		dimensions: dimensions,
		line: colorValues ? { color: colorValues, colorscale: colorScale } : {}
	};

	Plotly.newPlot('parallel-plot', [trace]);
}

function plotWorkerUsage(data) {
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
		line: { color: 'blue' }
	};

	let trace2 = {
		x: timestamps,
		y: realWorkers,
		mode: 'lines+markers',
		name: 'Real Workers',
		line: { color: 'red' }
	};

	let layout = {
		title: "Worker Usage Over Time",
		xaxis: { title: "Time", type: 'date' },
		yaxis: { title: "Number of Workers" },
		legend: { x: 0, y: 1 }
	};

	Plotly.newPlot('workerUsagePlot', [trace1, trace2], layout);
}

function plotCPUAndRAMUsage() {
    // Convert timestamps to human-readable format (optional)
    var timestamps = tab_main_worker_cpu_ram_csv_json.map(row => new Date(row[0] * 1000)); // Convert from Unix timestamp to Date object
    var ramUsage = tab_main_worker_cpu_ram_csv_json.map(row => row[1]);
    var cpuUsage = tab_main_worker_cpu_ram_csv_json.map(row => row[2]);

    // Create traces for the plot
    var trace1 = {
        x: timestamps,
        y: ramUsage,
        mode: 'lines+markers',
        name: 'RAM Usage (MB)',
        type: 'scatter'
    };

    var trace2 = {
        x: timestamps,
        y: cpuUsage,
        mode: 'lines+markers',
        name: 'CPU Usage (%)',
        type: 'scatter'
    };

    // Layout for the plot
    var layout = {
        title: 'CPU and RAM Usage Over Time',
        xaxis: {
            title: 'Timestamp',
            tickmode: 'array',
            tickvals: timestamps.filter((_, index) => index % Math.max(Math.floor(timestamps.length / 10), 1) === 0), // Reduce number of ticks
            ticktext: timestamps.filter((_, index) => index % Math.max(Math.floor(timestamps.length / 10), 1) === 0).map(t => t.toLocaleString()), // Convert timestamps to readable format
            tickangle: -45
        },
        yaxis: {
            title: 'Usage',
            rangemode: 'tozero'
        },
        legend: {
            x: 0.1,
            y: 0.9
        }
    };

    // Plot the data using Plotly
    var data = [trace1, trace2];
    Plotly.newPlot('mainWorkerCPURAM', data, layout);
}

function plotScatter2d() {
	var numericColumns = tab_results_headers_json.filter(col =>
		!special_col_names.includes(col) && !result_names.includes(col) &&
		tab_results_csv_json.every(row => !isNaN(parseFloat(row[tab_results_headers_json.indexOf(col)])))
	);

	if (numericColumns.length < 2) {
		console.error("Not enough columns for Scatter-Plots");
		return;
	}

	// Extrahiere RESULT-Spalte für die Farbgebung
	var resultIndex = tab_results_headers_json.indexOf(result_names[0]);
	var resultValues = tab_results_csv_json.map(row => parseFloat(row[resultIndex]));
	var minResult = Math.min(...resultValues);
	var maxResult = Math.max(...resultValues);

	// Erstelle Scatter-Plots für jede einzigartige Spaltenkombination
	var plotDiv = document.getElementById("plotScatter2d");
	plotDiv.innerHTML = "";

	for (let i = 0; i < numericColumns.length; i++) {
		for (let j = i + 1; j < numericColumns.length; j++) {
			let xCol = numericColumns[i];
			let yCol = numericColumns[j];

			let xIndex = tab_results_headers_json.indexOf(xCol);
			let yIndex = tab_results_headers_json.indexOf(yCol);

			let data = tab_results_csv_json.map(row => ({
				x: parseFloat(row[xIndex]),
				y: parseFloat(row[yIndex]),
				result: parseFloat(row[resultIndex])
			}));

			let colors = data.map(d => `rgb(${Math.round(255 * (d.result - minResult) / (maxResult - minResult))},
					     ${Math.round(255 * (1 - (d.result - minResult) / (maxResult - minResult)))},
					     0)`);

			let trace = {
				x: data.map(d => d.x),
				y: data.map(d => d.y),
				mode: 'markers',
				marker: { size: 10, color: colors },
				text: data.map(d => `Result: ${d.result}`),
				type: 'scatter'
			};

			let layout = {
				title: `${xCol} vs ${yCol}`,
				xaxis: { title: xCol },
				yaxis: { title: yCol },
				showlegend: false,
				coloraxis: { colorscale: [[0, 'green'], [1, 'red']] }
			};

			let subDiv = document.createElement("div");
			plotDiv.appendChild(subDiv);

			Plotly.newPlot(subDiv, [trace], layout);
		}
	}
}

function plotScatter3d() {
	var numericColumns = tab_results_headers_json.filter(col => 
		!special_col_names.includes(col) && !result_names.includes(col) &&
		tab_results_csv_json.every(row => !isNaN(parseFloat(row[tab_results_headers_json.indexOf(col)])))
	);

	if (numericColumns.length < 3) {
		console.error("Not enough numeric columns for 3D scatter plots");
		return;
	}

	var resultIndex = tab_results_headers_json.indexOf(result_names[0]);
	var resultValues = tab_results_csv_json.map(row => parseFloat(row[resultIndex]));
	var minResult = Math.min(...resultValues);
	var maxResult = Math.max(...resultValues);

	var plotDiv = document.getElementById("plotScatter3d");
	if (!plotDiv) {
		console.error("Div element with id 'plotScatter3d' not found");
		return;
	}
	plotDiv.innerHTML = "";

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
					result: parseFloat(row[resultIndex])
				}));

				let colors = data.map(d => `rgb(${Math.round(255 * (d.result - minResult) / (maxResult - minResult))}, 
					     ${Math.round(255 * (1 - (d.result - minResult) / (maxResult - minResult)))}, 
					     0)`);

				let trace = {
					x: data.map(d => d.x),
					y: data.map(d => d.y),
					z: data.map(d => d.z),
					mode: 'markers',
					marker: { size: 10, color: colors },
					text: data.map(d => `Result: ${d.result}`),
					type: 'scatter3d'
				};

				let layout = {
					title: `${xCol} vs ${yCol} vs ${zCol}`,
					scene: {
						xaxis: { title: xCol },
						yaxis: { title: yCol },
						zaxis: { title: zCol }
					},
					showlegend: false
				};

				let subDiv = document.createElement("div");
				subDiv.style.marginBottom = "20px"; // Add some spacing between plots
				plotDiv.appendChild(subDiv);

				Plotly.newPlot(subDiv, [trace], layout);
			}
		}
	}
}
