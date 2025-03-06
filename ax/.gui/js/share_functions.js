function get_graph_width() {
	var width = window.innerWidth || document.documentElement.clientWidth || document.body.clientWidth;
	return Math.floor(width * 0.9);
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
		sort: true
	}).render(document.getElementById(table_name));

	apply_theme_based_on_system_preferences();
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
	return data.filter(row => !row.includes(""));
}

function createParallelPlot(dataArray, headers, resultNames, ignoreColumns = []) {
	if($("#parallel-plot").data("loaded") == "true") {
		return;
	}
	dataArray = filterNonEmptyRows(dataArray);
	const ignoreSet = new Set(ignoreColumns);
	const numericalCols = [];
	const categoricalCols = [];
	const categoryMappings = {};

	headers.forEach((header, colIndex) => {
		if (ignoreSet.has(header)) return;

		const values = dataArray.map(row => row[colIndex]);
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

	if (resultNames.length === 1 && numericalCols.some(col => col.name.toLowerCase() === resultNames[0].toLowerCase())) {
		const resultCol = numericalCols.find(col => col.name.toLowerCase() === resultNames[0].toLowerCase());
		colorValues = dataArray.map(row => parseFloat(row[resultCol.index]));
		colorScale = [[0, 'green'], [1, 'red']];
	}

	const trace = {
		type: 'parcoords',
		dimensions: dimensions,
		line: colorValues ? { color: colorValues, colorscale: colorScale } : {}
	};

	dimensions.forEach(dim => {
		if (!dim.line) {
			dim.line = {};
		}
		if (!dim.line.color) {
			dim.line.color = 'rgba(169,169,169, 0.01)';
		}
	});

	let layout = {
		width: get_graph_width(),
		height: 800,
		paper_bgcolor: 'rgba(0,0,0,0)',
		plot_bgcolor: 'rgba(0,0,0,0)'
	}

	Plotly.newPlot('parallel-plot', [trace], layout);

	$("#parallel-plot").data("loaded", "true");
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
		legend: { x: 0, y: 1 },
		paper_bgcolor: 'rgba(0,0,0,0)',
		plot_bgcolor: 'rgba(0,0,0,0)'
	};

	Plotly.newPlot('workerUsagePlot', [trace1, trace2], layout);
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

	var layout = {
		title: 'CPU and RAM Usage Over Time',
		xaxis: {
			title: 'Timestamp',
			tickmode: 'array',
			tickvals: timestamps.filter((_, index) => index % Math.max(Math.floor(timestamps.length / 10), 1) === 0),
			ticktext: timestamps.filter((_, index) => index % Math.max(Math.floor(timestamps.length / 10), 1) === 0).map(t => t.toLocaleString()),
			tickangle: -45
		},
		yaxis: {
			title: 'Usage',
			rangemode: 'tozero'
		},
		legend: {
			x: 0.1,
			y: 0.9
		},
		paper_bgcolor: 'rgba(0,0,0,0)',
		plot_bgcolor: 'rgba(0,0,0,0)'
	};

	var data = [trace1, trace2];
	Plotly.newPlot('mainWorkerCPURAM', data, layout);
	$("#mainWorkerCPURAM").data("loaded", "true");
}

function plotScatter2d() {
	if ($("#plotScatter2d").data("loaded") == "true") {
		return;
	}
	var numericColumns = tab_results_headers_json.filter(col =>
		!special_col_names.includes(col) && !result_names.includes(col) &&
		tab_results_csv_json.every(row => !isNaN(parseFloat(row[tab_results_headers_json.indexOf(col)])))
	);

	if (numericColumns.length < 2) {
		console.error("Not enough columns for Scatter-Plots");
		return;
	}

	var resultIndex = tab_results_headers_json.findIndex(function(header) {
		return header.toLowerCase() === result_names[0].toLowerCase();
	});
	var resultValues = tab_results_csv_json.map(row => row[resultIndex]);
	var minResult = Math.min(...resultValues.filter(value => value !== null && value !== ""));
	var maxResult = Math.max(...resultValues.filter(value => value !== null && value !== ""));

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
				result: row[resultIndex] !== "" ? parseFloat(row[resultIndex]) : null
			}));

			let colors = data.map(d => {
				if (d.result === null) {
					return 'rgb(0, 0, 0)';
				} else {
					return `rgb(${Math.round(255 * (d.result - minResult) / (maxResult - minResult))},
			 ${Math.round(255 * (1 - (d.result - minResult) / (maxResult - minResult)))},
			 0)`;
				}
			});

			let trace = {
				x: data.map(d => d.x),
				y: data.map(d => d.y),
				mode: 'markers',
				marker: {
					size: 10,
					color: data.map(d => d.result),
					colorscale: [
						[0, 'green'],
						[1, 'red']
					],
					colorbar: {
						title: 'Result',
						tickvals: [minResult, maxResult],
						ticktext: [`${minResult}`, `${maxResult}`]
					}
				},
				text: data.map(d => d.result !== null ? `Result: ${d.result}` : 'No result'),
				type: 'scatter',
				showlegend: false
			};

			let layout = {
				title: `${xCol} vs ${yCol}`,
				xaxis: { title: xCol },
				yaxis: { title: yCol },
				showlegend: false,
				width: get_graph_width(),
				height: 800,
				paper_bgcolor: 'rgba(0,0,0,0)',
				plot_bgcolor: 'rgba(0,0,0,0)'
			};

			let subDiv = document.createElement("div");
			plotDiv.appendChild(subDiv);

			Plotly.newPlot(subDiv, [trace], layout);
		}
	}
	$("#plotScatter2d").data("loaded", "true");
}

function plotScatter3d() {
	if($("#plotScatter3d").data("loaded") == "true") {
		return;
	}
	var numericColumns = [];
	var categoricalColumns = {};

	tab_results_headers_json.forEach(col => {
		if (!special_col_names.includes(col) && !result_names.includes(col)) {
			let values = tab_results_csv_json.map(row => row[tab_results_headers_json.indexOf(col)]);
			let isNumeric = values.every(v => !isNaN(parseFloat(v)));

			if (isNumeric) {
				numericColumns.push(col);
			} else {
				categoricalColumns[col] = [...new Set(values)];
			}
		}
	});

	let allColumns = [...numericColumns, ...Object.keys(categoricalColumns)];

	if (allColumns.length < 3) {
		console.error("Not enough columns for 3D scatter plots");
		return;
	}

	var resultIndex = tab_results_headers_json.findIndex(header => header.toLowerCase() === result_names[0].toLowerCase());
	var resultValues = tab_results_csv_json.map(row => row[resultIndex]);
	var minResult = Math.min(...resultValues.filter(value => value !== null && value !== ""));
	var maxResult = Math.max(...resultValues.filter(value => value !== null && value !== ""));

	var plotDiv = document.getElementById("plotScatter3d");
	if (!plotDiv) {
		console.error("Div element with id 'plotScatter3d' not found");
		return;
	}
	plotDiv.innerHTML = "";

	for (let i = 0; i < allColumns.length; i++) {
		for (let j = i + 1; j < allColumns.length; j++) {
			for (let k = j + 1; k < allColumns.length; k++) {
				let xCol = allColumns[i];
				let yCol = allColumns[j];
				let zCol = allColumns[k];

				let xIndex = tab_results_headers_json.indexOf(xCol);
				let yIndex = tab_results_headers_json.indexOf(yCol);
				let zIndex = tab_results_headers_json.indexOf(zCol);

				let data = tab_results_csv_json.map(row => ({
					x: numericColumns.includes(xCol) ? parseFloat(row[xIndex]) : categoricalColumns[xCol].indexOf(row[xIndex]),
					y: numericColumns.includes(yCol) ? parseFloat(row[yIndex]) : categoricalColumns[yCol].indexOf(row[yIndex]),
					z: numericColumns.includes(zCol) ? parseFloat(row[zIndex]) : categoricalColumns[zCol].indexOf(row[zIndex]),
					result: row[resultIndex] !== "" ? parseFloat(row[resultIndex]) : null
				}));

				let colors = data.map(d => d.result === null ? 'rgb(0, 0, 0)' :
					`rgb(${Math.round(255 * (d.result - minResult) / (maxResult - minResult))},
					    ${Math.round(255 * (1 - (d.result - minResult) / (maxResult - minResult)))},
					    0)`);

				let trace = {
					x: data.map(d => d.x),
					y: data.map(d => d.y),
					z: data.map(d => d.z),
					mode: 'markers',
					marker: {
						size: 5,
						color: data.map(d => d.result),
						colorscale: [
							[0, 'green'],
							[1, 'red']
						],
						colorbar: {
							title: 'Result',
							tickvals: [minResult, maxResult],
							ticktext: [`${minResult}`, `${maxResult}`]
						}
					},
					text: data.map(d => d.result !== null ? `Result: ${d.result}` : 'No result'),
					type: 'scatter3d',
					showlegend: false
				};

				let layout = {
					title: `${xCol} vs ${yCol} vs ${zCol}`,
					scene: {
						xaxis: { title: xCol },
						yaxis: { title: yCol },
						zaxis: { title: zCol }
					},
					showlegend: false,
					width: get_graph_width(),
					height: 800,
					paper_bgcolor: 'rgba(0,0,0,0)',
					plot_bgcolor: 'rgba(0,0,0,0)'
				};

				let subDiv = document.createElement("div");
				subDiv.style.marginBottom = "20px";
				plotDiv.appendChild(subDiv);

				Plotly.newPlot(subDiv, [trace], layout);
			}
		}
	}
	$("#plotScatter3d").data("loaded", "true");
}

async function load_pareto_graph() {
	if($("#tab_pareto_fronts").data("loaded") == "true") {
		return;
	}

	var data = pareto_front_data;

	if (!data || typeof data !== "object") {
		console.error("Invalid data format for pareto_front_data");
		return;
	}

	if (!Object.keys(data).length) {
		console.warn("No data found in pareto_front_data");
		return;
	}

	let categories = Object.keys(data);
	let allMetrics = new Set();

	function extractMetrics(obj, prefix = "") {
		let keys = Object.keys(obj);
		for (let key of keys) {
			let newPrefix = prefix ? `${prefix} -> ${key}` : key;
			if (typeof obj[key] === "object" && !Array.isArray(obj[key])) {
				extractMetrics(obj[key], newPrefix);
			} else {
				if (!newPrefix.includes("param_dicts") && !newPrefix.includes(" -> sems -> ") && !newPrefix.includes("absolute_metrics")) {
					allMetrics.add(newPrefix);
				}
			}
		}
	}

	for (let cat of categories) {
		extractMetrics(data[cat]);
	}

	allMetrics = Array.from(allMetrics);

	function extractValues(obj, metricPath, values) {
		let parts = metricPath.split(" -> ");
		let data = obj;
		for (let part of parts) {
			if (data && typeof data === "object") {
				data = data[part];
			} else {
				return;
			}
		}
		if (Array.isArray(data)) {
			values.push(...data);
		}
	}

	let graphContainer = document.getElementById("pareto_front_graphs_container");
	graphContainer.innerHTML = "";

	var already_plotted = [];

	for (let i = 0; i < allMetrics.length; i++) {
		for (let j = i + 1; j < allMetrics.length; j++) {
			let xMetric = allMetrics[i];
			let yMetric = allMetrics[j];

			let xValues = [];
			let yValues = [];

			for (let cat of categories) {
				let metricData = data[cat];
				extractValues(metricData, xMetric, xValues);
				extractValues(metricData, yMetric, yValues);
			}

			xValues = xValues.filter(v => v !== undefined && v !== null);
			yValues = yValues.filter(v => v !== undefined && v !== null);

			let cleanXMetric = xMetric.replace(/.* -> /g, "");
			let cleanYMetric = yMetric.replace(/.* -> /g, "");

			let plot_key = `${cleanXMetric}-${cleanYMetric}`;

			if (xValues.length > 0 && yValues.length > 0 && xValues.length === yValues.length && !already_plotted.includes(plot_key)) {
				let div = document.createElement("div");
				div.id = `pareto_front_graph_${i}_${j}`;
				div.style.marginBottom = "20px";
				graphContainer.appendChild(div);

				let layout = {
					title: `${cleanXMetric} vs ${cleanYMetric}`,
					xaxis: { title: cleanXMetric },
					yaxis: { title: cleanYMetric },
					hovermode: "closest",
					paper_bgcolor: 'rgba(0,0,0,0)',
					plot_bgcolor: 'rgba(0,0,0,0)'
				};

				let trace = {
					x: xValues,
					y: yValues,
					mode: "markers",
					type: "scatter",
					name: `${cleanXMetric} vs ${cleanYMetric}`
				};

				Plotly.newPlot(div.id, [trace], layout);

				already_plotted.push(plot_key);
			}
		}
	}

	$("#tab_pareto_fronts").data("loaded", "true");
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
			line: { color: "red" }
		};

		const ramTrace = {
			x: timestamps,
			y: ramUsage,
			mode: "lines+markers",
			name: "RAM Usage (MB)",
			yaxis: "y2",
			line: { color: "blue" }
		};

		const layout = {
			title: `Worker CPU and RAM Usage - ${hostname}`,
			xaxis: { title: "Timestamp" },
			yaxis: {
				title: "CPU Usage (%)",
				side: "left",
				color: "red"
			},
			yaxis2: {
				title: "RAM Usage (MB)",
				side: "right",
				overlaying: "y",
				color: "blue"
			},
			showlegend: true,
			paper_bgcolor: 'rgba(0,0,0,0)',
			plot_bgcolor: 'rgba(0,0,0,0)'
		};

		Plotly.newPlot(chartId, [cpuTrace, ramTrace], layout);
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
					$("#" + pre_id).html(data.data);
					$("#" + pre_id).data("loaded", true);
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
			marker: { color: 'rgb(0, 255, 0)' },
		};
	});

	let layout = {
		title: 'Boxplot of Numerical Columns',
		yaxis: { title: 'Value' },
		xaxis: { title: 'Columns' },
		showlegend: false,
		width: get_graph_width(),
		height: 800,
		paper_bgcolor: 'rgba(0,0,0,0)',
		plot_bgcolor: 'rgba(0,0,0,0)'
	};

	Plotly.newPlot(plotDiv, traces, layout);
	$("#plotBoxplot").data("loaded", "true");
}

function plotHeatmap() {
	if ($("#plotHeatmap").data("loaded") == "true") {
		return;
	}
	var numericColumns = tab_results_headers_json.filter(col =>
		!special_col_names.includes(col) && !result_names.includes(col) &&
		tab_results_csv_json.every(row => !isNaN(parseFloat(row[tab_results_headers_json.indexOf(col)])))
	);

	if (numericColumns.length < 2) {
		console.error("Not enough columns for Heatmap");
		return;
	}

	var dataMatrix = numericColumns.map(col => {
		let index = tab_results_headers_json.indexOf(col);
		return tab_results_csv_json.map(row => parseFloat(row[index]));
	});

	var trace = {
		z: dataMatrix,
		x: numericColumns,
		y: numericColumns,
		colorscale: 'Viridis',
		type: 'heatmap',
	};

	var layout = {
		title: 'Correlation Heatmap',
		xaxis: { title: 'Columns' },
		yaxis: { title: 'Columns' },
		showlegend: false,
		width: get_graph_width(),
		height: 800,
		paper_bgcolor: 'rgba(0,0,0,0)',
		plot_bgcolor: 'rgba(0,0,0,0)'
	};

	var plotDiv = document.getElementById("plotHeatmap");
	plotDiv.innerHTML = "";

	Plotly.newPlot(plotDiv, [trace], layout);
	$("#plotHeatmap").data("loaded", "true");
}

function plotHistogram() {
	if ($("#plotHistogram").data("loaded") == "true") {
		return;
	}
	var numericColumns = tab_results_headers_json.filter(col =>
		!special_col_names.includes(col) && !result_names.includes(col) &&
		tab_results_csv_json.every(row => !isNaN(parseFloat(row[tab_results_headers_json.indexOf(col)])))
	);

	if (numericColumns.length < 1) {
		console.error("Not enough columns for Histogram");
		return;
	}

	var plotDiv = document.getElementById("plotHistogram");
	plotDiv.innerHTML = "";

	let traces = numericColumns.map(col => {
		let index = tab_results_headers_json.indexOf(col);
		let data = tab_results_csv_json.map(row => parseFloat(row[index]));

		return {
			x: data,
			type: 'histogram',
			name: col,
			opacity: 0.7,
			marker: { color: 'rgb(0, 255, 0)' },
			autobinx: true
		};
	});

	let layout = {
		title: 'Histogram of Numerical Columns',
		xaxis: { title: 'Value' },
		yaxis: { title: 'Frequency' },
		showlegend: true,
		barmode: 'overlay',
		width: get_graph_width(),
		height: 800,
		paper_bgcolor: 'rgba(0,0,0,0)',
		plot_bgcolor: 'rgba(0,0,0,0)'
	};

	Plotly.newPlot(plotDiv, traces, layout);
	$("#plotHistogram").data("loaded", "true");
}

function plotViolin() {
	if ($("#plotViolin").data("loaded") == "true") {
		return;
	}
	var numericColumns = tab_results_headers_json.filter(col =>
		!special_col_names.includes(col) && !result_names.includes(col) &&
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
			box: { visible: true },
			line: { color: 'rgb(0, 255, 0)' },
			marker: { color: 'rgb(0, 255, 0)' },
			meanline: { visible: true },
		};
	});

	let layout = {
		title: 'Violin Plot of Numerical Columns',
		yaxis: { title: 'Value' },
		xaxis: { title: 'Columns' },
		showlegend: false,
		width: get_graph_width(),
		height: 800,
		paper_bgcolor: 'rgba(0,0,0,0)',
		plot_bgcolor: 'rgba(0,0,0,0)'
	};

	Plotly.newPlot(plotDiv, traces, layout);
	$("#plotViolin").data("loaded", "true");
}

function plotBubbleChart() {
    if ($("#plotBubbleChart").data("loaded") == "true") {
        return;
    }
    var numericColumns = tab_results_headers_json.filter(col =>
        !special_col_names.includes(col) && !result_names.includes(col) &&
        tab_results_csv_json.every(row => !isNaN(parseFloat(row[tab_results_headers_json.indexOf(col)])))
    );

    if (numericColumns.length < 2) {
        console.error("Not enough columns for Bubble Chart");
        return;
    }

    var resultIndex = tab_results_headers_json.findIndex(function(header) {
        return result_names.includes(header.toLowerCase());
    });

    var plotDiv = document.getElementById("plotBubbleChart");
    plotDiv.innerHTML = "";

    let xCol = numericColumns[0];
    let yCol = numericColumns[1];
    let sizeCol = "run_time";  // Beispielgröße für Blasen

    let xIndex = tab_results_headers_json.indexOf(xCol);
    let yIndex = tab_results_headers_json.indexOf(yCol);
    let sizeIndex = tab_results_headers_json.indexOf(sizeCol);

    let data = tab_results_csv_json.map(row => ({
        x: parseFloat(row[xIndex]),
        y: parseFloat(row[yIndex]),
        size: parseFloat(row[sizeIndex]),
        result: row[resultIndex] !== "" ? parseFloat(row[resultIndex]) : null
    }));

    let trace = {
        x: data.map(d => d.x),
        y: data.map(d => d.y),
        mode: 'markers',
        marker: {
            size: data.map(d => d.size),
            color: data.map(d => d.result),
            colorscale: 'Viridis',
            showscale: true
        },
        type: 'scatter',
        showlegend: false
    };

    let layout = {
        title: `${xCol} vs ${yCol}`,
        xaxis: { title: xCol },
        yaxis: { title: yCol },
        showlegend: false,
        width: get_graph_width(),
        height: 800,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)'
    };

    Plotly.newPlot(plotDiv, [trace], layout);
    $("#plotBubbleChart").data("loaded", "true");
}

