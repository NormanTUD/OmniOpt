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
