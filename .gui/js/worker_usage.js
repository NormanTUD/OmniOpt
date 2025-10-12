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

