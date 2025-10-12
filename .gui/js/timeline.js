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

