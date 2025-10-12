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
