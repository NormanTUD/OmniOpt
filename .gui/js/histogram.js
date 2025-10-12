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
