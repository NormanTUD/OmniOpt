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

