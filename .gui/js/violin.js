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

