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
			showlegend: false
		};

		let subDiv = document.createElement("div");
		document.getElementById("plotResultEvolution").appendChild(subDiv);

		Plotly.newPlot(subDiv, [trace], add_default_layout_data(layout));
	});

	$("#plotResultEvolution").data("loaded", "true");
}
