function plotResultPairs() {
	if ($("#plotResultPairs").data("loaded") == "true") {
		return;
	}

	var plotDiv = document.getElementById("plotResultPairs");
	plotDiv.innerHTML = "";

	for (let i = 0; i < result_names.length; i++) {
		for (let j = i + 1; j < result_names.length; j++) {
			let xName = result_names[i];
			let yName = result_names[j];

			let xIndex = tab_results_headers_json.indexOf(xName);
			let yIndex = tab_results_headers_json.indexOf(yName);

			let data = tab_results_csv_json
				.filter(row => row[xIndex] !== "" && row[yIndex] !== "")
				.map(row => ({
					x: parseFloat(row[xIndex]),
					y: parseFloat(row[yIndex]),
					status: row[tab_results_headers_json.indexOf("trial_status")]
				}));

			let colors = data.map(d => d.status === "COMPLETED" ? 'green' : (d.status === "FAILED" ? 'red' : 'gray'));

			let trace = {
				x: data.map(d => d.x),
				y: data.map(d => d.y),
				mode: 'markers',
				marker: {
					size: get_marker_size(),
					color: colors
				},
				text: data.map(d => `Status: ${d.status}`),
				type: 'scatter',
				showlegend: false
			};

			let layout = {
				xaxis: {
					title: get_axis_title_data(xName)
				},
				yaxis: {
					title: get_axis_title_data(yName)
				},
				showlegend: false
			};

			let subDiv = document.createElement("div");
			plotDiv.appendChild(subDiv);

			Plotly.newPlot(subDiv, [trace], add_default_layout_data(layout));
		}
	}

	$("#plotResultPairs").data("loaded", "true");
}
