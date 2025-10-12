function plotHeatmap() {
	if ($("#plotHeatmap").data("loaded") === "true") return;

	let numericColumns = tab_results_headers_json.filter(col => {
		if (special_col_names.includes(col) || result_names.includes(col)) return false;
		if (col.toLowerCase().startsWith("oo_info_")) return false;
		let idx = tab_results_headers_json.indexOf(col);
		return tab_results_csv_json.every(row => {
			let v = parseFloat(row[idx]);
			return !isNaN(v) && isFinite(v);
		});
	});

	if (numericColumns.length < 2) {
		console.error("Not enough valid numeric columns for Heatmap");
		return;
	}

	let columnData = numericColumns.map(col => {
		let idx = tab_results_headers_json.indexOf(col);
		return tab_results_csv_json.map(row => parseFloat(row[idx]));
	});

	let dataMatrix = numericColumns.map((_, i) =>
		numericColumns.map((_, j) => {
			let vals = columnData[i].map((v, k) => (v + columnData[j][k]) / 2);
			return vals.reduce((a, b) => a + b, 0) / vals.length;
		})
	);

	let trace = {
		z: dataMatrix,
		x: numericColumns,
		y: numericColumns,
		colorscale: 'Viridis',
		type: 'heatmap'
	};

	let layout = {
		xaxis: { title: get_axis_title_data("Columns") },
		yaxis: { title: get_axis_title_data("Columns") },
		showlegend: false
	};

	let plotDiv = document.getElementById("plotHeatmap");
	plotDiv.innerHTML = "";

	Plotly.newPlot(plotDiv, [trace], add_default_layout_data(layout));
	$("#plotHeatmap").data("loaded", "true");
}
