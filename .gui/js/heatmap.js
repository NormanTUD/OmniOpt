function pearsonCorrelation(x, y) {
	let n = x.length;
	if (n < 3) return 0;

	let sumX = 0, sumY = 0;
	for (let i = 0; i < n; i++) {
		sumX += x[i];
		sumY += y[i];
	}
	let meanX = sumX / n;
	let meanY = sumY / n;

	let sumXY = 0, sumX2 = 0, sumY2 = 0;
	for (let i = 0; i < n; i++) {
		let dx = x[i] - meanX;
		let dy = y[i] - meanY;
		sumXY += dx * dy;
		sumX2 += dx * dx;
		sumY2 += dy * dy;
	}

	let denom = Math.sqrt(sumX2 * sumY2);
	if (denom === 0) return 0;
	return sumXY / denom;
}

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
			if (i === j) return 1;
			return pearsonCorrelation(columnData[i], columnData[j]);
		})
	);

	let trace = {
		z: dataMatrix,
		x: numericColumns,
		y: numericColumns,
		colorscale: [
			[0, '#313695'],
			[0.25, '#74add1'],
			[0.5, '#f7f7f7'],
			[0.75, '#f46d43'],
			[1, '#a50026']
		],
		zmin: -1,
		zmax: 1,
		type: 'heatmap',
		colorbar: {
			title: 'Pearson r',
			titlefont: { color: get_text_color() },
			tickfont: { color: get_text_color() }
		},
		text: dataMatrix.map(row => row.map(v => v.toFixed(3))),
		texttemplate: '%{text}',
		hovertemplate: '%{x} vs %{y}<br>r = %{z:.3f}<extra></extra>'
	};

	let layout = {
		title: 'Correlation Matrix',
		xaxis: { title: get_axis_title_data("Columns"), tickangle: -45 },
		yaxis: { title: get_axis_title_data("Columns") },
		showlegend: false
	};

	let plotDiv = document.getElementById("plotHeatmap");
	plotDiv.innerHTML = "";

	Plotly.newPlot(plotDiv, [trace], add_default_layout_data(layout));
	$("#plotHeatmap").data("loaded", "true");
}
