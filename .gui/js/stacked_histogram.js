function plotStackedHistogram() {
	if ($("#plotStackedHistogram").data("loaded") === "true") return;

	let genNodeIdx = tab_results_headers_json.indexOf("generation_node");
	if (genNodeIdx === -1) { console.error("generation_node not found for stacked histogram"); return; }

	let resultIdx = tab_results_headers_json.indexOf(result_names[0]);
	if (resultIdx === -1) { console.error("Result column not found for stacked histogram"); return; }

	let methodData = {};
	tab_results_csv_json.forEach(row => {
		let method = row[genNodeIdx];
		let val = parseFloat(row[resultIdx]);
		if (!method || isNaN(val)) return;
		if (!methodData[method]) methodData[method] = [];
		methodData[method].push(val);
	});

	let methods = Object.keys(methodData);
	if (methods.length < 1) { console.error("No generation methods found for stacked histogram"); return; }

	let plotDiv = document.getElementById("plotStackedHistogram");
	plotDiv.innerHTML = "";

	let colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'];

	let traces = methods.map((method, i) => ({
		x: methodData[method],
		type: 'histogram',
		name: method,
		opacity: 0.75,
		marker: { color: colors[i % colors.length] }
	}));

	let layout = {
		title: result_names[0] + " distribution by generation method",
		xaxis: { title: get_axis_title_data(result_names[0]) },
		yaxis: { title: get_axis_title_data("Count") },
		barmode: 'stack',
		showlegend: true,
		legend: { font: get_font_data() }
	};

	Plotly.newPlot(plotDiv, traces, add_default_layout_data(layout));
	$("#plotStackedHistogram").data("loaded", "true");
}
