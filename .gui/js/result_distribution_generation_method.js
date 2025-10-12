function plotResultsDistributionByGenerationMethod() {
	if ("true" === $("#plotResultsDistributionByGenerationMethod").data("loaded")) {
		return;
	}

	var res_col = result_names[0];
	var gen_method_col = "generation_node";

	var data = {};

	tab_results_csv_json.forEach(row => {
		var gen_method = row[tab_results_headers_json.indexOf(gen_method_col)];
		var result = row[tab_results_headers_json.indexOf(res_col)];

		if (!data[gen_method]) {
			data[gen_method] = [];
		}
		data[gen_method].push(result);
	});

	var traces = Object.keys(data).map(method => {
		return {
			y: data[method],
			type: 'box',
			name: method,
			boxpoints: 'outliers',
			jitter: 0.5,
			pointpos: 0
		};
	});

	var layout = {
		title: 'Distribution of Results by Generation Method',
		yaxis: {
			title: get_axis_title_data(res_col)
		},
		xaxis: {
			title: get_axis_title_data("Generation Method")
		},
		boxmode: 'group'
	};

	Plotly.newPlot("plotResultsDistributionByGenerationMethod", traces, add_default_layout_data(layout));
	$("#plotResultsDistributionByGenerationMethod").data("loaded", "true");
}

