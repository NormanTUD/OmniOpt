function plotExitCodesPieChart() {
	if ($("#plotExitCodesPieChart").data("loaded") == "true") {
		return;
	}

	var exitCodes = tab_results_csv_json.map(row => row[tab_results_headers_json.indexOf("exit_code")]);

	var exitCodeCounts = exitCodes.reduce(function(counts, exitCode) {
		counts[exitCode] = (counts[exitCode] || 0) + 1;
		return counts;
	}, {});

	var labels = Object.keys(exitCodeCounts);
	var values = Object.values(exitCodeCounts);

	var plotDiv = document.getElementById("plotExitCodesPieChart");
	plotDiv.innerHTML = "";

	var trace = {
		labels: labels,
		values: values,
		type: 'pie',
		hoverinfo: 'label+percent',
		textinfo: 'label+value',
		marker: {
			colors: ['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0']
		}
	};

	var layout = {
		title: 'Exit Code Distribution',
		showlegend: true
	};

	Plotly.newPlot(plotDiv, [trace], add_default_layout_data(layout));
	$("#plotExitCodesPieChart").data("loaded", "true");
}
