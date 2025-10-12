function plotGPUUsage() {
	if ($("#tab_gpu_usage").data("loaded") === "true") {
		return;
	}

	Object.keys(gpu_usage).forEach(node => {
		const nodeData = gpu_usage[node];

		var timestamps = [];
		var gpuUtilizations = [];
		var temperatures = [];

		nodeData.forEach(entry => {
			try {
				var timestamp = new Date(entry[0]* 1000);
				var utilization = parseFloat(entry[1]);
				var temperature = parseFloat(entry[2]);

				if (!isNaN(timestamp) && !isNaN(utilization) && !isNaN(temperature)) {
					timestamps.push(timestamp);
					gpuUtilizations.push(utilization);
					temperatures.push(temperature);
				} else {
					console.warn("Invalid data point:", entry);
				}
			} catch (error) {
				console.error("Error processing GPU data entry:", error, entry);
			}
		});

		var trace1 = {
			x: timestamps,
			y: gpuUtilizations,
			mode: 'lines+markers',
			marker: {
				size: get_marker_size(),
			},
			name: 'GPU Utilization (%)',
			type: 'scatter',
			yaxis: 'y1'
		};

		var trace2 = {
			x: timestamps,
			y: temperatures,
			mode: 'lines+markers',
			marker: {
				size: get_marker_size(),
			},
			name: 'GPU Temperature (°C)',
			type: 'scatter',
			yaxis: 'y2'
		};

		var layout = {
			title: 'GPU Usage Over Time - ' + node,
			xaxis: {
				title: get_axis_title_data("Timestamp", "date"),
				tickmode: 'array',
				tickvals: timestamps.filter((_, index) => index % Math.max(Math.floor(timestamps.length / 10), 1) === 0),
				ticktext: timestamps.filter((_, index) => index % Math.max(Math.floor(timestamps.length / 10), 1) === 0).map(t => t.toLocaleString()),
				tickangle: -45
			},
			yaxis: {
				title: get_axis_title_data("GPU Utilization (%)"),
				overlaying: 'y',
				rangemode: 'tozero'
			},
			yaxis2: {
				title: get_axis_title_data("GPU Temperature (°C)"),
				overlaying: 'y',
				side: 'right',
				position: 0.85,
				rangemode: 'tozero'
			},
			legend: {
				x: 0.1,
				y: 0.9
			}
		};

		var divId = 'gpu_usage_plot_' + node;

		if (!document.getElementById(divId)) {
			var div = document.createElement('div');
			div.id = divId;
			div.className = 'gpu-usage-plot';
			document.getElementById('tab_gpu_usage').appendChild(div);
		}

		var plotData = [trace1, trace2];

		Plotly.newPlot(divId, plotData, add_default_layout_data(layout));
	});

	$("#tab_gpu_usage").data("loaded", "true");
}
