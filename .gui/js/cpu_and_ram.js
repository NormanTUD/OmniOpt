async function plot_worker_cpu_ram() {
	if($("#worker_cpu_ram_pre").data("loaded") == "true") {
		return;
	}

	const logData = $("#worker_cpu_ram_pre").text();
	const regex = /^Unix-Timestamp: (\d+), Hostname: ([\w-]+), CPU: ([\d.]+)%, RAM: ([\d.]+) MB \/ ([\d.]+) MB$/;

	const hostData = {};

	logData.split("\n").forEach(line => {
		line = line.trim();
		const match = line.match(regex);
		if (match) {
			const timestamp = new Date(parseInt(match[1]) * 1000);
			const hostname = match[2];
			const cpu = parseFloat(match[3]);
			const ram = parseFloat(match[4]);

			if (!hostData[hostname]) {
				hostData[hostname] = { timestamps: [], cpuUsage: [], ramUsage: [] };
			}

			hostData[hostname].timestamps.push(timestamp);
			hostData[hostname].cpuUsage.push(cpu);
			hostData[hostname].ramUsage.push(ram);
		}
	});

	if (!Object.keys(hostData).length) {
		console.log("No valid data found");
		return;
	}

	const container = document.getElementById("cpuRamWorkerChartContainer");
	container.innerHTML = "";

	var i = 1;

	Object.entries(hostData).forEach(([hostname, { timestamps, cpuUsage, ramUsage }], index) => {
		const chartId = `workerChart_${index}`;
		const chartDiv = document.createElement("div");
		chartDiv.id = chartId;
		chartDiv.style.marginBottom = "40px";
		container.appendChild(chartDiv);

		const cpuTrace = {
			x: timestamps,
			y: cpuUsage,
			mode: "lines+markers",
			name: "CPU Usage (%)",
			yaxis: "y1",
			line: {
				color: "red"
			}
		};

		const ramTrace = {
			x: timestamps,
			y: ramUsage,
			mode: "lines+markers",
			name: "RAM Usage (MB)",
			yaxis: "y2",
			line: {
				color: "blue"
			}
		};

		const layout = {
			title: `Worker CPU and RAM Usage - ${hostname}`,
			xaxis: {
				title: get_axis_title_data("Timestamp", "date")
			},
			yaxis: {
				title: get_axis_title_data("CPU Usage (%)"),
				side: "left",
				color: "red"
			},
			yaxis2: {
				title: get_axis_title_data("RAM Usage (MB)"),
				side: "right",
				overlaying: "y",
				color: "blue"
			},
			showlegend: true
		};

		Plotly.newPlot(chartId, [cpuTrace, ramTrace], add_default_layout_data(layout));
		i++;
	});

	$("#plot_worker_cpu_ram_button").remove();
	$("#worker_cpu_ram_pre").data("loaded", "true");
}


function plotCPUAndRAMUsage() {
	if($("#mainWorkerCPURAM").data("loaded") == "true") {
		return;
	}
	var timestamps = tab_main_worker_cpu_ram_csv_json.map(row => new Date(row[0] * 1000));
	var ramUsage = tab_main_worker_cpu_ram_csv_json.map(row => row[1]);
	var cpuUsage = tab_main_worker_cpu_ram_csv_json.map(row => row[2]);

	var trace1 = {
		x: timestamps,
		y: cpuUsage,
		mode: 'lines+markers',
		marker: {
			size: get_marker_size(),
		},
		name: 'CPU Usage (%)',
		type: 'scatter',
		yaxis: 'y1'
	};

	var trace2 = {
		x: timestamps,
		y: ramUsage,
		mode: 'lines+markers',
		marker: {
			size: get_marker_size(),
		},
		name: 'RAM Usage (MB)',
		type: 'scatter',
		yaxis: 'y2'
	};

	var layout = {
		title: 'CPU and RAM Usage Over Time',
		xaxis: {
			title: get_axis_title_data("Timestamp", "date"),
			tickmode: 'array',
			tickvals: timestamps.filter((_, index) => index % Math.max(Math.floor(timestamps.length / 10), 1) === 0),
			ticktext: timestamps.filter((_, index) => index % Math.max(Math.floor(timestamps.length / 10), 1) === 0).map(t => t.toLocaleString()),
			tickangle: -45
		},
		yaxis: {
			title: get_axis_title_data("CPU Usage (%)"),
			rangemode: 'tozero'
		},
		yaxis2: {
			title: get_axis_title_data("RAM Usage (MB)"),
			overlaying: 'y',
			side: 'right',
			rangemode: 'tozero'
		},
		legend: {
			x: 0.1,
			y: 0.9
		}
	};

	var data = [trace1, trace2];
	Plotly.newPlot('mainWorkerCPURAM', data, add_default_layout_data(layout));
	$("#mainWorkerCPURAM").data("loaded", "true");
}

