function plotJobInfoCharts() {
	if ($("#plotJobInfoCharts").data("loaded") === "true") return;

	let plotDiv = document.getElementById("plotJobInfoCharts");
	plotDiv.innerHTML = "";

	let hasRuntime = false;

	if (typeof tab_job_infos_headers_json !== "undefined" && typeof tab_job_infos_csv_json !== "undefined") {
		let headers = tab_job_infos_headers_json;
		let rows = tab_job_infos_csv_json;

		let runTimeIdx = headers.indexOf("run_time");
		let exitCodeIdx = headers.indexOf("exit_code");
		let hostnameIdx = headers.indexOf("hostname");

		if (runTimeIdx !== -1) {
			let runTimes = rows.map(row => parseFloat(row[runTimeIdx])).filter(v => !isNaN(v) && v > 0);

			if (runTimes.length >= 2) {
				hasRuntime = true;

				let rtDiv = document.createElement("div");
				plotDiv.appendChild(rtDiv);
				Plotly.newPlot(rtDiv, [{
					x: runTimes,
					type: 'histogram',
					marker: { color: '#4a90d9' },
					nbinsx: Math.min(30, Math.max(5, Math.ceil(Math.sqrt(runTimes.length))))
				}], add_default_layout_data({
					title: 'Run Time Distribution',
					xaxis: { title: get_axis_title_data("Run Time (s)") },
					yaxis: { title: get_axis_title_data("Count") }
				}));

				if (exitCodeIdx !== -1) {
					let ecData = {};
					rows.forEach(row => {
						let rt = parseFloat(row[runTimeIdx]);
						let ec = row[exitCodeIdx];
						if (isNaN(rt) || rt <= 0 || !ec || ec === "None") return;
						if (!ecData[ec]) ecData[ec] = [];
						ecData[ec].push(rt);
					});

					let ecKeys = Object.keys(ecData);
					if (ecKeys.length >= 1) {
						let ecDiv = document.createElement("div");
						plotDiv.appendChild(ecDiv);
						let ecTraces = ecKeys.map(ec => ({
							y: ecData[ec],
							type: 'violin',
							name: 'Exit ' + ec,
							box: { visible: true },
							meanline: { visible: true }
						}));
						Plotly.newPlot(ecDiv, ecTraces, add_default_layout_data({
							title: 'Run Time by Exit Code',
							yaxis: { title: get_axis_title_data("Run Time (s)") },
							xaxis: { title: get_axis_title_data("Exit Code") },
							showlegend: false
						}));
					}
				}

				if (hostnameIdx !== -1) {
					let hostData = {};
					rows.forEach(row => {
						let rt = parseFloat(row[runTimeIdx]);
						let host = row[hostnameIdx];
						if (isNaN(rt) || rt <= 0 || !host) return;
						if (!hostData[host]) hostData[host] = [];
						hostData[host].push(rt);
					});

					let hostKeys = Object.keys(hostData);
					if (hostKeys.length >= 2) {
						let hostDiv = document.createElement("div");
						plotDiv.appendChild(hostDiv);
						let hostTraces = hostKeys.map(h => ({
							y: hostData[h],
							type: 'box',
							name: h
						}));
						Plotly.newPlot(hostDiv, hostTraces, add_default_layout_data({
							title: 'Run Time by Hostname',
							yaxis: { title: get_axis_title_data("Run Time (s)") },
							xaxis: { title: get_axis_title_data("Hostname") },
							showlegend: false
						}));
					}
				}
			}
		}
	}

	if (!hasRuntime) {
		plotDiv.innerHTML = '<p style="text-align:center; padding: 20px;">No job_infos.csv data available with run_time information.</p>';
	}

	$("#plotJobInfoCharts").data("loaded", "true");
}
