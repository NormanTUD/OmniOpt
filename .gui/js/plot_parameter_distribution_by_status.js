function plotParameterDistributionsByStatus() {
	const container = document.getElementById('parameter_by_status_distribution');
	if (!container) {
		console.error("No container with id 'parameter_by_status_distribution' found.");
		return null;
	}

	if ($(container).data("loaded") === "true") {
		return;
	}

	if (
		typeof special_col_names === "undefined" ||
		typeof result_names === "undefined" ||
		typeof result_min_max === "undefined" ||
		typeof tab_results_headers_json === "undefined" ||
		typeof tab_results_csv_json === "undefined"
	) {
		console.error("Missing one or more required global variables.");
		return null;
	}

	if (
		!Array.isArray(special_col_names) ||
		!Array.isArray(result_names) ||
		!Array.isArray(result_min_max) ||
		!Array.isArray(tab_results_headers_json) ||
		!Array.isArray(tab_results_csv_json)
	) {
		console.error("All inputs must be arrays.");
		return null;
	}

	container.innerHTML = "";

	const statusIndex = tab_results_headers_json.indexOf("trial_status");
	if (statusIndex < 0) {
		container.textContent = "No 'trial_status' found in data.";
		return null;
	}

	const trialStatuses = [...new Set(tab_results_csv_json.map(row => row[statusIndex]))].filter(s => s != null);
	const paramCols = tab_results_headers_json.filter(col =>
		!special_col_names.includes(col) &&
		!result_names.includes(col)
	);

	for (const param of paramCols) {
		const paramIndex = tab_results_headers_json.indexOf(param);
		if (paramIndex < 0) continue;

		const traces = [];

		trialStatuses.forEach((status) => {
			const filteredValues = tab_results_csv_json
				.filter(row => row[statusIndex] === status)
				.map(row => row[paramIndex])
				.filter(val => val !== "" && val != null && !isNaN(val))
				.map(Number);

			if (filteredValues.length >= 1) {
				const nbins = 20;
				traces.push({
					type: 'histogram',
					x: filteredValues,
					name: status,
					opacity: 0.6,
					xbingroup: 0,
					marker: {color: getColorForStatus(status)},
					nbinsx: nbins,
				});
			}
		});

		if (traces.length > 0) {
			const h2 = document.createElement('h2');
			if(!param.startsWith("OO_Info_")) {
				h2.textContent = `Histogram: ${param}`;
				container.appendChild(h2);

				const plotDiv = document.createElement('div');
				plotDiv.style.marginBottom = '30px';
				container.appendChild(plotDiv);

				Plotly.newPlot(plotDiv, traces, {
					barmode: 'overlay',  // 'stack' oder 'overlay'
					xaxis: {
						title: { text: String(param) },  // Sicherstellen, dass es ein Textobjekt ist
						automargin: true,
						tickangle: -45,                  // Optional: bessere Lesbarkeit
						titlefont: { size: 16 }          // Optional: größerer Titel
					},
					yaxis: {
						title: { text: 'Count' },        // Titel explizit als Objekt angeben
						automargin: true,
						titlefont: { size: 16 }          // Optional: größerer Titel
					},
					margin: {
						l: 60,
						r: 30,
						t: 30,
						b: 80                            // genug Platz für x-Achsentitel
					},
					legend: {
						orientation: "h"
					}
				}, {
					responsive: true
				});

			}
		}
	}

	$(container).data("loaded", "true");

	function getColorForStatus(status) {
		const baseAlpha = 0.5;
		switch(status.toUpperCase()) {
			case 'FAILED':      return `rgba(214, 39, 40, ${baseAlpha})`;
			case 'COMPLETED':   return `rgba(44, 160, 44, ${baseAlpha})`;
			case 'ABANDONED':   return `rgba(255, 215, 0, ${baseAlpha})`;
			case 'RUNNING':     return `rgba(50, 50, 44, ${baseAlpha})`;
			default:
				const otherColors = [
					`rgba(31, 119, 180, ${baseAlpha})`,
					`rgba(255, 127, 14, ${baseAlpha})`,
					`rgba(148, 103, 189, ${baseAlpha})`,
					`rgba(140, 86, 75, ${baseAlpha})`,
					`rgba(227, 119, 194, ${baseAlpha})`,
					`rgba(127, 127, 127, ${baseAlpha})`,
					`rgba(188, 189, 34, ${baseAlpha})`,
					`rgba(23, 190, 207, ${baseAlpha})`
				];
				let hash = 0;
				for (let i = 0; i < status.length; i++) {
					hash = status.charCodeAt(i) + ((hash << 5) - hash);
				}
				const index = Math.abs(hash) % otherColors.length;
				return otherColors[index];
		}
	}

	resizePlotlyCharts();
}
