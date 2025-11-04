function plotResultEvolution() {
	if ($("#plotResultEvolution").data("loaded") == "true") {
		return;
	}

	result_names.forEach(resultName => {
		let xColumnIndex = tab_results_headers_json.indexOf("trial_index");
		let resultIndex = tab_results_headers_json.indexOf(resultName);

		let filteredData = tab_results_csv_json.map(row => {
			let x = parseFloat(row[xColumnIndex]);
			let y = parseFloat(row[resultIndex]);
			if (!isNaN(x) && !isNaN(y)) {
				return { x, y };
			}
			return null;
		}).filter(d => d !== null);

		if (filteredData.length === 0) return;

		filteredData.sort((a, b) => a.x - b.x);

		let xData = filteredData.map(d => d.x);
		let yData = filteredData.map(d => d.y);

		let N = xData.length;
		let sumX = xData.reduce((a, b) => a + b, 0);
		let sumY = yData.reduce((a, b) => a + b, 0);
		let sumXY = xData.reduce((sum, xi, idx) => sum + xi * yData[idx], 0);
		let sumX2 = xData.reduce((sum, xi) => sum + xi * xi, 0);

		let denominator = (N * sumX2 - sumX * sumX);
		let a = denominator !== 0 ? (N * sumXY - sumX * sumY) / denominator : 0;
		let b = (sumY - a * sumX) / N;

		let fitYData = xData.map(x => a * x + b);

		let traceData = {
			x: xData,
			y: yData,
			mode: 'lines+markers',
			name: resultName,
			line: { shape: 'linear', color: 'blue' },
			marker: { size: get_marker_size() }
		};

		let traceFit = {
			x: xData,
			y: fitYData,
			mode: 'lines',
			name: resultName + ' Fit',
			line: { dash: 'dash', color: 'red' }
		};

		// --- Layout ---
		let layout = {
			title: `Evolution of ${resultName} over time`,
			xaxis: { title: get_axis_title_data("Trial-Index") },
			yaxis: { title: get_axis_title_data(resultName) },
			showlegend: true
		};

		let subDiv = document.createElement("div");
		subDiv.style.marginBottom = "30px";
		document.getElementById("plotResultEvolution").appendChild(subDiv);

		Plotly.newPlot(subDiv, [traceData, traceFit], add_default_layout_data(layout));

		let formulaDiv = document.createElement("div");
		formulaDiv.style.marginTop = "5px";
		formulaDiv.style.fontSize = "14px";
		formulaDiv.innerHTML = `Fit equation: \\(y = ${a.toFixed(3)} x + ${b.toFixed(3)}\\)`;
		subDiv.appendChild(formulaDiv);

		if (window.MathJax) {
			MathJax.typesetPromise([formulaDiv]);
		}
	});

	$("#plotResultEvolution").data("loaded", "true");
}
