function plotResultEvolution() {
	if ($("#plotResultEvolution").data("loaded") === "true") return;

	result_names.forEach(resultName => {
		let xColumnIndex = tab_results_headers_json.indexOf("trial_index");
		let resultIndex = tab_results_headers_json.indexOf(resultName);

		let filteredData = tab_results_csv_json.map(row => {
			let x = parseFloat(row[xColumnIndex]);
			let y = parseFloat(row[resultIndex]);
			if (!isNaN(x) && !isNaN(y)) return { x, y };
			return null;
		}).filter(d => d !== null);

		if (filteredData.length === 0) return;

		filteredData.sort((a, b) => a.x - b.x);

		let xData = filteredData.map(d => d.x);
		let yData = filteredData.map(d => d.y);

		let linearFit = calculateLinearFit(xData, yData);
		let loessY = loessFit(xData, yData);

		plotSingleRun(subDivContainer(), xData, yData, linearFit, loessY, resultName);
	});

	$("#plotResultEvolution").data("loaded", "true");
}

// ------------------- Helper Functions -------------------

function calculateLinearFit(x, y) {
	let N = x.length;
	let sumX = x.reduce((a, b) => a + b, 0);
	let sumY = y.reduce((a, b) => a + b, 0);
	let sumXY = x.reduce((sum, xi, idx) => sum + xi * y[idx], 0);
	let sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);

	let denominator = N * sumX2 - sumX * sumX;
	let a = denominator !== 0 ? (N * sumXY - sumX * sumY) / denominator : 0;
	let b = (sumY - a * sumX) / N;

	// R^2
	let yMean = sumY / N;
	let ssTot = y.reduce((s, yi) => s + Math.pow(yi - yMean, 2), 0);
	let ssRes = y.reduce((s, yi, idx) => s + Math.pow(yi - (a * x[idx] + b), 2), 0);
	let r2 = ssTot !== 0 ? 1 - ssRes / ssTot : 1;

	return { a, b, r2, fitY: x.map(xi => a * xi + b) };
}

function loessFit(x, y, bandwidth = 0.3) {
	let n = x.length;
	let ypred = [];
	for (let i = 0; i < n; i++) {
		let xi = x[i];
		let distances = x.map(v => Math.abs(v - xi));
		let radius = distances.slice().sort((a, b) => a - b)[Math.floor(bandwidth * n)];
		let weights = x.map(v => {
			let u = Math.abs(v - xi) / radius;
			return u < 1 ? Math.pow(1 - u * u, 2) : 0;
		});
		let sw = weights.reduce((a, b) => a + b, 0);
		let xw = x.reduce((s, v, j) => s + weights[j] * v, 0) / sw;
		let yw = y.reduce((s, v, j) => s + weights[j] * v, 0) / sw;
		let num = 0, den = 0;
		for (let j = 0; j < n; j++) {
			num += weights[j] * (x[j] - xw) * (y[j] - yw);
			den += weights[j] * Math.pow(x[j] - xw, 2);
		}
		let beta = den !== 0 ? num / den : 0;
		let alpha = yw - beta * xw;
		ypred.push(alpha + beta * xi);
	}
	return ypred;
}

function subDivContainer() {
	let subDiv = document.createElement("div");
	subDiv.style.marginBottom = "30px";
	document.getElementById("plotResultEvolution").appendChild(subDiv);
	return subDiv;
}

function plotSingleRun(subDiv, xData, yData, linearFit, loessY, resultName) {
	let traceData = {
		x: xData,
		y: yData,
		mode: 'lines+markers',
		name: resultName,
		line: { shape: 'linear', color: 'blue' },
		marker: { size: get_marker_size() }
	};

	let traceLinear = {
		x: xData,
		y: linearFit.fitY,
		mode: 'lines',
		name: resultName + ' Linear Fit',
		line: { dash: 'dash', color: 'red' }
	};

	let traceLoess = {
		x: xData,
		y: loessY,
		mode: 'lines',
		name: resultName + ' LOESS',
		line: { dash: 'dot', color: 'gray' }
	};

	let layout = {
		title: `Evolution of ${resultName} over time`,
		xaxis: { title: get_axis_title_data("Trial-Index") },
		yaxis: { title: get_axis_title_data(resultName) },
		showlegend: true
	};

	Plotly.newPlot(subDiv, [traceData, traceLinear, traceLoess], add_default_layout_data(layout));

	let formulaDiv = document.createElement("div");
	formulaDiv.style.marginTop = "5px";
	formulaDiv.style.fontSize = "14px";

	let trendScore = (linearFit.a * linearFit.r2).toFixed(3);

	formulaDiv.innerHTML = `
	<b>Linear Fit:</b> \\(y = ${linearFit.a.toFixed(3)} x + ${linearFit.b.toFixed(3)}\\)<br>
	<b>R²:</b> ${linearFit.r2.toFixed(3)}<br>
	<b>Trend Score:</b> ${trendScore} (${trendScore < 0 ? "Good" : "Bad/Stable"})<br>
	<b>LOESS:</b> smoothed trend (visual)
	`;

	subDiv.appendChild(formulaDiv);

	if (window.MathJax) MathJax.typesetPromise([formulaDiv]);
}
