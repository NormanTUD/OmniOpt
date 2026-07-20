function plotKdeHistogram() {
	if ($("#plotKdeHistogram").data("loaded") === "true") return;

	let resultIndex = tab_results_headers_json.indexOf(result_names[0]);
	if (resultIndex === -1) { console.error("Result column not found for KDE histogram"); return; }

	let resultValues = tab_results_csv_json.map(row => parseFloat(row[resultIndex])).filter(v => !isNaN(v));
	if (resultValues.length < 2) { console.error("Not enough result values for KDE histogram"); return; }

	let rMin = Math.min(...resultValues);
	let rMax = Math.max(...resultValues);
	let numBins = Math.min(10, Math.max(3, Math.ceil(Math.sqrt(resultValues.length))));

	let binWidth = (rMax - rMin) / numBins;
	if (binWidth === 0) binWidth = 1;

	function getResultBin(val) {
		let bin = Math.floor((val - rMin) / binWidth);
		if (bin >= numBins) bin = numBins - 1;
		if (bin < 0) bin = 0;
		return bin;
	}

	let colorScale = [
		'#1a9850', '#66bd63', '#a6d96a', '#d9ef8b',
		'#fee08b', '#fdae61', '#f46d43', '#d73027',
		'#a50026', '#67001f'
	];

	function binColor(binIdx) {
		return colorScale[binIdx % colorScale.length];
	}

	let numericColumns = tab_results_headers_json.filter(col => {
		if (special_col_names.includes(col) || result_names.includes(col)) return false;
		if (col.toLowerCase().startsWith("oo_info_")) return false;
		let idx = tab_results_headers_json.indexOf(col);
		return tab_results_csv_json.every(row => {
			let v = parseFloat(row[idx]);
			return !isNaN(v) && isFinite(v);
		});
	});

	if (numericColumns.length < 1) {
		console.error("Not enough numeric parameter columns for KDE histogram");
		return;
	}

	let plotDiv = document.getElementById("plotKdeHistogram");
	plotDiv.innerHTML = "";

	numericColumns.forEach(col => {
		let colIdx = tab_results_headers_json.indexOf(col);
		let colData = tab_results_csv_json.map(row => parseFloat(row[colIdx]));
		let validPairs = tab_results_csv_json
			.map(row => ({ param: parseFloat(row[colIdx]), result: parseFloat(row[resultIndex]) }))
			.filter(p => !isNaN(p.param) && !isNaN(p.result));

		if (validPairs.length < 2) return;

		let pMin = Math.min(...validPairs.map(p => p.param));
		let pMax = Math.max(...validPairs.map(p => p.param));
		let pBinCount = Math.min(20, Math.max(5, Math.ceil(Math.sqrt(validPairs.length))));
		let pBinWidth = (pMax - pMin) / pBinCount;
		if (pBinWidth === 0) return;

		let subDiv = document.createElement("div");
		plotDiv.appendChild(subDiv);

		let traces = [];
		for (let b = 0; b < numBins; b++) {
			let binLabel = (rMin + b * binWidth).toFixed(2) + " - " + (rMin + (b + 1) * binWidth).toFixed(2);
			let binValues = validPairs
				.filter(p => getResultBin(p.result) === b)
				.map(p => p.param);

			if (binValues.length === 0) continue;

			traces.push({
				x: binValues,
				type: 'histogram',
				name: binLabel,
				opacity: 0.75,
				marker: { color: binColor(b) },
				xbins: { start: pMin, end: pMax, size: pBinWidth },
				nbinsx: pBinCount
			});
		}

		if (traces.length === 0) return;

		let layout = {
			title: col + " (colored by " + result_names[0] + " range)",
			xaxis: { title: get_axis_title_data(col) },
			yaxis: { title: get_axis_title_data("Count") },
			barmode: 'stack',
			showlegend: true,
			legend: { font: get_font_data() }
		};

		Plotly.newPlot(subDiv, traces, add_default_layout_data(layout));
	});

	$("#plotKdeHistogram").data("loaded", "true");
}
