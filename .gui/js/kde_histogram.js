async function plotKdeHistogram() {
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

	const headers = tab_results_headers_json;
	const rows = tab_results_csv_json;
	const headerIndex = new Map();
	const specialSet = new Set(special_col_names);
	const resultSet = new Set(result_names);

	for (let i = 0; i < headers.length; i++) {
		headerIndex.set(headers[i], i);
	}

	let numericColumns = [];

	for (let c = 0; c < headers.length; c++) {
		const col = headers[c];

		if (specialSet.has(col) || resultSet.has(col) || col.toLowerCase().startsWith("oo_info_")) {
			continue;
		}

		const idx = headerIndex.get(col);
		let isNumeric = true;

		for (let r = 0; r < rows.length; r++) {
			const v = parseFloat(rows[r][idx]);

			if (isNaN(v) || !isFinite(v)) {
				isNumeric = false;
				break;
			}
		}

		if (isNumeric) {
			numericColumns.push(col);
		}
	}

	if (numericColumns.length < 1) {
		console.error("Not enough numeric parameter columns for KDE histogram");
		return;
	}

	let plotDiv = document.getElementById("plotKdeHistogram");
	let subDivIndex = 0;
	window.__kdeRenderToken = (window.__kdeRenderToken || 0) + 1;
	const renderToken = window.__kdeRenderToken;

	for (const col of numericColumns) {
		if (renderToken !== window.__kdeRenderToken) return;

		let colIdx = headers.indexOf(col);
		let validPairs = [];

		for (let r = 0; r < rows.length; r++) {
			const param = parseFloat(rows[r][colIdx]);
			const result = parseFloat(rows[r][resultIndex]);

			if (!isNaN(param) && !isNaN(result)) {
				validPairs.push({ param: param, result: result });
			}
		}

		if (validPairs.length < 2) continue;

		const maxKdePoints = 3000;

		if (validPairs.length > maxKdePoints) {
			const sampledPairs = new Array(maxKdePoints);
			const stride = validPairs.length / maxKdePoints;

			for (let i = 0; i < maxKdePoints; i++) {
				sampledPairs[i] = validPairs[Math.floor(i * stride)];
			}

			validPairs = sampledPairs;
		}

		let pMin = Infinity;
		let pMax = -Infinity;
		let bins = new Array(numBins);

		for (let b = 0; b < numBins; b++) {
			bins[b] = [];
		}

		for (let i = 0; i < validPairs.length; i++) {
			const pair = validPairs[i];
			if (pair.param < pMin) pMin = pair.param;
			if (pair.param > pMax) pMax = pair.param;
			bins[getResultBin(pair.result)].push(pair.param);
		}

		let pBinCount = Math.min(20, Math.max(5, Math.ceil(Math.sqrt(validPairs.length))));
		let pBinWidth = (pMax - pMin) / pBinCount;
		if (pBinWidth === 0) continue;

		let subDiv = plotDiv.children[subDivIndex];

		if (!subDiv) {
			subDiv = document.createElement("div");
			plotDiv.appendChild(subDiv);
		}

		subDivIndex++;

		let traces = [];
		for (let b = 0; b < numBins; b++) {
			let binValues = bins[b];

			if (binValues.length === 0) continue;

			let binLabel = (rMin + b * binWidth).toFixed(2) + " - " + (rMin + (b + 1) * binWidth).toFixed(2);

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

		if (traces.length === 0) continue;

		let layout = {
			title: col + " (colored by " + result_names[0] + " range)",
			xaxis: { title: get_axis_title_data(col) },
			yaxis: { title: get_axis_title_data("Count") },
			barmode: 'stack',
			showlegend: true,
			legend: { font: get_font_data() }
		};

		if (typeof Plotly.react === "function" && subDiv.classList.contains("js-plotly-plot")) {
			await Plotly.react(subDiv, traces, add_default_layout_data(layout));
		} else {
			await Plotly.newPlot(subDiv, traces, add_default_layout_data(layout));
		}

		await new Promise(resolve => setTimeout(resolve, 0));
	}

	if (renderToken !== window.__kdeRenderToken) return;

	while (plotDiv.children.length > subDivIndex) {
		plotDiv.removeChild(plotDiv.lastChild);
	}

	$("#plotKdeHistogram").data("loaded", "true");
}
