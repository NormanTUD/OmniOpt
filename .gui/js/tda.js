"use strict";

/*
 * Topological data analysis for shared OmniOpt runs.
 *
 * The point cloud is built from the numeric columns of results.csv.  Special
 * columns and all OO_Info_* columns are excluded automatically.  We build the
 * Vietoris-Rips filtration from Euclidean distances and compute persistent
 * homology in dimensions 0 and 1: H0 bars show connected components, H1 bars
 * show loops/cycles.  The Betti curves are the ranks of these homology groups
 * as the filtration radius grows.
 */

var _TDA_DEFAULT_POINTS = 60;
var _TDA_MAX_POINTS = 80;

function _tdaIsSpecialColumn(name) {
	if (typeof special_col_names !== "undefined" && Array.isArray(special_col_names) && special_col_names.includes(name)) {
		return true;
	}
	return typeof name === "string" && name.startsWith("OO_Info");
}

function _tdaAsNumber(value) {
	if (value === null || value === undefined || value === "") {
		return NaN;
	}
	var number = Number(value);
	return isFinite(number) ? number : NaN;
}

function _tdaColumnInfo() {
	var headers = tab_results_headers_json;
	var rows = tab_results_csv_json;
	var info = [];
	var inspectedRows = Math.min(rows.length, 5000);

	for (var column = 0; column < headers.length; column++) {
		var name = headers[column];
		if (_tdaIsSpecialColumn(name)) {
			continue;
		}

		var finiteValues = 0;
		var distinctValues = new Set();
		for (var row = 0; row < inspectedRows; row++) {
			var value = _tdaAsNumber(rows[row][column]);
			if (isFinite(value)) {
				finiteValues++;
				if (distinctValues.size < 100) {
					distinctValues.add(value);
				}
			}
		}

		if (finiteValues >= Math.max(4, Math.ceil(inspectedRows * 0.8)) && distinctValues.size >= 2) {
			info.push({ name: name, column: column });
		}
	}
	return info;
}

function _tdaSampledPoints(columns, maxPoints) {
	var headers = tab_results_headers_json;
	var rows = tab_results_csv_json;
	var points = [];

	for (var row = 0; row < rows.length; row++) {
		var point = new Array(columns.length);
		var usable = true;
		for (var dimension = 0; dimension < columns.length; dimension++) {
			var value = _tdaAsNumber(rows[row][columns[dimension].column]);
			if (!isFinite(value)) {
				usable = false;
				break;
			}
			point[dimension] = value;
		}
		if (usable) {
			points.push(point);
		}
	}

	if (points.length <= maxPoints) {
		return points;
	}

	var sampled = [];
	var step = points.length / maxPoints;
	for (var index = 0; index < maxPoints; index++) {
		sampled.push(points[Math.floor(index * step)]);
	}
	return sampled;
}

function _tdaNormalize(points, mode) {
	if (!points.length || mode === "none") {
		return points;
	}

	var dimensions = points[0].length;
	var means = new Array(dimensions).fill(0);
	var mins = new Array(dimensions).fill(Infinity);
	var maxs = new Array(dimensions).fill(-Infinity);

	for (var p = 0; p < points.length; p++) {
		for (var d = 0; d < dimensions; d++) {
			means[d] += points[p][d];
			if (points[p][d] < mins[d]) mins[d] = points[p][d];
			if (points[p][d] > maxs[d]) maxs[d] = points[p][d];
		}
	}
	for (var d = 0; d < dimensions; d++) {
		means[d] /= points.length;
	}

	var stds = new Array(dimensions).fill(0);
	if (mode === "zscore") {
		for (var p = 0; p < points.length; p++) {
			for (var d = 0; d < dimensions; d++) {
				var diff = points[p][d] - means[d];
				stds[d] += diff * diff;
			}
		}
		for (var d = 0; d < dimensions; d++) {
			stds[d] = Math.sqrt(stds[d] / points.length) || 1;
		}
	}

	return points.map(function (point) {
		return point.map(function (value, dimension) {
			if (mode === "zscore") {
				return (value - means[dimension]) / stds[dimension];
			}
			var range = maxs[dimension] - mins[dimension];
			return range > 0 ? (value - mins[dimension]) / range : 0;
		});
	});
}

function _tdaSymmetricDifference(first, second) {
	var merged = [];
	var i = 0;
	var j = 0;
	while (i < first.length && j < second.length) {
		if (first[i] < second[j]) {
			merged.push(first[i++]);
		} else if (first[i] > second[j]) {
			merged.push(second[j++]);
		} else {
			i++;
			j++;
		}
	}
	while (i < first.length) merged.push(first[i++]);
	while (j < second.length) merged.push(second[j++]);
	return merged;
}

function _tdaPersistence(points) {
	var n = points.length;
	var distances = new Float64Array(n * n);
	var edges = [];

	for (var i = 0; i < n; i++) {
		for (var j = i + 1; j < n; j++) {
			var squared = 0;
			for (var d = 0; d < points[i].length; d++) {
				var difference = points[i][d] - points[j][d];
				squared += difference * difference;
			}
			var distance = Math.sqrt(squared);
			distances[i * n + j] = distance;
			distances[j * n + i] = distance;
			edges.push({ i: i, j: j, distance: distance });
		}
	}

	edges.sort(function (a, b) {
		return a.distance - b.distance || a.i - b.i || a.j - b.j;
	});

	var parent = new Int32Array(n);
	var rank = new Uint8Array(n);
	for (var v = 0; v < n; v++) parent[v] = v;

	function find(value) {
		while (parent[value] !== value) {
			parent[value] = parent[parent[value]];
			value = parent[value];
		}
		return value;
	}

	function union(a, b) {
		a = find(a);
		b = find(b);
		if (a === b) return false;
		if (rank[a] < rank[b]) {
			parent[a] = b;
		} else if (rank[a] > rank[b]) {
			parent[b] = a;
		} else {
			parent[b] = a;
			rank[a]++;
		}
		return true;
	}

	var h0 = [];
	var edgeBirth = new Float64Array(edges.length);
	var positiveEdge = new Uint8Array(edges.length);
	for (var e = 0; e < edges.length; e++) {
		if (union(edges[e].i, edges[e].j)) {
			h0.push({ birth: 0, death: edges[e].distance });
		} else {
			positiveEdge[e] = 1;
			edgeBirth[e] = edges[e].distance;
		}
	}
	h0.push({ birth: 0, death: Infinity });

	var edgeIndex = new Int32Array(n * n);
	edgeIndex.fill(-1);
	for (var e = 0; e < edges.length; e++) {
		edgeIndex[edges[e].i * n + edges[e].j] = e;
		edgeIndex[edges[e].j * n + edges[e].i] = e;
	}

	var triangles = [];
	for (var i = 0; i < n; i++) {
		for (var j = i + 1; j < n; j++) {
			var edgeIJ = edgeIndex[i * n + j];
			for (var k = j + 1; k < n; k++) {
				var edgeIK = edgeIndex[i * n + k];
				var edgeJK = edgeIndex[j * n + k];
				var birth = Math.max(edges[edgeIJ].distance, edges[edgeIK].distance, edges[edgeJK].distance);
				triangles.push([birth, edgeIJ, edgeIK, edgeJK]);
			}
		}
	}

	triangles.sort(function (a, b) {
		return a[0] - b[0] || a[1] - b[1] || a[2] - b[2] || a[3] - b[3];
	});

	var lowColumns = new Map();
	var h1 = [];
	var killedPositiveEdge = new Uint8Array(edges.length);

	for (var t = 0; t < triangles.length; t++) {
		var triangle = triangles[t];
		var boundary = [triangle[1], triangle[2], triangle[3]].sort(function (a, b) { return a - b; });

		while (boundary.length) {
			var pivot = boundary[boundary.length - 1];
			var previous = lowColumns.get(pivot);
			if (!previous) {
				break;
			}
			boundary = _tdaSymmetricDifference(boundary, previous);
		}

		if (boundary.length) {
			var pivotEdge = boundary[boundary.length - 1];
			if (positiveEdge[pivotEdge]) {
				h1.push({ birth: edgeBirth[pivotEdge], death: triangle[0] });
				killedPositiveEdge[pivotEdge] = 1;
			}
			lowColumns.set(pivotEdge, boundary);
		}
	}

	for (var e = 0; e < edges.length; e++) {
		if (positiveEdge[e] && !killedPositiveEdge[e]) {
			h1.push({ birth: edgeBirth[e], death: Infinity });
		}
	}

	return {
		h0: h0,
		h1: h1,
		maxDistance: edges.length ? edges[edges.length - 1].distance : 0,
		edgeCount: edges.length,
		triangleCount: triangles.length
	};
}

function _tdaBettiCurves(h0, h1, cutoff) {
	var samples = 120;
	var xs = [];
	var beta0 = [];
	var beta1 = [];

	for (var sample = 0; sample < samples; sample++) {
		var epsilon = cutoff * sample / (samples - 1);
		var components = 0;
		var cycles = 0;
		for (var i = 0; i < h0.length; i++) {
			if (h0[i].birth <= epsilon && epsilon < h0[i].death) components++;
		}
		for (var i = 0; i < h1.length; i++) {
			if (h1[i].birth <= epsilon && epsilon < h1[i].death) cycles++;
		}
		xs.push(epsilon);
		beta0.push(components);
		beta1.push(cycles);
	}
	return { xs: xs, beta0: beta0, beta1: beta1 };
}

function _tdaBarcodeTrace(bars, offset, name, color, cutoff) {
	var xs = [];
	var ys = [];
	for (var i = 0; i < bars.length; i++) {
		var death = isFinite(bars[i].death) ? Math.min(bars[i].death, cutoff) : cutoff;
		xs.push(bars[i].birth, death, null);
		ys.push(offset + i, offset + i, null);
	}
	return {
		x: xs,
		y: ys,
		mode: "lines",
		type: "scatter",
		name: name,
		line: { color: color, width: 6 },
		hoverinfo: "skip",
		showlegend: true
	};
}

function _tdaShowMessage(message) {
	var summary = document.getElementById("tda_summary");
	if (summary) {
		summary.textContent = message;
	}
}

function _tdaPlot(results, cutoff, normalizationLabel) {
	var curves = _tdaBettiCurves(results.h0, results.h1, cutoff);
	var maxBeta1 = 0;
	var maxBeta1Index = 0;
	for (var i = 0; i < curves.beta1.length; i++) {
		if (curves.beta1[i] > maxBeta1) {
			maxBeta1 = curves.beta1[i];
			maxBeta1Index = i;
		}
	}
	var maxBeta1Epsilon = curves.xs[maxBeta1Index] || 0;
	var h0Start = curves.beta0.length ? curves.beta0[0] : 0;
	var h0End = curves.beta0.length ? curves.beta0[curves.beta0.length - 1] : 0;
	var finiteH1 = results.h1.filter(function (bar) { return isFinite(bar.death); });
	var longestH1 = 0;
	for (var i = 0; i < finiteH1.length; i++) {
		longestH1 = Math.max(longestH1, finiteH1[i].death - finiteH1[i].birth);
	}

	var bettiTraces = [
		{ x: curves.xs, y: curves.beta0, mode: "lines", name: "H0 (components)", line: { color: "#2e7d32", width: 3 } },
		{ x: curves.xs, y: curves.beta1, mode: "lines", name: "H1 (loops)", line: { color: "#ef6c00", width: 3 } }
	];
	var filtrationAxis = "Filtration radius ε (" + normalizationLabel + " distance)";
	var bettiLayout = add_default_layout_data({
		title: "Betti curves of the Vietoris-Rips filtration",
		xaxis: { title: get_axis_title_data(filtrationAxis), type: "linear" },
		yaxis: { title: get_axis_title_data("Betti number (number of features)") },
		legend: { orientation: "h" },
		hovermode: "x unified"
	});
	Plotly.newPlot("tda_betti_plot", bettiTraces, bettiLayout, { responsive: true });

	var bettiExplanation = document.getElementById("tda_betti_explanation");
	if (bettiExplanation) {
		bettiExplanation.textContent =
			"β₀ counts connected components and starts at " + h0Start + "; it drops whenever clusters merge. " +
			"β₁ counts independent loops. β₀ ends at " + h0End + " and β₁ peaks at " + maxBeta1 +
			" around ε ≈ " + maxBeta1Epsilon.toFixed(3) + ". " +
			"A high β₁ over a long interval indicates ring-like structure, while a peak that disappears immediately is usually sampling noise.";
	}

	var h0 = results.h0.slice().sort(function (a, b) {
		var da = isFinite(a.death) ? a.death : Infinity;
		var db = isFinite(b.death) ? b.death : Infinity;
		return da - db || a.birth - b.birth;
	});
	var h1 = results.h1.slice().sort(function (a, b) {
		return a.birth - b.birth || a.death - b.death;
	});
	var barcodeTraces = [
		_tdaBarcodeTrace(h0, 0, "H0 components", "#2e7d32", cutoff),
		_tdaBarcodeTrace(h1, h0.length + 2, "H1 loops", "#ef6c00", cutoff)
	];
	var barcodeLayout = add_default_layout_data({
		title: "Persistence barcode (infinite bars are clipped at the dashed cutoff)",
		xaxis: { title: get_axis_title_data(filtrationAxis), type: "linear" },
		yaxis: { title: get_axis_title_data("Homology generators (H0 components, then H1 loops)"), showticklabels: false },
		legend: { orientation: "h" },
		shapes: [{
			type: "line",
			x0: cutoff,
			x1: cutoff,
			y0: 0,
			y1: 1,
			yref: "paper",
			line: { color: "#888", width: 1, dash: "dash" }
		}]
	});
	Plotly.newPlot("tda_barcode_plot", barcodeTraces, barcodeLayout, { responsive: true });

	var barcodeExplanation = document.getElementById("tda_barcode_explanation");
	if (barcodeExplanation) {
		barcodeExplanation.textContent =
			"Each horizontal segment is one topological feature. The left end is its birth radius and the right end is the radius where it disappears. " +
			"Green segments are H0 connected components; orange segments are H1 loops. " +
			"There are " + finiteH1.length + " finite H1 bar(s); the longest H1 bar has persistence " +
			longestH1.toFixed(3) + ". Long bars are strong features, short bars are usually noise. " +
			"The infinite H0 component is clipped at the dashed cutoff line.";
	}
}

function _tdaRun() {
	var root = document.getElementById("tda_root");
	if (!root) return;

	var selection = Array.from(root.querySelectorAll("input[data-tda-column]:checked")).map(function (checkbox) {
		return Number(checkbox.dataset.tdaColumn);
	});
	var columnInfo = root._tdaColumnInfo || [];
	var selectedColumns = columnInfo.filter(function (entry) { return selection.includes(entry.column); });

	if (selectedColumns.length === 0) {
		_tdaShowMessage("Please select at least one numeric data column.");
		return;
	}

	var maxPoints = parseInt(root.querySelector("#tda_max_points").value, 10);
	if (!isFinite(maxPoints)) maxPoints = _TDA_DEFAULT_POINTS;
	maxPoints = Math.max(8, Math.min(_TDA_MAX_POINTS, maxPoints));

	if (root._tdaRunning) {
		return;
	}
	root._tdaRunning = true;

	var normalization = root.querySelector("#tda_normalization").value;
	_tdaShowMessage("Computing Vietoris-Rips filtration ...");

	setTimeout(function () {
		try {
		var points = _tdaSampledPoints(selectedColumns, maxPoints);
		if (points.length < 4) {
			_tdaShowMessage("Not enough complete numeric rows for topological analysis (need at least 4).");
			return;
		}

		var normalized = _tdaNormalize(points, normalization);
		var results = _tdaPersistence(normalized);
		var cutoff = results.maxDistance > 0 ? results.maxDistance * 1.05 : 1;

		var normalizationLabel = normalization === "none" ? "unnormalized" : (normalization === "minmax" ? "min-max normalized" : "z-score normalized");
		_tdaPlot(results, cutoff, normalizationLabel);

		var h0Finite = results.h0.filter(function (bar) { return isFinite(bar.death); }).length;
		var h1Finite = results.h1.filter(function (bar) { return isFinite(bar.death); }).length;
		_tdaShowMessage(
			"Using " + normalized.length + " point(s), " +
			selectedColumns.length + " dimension(s). " +
			"H0: " + h0Finite + " finite bars + 1 infinite component. " +
			"H1: " + h1Finite + " finite loop bar(s). " +
			"Rips complex: " + results.edgeCount + " edges, " + results.triangleCount + " triangles. " +
			"Special and OO_Info columns are automatically excluded."
		);
		} finally {
			root._tdaRunning = false;
		}
	}, 20);
}

function initializeTDA() {
	var root = document.getElementById("tda_root");
	if (!root || root.dataset.tdaInitialized === "true") {
		return;
	}

	if (
		typeof tab_results_headers_json === "undefined" ||
		typeof tab_results_csv_json === "undefined" ||
		!Array.isArray(tab_results_headers_json) ||
		!Array.isArray(tab_results_csv_json)
	) {
		root.textContent = "No results.csv data is available for topological analysis.";
		return;
	}

	var columnInfo = _tdaColumnInfo();
	root._tdaColumnInfo = columnInfo;
	root.dataset.tdaInitialized = "true";
	root.innerHTML = "";

	var intro = document.createElement("p");
	intro.textContent = "Persistent homology of the numeric results as a Vietoris-Rips point cloud. H0 bars are connected components, H1 bars are loops. Special and OO_Info columns are automatically disabled.";
	root.appendChild(intro);

	var controls = document.createElement("div");
	controls.style.display = "flex";
	controls.style.flexWrap = "wrap";
	controls.style.gap = "16px";
	controls.style.alignItems = "flex-start";
	controls.style.marginBottom = "12px";

	var summary = document.createElement("div");
	summary.id = "tda_summary";
	summary.style.marginBottom = "12px";
	root.appendChild(summary);

	var columnsDetails = document.createElement("details");
	columnsDetails.open = true;
	columnsDetails.style.minWidth = "260px";
	var columnsSummary = document.createElement("summary");
	columnsSummary.textContent = "Data dimensions";
	columnsDetails.appendChild(columnsSummary);
	var columnsList = document.createElement("div");
	columnsList.style.maxHeight = "220px";
	columnsList.style.overflowY = "auto";
	columnsList.style.border = "1px solid #bbb";
	columnsList.style.borderRadius = "6px";
	columnsList.style.padding = "8px";
	columnsDetails.appendChild(columnsList);

	if (columnInfo.length === 0) {
		_tdaShowMessage("No numeric, non-special result columns are available.");
		return;
	}

	columnInfo.forEach(function (entry) {
		var label = document.createElement("label");
		label.style.display = "block";
		label.style.whiteSpace = "nowrap";
		var checkbox = document.createElement("input");
		checkbox.type = "checkbox";
		checkbox.checked = true;
		checkbox.dataset.tdaColumn = entry.column;
		label.appendChild(checkbox);
		label.appendChild(document.createTextNode(" " + entry.name));
		columnsList.appendChild(label);
	});

	var settings = document.createElement("div");
	settings.style.display = "flex";
	settings.style.flexDirection = "column";
	settings.style.gap = "8px";

	var normalizationLabel = document.createElement("label");
	normalizationLabel.textContent = "Normalization: ";
	var normalization = document.createElement("select");
	normalization.id = "tda_normalization";
	[["zscore", "z-score (recommended)"], ["minmax", "min-max"], ["none", "none"]].forEach(function (optionData) {
		var option = document.createElement("option");
		option.value = optionData[0];
		option.textContent = optionData[1];
		normalization.appendChild(option);
	});
	normalizationLabel.appendChild(normalization);
	settings.appendChild(normalizationLabel);

	var pointsLabel = document.createElement("label");
	pointsLabel.textContent = "Max points: ";
	var pointsInput = document.createElement("input");
	pointsInput.type = "number";
	pointsInput.id = "tda_max_points";
	pointsInput.min = "8";
	pointsInput.max = String(_TDA_MAX_POINTS);
	pointsInput.value = String(_TDA_DEFAULT_POINTS);
	pointsInput.style.width = "80px";
	pointsLabel.appendChild(pointsInput);
	settings.appendChild(pointsLabel);

	var runButton = document.createElement("button");
	runButton.type = "button";
	runButton.textContent = "Compute persistent homology";
	runButton.addEventListener("click", _tdaRun);
	settings.appendChild(runButton);

	var toggleButtons = document.createElement("div");
	toggleButtons.style.display = "flex";
	toggleButtons.style.gap = "6px";
	var selectAll = document.createElement("button");
	selectAll.type = "button";
	selectAll.textContent = "Select all";
	selectAll.addEventListener("click", function () {
		columnsList.querySelectorAll("input[type=checkbox]").forEach(function (checkbox) { checkbox.checked = true; });
	});
	var selectNone = document.createElement("button");
	selectNone.type = "button";
	selectNone.textContent = "Select none";
	selectNone.addEventListener("click", function () {
		columnsList.querySelectorAll("input[type=checkbox]").forEach(function (checkbox) { checkbox.checked = false; });
	});
	toggleButtons.appendChild(selectAll);
	toggleButtons.appendChild(selectNone);
	settings.appendChild(toggleButtons);

	controls.appendChild(columnsDetails);
	controls.appendChild(settings);
	root.appendChild(controls);

	summary.textContent = columnInfo.length + " usable numeric dimension(s) found.";

	var bettiHeading = document.createElement("h3");
	bettiHeading.textContent = "Betti curves (H0 / H1)";
	root.appendChild(bettiHeading);
	var bettiPlot = document.createElement("div");
	bettiPlot.id = "tda_betti_plot";
	bettiPlot.className = "invert_in_dark_mode";
	root.appendChild(bettiPlot);
	var bettiExplanation = document.createElement("p");
	bettiExplanation.id = "tda_betti_explanation";
	bettiExplanation.className = "tda_explanation";
	root.appendChild(bettiExplanation);

	var barcodeHeading = document.createElement("h3");
	barcodeHeading.textContent = "Persistence barcode";
	root.appendChild(barcodeHeading);
	var barcodePlot = document.createElement("div");
	barcodePlot.id = "tda_barcode_plot";
	barcodePlot.className = "invert_in_dark_mode";
	root.appendChild(barcodePlot);
	var barcodeExplanation = document.createElement("p");
	barcodeExplanation.id = "tda_barcode_explanation";
	barcodeExplanation.className = "tda_explanation";
	root.appendChild(barcodeExplanation);

	_tdaRun();
}
