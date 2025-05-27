async function load_pareto_graph() {
	if ($("#tab_pareto_fronts").data("loaded") === "true") return;

	const data = pareto_front_data;
	if (!data || typeof data !== "object" || Object.keys(data).length === 0) {
		console.error("Invalid or empty pareto_front_data");
		return;
	}

	const categories = Object.keys(data);
	const allMetrics = getAllMetrics(data, categories);

	const graphContainer = document.getElementById("pareto_front_graphs_container");
	graphContainer.classList.add("invert_in_dark_mode");
	graphContainer.innerHTML = "";

	const alreadyPlotted = new Set();

	for (let i = 0; i < allMetrics.length; i++) {
		for (let j = i + 1; j < allMetrics.length; j++) {
			const xMetric = allMetrics[i];
			const yMetric = allMetrics[j];

			const { xValues, yValues } = getValuesForMetrics(data, categories, xMetric, yMetric);

			if (xValues.length > 0 && yValues.length > 0 && xValues.length === yValues.length) {
				const cleanX = cleanMetricName(xMetric);
				const cleanY = cleanMetricName(yMetric);
				const plotKeyOne = `${cleanX}-${cleanY}`;
				const plotKeyTwo = `${cleanY}-${cleanX}`;

				if (alreadyPlotted.has(plotKeyOne) || alreadyPlotted.has(plotKeyTwo)) continue;

				createPlot(graphContainer, i, j, cleanX, cleanY, xValues, yValues);

				alreadyPlotted.add(plotKeyOne);
				alreadyPlotted.add(plotKeyTwo);
			}
		}
	}

	if (typeof apply_theme_based_on_system_preferences === "function") {
		apply_theme_based_on_system_preferences();
	}

	$("#tab_pareto_fronts").data("loaded", "true");
}

// Extrahiert alle Metriken rekursiv
function getAllMetrics(data, categories) {
	const allMetrics = new Set();

	function extractMetrics(obj, prefix = "") {
		Object.keys(obj).forEach(key => {
			const newPrefix = prefix ? `${prefix} -> ${key}` : key;
			if (typeof obj[key] === "object" && !Array.isArray(obj[key])) {
				extractMetrics(obj[key], newPrefix);
			} else {
				if (
					!newPrefix.includes("param_dicts") &&
					!newPrefix.includes(" -> sems -> ") &&
					!newPrefix.includes("absolute_metrics")
				) {
					allMetrics.add(newPrefix);
				}
			}
		});
	}

	categories.forEach(cat => extractMetrics(data[cat]));
	return Array.from(allMetrics);
}

// Extrahiert Werte aus CSV nach Indices und Spaltenname
function extractValuesFromCSV(indices, columnName) {
	// Finde die Spaltenindex für den gesuchten Spaltennamen
	const trial_index_idx = tab_results_headers_json.indexOf("trial_index");

	let rowIndices = [];

	for (let row_nr = 0; row_nr < tab_results_csv_json.length; row_nr++) {
		let row = tab_results_csv_json[row_nr];

		if (indices.includes(row[trial_index_idx])) {
			rowIndices.push(row_nr);
		}
	}

	const colIdx = tab_results_headers_json.indexOf(columnName);
	if (colIdx === -1) {
		// Wenn Spalte nicht gefunden wurde, gebe leeres Array zurück
		return [];
	}

	const values = [];
	// Durchlaufe alle Indices, die ausgewertet werden sollen
	for (let i = 0; i < rowIndices.length; i++) {
		const rowIdx = rowIndices[i];
		let res = tab_results_csv_json[rowIdx][colIdx];
		log(`columnName: ${columnName}, colIdx: ${colIdx}, rowIdx: ${rowIdx}, res = ${res}`);
		values.push(res);
	}

	return values;
}


// Extrahiert Werte aus Objekt anhand von Metric-Pfad
function extractValues(obj, metricPath) {
	const parts = metricPath.split(" -> ");
	let data = obj;

	for (const part of parts) {
		if (data && typeof data === "object") {
			data = data[part];
		} else {
			return [];
		}
	}
	return Array.isArray(data) ? data : [];
}

// Bereitet die Werte-Arrays für x- und y-Metriken auf
function getValuesForMetrics(data, categories, xMetric, yMetric) {
	let xValues = [];
	let yValues = [];

	const cleanX = cleanMetricName(xMetric);
	const cleanY = cleanMetricName(yMetric);

	if (cleanX === cleanY) return { xValues, yValues };

	for (const cat of categories) {
		const subCats = Object.keys(data[cat]);
		for (const sub of subCats) {
			const block = data[cat][sub];
			if (cleanX === "idxs" || cleanY === "idxs") continue;

			if (Array.isArray(block["idxs"])) {
				const idxs = block["idxs"];
				xValues.push(...extractValuesFromCSV(idxs, cleanX));
				yValues.push(...extractValuesFromCSV(idxs, cleanY));
			} else {
				xValues.push(...extractValues(block, xMetric));
				yValues.push(...extractValues(block, yMetric));
			}
		}
	}

	return { xValues, yValues };
}

// Erzeugt den Plot im Container
function createPlot(container, i, j, cleanX, cleanY, xValues, yValues) {
	const div = document.createElement("div");
	div.id = `pareto_front_graph_${i}_${j}`;
	div.style.marginBottom = "20px";
	container.appendChild(div);

	const layout = {
		title: `${cleanX} vs ${cleanY}`,
		xaxis: { title: get_axis_title_data(cleanX) },
		yaxis: { title: get_axis_title_data(cleanY) },
		hovermode: "closest",
	};

	const trace = {
		x: xValues,
		y: yValues,
		mode: "markers",
		marker: { size: get_marker_size() },
		type: "scatter",
		name: `${cleanX} vs ${cleanY}`,
	};

	Plotly.newPlot(div.id, [trace], add_default_layout_data(layout));
}

// Hilfsfunktion: Säubert den Metric-Namen
function cleanMetricName(metric) {
	return metric.replace(/.* -> /g, "");
}
