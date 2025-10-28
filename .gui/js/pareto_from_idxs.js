"use strict";

function get_row_by_index(idx) {
	if (!Object.keys(window).includes("tab_results_csv_json")) {
		error("tab_results_csv_json is not defined");
		return;
	}

	if (!Object.keys(window).includes("tab_results_headers_json")) {
		error("tab_results_headers_json is not defined");
		return;
	}


	var trial_index_col_idx = tab_results_headers_json.indexOf("trial_index");

	if(trial_index_col_idx == -1) {
		error(`"trial_index" could not be found in tab_results_headers_json. Cannot continue`);

		return null;
	}

	for (var i = 0; i < tab_results_csv_json.length; i++) {
		var row = tab_results_csv_json[i];
		var trial_index = row[trial_index_col_idx];

		if (trial_index == idx) {
			return row;
		}
	}

	return null;
}

function load_pareto_graph_from_idxs () {
	// Schnittstelle bleibt gleich: keine Argumente, alte Fehlertexte beibehalten.
	// Wenn pareto_idxs nicht gesetzt ist, versuchen wir, ihn hier berechnet zu bekommen.
	if (!Object.keys(window).includes("pareto_idxs")) {
		// versuche Pareto in JS zu berechnen, falls die notwendigen Daten da sind
		try {
			if (typeof computeParetoIdxsFromTabResults === "function") {
				var computed = computeParetoIdxsFromTabResults();
				if (computed && typeof computed === "object") {
					window.pareto_idxs = computed;
				}
			}
		} catch (e) {
			// ignore, wir fallen durch zu der existierenden Fehlerbehandlung weiter unten
		}
	}

	if (!Object.keys(window).includes("pareto_idxs")) {
		error("pareto_idxs is not defined");
		return;
	}

	if (!Object.keys(window).includes("tab_results_csv_json")) {
		error("tab_results_csv_json is not defined");
		return;
	}

	if (!Object.keys(window).includes("tab_results_headers_json")) {
		error("tab_results_headers_json is not defined");
		return;
	}

	if(pareto_idxs === null) {
		var err_msg = "pareto_idxs is null. Cannot plot or create tables from empty data. This can be caused by a defective <tt>pareto_idxs.json</tt> file. Please try reloading, or re-calculating the pareto-front and re-submitting if this problem persists.";
		$("#pareto_from_idxs_table").html(`<div class="caveat alarm">${err_msg}</div>`);
		return;	
	}

	var table = get_pareto_table_data_from_idx();

	var html_tables = createParetoTablesFromData(table);

	$("#pareto_from_idxs_table").html(html_tables);

	renderParetoFrontPlots(table);

	if (typeof apply_theme_based_on_system_preferences === 'function') {
		apply_theme_based_on_system_preferences();
	}
}

function renderParetoFrontPlots(data) {
	try {
		let container = document.getElementById("pareto_front_idxs_plot_container");
		if (!container) {
			console.error("DIV with id 'pareto_front_idxs_plot_container' not found.");
			return;
		}

		container.innerHTML = "";

		if(data === undefined || data === null) {
			var err_msg = "There was an error getting the data for Pareto-Fronts. See the developer's console to see further details.";
			$("#pareto_from_idxs_table").html(`<div class="caveat alarm">${err_msg}</div>`);
			return;
		}

		Object.keys(data).forEach((key, idx) => {
			if (!key.startsWith("Pareto front for ")) return;

			let label = key.replace("Pareto front for ", "");
			let [xKey, yKey] = label.split("/");

			if (!xKey || !yKey) {
				console.warn("Could not extract two objectives from key:", key);
				return;
			}

			let entries = data[key];
			let x = [];
			let y = [];
			let hoverTexts = [];

			entries.forEach((entry) => {
				let results = entry.results || {};
				let values = entry.values || {};

				let xVal = (results[xKey] || [])[0];
				let yVal = (results[yKey] || [])[0];

				if (xVal === undefined || yVal === undefined) {
					console.warn("Missing values for", xKey, yKey, "in", entry);
					return;
				}

				x.push(xVal);
				y.push(yVal);

				let hoverInfo = [];

				if ("trial_index" in values) {
					hoverInfo.push(`<b>Trial Index:</b> ${values.trial_index[0]}`);
				}

				Object.keys(values)
					.filter(k => k !== "trial_index")
					.sort()
					.forEach(k => {
						hoverInfo.push(`<b>${k}:</b> ${values[k][0]}`);
					});

				Object.keys(results)
					.sort()
					.forEach(k => {
						hoverInfo.push(`<b>${k}:</b> ${results[k][0]}`);
					});

				hoverTexts.push(hoverInfo.join("<br>"));
			});

			let wrapper = document.createElement("div");
			wrapper.style.marginBottom = "30px";

			let titleEl = document.createElement("h3");
			titleEl.textContent = `Pareto Front: ${xKey} (${getMinMaxByResultName(xKey)}) vs ${yKey} (${getMinMaxByResultName(yKey)})`;
			wrapper.appendChild(titleEl);

			let divId = `pareto_plot_${idx}`;
			let plotDiv = document.createElement("div");
			plotDiv.id = divId;
			plotDiv.style.width = "100%";
			plotDiv.style.height = "400px";

			wrapper.appendChild(plotDiv);
			container.appendChild(wrapper);

			let trace = {
				x: x,
				y: y,
				text: hoverTexts,
				hoverinfo: "text",
				mode: "markers",
				type: "scatter",
				marker: {
					size: 8,
					color: 'rgb(31, 119, 180)',
					line: {
						width: 1,
						color: 'black'
					}
				},
				name: label
			};

			let layout = {
				xaxis: { title: { text: xKey } },
				yaxis: { title: { text: yKey } },
				margin: { t: 10, l: 60, r: 20, b: 50 },
				hovermode: "closest",
				showlegend: false
			};

			Plotly.newPlot(divId, [trace], add_default_layout_data(layout, 1));
		});
	} catch (e) {
		console.error("Error while rendering Pareto front plots:", e);
	}
}

function createParetoTablesFromData(data) {
	try {
		var container = document.createElement("div");

		var parsedData;
		try {
			parsedData = typeof data === "string" ? JSON.parse(data) : data;
		} catch (e) {
			console.error("JSON parsing failed:", e);
			return container;
		}

		for (var sectionTitle in parsedData) {
			if (!parsedData.hasOwnProperty(sectionTitle)) {
				continue;
			}

			var sectionData = parsedData[sectionTitle];
			var heading = document.createElement("h2");
			heading.textContent = sectionTitle;
			container.appendChild(heading);

			var table = document.createElement("table");
			table.style.borderCollapse = "collapse";
			table.style.marginBottom = "2em";
			table.style.width = "100%";

			var thead = document.createElement("thead");
			var headerRow = document.createElement("tr");

			var allValueKeys = new Set();
			var allResultKeys = new Set();

			sectionData.forEach(entry => {
				var values = entry.values || {};
				var results = entry.results || {};

				Object.keys(values).forEach(key => {
					allValueKeys.add(key);
				});

				Object.keys(results).forEach(key => {
					allResultKeys.add(key);
				});
			});

			var sortedValueKeys = Array.from(allValueKeys).sort();
			var sortedResultKeys = Array.from(allResultKeys).sort();

			if (sortedValueKeys.includes("trial_index")) {
				sortedValueKeys = sortedValueKeys.filter(k => k !== "trial_index");
				sortedValueKeys.unshift("trial_index");
			}

			var allColumns = [...sortedValueKeys, ...sortedResultKeys];

			allColumns.forEach(col => {
				var th = document.createElement("th");
				th.textContent = col;
				th.style.border = "1px solid black";
				th.style.padding = "4px";
				headerRow.appendChild(th);
			});

			thead.appendChild(headerRow);
			table.appendChild(thead);

			var tbody = document.createElement("tbody");

			sectionData.forEach(entry => {
				var tr = document.createElement("tr");

				allColumns.forEach(col => {
					var td = document.createElement("td");
					td.style.border = "1px solid black";
					td.style.padding = "4px";

					var value = null;

					if (col in entry.values) {
						value = entry.values[col];
					} else if (col in entry.results) {
						value = entry.results[col];
					}

					if (Array.isArray(value)) {
						td.textContent = value.join(", ");
					} else {
						td.textContent = value !== null && value !== undefined ? value : "";
					}

					tr.appendChild(td);
				});

				tbody.appendChild(tr);
			});

			table.appendChild(tbody);
			container.appendChild(table);
		}

		return container;
	} catch (err) {
		console.error("Unexpected error:", err);
		var errorDiv = document.createElement("div");
		errorDiv.textContent = "Error generating tables.";
		return errorDiv;
	}
}

function get_pareto_table_data_from_idx () {
	if (!Object.keys(window).includes("pareto_idxs")) {
		error("pareto_idxs is not defined");
		return;
	}

	if (!Object.keys(window).includes("tab_results_csv_json")) {
		error("tab_results_csv_json is not defined");
		return;
	}

	if (!Object.keys(window).includes("tab_results_headers_json")) {
		error("tab_results_headers_json is not defined");
		return;
	}

	var x_keys = Object.keys(pareto_idxs);

	var tables = {};

	for (var i = 0; i < x_keys.length; i++) {
		var x_key = x_keys[i];
		var y_keys = Object.keys(pareto_idxs[x_key]);

		for (var j = 0; j < y_keys.length; j++) {
			var y_key = y_keys[j];

			var indices = pareto_idxs[x_key][y_key];

			for (var k = 0; k < indices.length; k++) {
				var idx = indices[k];
				var row = get_row_by_index(idx);

				if(row === null) {
					error(`Error getting the row for index ${idx}`);
					return;
				}

				var row_dict = {
					"results": {},
					"values": {},
				};

				for (var l = 0; l < tab_results_headers_json.length; l++) {
					var header = tab_results_headers_json[l];

					if (!special_col_names.includes(header) || header == "trial_index") {
						var val = row[l];

						if (result_names.includes(header)) {
							if (!Object.keys(row_dict["results"]).includes(header)) {
								row_dict["results"][header] = [];
							}

							row_dict["results"][header].push(val);
						} else {
							if (!Object.keys(row_dict["values"]).includes(header)) {
								row_dict["values"][header] = [];
							}
							row_dict["values"][header].push(val);
						}
					}
					
				}

				var table_key = `Pareto front for ${x_key}/${y_key}`;

				if(!Object.keys(tables).includes(table_key)) {
					tables[table_key] = [];
				}

				tables[table_key].push(row_dict);
			}

		}
	}

	return tables;
}

function getMinMaxByResultName(resultName) {
	try {
		if (typeof resultName !== "string") {
			error("Parameter resultName must be a string");
			return;
		}

		if (!Array.isArray(result_names)) {
			error("Global variable result_names is not an array or undefined");
			return;
		}
		if (!Array.isArray(result_min_max)) {
			error("Global variable result_min_max is not an array or undefined");
			return;
		}

		if (result_names.length !== result_min_max.length) {
			error("Global arrays result_names and result_min_max must have the same length");
			return;
		}

		var index = result_names.indexOf(resultName);
		if (index === -1) {
			error("Result name '" + resultName + "' not found in result_names");
			return;
		}

		var minMaxValue = result_min_max[index];

		if (minMaxValue !== "min" && minMaxValue !== "max") {
			error("Value for result name '" + resultName + "' is invalid: expected 'min' or 'max'");
			return;
		}

		return minMaxValue;

	} catch (e) {
		error("Unexpected error: " + e.message);
	}
}

/* ----------------- Neue Pareto-Funktionen in JS (Integration, Schnittstellen unverändert) ----------------- */

/*
  computeParetoIdxsFromTabResults()
    - Liest tab_results_headers_json und tab_results_csv_json sowie result_names/result_min_max
    - Filtert nur COMPLETED trial_status
    - Berechnet Pareto-Fronten für alle Paare result_names[i] vs result_names[j] (i != j)
    - Liefert ein Objekt in der Form { "X": { "Y": [trial_index, ...], ... }, ... }
    - Gibt null zurück bei fatalen Fehlern.
*/
function computeParetoIdxsFromTabResults() {
	try {
		// prüfe nötige globale arrays
		if (typeof tab_results_headers_json === "undefined" || !Array.isArray(tab_results_headers_json)) {
			if (typeof error === "function") error("computeParetoIdxsFromTabResults: tab_results_headers_json is missing or not an array.");
			else console.error("computeParetoIdxsFromTabResults: tab_results_headers_json is missing or not an array.");
			return null;
		}
		if (typeof tab_results_csv_json === "undefined" || !Array.isArray(tab_results_csv_json)) {
			if (typeof error === "function") error("computeParetoIdxsFromTabResults: tab_results_csv_json is missing or not an array.");
			else console.error("computeParetoIdxsFromTabResults: tab_results_csv_json is missing or not an array.");
			return null;
		}
		if (typeof result_names === "undefined" || !Array.isArray(result_names) || result_names.length === 0) {
			if (typeof error === "function") error("computeParetoIdxsFromTabResults: result_names must be a non-empty array.");
			else console.error("computeParetoIdxsFromTabResults: result_names must be a non-empty array.");
			return null;
		}
		if (typeof result_min_max === "undefined" || !Array.isArray(result_min_max) || result_min_max.length !== result_names.length) {
			if (typeof error === "function") error("computeParetoIdxsFromTabResults: result_min_max must be an array of same length as result_names.");
			else console.error("computeParetoIdxsFromTabResults: result_min_max must be an array of same length as result_names.");
			return null;
		}
		// special_col_names optional, wir greifen darauf zu wenn vorhanden
		var useSpecial = (typeof special_col_names !== "undefined" && Array.isArray(special_col_names));

		// header -> index map
		var headerIndex = {};
		for (var i = 0; i < tab_results_headers_json.length; i++) {
			headerIndex[tab_results_headers_json[i]] = i;
		}

		if (!headerIndex.hasOwnProperty("trial_index")) {
			if (typeof error === "function") error("computeParetoIdxsFromTabResults: 'trial_index' header not found in tab_results_headers_json.");
			else console.error("computeParetoIdxsFromTabResults: 'trial_index' header not found in tab_results_headers_json.");
			return null;
		}
		var trialIndexHeaderIdx = headerIndex["trial_index"];

		// prepare structured rows (nur COMPLETED, nur numeric results vorhanden)
		var structuredRows = [];
		for (var r = 0; r < tab_results_csv_json.length; r++) {
			var row = tab_results_csv_json[r];
			if (!Array.isArray(row)) continue;

			// trial index
			var trialIdxVal = row[trialIndexHeaderIdx];

			// trial_status prüfen falls vorhanden
			var status = null;
			if (headerIndex.hasOwnProperty("trial_status")) {
				var rawSt = row[headerIndex["trial_status"]];
				if (typeof rawSt === "string") status = rawSt.trim().toUpperCase();
				else if (rawSt === null || typeof rawSt === "undefined") status = "";
				else status = String(rawSt).trim().toUpperCase();
			} else {
				status = "COMPLETED";
			}

			if (status !== "COMPLETED") continue;

			// parse numeric results
			var numericResults = {};
			var skip = false;
			for (var ri = 0; ri < result_names.length; ri++) {
				var rname = result_names[ri];
				var hdrIdx;
				if (headerIndex.hasOwnProperty(rname)) {
					hdrIdx = headerIndex[rname];
				} else {
					// case-insensitive fallback
					hdrIdx = -1;
					for (var hh = 0; hh < tab_results_headers_json.length; hh++) {
						if (String(tab_results_headers_json[hh]).toLowerCase() === String(rname).toLowerCase()) {
							hdrIdx = hh;
							headerIndex[rname] = hh; // cache mapping
							break;
						}
					}
					if (hdrIdx === -1) {
						console.warn("computeParetoIdxsFromTabResults: header for result '" + rname + "' not found. Skipping row.");
						skip = true;
						break;
					}
				}
				var rawVal = row[hdrIdx];
				var parsed = parseFloat(rawVal);
				if (!isFinite(parsed)) {
					skip = true;
					break;
				}
				numericResults[rname] = parsed;
			}
			if (skip) continue;

			structuredRows.push({
				trial_index: trialIdxVal,
				rowIdx: r,
				results: numericResults
			});
		}

		if (structuredRows.length === 0) {
			// keine gültigen Zeilen
			console.warn("computeParetoIdxsFromTabResults: no suitable completed rows found for pareto calculation.");
			return {};
		}

		// Hilfsfunktionen für Dominanzprüfung
		function _isFiniteNumber(v) {
			return typeof v === "number" && isFinite(v);
		}

		function _pointDominates(xi, yi, xj, yj, xMinimize, yMinimize) {
			var xBetterEq, xStrict;
			var yBetterEq, yStrict;

			if (xMinimize) {
				xBetterEq = xj <= xi;
				xStrict = xj < xi;
			} else {
				xBetterEq = xj >= xi;
				xStrict = xj > xi;
			}

			if (yMinimize) {
				yBetterEq = yj <= yi;
				yStrict = yj < yi;
			} else {
				yBetterEq = yj >= yi;
				yStrict = yj > yi;
			}

			return (xBetterEq && yBetterEq && (xStrict || yStrict));
		}

		function paretoFrontIndices(xArr, yArr, xMinimize, yMinimize) {
			if (!Array.isArray(xArr) || !Array.isArray(yArr)) {
				throw new Error("paretoFrontIndices: input must be arrays");
			}
			if (xArr.length !== yArr.length) {
				throw new Error("paretoFrontIndices: x and y must have same length");
			}
			var n = xArr.length;
			var isDom = new Array(n);
			for (var i = 0; i < n; i++) isDom[i] = false;

			for (var i = 0; i < n; i++) {
				if (isDom[i]) continue;
				var xi = xArr[i], yi = yArr[i];
				if (!_isFiniteNumber(xi) || !_isFiniteNumber(yi)) {
					isDom[i] = true;
					continue;
				}
				for (var j = 0; j < n; j++) {
					if (i === j) continue;
					var xj = xArr[j], yj = yArr[j];
					if (!_isFiniteNumber(xj) || !_isFiniteNumber(yj)) continue;
					if (_pointDominates(xi, yi, xj, yj, xMinimize, yMinimize)) {
						isDom[i] = true;
						break;
					}
				}
			}
			var res = [];
			for (var k = 0; k < n; k++) if (!isDom[k]) res.push(k);
			return res;
		}

		// Berechne für jedes Paar
		var paretoIdxs = {};
		for (var xi = 0; xi < result_names.length; xi++) {
			var xName = result_names[xi];
			paretoIdxs[xName] = paretoIdxs[xName] || {};
			for (var yi = 0; yi < result_names.length; yi++) {
				if (yi === xi) continue;
				var yName = result_names[yi];

				var xVals = [];
				var yVals = [];
				var trialIndices = [];

				for (var s = 0; s < structuredRows.length; s++) {
					var rowObj = structuredRows[s];
					if (!rowObj.results.hasOwnProperty(xName) || !rowObj.results.hasOwnProperty(yName)) continue;
					xVals.push(rowObj.results[xName]);
					yVals.push(rowObj.results[yName]);
					trialIndices.push(rowObj.trial_index);
				}

				if (xVals.length === 0) {
					paretoIdxs[xName][yName] = [];
					continue;
				}

				var xMin = (String(result_min_max[xi]).toLowerCase() === "min");
				var yMin = (String(result_min_max[yi]).toLowerCase() === "min");

				var nd;
				try {
					nd = paretoFrontIndices(xVals, yVals, xMin, yMin);
				} catch (err) {
					console.error("paretoFrontIndices failed for", xName, yName, err);
					paretoIdxs[xName][yName] = [];
					continue;
				}

				var ndTrial = nd.map(function(localIdx) {
					return trialIndices[localIdx];
				});

				// sortiere deterministisch nach x aufsteigend (like Python)
				try {
					var combined = ndTrial.map(function(ti) {
						var local = trialIndices.indexOf(ti);
						return { ti: ti, x: xVals[local] };
					});
					combined.sort(function(a, b) {
						if (a.x < b.x) return -1;
						if (a.x > b.x) return 1;
						return 0;
					});
					ndTrial = combined.map(function(c) { return c.ti; });
				} catch (e) {
					// ignore
				}

				paretoIdxs[xName][yName] = ndTrial;
			}
		}

		return paretoIdxs;
	} catch (ex) {
		console.error("computeParetoIdxsFromTabResults: unexpected error:", ex);
		return null;
	}
}

/*
  computeAndRenderParetoFromResults()
    - Komfortfunktion: berechnet pareto_idxs, setzt window.pareto_idxs und ruft load_pareto_graph_from_idxs()
    - Schnittstelle bleibt: keine Argumente, keine Änderung in externem Aufrufverhalten.
*/
function computeAndRenderParetoFromResults() {
	try {
		var computed = computeParetoIdxsFromTabResults();
		if (computed === null) {
			if (typeof error === "function") error("computeAndRenderParetoFromResults: Pareto computation failed.");
			else console.error("computeAndRenderParetoFromResults: Pareto computation failed.");
			return;
		}
		try {
			window.pareto_idxs = computed;
		} catch (e) {
			this.pareto_idxs = computed;
		}
		if (typeof load_pareto_graph_from_idxs === "function") {
			load_pareto_graph_from_idxs();
		} else {
			console.warn("computeAndRenderParetoFromResults: load_pareto_graph_from_idxs is not defined; pareto_idxs is set though.");
		}
	} catch (err) {
		console.error("computeAndRenderParetoFromResults unexpected error:", err);
	}
}
