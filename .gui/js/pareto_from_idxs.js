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

	apply_theme_based_on_system_preferences();
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
