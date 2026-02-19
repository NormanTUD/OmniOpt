"use strict";

function plotArmDistribution() {
	if ($("#plotArmDistribution").data("loaded") === "true") return;

	if (
		typeof arm_evals_headers_json === "undefined" ||
		typeof arm_evals_csv_json === "undefined" ||
		!Array.isArray(arm_evals_headers_json) ||
		!Array.isArray(arm_evals_csv_json)
	) {
		console.error("plotArmDistribution: arm_evals data not available.");
		$("#plotArmDistribution").html("<p>No arm evaluation data available.</p>");
		return;
	}

	var headers = arm_evals_headers_json;
	var data = arm_evals_csv_json;

	if (headers.length === 0 || data.length === 0) {
		$("#plotArmDistribution").html("<p>Arm evaluation data is empty.</p>");
		return;
	}

	var trialIndexCol = headers.indexOf("trial_index");
	var subArmCol = headers.indexOf("sub_arm_nr");

	if (trialIndexCol === -1) {
		console.error("plotArmDistribution: 'trial_index' column not found.");
		$("#plotArmDistribution").html("<p>Error: 'trial_index' column missing in arm evaluation data.</p>");
		return;
	}

	// --- identify numeric result columns ---
	var resultCols = [];
	for (var i = 0; i < headers.length; i++) {
		if (headers[i] === "trial_index" || headers[i] === "sub_arm_nr") continue;
		var allNumeric = true;
		for (var r = 0; r < data.length; r++) {
			var v = data[r][i];
			if (v === "" || v === null || v === undefined) continue;
			if (isNaN(parseFloat(v))) { allNumeric = false; break; }
		}
		if (allNumeric) {
			resultCols.push({ name: headers[i], index: i });
		}
	}

	if (resultCols.length === 0) {
		$("#plotArmDistribution").html("<p>No numeric result columns found in arm evaluation data.</p>");
		return;
	}

	// --- group rows by trial_index ---
	var grouped = {};
	for (var r = 0; r < data.length; r++) {
		var ti = parseInt(data[r][trialIndexCol], 10);
		if (isNaN(ti)) continue;
		if (!(ti in grouped)) grouped[ti] = [];
		grouped[ti].push(data[r]);
	}

	var trialIndices = Object.keys(grouped).map(Number).sort(function (a, b) { return a - b; });

	if (trialIndices.length === 0) {
		$("#plotArmDistribution").html("<p>No valid trial indices found.</p>");
		return;
	}

	var plotDiv = document.getElementById("plotArmDistribution");
	plotDiv.innerHTML = "";

	// --- helper: compute descriptive stats ---
	function computeStats(values) {
		if (!values || values.length === 0) return null;
		var n = values.length;
		var sum = 0;
		for (var i = 0; i < n; i++) sum += values[i];
		var mean = sum / n;
		var ssq = 0;
		for (var i = 0; i < n; i++) ssq += (values[i] - mean) * (values[i] - mean);
		var std = n > 1 ? Math.sqrt(ssq / (n - 1)) : 0;
		var sorted = values.slice().sort(function (a, b) { return a - b; });
		var median = n % 2 === 0
			? (sorted[n / 2 - 1] + sorted[n / 2]) / 2
			: sorted[Math.floor(n / 2)];
		return { mean: mean, std: std, min: sorted[0], max: sorted[n - 1], median: median, count: n };
	}

	// --- helper: get values for a trial + column ---
	function trialValues(ti, colIdx) {
		return grouped[ti]
			.map(function (row) { return parseFloat(row[colIdx]); })
			.filter(function (v) { return !isNaN(v); });
	}

	// =====================================================
	//  Per-result-column visualizations
	// =====================================================
	resultCols.forEach(function (col) {
		var sectionHeader = document.createElement("h3");
		sectionHeader.textContent = "Result: " + col.name;
		sectionHeader.style.marginTop = "40px";
		plotDiv.appendChild(sectionHeader);

		var statsMap = {};
		trialIndices.forEach(function (ti) {
			statsMap[ti] = computeStats(trialValues(ti, col.index));
		});

		var valid = trialIndices.filter(function (ti) { return statsMap[ti] !== null; });

		// ---- Plot 1: Mean ± Std with individual points ----
		var meanDiv = document.createElement("div");
		meanDiv.style.marginBottom = "20px";
		plotDiv.appendChild(meanDiv);

		var ptsX = [], ptsY = [];
		valid.forEach(function (ti) {
			trialValues(ti, col.index).forEach(function (v) {
				ptsX.push(ti);
				ptsY.push(v);
			});
		});

		var traces = [];

		// individual evaluations
		traces.push({
			x: ptsX,
			y: ptsY,
			type: "scatter",
			mode: "markers",
			name: "Individual evals",
			marker: { size: get_marker_size() + 2, color: "rgba(220,80,80,0.55)", symbol: "circle" }
		});

		// mean ± std
		traces.push({
			x: valid,
			y: valid.map(function (ti) { return statsMap[ti].mean; }),
			error_y: {
				type: "data",
				array: valid.map(function (ti) { return statsMap[ti].std; }),
				visible: true,
				color: "rgba(0,0,180,0.6)"
			},
			type: "scatter",
			mode: "markers+lines",
			name: "Mean ± Std Dev",
			marker: { size: get_marker_size() + 4, color: "blue" },
			line: { color: "rgba(0,0,180,0.25)", dash: "dot" }
		});

		// overlay aggregated value from results.csv if available
		if (typeof tab_results_headers_json !== "undefined" && typeof tab_results_csv_json !== "undefined") {
			var mti = tab_results_headers_json.indexOf("trial_index");
			var mri = tab_results_headers_json.indexOf(col.name);
			if (mti !== -1 && mri !== -1) {
				var mx = [], my = [];
				tab_results_csv_json.forEach(function (row) {
					var t = parseFloat(row[mti]);
					var v = parseFloat(row[mri]);
					if (!isNaN(t) && !isNaN(v) && (t in grouped)) { mx.push(t); my.push(v); }
				});
				if (mx.length > 0) {
					traces.push({
						x: mx, y: my,
						type: "scatter", mode: "markers",
						name: "Aggregated (results.csv)",
						marker: { size: get_marker_size() + 4, color: "green", symbol: "diamond" }
					});
				}
			}
		}

		var layout1 = {
			title: col.name + " — Mean ± Std Dev per Arm",
			xaxis: { title: get_axis_title_data("Trial Index"), dtick: 1 },
			yaxis: { title: get_axis_title_data(col.name) },
			showlegend: true
		};
		Plotly.newPlot(meanDiv, traces, add_default_layout_data(layout1));

		// ---- Plot 2: Box plot per trial ----
		var anyMulti = valid.some(function (ti) { return statsMap[ti].count > 1; });
		if (anyMulti) {
			var boxDiv = document.createElement("div");
			boxDiv.style.marginBottom = "30px";
			plotDiv.appendChild(boxDiv);

			var boxTraces = valid.map(function (ti) {
				return {
					y: trialValues(ti, col.index),
					type: "box",
					name: "Trial " + ti,
					boxpoints: "all",
					jitter: 0.35,
					pointpos: -1.6,
					marker: { size: get_marker_size() }
				};
			});

			var layout2 = {
				title: col.name + " — Box Plot per Arm",
				yaxis: { title: get_axis_title_data(col.name) },
				showlegend: true
			};
			Plotly.newPlot(boxDiv, boxTraces, add_default_layout_data(layout2));
		}
	});

	// =====================================================
	//  Summary statistics table
	// =====================================================
	var tblHeader = document.createElement("h3");
	tblHeader.textContent = "Summary Statistics";
	tblHeader.style.marginTop = "40px";
	plotDiv.appendChild(tblHeader);

	var tHeaders = ["trial_index", "n_evals"];
	resultCols.forEach(function (col) {
		["mean", "std", "min", "max", "median"].forEach(function (s) {
			tHeaders.push(col.name + "_" + s);
		});
	});

	var tData = [];
	trialIndices.forEach(function (ti) {
		var row = [ti, grouped[ti].length];
		resultCols.forEach(function (col) {
			var st = computeStats(trialValues(ti, col.index));
			if (st) {
				row.push(parseFloat(st.mean.toFixed(6)));
				row.push(parseFloat(st.std.toFixed(6)));
				row.push(parseFloat(st.min.toFixed(6)));
				row.push(parseFloat(st.max.toFixed(6)));
				row.push(parseFloat(st.median.toFixed(6)));
			} else {
				row.push("N/A", "N/A", "N/A", "N/A", "N/A");
			}
		});
		tData.push(row);
	});

	var tblContainerId = "armDistributionStatsTable";
	var tblContainer = document.createElement("div");
	tblContainer.id = tblContainerId;
	tblContainer.style.marginTop = "10px";
	plotDiv.appendChild(tblContainer);

	if (typeof gridjs !== "undefined" && typeof gridjs.Grid === "function") {
		try {
			new gridjs.Grid({
				columns: tHeaders,
				data: tData,
				search: true,
				sort: true
			}).render(tblContainer);
		} catch (e) {
			console.error("gridjs error in arm distribution:", e);
			_armDistFallbackTable(tblContainer, tHeaders, tData);
		}
	} else {
		_armDistFallbackTable(tblContainer, tHeaders, tData);
	}

	// =====================================================
	//  Linked arm info from results.csv
	// =====================================================
	if (typeof tab_results_headers_json !== "undefined" && typeof tab_results_csv_json !== "undefined") {
		var mainTrialIdx = tab_results_headers_json.indexOf("trial_index");
		if (mainTrialIdx !== -1) {
			var armNameIdx = tab_results_headers_json.indexOf("arm_name");
			var genNodeIdx = tab_results_headers_json.indexOf("generation_node");
			var statusIdx = tab_results_headers_json.indexOf("trial_status");

			var infoHeaders = ["trial_index"];
			if (armNameIdx !== -1) infoHeaders.push("arm_name");
			if (genNodeIdx !== -1) infoHeaders.push("generation_node");
			if (statusIdx !== -1) infoHeaders.push("trial_status");

			// parameter columns
			var paramCols = [];
			tab_results_headers_json.forEach(function (h, idx) {
				if (typeof special_col_names !== "undefined" && special_col_names.indexOf(h) !== -1) return;
				if (typeof result_names !== "undefined" && result_names.indexOf(h) !== -1) return;
				if (h.indexOf("OO_Info_") === 0) return;
				paramCols.push({ name: h, index: idx });
				infoHeaders.push(h);
			});

			var infoData = [];
			tab_results_csv_json.forEach(function (row) {
				var t = parseFloat(row[mainTrialIdx]);
				if (isNaN(t) || !(t in grouped)) return;
				var ir = [t];
				if (armNameIdx !== -1) ir.push(row[armNameIdx]);
				if (genNodeIdx !== -1) ir.push(row[genNodeIdx]);
				if (statusIdx !== -1) ir.push(row[statusIdx]);
				paramCols.forEach(function (pc) { ir.push(row[pc.index]); });
				infoData.push(ir);
			});

			if (infoData.length > 0) {
				var aiHeader = document.createElement("h3");
				aiHeader.textContent = "Arm Parameters (from results.csv)";
				aiHeader.style.marginTop = "40px";
				plotDiv.appendChild(aiHeader);

				var aiContainer = document.createElement("div");
				aiContainer.id = "armDistributionArmInfoTable";
				plotDiv.appendChild(aiContainer);

				if (typeof gridjs !== "undefined" && typeof gridjs.Grid === "function") {
					try {
						new gridjs.Grid({
							columns: infoHeaders,
							data: infoData,
							search: true,
							sort: true
						}).render(aiContainer);
					} catch (e) {
						console.error("gridjs error in arm info:", e);
						_armDistFallbackTable(aiContainer, infoHeaders, infoData);
					}
				} else {
					_armDistFallbackTable(aiContainer, infoHeaders, infoData);
				}
			}
		}
	}

	$("#plotArmDistribution").data("loaded", "true");

	if (typeof apply_theme_based_on_system_preferences === "function") {
		apply_theme_based_on_system_preferences();
	}
}

// Fallback plain HTML table when gridjs is not available
function _armDistFallbackTable(container, headers, rows) {
	var table = document.createElement("table");
	table.style.borderCollapse = "collapse";
	table.style.width = "100%";

	var thead = document.createElement("thead");
	var hr = document.createElement("tr");
	headers.forEach(function (h) {
		var th = document.createElement("th");
		th.textContent = h;
		th.style.border = "1px solid #aaa";
		th.style.padding = "6px 10px";
		th.style.textAlign = "left";
		hr.appendChild(th);
	});
	thead.appendChild(hr);
	table.appendChild(thead);

	var tbody = document.createElement("tbody");
	rows.forEach(function (rowData) {
		var tr = document.createElement("tr");
		rowData.forEach(function (cell) {
			var td = document.createElement("td");
			td.textContent = cell;
			td.style.border = "1px solid #ccc";
			td.style.padding = "4px 10px";
			tr.appendChild(td);
		});
		tbody.appendChild(tr);
	});
	table.appendChild(tbody);
	container.appendChild(table);
}
