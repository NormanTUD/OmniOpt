"use strict";

var hashcache = [];
var max_nr_ticks = 1000;

function get_width() {
	return Math.max(1200, parseInt(0.95 * window.innerWidth));
}

function get_height() {
	return Math.max(800, 0.9 * window.innerHeight);
}

function isIntegerOrFloat(value) {
	return /^\d+(\.\d*)?$/.test(value);
}

function convertToIntAndFilter(array) {
	var result = [];

	for (var i = 0; i < array.length; i++) {
		var obj = array[i];
		var values = Object.values(obj);
		var isConvertible = values.every(isIntegerOrFloat);

		if (isConvertible) {
			var intValues = values.map(Number);
			result.push(intValues);
		}
	}

	return result;
}

function getColor(value, minResult, maxResult) {
	var normalized = (value - minResult) / (maxResult - minResult);
	var red = Math.floor(normalized * 255);
	var green = Math.floor((1 - normalized) * 255);
	return `rgb(${red},${green},0)`;
}

function isNumeric(value) {
	return !isNaN(value) && isFinite(value);
}

function getUniqueValues(arr) {
	return [...new Set(arr)];
}

function every_array_element_is_a_number (arr) {
	for (var i = 0; i < arr.length; i++) {
		if (isNaN(arr[i]) || typeof(arr[i]) != "number") {
			return false;
		}
	}

	return true;
}

function mapStrings(values, minNumericValue) {
	var uniqueStrings = [...new Set(values.filter(v => isNaN(parseFloat(v))))];
	uniqueStrings.sort(); // Alphabetically sort the strings
	var stringMapping = {};
	var baseNegativeValue = minNumericValue - uniqueStrings.length - 1;
	uniqueStrings.forEach((str, idx) => {
		stringMapping[str] = baseNegativeValue - idx;
	});
	return stringMapping;
}

async function newPlot3d (name, trace3d, layout3d) {
	Plotly.newPlot(name, [trace3d], layout3d);

	return true;
}

async function scatter_3d(_paramKeys, _results_csv_json, minResult, maxResult, resultValues, mappingKeyNameToIndex) {
	showSpinnerOverlay("Plotting 3d scatter...");
	var already_existing_plots = [];

	$("#scatter_plot_3d_container").html("");

	var promises = [];

	if (_paramKeys.length >= 3 && _paramKeys.length <= 6) {
		var data_md5 = md5(JSON.stringify(_results_csv_json));

		if ($("#scatter_plot_3d_container").data("md5") == data_md5) {
			return;
		}

		for (var i = 0; i < _paramKeys.length; i++) {
			for (var j = i + 1; j < _paramKeys.length; j++) {
				for (var k = j + 1; k < _paramKeys.length; k++) {
					if (!$("#scatter_plot_3d_container").length) {
						add_tab("scatter_plot_3d", "3d-Scatter-Plot", "<div id='scatter_plot_3d_container'></div>");
					}

					var map_x = mappingKeyNameToIndex[_paramKeys[i]];
					var map_y = mappingKeyNameToIndex[_paramKeys[j]];
					var map_z = mappingKeyNameToIndex[_paramKeys[k]];

					var x_name = _paramKeys[i];
					var y_name = _paramKeys[j];
					var z_name = _paramKeys[k];

					var _key = [x_name, y_name, z_name].sort().join("!!!");
					if (already_existing_plots.includes(_key)) {
						warn(`Key already exists: ${_key}`);
						continue;
					}

					var xValuesRaw = _results_csv_json.map(row => row[map_x]);
					var yValuesRaw = _results_csv_json.map(row => row[map_y]);
					var zValuesRaw = _results_csv_json.map(row => row[map_z]);

					var minXValue = getMinNumericValue(xValuesRaw);
					var minYValue = getMinNumericValue(yValuesRaw);
					var minZValue = getMinNumericValue(zValuesRaw);

					var stringMappingX = mapStrings(xValuesRaw, minXValue);
					var stringMappingY = mapStrings(yValuesRaw, minYValue);
					var stringMappingZ = mapStrings(zValuesRaw, minZValue);

					var xValues = [];
					var yValues = [];
					var zValues = [];
					var hoverText = [];

					_results_csv_json.forEach(function(row) {
						var xParsed = parseFloat(row[map_x]);
						var xValue = isNaN(xParsed) ? stringMappingX[row[map_x]] : xParsed;
						xValues.push(xValue);

						var yParsed = parseFloat(row[map_y]);
						var yValue = isNaN(yParsed) ? stringMappingY[row[map_y]] : yParsed;
						yValues.push(yValue);

						var zParsed = parseFloat(row[map_z]);
						var zValue = isNaN(zParsed) ? stringMappingZ[row[map_z]] : zParsed;
						zValues.push(zValue);

						hoverText.push(`x: ${row[map_x]}, y: ${row[map_y]}, z: ${row[map_z]}`);
					});

					function color_curried(value) {
						return getColor(value, minResult, maxResult);
					}

					var colors = resultValues.map(color_curried);

					var trace3d = {
						x: xValues,
						y: yValues,
						z: zValues,
						mode: "markers",
						type: "scatter3d",
						marker: {
							size: 5,
							color: resultValues,
							colorscale: [
								[0, "rgb(0, 255, 0)"],
								[0.5, "rgb(255, 255, 0)"],
								[1, "rgb(255, 0, 0)"]
							],
							cmin: minResult,
							cmax: maxResult,
							colorbar: {
								title: "Result Value",
								tickvals: [minResult, maxResult],
								ticktext: [`Min (${minResult})`, `Max (${maxResult})`],
								len: 0.8
							}
						},
						text: hoverText,
						hoverinfo: "text",
						showlegend: false
					};

					var xAxisConfig = getAxisConfigScatter3d(stringMappingX, xValuesRaw, minXValue, !isNaN(minXValue));
					var yAxisConfig = getAxisConfigScatter3d(stringMappingY, yValuesRaw, minYValue, !isNaN(minYValue));
					var zAxisConfig = getAxisConfigScatter3d(stringMappingZ, zValuesRaw, minZValue, !isNaN(minZValue));

					var layout3d = {
						title: `3D Scatter Plot: ${x_name} vs ${y_name} vs ${z_name}`,
						width: get_width(),
						height: get_height(),
						autosize: false,
						margin: {
							l: 50,
							r: 50,
							b: 100,
							t: 100,
							pad: 4
						},
						scene: {
							xaxis: {
								title: x_name,
								tickvals: xAxisConfig.tickvals,
								ticktext: xAxisConfig.ticktext
							},
							yaxis: {
								title: y_name,
								tickvals: yAxisConfig.tickvals,
								ticktext: yAxisConfig.ticktext
							},
							zaxis: {
								title: z_name,
								tickvals: zAxisConfig.tickvals,
								ticktext: zAxisConfig.ticktext
							}
						},
						paper_bgcolor: "rgba(0,0,0,0)",
						plot_bgcolor: "rgba(0,0,0,0)",
						showlegend: false,
						legend: {
							x: 0.1,
							y: 1.1,
							orientation: "h"
						}
					};

					var new_plot_div = $(`<div class='share_graph scatter-plot' id='scatter-plot-3d-${x_name}_${y_name}_${z_name}' style='width:${get_width()}px;height:${get_height()}px;'></div>`);
					if ($("#scatter_plot_3d_container").length) {
						$("#scatter_plot_3d_container").append(new_plot_div);
						promises.push(newPlot3d(`scatter-plot-3d-${x_name}_${y_name}_${z_name}`, trace3d, layout3d));
						already_existing_plots.push(_key);
					} else {
						error("Cannot find #scatter_plot_3d_container");
					}
				}
			}
		}
	}

	await Promise.all(promises);

	$("#scatter_plot_3d_container").data("md5", data_md5);
}

function reduceNumericTicks(tickvals, ticktext, maxTicks) {
	const step = Math.ceil(tickvals.length / maxTicks);
	return {
		tickvals: tickvals.filter((_, i) => i % step === 0),
		ticktext: ticktext.filter((_, i) => i % step === 0)
	};
}

function getAxisConfigScatter2d(stringMapping, rawValues, minValue, isNumeric) {
	var tickvals = [];
	var ticktext = [];

	Object.entries(stringMapping).forEach(([key, mappedValue]) => {
		tickvals.push(mappedValue);
		ticktext.push(key);
	});

	if (isNumeric) {
		rawValues.forEach(val => {
			var parsed = parseFloat(val);
			if (!isNaN(parsed)) {
				tickvals.push(parsed);
				ticktext.push(String(parsed));
			}
		});
		// Reduce tick count if too many numeric values
		return reduceNumericTicks(tickvals, ticktext, 10); // Allow a max of 10 ticks
	}

	return { tickvals, ticktext };
}

function getAxisConfigScatter3d(stringMapping, rawValues, minValue, isNumeric, maxTicks = 10) {
	var tickvals = [];
	var ticktext = [];
	Object.entries(stringMapping).forEach(([key, mappedValue]) => {
		tickvals.push(mappedValue);
		ticktext.push(key);
	});

	if (isNumeric) {
		let numericValues = Array.from(new Set(rawValues.filter(v => !isNaN(parseFloat(v))).map(parseFloat)));
		numericValues.sort((a, b) => a - b);

		let interval = Math.ceil(numericValues.length / maxTicks);
		numericValues.forEach((val, index) => {
			if (index % interval === 0) {
				tickvals.push(val);
				ticktext.push(String(val));
			}
		});
	}

	return { tickvals, ticktext };
}

function getMinNumericValue(values) {
	return Math.min(...values.filter(v => !isNaN(parseFloat(v))));
}

function scatter(_paramKeys, _results_csv_json, minResult, maxResult, resultValues, mappingKeyNameToIndex) {
	showSpinnerOverlay("Plotting 2d scatter...");
	var already_existing_plots = [];
	var data_md5 = md5(JSON.stringify(_results_csv_json));

	if ($("#scatter_plot_2d_container").data("md5") == data_md5) {
		return;
	}

	$("#scatter_plot_2d_container").html("");

	for (var i = 0; i < _paramKeys.length; i++) {
		for (var j = i + 1; j < _paramKeys.length; j++) {
			if (!$("#scatter_plot_2d_container").length) {
				add_tab("scatter_plot_2d", "2d-Scatter-Plot", "<div id='scatter_plot_2d_container'></div>");
			}

			var map_x = mappingKeyNameToIndex[_paramKeys[i]];
			var map_y = mappingKeyNameToIndex[_paramKeys[j]];

			var x_name = _paramKeys[i];
			var y_name = _paramKeys[j];

			var _key = [x_name, y_name].sort().join("!!!");

			if (already_existing_plots.includes(_key)) {
				warn(`Key already exists: ${_key}`);
				continue;
			}

			var xValuesRaw = _results_csv_json.map(row => row[map_x]);
			var yValuesRaw = _results_csv_json.map(row => row[map_y]);

			var minXValue = getMinNumericValue(xValuesRaw);
			var minYValue = getMinNumericValue(yValuesRaw);

			var stringMappingX = mapStrings(xValuesRaw, minXValue);
			var stringMappingY = mapStrings(yValuesRaw, minYValue);

			var xValues = [];
			var yValues = [];
			var hoverText = [];

			_results_csv_json.forEach(function (row) {
				var xParsed = parseFloat(row[map_x]);
				var xValue = isNaN(xParsed) ? stringMappingX[row[map_x]] : xParsed;
				xValues.push(xValue);

				var yParsed = parseFloat(row[map_y]);
				var yValue = isNaN(yParsed) ? stringMappingY[row[map_y]] : yParsed;
				yValues.push(yValue);

				hoverText.push(`x: ${row[map_x]}, y: ${row[map_y]}`);
			});

			function color_curried(value) {
				return getColor(value, minResult, maxResult);
			}

			var colors = resultValues.map(color_curried);

			var uniqueValues = Array.from(new Set(resultValues)).sort((a, b) => a - b);
			var customColorscale = uniqueValues.map(value => {
				return [(value - minResult) / (maxResult - minResult), color_curried(value)];
			});

			var trace2d = {
				x: xValues,
				y: yValues,
				mode: "markers",
				type: "scatter",
				marker: {
					color: colors,
					size: 15,
					sizemode: "diameter",
					sizeref: 1
				},
				text: hoverText,
				hoverinfo: "text"
			};


			var colorScaleTrace = {
				x: [null],
				y: [null],
				type: "scatter",
				mode: "markers",
				marker: {
					color: [minResult, maxResult],
					colorscale: customColorscale,
					cmin: minResult,
					cmax: maxResult,
					showscale: true,
					colorbar: {
						title: "Result Values",
						titleside: "right"
					}
				},
				hoverinfo: "none"
			};

			var xAxisConfig = getAxisConfigScatter2d(stringMappingX, xValuesRaw, minXValue, !isNaN(minXValue));
			var yAxisConfig = getAxisConfigScatter2d(stringMappingY, yValuesRaw, minYValue, !isNaN(minYValue));

			var layout2d = {
				title: `Scatter Plot: ${x_name} vs ${y_name}`,
				xaxis: {
					title: x_name,
					tickvals: xAxisConfig.tickvals,
					ticktext: xAxisConfig.ticktext,
					tickangle: -45 // Rotate tick labels for better readability
				},
				yaxis: {
					title: y_name,
					tickvals: yAxisConfig.tickvals,
					ticktext: yAxisConfig.ticktext,
					tickangle: -45 // Rotate tick labels for better readability
				},
				paper_bgcolor: "rgba(0,0,0,0)",
				plot_bgcolor: "rgba(0,0,0,0)",
				showlegend: false
			};

			var new_plot_div = $(`<div class='share_graph scatter-plot' id='scatter-plot-${x_name}_${y_name}' style='width:${get_width()}px;height:${get_height()}px;'></div>`);
			$("#scatter_plot_2d_container").append(new_plot_div);

			if ($("#scatter_plot_2d_container").length) {
				Plotly.newPlot(`scatter-plot-${x_name}_${y_name}`, [trace2d, colorScaleTrace], layout2d);
				already_existing_plots.push(_key);
			} else {
				error("Cannot find #scatter_plot_2d_container");
			}
		}
	}

	$("#scatter_plot_2d_container").data("md5", data_md5);
}

async function load_results () {
	showSpinnerOverlay("Loading results...");
	var data = await fetchJsonFromUrlFilenameOnly("results.csv");
	if(!data) {
		warn("load_results: Could not fetch results.csv");
		return;
	}

	if(!Object.keys(data).includes("raw")) {
		//warn(`load_results: Could not plot seemingly empty data: no raw found`);
		return;
	}

	add_tab("results", "Results", "<div id='results_csv'></div>");

	$("#results_csv").html(`<pre class="stdout_file invert_in_dark_mode autotable">${data.raw}</pre>${copy_button("stdout_file")}`);
}

function isFullyNumeric(values) {
	return values.every(value => !isNaN(parseFloat(value)) && isFinite(value));
}

async function plot_all_possible () {
	showSpinnerOverlay("Trying to plot all possible plots...");

	var _results_csv_json = await fetchJsonFromUrlFilenameOnly("results.csv");

	if(!_results_csv_json) {
		return;
	}

	if(!Object.keys(_results_csv_json).includes("data")) {
		warn("plot_all_possible: Could not plot seemingly empty _results_csv_json: no data found");
		return;
	}

	if(!Object.keys(_results_csv_json).includes("data") && !results_csv_json.data.length) {
		warn("plot_all_possible: Could not plot seemingly empty _results_csv_json");
		return;
	}

	convertToIntAndFilter(_results_csv_json.data.map(Object.values));

	var header_line = _results_csv_json.data.shift();

	var mappingKeyNameToIndex = {};
	var paramKeys = [];

	for (var i = 0; i < header_line.length; i++) {
		var this_element = header_line[i];

		if(!["trial_index", "arm_name", "trial_status", "generation_method", "result"].includes(this_element)) {
			paramKeys.push(this_element);
			mappingKeyNameToIndex[this_element] = i;
		}
	}

	var result_idx = header_line.indexOf("result");

	if(result_idx < 0) {
		//error("Cannot find result column index!");
		return;
	}

	var resultValues = _results_csv_json.data.map(function(row) {
		return parseFloat(row[result_idx]);
	});

	resultValues = resultValues.filter(function (value) {
		return !Number.isNaN(value);
	});

	var minResult = Math.min.apply(null, resultValues);
	var maxResult = Math.max.apply(null, resultValues);

	scatter_3d(paramKeys, _results_csv_json.data, minResult, maxResult, resultValues, mappingKeyNameToIndex);
	scatter(paramKeys, _results_csv_json.data, minResult, maxResult, resultValues, mappingKeyNameToIndex);

	apply_theme_based_on_system_preferences();
}

function convertUnixTimeToReadable(unixTime) {
	var date = new Date(unixTime * 1000);
	return date.toLocaleString();
}

async function load_parameter () {
	showSpinnerOverlay("Loading parameters...");
	var data = await fetchJsonFromUrlFilenameOnly("parameters.txt", 1);
	if(!data) {
		return;
	}

	if(!Object.keys(data).includes("raw")) {
		warn("load_parameter: Could not plot seemingly empty data: no raw found");
		return;
	}

	if (data.raw != "null" && data.raw !== null) {
		$(".parameters_txt").html(`<pre>${removeAnsiCodes(data.raw).replaceAll(/\s*$/g, "")}</pre>`);
	}
}

async function get_result_names_data () {
	var result_names_data = await fetchJsonFromUrlFilenameOnly("result_names.txt", 1);

	var result_names = ["RESULT"];

	if(result_names_data && Object.keys(result_names_data).includes("raw") && result_names_data["raw"] != "null" && result_names_data !== null) {
		var parsed_json = result_names_data["raw"];

		result_names = parsed_json.split(/\r?\n/);
	}

	let trimmedStrings = result_names.map(function(str) {
		return str.trim();
	});

	return trimmedStrings;
}

function get_checkmark_if_contains_result(str, result_names) {
	try {
		if (!Array.isArray(result_names) || result_names.length === 0) {
			console.error("Error: result_names must be a non-empty array.");
			return "❌";
		}

		var escapedResultNames = result_names.map(name => name.toLowerCase().replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
		var regexPattern = `(${escapedResultNames.join("|")}):\\s*[+-]?\\d+(\\.\\d+)?`;
		var regex = new RegExp(regexPattern, "gi");

		var foundResults = [];
		var match;
		while ((match = regex.exec(str.toLowerCase())) !== null) {
			foundResults.push(match[1]);
		}

		var checkmarks = result_names.map(name => (foundResults.includes(name.toLowerCase()) ? "✅" : "❌"));

		if (checkmarks.every(mark => mark === "✅")) {
			return "✅";
		}

		if (checkmarks.every(mark => mark === "❌")) {
			return "❌";
		}

		return checkmarks.join("");
	} catch (error) {
		console.error("An error occurred:", error);
		return "❌";
	}
}

async function load_out_files () {
	showSpinnerOverlay("Loading out files...");
	var urlParams = new URLSearchParams(window.location.search);

	var data = await fetchJsonFromUrl(`get_out_files.php?user_id=${urlParams.get("user_id")}&experiment_name=${urlParams.get("experiment_name")}&run_nr=${urlParams.get("run_nr")}`);

	if(!data) {
		return;
	}

	if(!Object.keys(data).includes("data")) {
		warn("load_out_files: Could not plot seemingly empty data: no data found");
		return;
	}

	var main_tabs_div_id = "internal_out_files_content";

	if(data.data) {
		var awaits = [];
		var maxConcurrentRequests = 5;
		var got_data = [];

		for (var i = 0; i < data.data.length; i += maxConcurrentRequests) {
			var batchRequests = [];

			for (var j = i; j < i + maxConcurrentRequests && j < data.data.length; j++) {
				showSpinnerOverlay(`Loading log ${j + 1}/${data.data.length}`);

				if (!$("#" + main_tabs_div_id).length) {
					add_tab("out_files", "Out-Files", `
						<div id='${main_tabs_div_id}'>
						    <div>
							<ul class="nav nav-tabs"></ul>
						    </div>
						</div>
					    `);
				}

				var _fn = data.data[j].replaceAll(/.*\//g, ""); // Clean up filename
				var requestPromise = fetchJsonFromUrl(`get_out_files.php?user_id=${urlParams.get("user_id")}&experiment_name=${urlParams.get("experiment_name")}&run_nr=${urlParams.get("run_nr")}&fn=${_fn}`);

				batchRequests.push(requestPromise);
			}

			awaits.push(...batchRequests);
			Promise.all(batchRequests).then(batchData => {
				got_data.push(...batchData);
			});

			await Promise.all(batchRequests);
		}

		var result_names = await get_result_names_data();

		for (var i = 0; i < data.data.length; i++) {
			var _d = got_data[i];
			showSpinnerOverlay(`Adding log tab ${j + 1}/${data.data.length}`);

			if(Object.keys(_d).includes("error")) {
				error(_d.error);
			} else {
				var _new_tab_id = `out_files_${md5(_d.data + _fn)}`;
				if($("#" + _new_tab_id).length == 0) {
					var _fn = data.data[i].replaceAll(/.*\//g, ""); // Clean up filename
					showSpinnerOverlay(`Loading log ${_fn} (${i + 1}/${got_data.length})...`);
						var _new_tab_title = `${_fn.replace("_0_log.out", "")} <span>${get_checkmark_if_contains_result(_d.data, result_names)}</span>`;
						var ansi_html_data = ansi_to_html(_d.data);
						var _new_tab_content =
							`<div class='out_file_internal' id='out_file_content_${md5(_d.data + _fn)}_internal'>
								<pre style='color: lightgreen; background-color: black;' class='invert_in_dark_mode'>${ansi_html_data}</pre>
							</div>`;

						add_tab(_new_tab_id, _new_tab_title, _new_tab_content, "#" + main_tabs_div_id, false);
					}

			}

		}

		$("#" + main_tabs_div_id).tabs("refresh");

		open_first_tab_when_none_is_open(main_tabs_div_id);

		convert_ansi_to_html();
	}
}

async function load_evaluation_errors_and_oo_errors () {
	showSpinnerOverlay("Loading evaluation- and OmniOpt2-errors...");
	var p = [];
	p.push(_load_evaluation_errors_and_oo_errors("oo_errors", "Evaluation Errors", "oo_errors.txt", "oo_errors"));

	for (var i = 0; i < p.length; i++) {
		await p[i];
	}
}

async function _load_evaluation_errors_and_oo_errors (tab_div, title, _fn, _divname) {
	//debug_function("_load_evaluation_errors_and_oo_errors()");
	var data = await fetchJsonFromUrlFilenameOnly(`${_fn}`);
	if(!data) {
		return;
	}

	if(!Object.keys(data).includes("raw")) {
		//warn(`_load_evaluation_errors_and_oo_errors: Could not plot seemingly empty data: no raw found`);
		return;
	}

	add_tab(tab_div, title, `<div id='${_divname}'></div>`);

	if($(`#${_divname}`).length == 0) {
		error(`Could not find #${_divname}`);
	} else {
		$(`#${_divname}`).html(`<pre style="white-space: preserve-breaks;">${ansi_to_html(data.raw)}</pre>`);
	}
}

async function load_progressbar_log() {
	showSpinnerOverlay("Loading progressbar-log...");
	var data = await fetchJsonFromUrlFilenameOnly("progressbar");
	if(!data) {
		return;
	}

	if(!Object.keys(data).includes("raw")) {
		//warn(`load_progressbar_log: Could not plot seemingly empty data: no raw found`);
		return;
	}

	add_tab("progressbar_log", "Progressbar-Log", "<div id='progressbar_log_element'></div>");

	if($("#progressbar_log_element").length == 0) {
		error("Could not find #progressbar_log_element");
	} else {
		var converted = ansi_to_html(removeLinesStartingWith(data.raw, "P7;1;75", "-$$$$$-$$$$$"));
		const removeTrailingWhitespaces = (str) => str.split("\n").map(line => line.replace(/\s+$/, "")).join("\n");
		converted = removeTrailingWhitespaces(converted);
		$("#progressbar_log_element").html(`<pre class="progressbar_log_class" style='color: white; background-color: black; white-space: break-spaces;'>${converted}</pre>${copy_button("progressbar_log_class")}`);
		$(".progressbar_log_class").addClass("invert_in_dark_mode");
	}
}

async function load_trial_index_to_params_log () {
	showSpinnerOverlay("Loading trial index to params log...");
	var data = await fetchJsonFromUrlFilenameOnly("trial_index_to_params");
	if(!data) {
		return;
	}

	if(!Object.keys(data).includes("raw")) {
		//warn(`load_trial_index_to_params_log: Could not plot seemingly empty data: no raw found`);
		return;
	}

	add_tab("trial_index_to_params", "Trial Index to Param", "<div id='trial_index_to_params_element'></div>");

	if($("#trial_index_to_params_element").length == 0) {
		error("Could not find #trial_index_to_params_element");
	} else {
		var converted = ansi_to_html(removeLinesStartingWith(data.raw, "P7;1;75", "-$$$$$-$$$$$"));
		const removeTrailingWhitespaces = (str) => str.split("\n").map(line => line.replace(/\s+$/, "")).join("\n");
		converted = removeTrailingWhitespaces(converted);
		$("#trial_index_to_params_element").html(`<pre class="trial_index_to_params_element_class" style='color: white; background-color: black; white-space: break-spaces;'>${converted}</pre>${copy_button("trial_index_to_params_element_class")}`);
		$(".trial_index_to_params_element_class").addClass("invert_in_dark_mode");
	}
}

async function load_install_errors() {
	showSpinnerOverlay("Loading install-errors...");
	var data = await fetchJsonFromUrlFilenameOnly("install_errors");
	if(!data) {
		return;
	}

	if(!Object.keys(data).includes("raw")) {
		//warn(`load_install_errors: Could not plot seemingly empty data: no raw found`);
		return;
	}

	add_tab("install_errors", "Install errors", "<div id='install_errors_element'></div>");

	if($("#install_errors_element").length == 0) {
		error("Could not find #install_errors_element");
	} else {
		var converted = ansi_to_html(removeLinesStartingWith(data.raw, "P7;1;75", "-$$$$$-$$$$$"));
		const removeTrailingWhitespaces = (str) => str.split("\n").map(line => line.replace(/\s+$/, "")).join("\n");
		converted = removeTrailingWhitespaces(converted);
		$("#install_errors_element").html(`<pre class="install_errors_class" class='invert_in_dark_mode' style='color: white; background-color: black; white-space: break-spaces;'>${converted}</pre>${copy_button("install_errors_class")}`);
	}
}

function injectStyles() {
	const styles = `
		#searchContainer {
			margin-bottom: 10px;
		}
		#debug_search {
			padding: 5px;
			width: 100%;
			box-sizing: border-box;
		}
		.stacktrace {
		    white-space: pre-wrap;
		}
	`;
	const styleSheet = document.createElement("style");
	styleSheet.type = "text/css";
	styleSheet.innerText = styles;
	document.head.appendChild(styleSheet);
}

function createTable(data, id) {
	const tableContainer = document.createElement("div");
	const table = document.createElement("table");
	const headerRow = document.createElement("tr");
	const headers = ["Time", "Message", "Function Stack"];

	headers.forEach(header => {
		const th = document.createElement("th");
		th.innerText = header;
		headerRow.appendChild(th);
		headerRow.classList.add("invert_in_dark_mode");
	});
	table.appendChild(headerRow);

	data.forEach(item => {
		const row = document.createElement("tr");
		const timeCell = document.createElement("td");
		const msgCell = document.createElement("td");
		const stackCell = document.createElement("td");
		stackCell.classList.add("stacktrace");

		timeCell.innerText = item.time;
		msgCell.innerHTML = `<samp>${item.msg}</samp>`;

		if(Object.keys(item).includes("function_stack")) {
			const formattedStack = item.function_stack
				.map(func => `<samp>${func.function} (Line ${func.line_number})</samp>`)
				.join("\n");

			stackCell.innerHTML = formattedStack;

			row.appendChild(timeCell);
			row.appendChild(msgCell);
			row.appendChild(stackCell);
			table.appendChild(row);
		}
	});

	tableContainer.appendChild(table);
	$("#" + id).html(tableContainer);
	addSearchFunctionality(tableContainer, table);
}

function parseLogData(logData) {
	const lines = logData.split("\n");

	const jsonData = [];
	let jsonBuffer = "";

	lines.forEach((line, index) => {
		if (line.trim() === "") return;

		jsonBuffer += line;

		try {
			const jsonObject = JSON.parse(jsonBuffer);
			jsonData.push(jsonObject);
			jsonBuffer = "";
		} catch (error) {
		}
	});

	if (jsonBuffer.trim() !== "") {
		console.warn("Warning: Incomplete JSON object found at end of log data.");
	}

	return jsonData;
}

function addSearchFunctionality(tableContainer, table) {
	const searchContainer = document.createElement("div");
	searchContainer.id = "searchContainer";

	const searchInput = document.createElement("input");
	searchInput.type = "text";
	searchInput.id = "debug_search";
	searchInput.placeholder = "Search...";
	searchContainer.appendChild(searchInput);
	tableContainer.insertBefore(searchContainer, table);

	searchInput.addEventListener("input", function() {
		const filter = searchInput.value.toLowerCase();
		const rows = table.getElementsByTagName("tr");

		for (let i = 1; i < rows.length; i++) {
			const cells = rows[i].getElementsByTagName("td");
			let rowContainsSearchTerm = false;

			for (let j = 0; j < cells.length; j++) {
				if (cells[j].innerText.toLowerCase().includes(filter)) {
					rowContainsSearchTerm = true;
					break;
				}
			}

			rows[i].style.display = rowContainsSearchTerm ? "" : "none";
		}
	});
}

async function load_debug_log() {
	showSpinnerOverlay("Loading debug-log...");
	var data = await fetchJsonFromUrlFilenameOnly("log");
	if(!data) {
		return;
	}

	if(!Object.keys(data).includes("raw")) {
		//warn(`load_debug_log: Could not plot seemingly empty data: no raw found`);
		return;
	}

	add_tab("internal_log", "Debug-Log", "<div id='internal_log_element'></div>");

	if($("#internal_log_element").length == 0) {
		error("Could not find #internal_log_element");
	} else {
		var converted = ansi_to_html(removeLinesStartingWith(data.raw, "P7;1;75", "-$$$$$-$$$$$"));
		const removeTrailingWhitespaces = (str) => str.split("\n").map(line => line.replace(/\s+$/, "")).join("\n");
		converted = removeTrailingWhitespaces(converted);
		$("#internal_log_element").html(`<pre style="display: none" class="internal_log_element_class" class='invert_in_dark_mode' style='color: white; background-color: black; white-space: break-spaces;'>${converted}</pre>${copy_button("internal_log_element_class")}<div id="internal_log_table"></div>`);

		var id = "internal_log_table";

		var parsed_log_data = parseLogData(data.raw);

		if(parsed_log_data && Object.keys(parsed_log_data)) {
			injectStyles();
			createTable(parsed_log_data, id);
		} else {
			$("#internal_log_element_class").show();
			$("#internal_log_table").hide();
		}
	}
}

async function load_outfile () {
	showSpinnerOverlay("Loading outfile...");
	var data = await fetchJsonFromUrlFilenameOnly("outfile", false, true);
	if(!data) {
		return;
	}

	if(!Object.keys(data).includes("raw")) {
		//warn(`load_outfile: Could not plot seemingly empty data: no raw found`);
		return;
	}

	add_tab("outfile", "Main Log", "<div id='outfile'></div>");

	if($("#outfile").length == 0) {
		error("Could not find #outfile");
	} else {
		var converted = ansi_to_html(removeLinesStartingWith(data.raw, "P7;1;75", "-$$$$$-$$$$$"));
		const removeTrailingWhitespaces = (str) => str.split("\n").map(line => line.replace(/\s+$/, "")).join("\n");
		converted = removeTrailingWhitespaces(converted);
		$("#outfile").html(`<pre class="main_outfile" style='color: white; background-color: black; white-space: break-spaces;'>${converted}</pre>${copy_button("main_outfile")}`);
		$(".main_outfile").addClass("invert_in_dark_mode");
	}
}

async function load_next_trials () {
	showSpinnerOverlay("Loading next-trials-log...");
	var urlParams = new URLSearchParams(window.location.search);

	var data = await fetchJsonFromUrl(`get_next_trials.php?user_id=${urlParams.get("user_id")}&experiment_name=${urlParams.get("experiment_name")}&run_nr=${urlParams.get("run_nr")}`);
	if(!data) {
		return;
	}

	if(!Object.keys(data).includes("raw")) {
		//warn(`load_next_trials: Could not plot seemingly empty data: no raw found`);
		return;
	}

	add_tab("next_trials", "Next-Trials", "<div id='next_trials_csv'></div>");

	$("#next_trials_csv").html(`${data.raw}`);
}

async function load_job_infos () {
	showSpinnerOverlay("Loading job-infos...");
	var data = await fetchJsonFromUrlFilenameOnly("job_infos.csv");
	if(!data) {
		return;
	}

	if(!Object.keys(data).includes("raw")) {
		//warn(`load_job_infos: Could not plot seemingly empty data: no raw found`);
		return;
	}

	add_tab("job_infos", "Job-Infos", "<div id='job_infos_csv'></div>");
	$("#job_infos_csv").html(`<pre class="stdout_file invert_in_dark_mode autotable">${data.raw}</pre>${copy_button("stdout_file")}`);
}

async function load_pareto_graph () {
	showSpinnerOverlay("Loading pareto graph (if available)");

	var data = await fetchJsonFromUrlFilenameOnly("pareto_front_data.json");

	if (!Object.keys(data).includes("raw")) {
		return;
	}

	if (data.raw != "null" && data.raw !== null) {
		add_tab("pareto_front_graphs", "Pareto-Front", "<div id='pareto_front_graphs_container'><div>");

		let parsedData = JSON.parse(data.raw);
		let categories = Object.keys(parsedData);
		let allMetrics = new Set();

		function extractMetrics(obj, prefix = "") {
			let keys = Object.keys(obj);
			for (let key of keys) {
				let newPrefix = prefix ? `${prefix} -> ${key}` : key;
				if (typeof obj[key] === "object" && !Array.isArray(obj[key])) {
					extractMetrics(obj[key], newPrefix);
				} else {
					if (!newPrefix.includes("param_dicts") && !newPrefix.includes(" -> sems -> ") && !newPrefix.includes("absolute_metrics")) {
						allMetrics.add(newPrefix);
					}
				}
			}
		}

		for (let cat of categories) {
			extractMetrics(parsedData[cat]);
		}

		allMetrics = Array.from(allMetrics);

		function extractValues(obj, metricPath, values) {
			let parts = metricPath.split(" -> ");
			let data = obj;
			for (let part of parts) {
				if (data && typeof data === "object") {
					data = data[part];
				} else {
					return;
				}
			}
			if (Array.isArray(data)) {
				values.push(...data);
			}
		}

		let graphContainer = document.getElementById("pareto_front_graphs_container");
		graphContainer.innerHTML = "";

		var already_plotted = [];

		for (let i = 0; i < allMetrics.length; i++) {
			for (let j = i + 1; j < allMetrics.length; j++) {
				let xMetric = allMetrics[i];
				let yMetric = allMetrics[j];

				let xValues = [];
				let yValues = [];

				for (let cat of categories) {
					let metricData = parsedData[cat];
					extractValues(metricData, xMetric, xValues);
					extractValues(metricData, yMetric, yValues);
				}

				xValues = xValues.filter(v => v !== undefined && v !== null);
				yValues = yValues.filter(v => v !== undefined && v !== null);

				let cleanXMetric = xMetric.replace(/.* -> /g, "");
				let cleanYMetric = yMetric.replace(/.* -> /g, "");

				let plot_key = `${cleanXMetric}-${cleanYMetric}`

				if (xValues.length > 0 && yValues.length > 0 && xValues.length === yValues.length && !already_plotted.includes(plot_key)) {
					let div = document.createElement("div");
					div.id = `pareto_front_graph_${i}_${j}`;
					div.style.marginBottom = "20px";
					graphContainer.appendChild(div);



					let layout = {
						title: `${cleanXMetric} vs ${cleanYMetric}`,
						xaxis: { title: cleanXMetric },
						yaxis: { title: cleanYMetric },
						hovermode: "closest"
					};

					let trace = {
						x: xValues,
						y: yValues,
						mode: "markers",
						type: "scatter",
						name: `${cleanXMetric} vs ${cleanYMetric}`
					};

					Plotly.newPlot(div.id, [trace], layout);

					already_plotted.push(plot_key);
				}
			}
		}

		var pareto_data = await fetchJsonFromUrlFilenameOnly("pareto_front_table.txt");

		if(!Object.keys(data).includes("raw")) {
			//warn(`load_best_result: Could not plot seemingly empty data: no raw found`);
			return;
		}

		if (Object.keys(pareto_data).includes("raw") && pareto_data.raw != "null" && pareto_data.raw !== null) {
			$("#pareto_front_graphs_container").append(`<pre>${removeAnsiCodes(pareto_data.raw)}</pre>`);
		}
	}
}

async function load_best_result () {
	showSpinnerOverlay("Loading best results...");
	var data = await fetchJsonFromUrlFilenameOnly("best_result.txt");
	if(!data) {
		return;
	}
}

async function plot_planned_vs_real_worker_over_time () {
	showSpinnerOverlay("Plotting planned vs. real workers...");
	var data = await fetchJsonFromUrlFilenameOnly("worker_usage.csv");
	if(!data) {
		return;
	}

	if(!Object.keys(data).includes("data")) {
		//warn(`plot_planned_vs_real_worker_over_time: Could not plot seemingly empty data: no data found`);
		return;
	}

	if(!data.data.length) {
		warn("plot_planned_vs_real_worker_over_time: Could not plot seemingly empty data");
		return;
	}

	convertToIntAndFilter(data.data.map(Object.values));

	replaceZeroWithNull(data.data);

	var unixTime = data.data.map(row => row[0]);
	var readableTime = unixTime.map(convertUnixTimeToReadable);
	var plannedWorkers = data.data.map(row => row[1]);
	var actualWorkers = data.data.map(row => row[2]);

	var tracePlanned = {
		x: readableTime,
		y: plannedWorkers,
		mode: "lines",
		name: "Planned Worker"
	};

	var traceActual = {
		x: readableTime,
		y: actualWorkers,
		mode: "lines",
		name: "Real Worker"
	};

	var layout = {
		title: "Planned vs. real worker over time",
		xaxis: {
			title: "Date"
		},
		yaxis: {
			title: "Nr. Worker"
		},
		width: get_width(),
		height: get_height(),
		paper_bgcolor: "rgba(0,0,0,0)",
		plot_bgcolor: "rgba(0,0,0,0)",

		showlegend: true,
		legend: {
			x: 0.1,
			y: 1.1,
			orientation: "h"
		},
	};

	add_tab("worker_usage", "Worker-Usage", "<div id='worker_usage_plot'></div><div id='worker_usage_raw'></div>");
	Plotly.newPlot("worker_usage_plot", [tracePlanned, traceActual], layout);

	$("#worker_usage_raw").html(`<pre class="stdout_file invert_in_dark_mode autotable">${data.raw}</pre>${copy_button("stdout_file")}`);
}

async function plot_cpu_ram_graph() {
	showSpinnerOverlay("Plotting CPU/RAM Graph...");
	var cpu_ram_usage_json = await fetchJsonFromUrlFilenameOnly("cpu_ram_usage.csv");
	if (!cpu_ram_usage_json) {
		return;
	}

	if (!Object.keys(cpu_ram_usage_json).includes("data")) {
		warn("plot_cpu_ram_graph: Could not plot seemingly empty cpu_ram_usage_json: no data found");
		return;
	}

	if (!Object.keys(cpu_ram_usage_json).includes("data") || !cpu_ram_usage_json.data.length) {
		warn("plot_cpu_ram_graph: Could not plot seemingly empty cpu_ram_usage_json");
		return;
	}

	convertToIntAndFilter(cpu_ram_usage_json.data.map(Object.values));

	replaceZeroWithNull(cpu_ram_usage_json.data);

	const validCpuEntries = cpu_ram_usage_json.data.filter(entry => entry[2] !== null && entry[2] !== undefined);

	// Filtered timestamps and CPU usage data
	const timestamps_cpu = validCpuEntries.map(entry => new Date(entry[0] * 1000));
	const cpuUsage = validCpuEntries.map(entry => entry[2]);

	// RAM data remains the same
	const timestamps_ram = cpu_ram_usage_json.data.map(entry => new Date(entry[0] * 1000));
	const ramUsage = cpu_ram_usage_json.data.map(entry => entry[1]);

	// RAM Usage Plot
	const ramTrace = {
		x: timestamps_ram,
		y: ramUsage,
		type: "scatter",
		mode: "lines",
		name: "RAM Usage (MB)",
		line: { color: "lightblue" }
	};

	// CPU Usage Plot
	const cpuTrace = {
		x: timestamps_cpu,
		y: cpuUsage,
		type: "scatter",
		mode: "lines",
		name: "CPU Usage (%)",
		line: { color: "orange" }
	};

	const ramLayout = {
		title: "RAM Usage Over Time by the main worker",
		xaxis: {
			title: "Time",
			type: "date"
		},
		yaxis: {
			title: "RAM Usage (MB)",
			showline: true
		},
		showlegend: true,
		legend: {
			x: 0.1,
			y: 1.1,
			orientation: "h"
		},
		paper_bgcolor: "rgba(0,0,0,0)",
		plot_bgcolor: "rgba(0,0,0,0)"
	};

	const cpuLayout = {
		title: "CPU Usage Over Time by the main worker",
		xaxis: {
			title: "Time",
			type: "date"
		},
		yaxis: {
			title: "CPU Usage (%)",
			showline: true
		},
		showlegend: true,
		legend: {
			x: 0.1,
			y: 1.1,
			orientation: "h"
		},
		paper_bgcolor: "rgba(0,0,0,0)",
		plot_bgcolor: "rgba(0,0,0,0)"
	};

	add_tab("cpu_ram_usage", "CPU/RAM Usage", `
		<div id='cpuRamChartContainer'>
		    <div id='ramChart'></div>
		    <div id='cpuChart'></div>
		    <div id='cpuRamChartRawData'></div>
		</div>
	`);

	if ($("#ramChart").length) {
		Plotly.newPlot("ramChart", [ramTrace], ramLayout);
	}

	if ($("#cpuChart").length) {
		Plotly.newPlot("cpuChart", [cpuTrace], cpuLayout);
	}

	$("#cpuRamChartRawData").html(`<pre class="stdout_file invert_in_dark_mode autotable">${cpu_ram_usage_json.raw}</pre>${copy_button("stdout_file")}`);
}

async function plot_worker_cpu_ram() {
	showSpinnerOverlay("Loading worker CPU/RAM plot");
	var data = await fetchJsonFromUrlFilenameOnly("eval_nodes_cpu_ram_logs.txt");

	if (!data || !Object.keys(data).includes("raw")) {
		removeSpinnerOverlay();
		console.log("No data or data.raw");
		return;
	}

	const logData = data.raw;
	const regex = /^Unix-Timestamp: (\d+), Hostname: ([\w-]+), CPU: ([\d.]+)%, RAM: ([\d.]+) MB \/ ([\d.]+) MB$/;

	const hostData = {};

	logData.split("\n").forEach(line => {
		line = line.trim();
		const match = line.match(regex);
		if (match) {
			const timestamp = new Date(parseInt(match[1]) * 1000);
			const hostname = match[2];
			const cpu = parseFloat(match[3]);
			const ram = parseFloat(match[4]);

			if (!hostData[hostname]) {
				hostData[hostname] = { timestamps: [], cpuUsage: [], ramUsage: [] };
			}

			hostData[hostname].timestamps.push(timestamp);
			hostData[hostname].cpuUsage.push(cpu);
			hostData[hostname].ramUsage.push(ram);
		}
	});

	if (!Object.keys(hostData).length) {
		removeSpinnerOverlay();
		console.log("No valid data found");
		return;
	}

	add_tab("worker_cpu_ram_usage", "Worker CPU/RAM Usage", `
		<div id='cpuRamWorkerChartContainer'></div>
	`);

	const container = document.getElementById("cpuRamWorkerChartContainer");
	container.innerHTML = "";

	Object.entries(hostData).forEach(([hostname, { timestamps, cpuUsage, ramUsage }], index) => {
		const chartId = `workerChart_${index}`;
		const chartDiv = document.createElement("div");
		chartDiv.id = chartId;
		chartDiv.style.marginBottom = "40px";
		container.appendChild(chartDiv);

		const cpuTrace = {
			x: timestamps,
			y: cpuUsage,
			mode: "lines+markers",
			name: "CPU Usage (%)",
			yaxis: "y1",
			line: { color: "red" }
		};

		const ramTrace = {
			x: timestamps,
			y: ramUsage,
			mode: "lines+markers",
			name: "RAM Usage (MB)",
			yaxis: "y2",
			line: { color: "blue" }
		};

		const layout = {
			title: `Worker CPU and RAM Usage - ${hostname}`,
			xaxis: { title: "Timestamp" },
			yaxis: {
				title: "CPU Usage (%)",
				side: "left",
				color: "red"
			},
			yaxis2: {
				title: "RAM Usage (MB)",
				side: "right",
				overlaying: "y",
				color: "blue"
			},
			showlegend: true
		};

		Plotly.newPlot(chartId, [cpuTrace, ramTrace], layout);
	});

	removeSpinnerOverlay();
}

function replaceZeroWithNull(arr) {
	if (Array.isArray(arr)) {
		for (let i = 0; i < arr.length; i++) {
			if (Array.isArray(arr[i])) {
				replaceZeroWithNull(arr[i]);
			} else if (arr[i] === 0) {
				arr[i] = null;
			}
		}
	}
}

async function fetchJsonFromUrl(url) {
	//debug_function(`fetchJsonFromUrl("${url}")`);
	try {
		const response = await fetch(url);
		if (!response.ok) {
			throw new Error("Network response was not ok: " + response.statusText);
		}
		const data = await response.json();
		if(Object.keys(data).includes("hash")) {
			var hash = data.hash;
			if(!hashcache.includes(hash)) {
				hashcache.push(hash);
				return data;
			}
		}

		return data;
	} catch (error) {
		return null;
	}

	return null;
}

async function _get_overview_data () {
	//debug_function("_get_overview_data()");
	var urlParams = new URLSearchParams(window.location.search);

	var _res = await fetchJsonFromUrl(`get_overview_data.php?user_id=${urlParams.get("user_id")}&experiment_name=${urlParams.get("experiment_name")}&run_nr=${urlParams.get("run_nr")}`);

	return _res;
}

async function load_experiment_overview () {
	showSpinnerOverlay("Loading experiment-overview data...");

	var res = await fetchJsonFromUrlFilenameOnly("experiment_overview.txt", true);

	if(!Object.keys(res).includes("error")) {
		add_tab("experiment_overview_data", "Experiment-Overview", "<div class='experiment_overview'></div>");
		$(".experiment_overview").html(`<pre>${res.raw}</pre>`);
	} else {
		$(".experiment_overview").html(`Error: <span class="error_line invert_in_dark_mode">${res.error}</span>`);
	}
}

async function load_arg_overview () {
	showSpinnerOverlay("Loading arg-overview data...");

	var res = await fetchJsonFromUrlFilenameOnly("args_overview.txt", true);

	if(!Object.keys(res).includes("error")) {
		add_tab("arg_overview_data", "Args-Overview", "<div class='arg_overview'></div>");
		$(".arg_overview").html(`<pre>${res.raw}</pre>`);
	} else {
		$(".arg_overview").html(`Error: <span class="error_line invert_in_dark_mode">${res.error}</span>`);
	}
}

async function load_overview_data() {
	showSpinnerOverlay("Loading overview data...");

	add_tab("overview_data", "Overview", "<div class='best_result_txt'></div><div class='parameters_txt'></div><div class='overview_table'></div>");

	var res = await _get_overview_data();

	//log(res);
	if(!Object.keys(res).includes("error")) {
		// Create a table
		var table = document.createElement("table");
		table.style.borderCollapse = "collapse";

		// Create table headers
		var headerRow = document.createElement("tr");
		["Failed", "Running", "Succeeded", "Total"].forEach(function (heading) {
			var th = document.createElement("th");
			th.style.border = "1px solid black";
			th.style.padding = "8px";
			th.classList.add("invert_in_dark_mode");
			th.textContent = heading;
			headerRow.appendChild(th);
		});
		table.appendChild(headerRow);

		// Create a data row
		var dataRow = document.createElement("tr");
		[res.failed, res.running, res.succeeded, res.total].forEach(function (value) {
			var td = document.createElement("td");
			td.style.border = "1px solid black";
			td.style.padding = "8px";
			td.textContent = value;
			dataRow.appendChild(td);
		});
		table.appendChild(dataRow);

		// Insert table into the #overview_table element
		$(".overview_table").html(table);
	} else {
		$(".overview_table").html(`Error: <span class="error_line invert_in_dark_mode">${res.error}</span>`);
	}

	try {
		var ui_url = await fetchJsonFromUrlFilenameOnly("ui_url.txt");

		if(ui_url && Object.keys(ui_url).includes("raw")) {
			var raw_url = ui_url.raw;

			var link_code = `<a id='link_to_gui_element' target='_blank' href='${raw_url}'>Link to the GUI page with all the settings of this job</a><br>`;

			if($("#link_to_gui_element").length == 0) {
				$("#overview_data-content").prepend(link_code);
			}
		}
	} catch (e) {
		console.error(e);
	}
}

async function fetchJsonFromUrlFilenameOnly(filename, remove_ansi=false, parse_ansi=false) {
	//debug_function(`fetchJsonFromUrlFilenameOnly('${filename}')`);
	var urlParams = new URLSearchParams(window.location.search);

	var url = `share_to_csv.php?user_id=${urlParams.get("user_id")}&experiment_name=${urlParams.get("experiment_name")}&run_nr=${urlParams.get("run_nr")}&filename=${filename}`;

	if(remove_ansi) {
		url = url + "&remove_ansi=1";
	}

	if(parse_ansi) {
		url = url + "&parse_ansi=1";
	}

	var _res = await fetchJsonFromUrl(url);

	return _res;
}

function convert_ansi_to_html () {
	$(".convert_ansi_to_html").each(function (i, e) {
		var html = e.innerHTML;

		var res = ansi_to_html(html);

		e.innerHTML = res;
	});
}

async function load_all_data() {
	showSpinnerOverlay("Loading data...");
	var urlParams = new URLSearchParams(window.location.search);

	if(urlParams.get("user_id") && urlParams.get("experiment_name") && !isNaN(parseInt(urlParams.get("run_nr")))) {
		var promises = [];

		promises.push(load_overview_data());
		promises.push(load_evaluation_errors_and_oo_errors());
		promises.push(load_pareto_graph());
		promises.push(load_best_result());
		promises.push(load_job_infos());
		promises.push(load_next_trials());
		promises.push(load_results());
		promises.push(load_outfile());
		promises.push(load_debug_log());
		promises.push(load_install_errors());
		promises.push(load_trial_index_to_params_log());
		promises.push(load_progressbar_log());
		promises.push(load_parameter());
		promises.push(load_arg_overview());
		promises.push(load_experiment_overview());

		promises.push(plot_all_possible());
		promises.push(plot_cpu_ram_graph());
		promises.push(plot_parallel_plot());
		promises.push(plot_worker_cpu_ram());
		promises.push(plot_planned_vs_real_worker_over_time());

		for (var i = 0; i < promises.length; i++) {
			await promises[i];
		}

		await load_out_files();

		initialize_autotables();

		removeSpinnerOverlay();

		//log("Loaded page");

		link_share_main();
	}
}

async function refresh() {
	await load_all_data();
	$("#refresh_button").text("Refresh");
}

function copy_button (name_to_search_for) {
	if(!name_to_search_for) {
		error("Empty name_to_search_for in copy_button");
		console.trace();
		return "";
	}

	return `<button class='copy_to_clipboard_button invert_in_dark_mode' onclick='find_closest_element_behind_and_copy_content_to_clipboard(this, "${name_to_search_for}")'>📋 Copy raw data to clipboard</button>`;
}

function link_share_main () {
	document.querySelectorAll('.main_outfile span').forEach(span => {
		const urlPattern = /(https?:\/\/[^\s]+)/g;
		const text = span.textContent;

		if (urlPattern.test(text)) {
			const newContent = text.replace(urlPattern, match => 
				`<a href="${match}" target="_blank" rel="noopener noreferrer">${match}</a>`
			);
			const newSpan = document.createElement('span');
			newSpan.innerHTML = newContent;
			span.replaceWith(newSpan);
		}
	});

}
