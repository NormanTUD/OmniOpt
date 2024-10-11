"use strict";

var hashcache = [];
var max_nr_ticks = 1000;

function get_width() {
	return Math.max(1200, parseInt(0.95 * window.innerWidth));
}

function get_height() {
	return Math.max(800, 0.9 * window.innerHeight)
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

function scatter_3d (_paramKeys, _results_csv_json, minResult, maxResult, resultValues, mappingKeyNameToIndex) {
	//debug_function("scatter_3d()");
	var already_existing_plots = [];
	var data_md5 = md5(JSON.stringify(_results_csv_json));

	if($('#scatter_plot_3d_container').data("md5") == data_md5) {
		return;
	}

	$('#scatter_plot_3d_container').html("");

	if (_paramKeys.length >= 3 && _paramKeys.length <= 6) {
		for (var i = 0; i < _paramKeys.length; i++) {
			for (var j = i + 1; j < _paramKeys.length; j++) {
				for (var k = j + 1; k < _paramKeys.length; k++) {
					if(!$("#scatter_plot_3d_container").length) {
						add_tab("scatter_plot_3d", "3d-Scatter-Plot", "<div id='scatter_plot_3d_container'></div>")
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

					// Function to map string values to unique negative numbers starting just below the minimum numeric value
					function mapStrings(values, minNumericValue) {
						var uniqueStrings = [...new Set(values.filter(v => isNaN(parseFloat(v))))];
						uniqueStrings.sort(); // Alphabetically sort the strings
						var stringMapping = {};
						// Start string values just below the minimum numeric value
						var baseNegativeValue = minNumericValue - uniqueStrings.length - 1;
						uniqueStrings.forEach((str, idx) => {
							stringMapping[str] = baseNegativeValue - idx;
						});
						return stringMapping;
					}

					// Get minimum numeric value for each axis
					function getMinNumericValue(values) {
						return Math.min(...values.filter(v => !isNaN(parseFloat(v))));
					}

					var xValuesRaw = _results_csv_json.map(row => row[map_x]);
					var yValuesRaw = _results_csv_json.map(row => row[map_y]);
					var zValuesRaw = _results_csv_json.map(row => row[map_z]);

					var minXValue = getMinNumericValue(xValuesRaw);
					var minYValue = getMinNumericValue(yValuesRaw);
					var minZValue = getMinNumericValue(zValuesRaw);

					// Map strings to negative values and store the original string for hover tooltips
					var stringMappingX = mapStrings(xValuesRaw, minXValue);
					var stringMappingY = mapStrings(yValuesRaw, minYValue);
					var stringMappingZ = mapStrings(zValuesRaw, minZValue);

					var xValues = [];
					var yValues = [];
					var zValues = [];
					var hoverText = [];

					_results_csv_json.forEach(function(row) {
						// Handle x-axis
						var xParsed = parseFloat(row[map_x]);
						var xValue = isNaN(xParsed) ? stringMappingX[row[map_x]] : xParsed;
						xValues.push(xValue);

						// Handle y-axis
						var yParsed = parseFloat(row[map_y]);
						var yValue = isNaN(yParsed) ? stringMappingY[row[map_y]] : yParsed;
						yValues.push(yValue);

						// Handle z-axis
						var zParsed = parseFloat(row[map_z]);
						var zValue = isNaN(zParsed) ? stringMappingZ[row[map_z]] : zParsed;
						zValues.push(zValue);

						// Hover text with the original values
						hoverText.push(`x: ${row[map_x]}, y: ${row[map_y]}, z: ${row[map_z]}`);
					});

					// Color function for markers
					function color_curried(value) {
						return getColor(value, minResult, maxResult);
					}
					var colors = resultValues.map(color_curried);

					// Plotly trace for 3D scatter plot
					var trace3d = {
						x: xValues,
						y: yValues,
						z: zValues,
						mode: 'markers',
						type: 'scatter3d',
						marker: {
							size: 5, // Set marker size
							color: resultValues, // Use resultValues for color mapping
							colorscale: [
								[0, 'rgb(0, 255, 0)'],  // Green at the lowest value (0)
								[0.5, 'rgb(255, 255, 0)'],  // Yellow in the middle (0.5)
								[1, 'rgb(255, 0, 0)']  // Red at the highest value (1)
							], // Custom color scale (green -> yellow -> red)
							cmin: minResult, // Minimum result value
							cmax: maxResult, // Maximum result value
							colorbar: {
								title: 'Result Value', // Title of the colorbar
								tickvals: [minResult, maxResult], // Show only min and max ticks
								ticktext: [`Min (${minResult})`, `Max (${maxResult})`], // Label for min and max
								len: 0.8 // Length of the colorbar
							}
						},
						text: hoverText, // Show the original values in hover info
						hoverinfo: 'text',
						showlegend: false // No need for a legend here
					};

					// Custom axis labels: tickvals (numeric + mapped string) and ticktext (display string/number)
					function getAxisConfig(stringMapping, rawValues, minValue, isNumeric) {
						var tickvals = [];
						var ticktext = [];

						// Handle string values (always show all strings)
						Object.entries(stringMapping).forEach(([key, mappedValue]) => {
							tickvals.push(mappedValue);
							ticktext.push(key);
						});

						// Handle numeric values (only reduce ticks for numeric values)
						if (isNumeric) {
							rawValues.forEach(val => {
								var parsed = parseFloat(val);
								if (!isNaN(parsed)) {
									tickvals.push(parsed);
									ticktext.push(String(parsed));
								}
							});
						}

						return { tickvals, ticktext };
					}

					var xAxisConfig = getAxisConfig(stringMappingX, xValuesRaw, minXValue, !isNaN(minXValue));
					var yAxisConfig = getAxisConfig(stringMappingY, yValuesRaw, minYValue, !isNaN(minYValue));
					var zAxisConfig = getAxisConfig(stringMappingZ, zValuesRaw, minZValue, !isNaN(minZValue));

					// Layout for 3D scatter plot
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
						paper_bgcolor: 'rgba(0,0,0,0)',
						plot_bgcolor: 'rgba(0,0,0,0)',
						showlegend: false,
						legend: {
							x: 0.1,
							y: 1.1,
							orientation: 'h'
						}
					};

					// Create a new div for the plot
					var new_plot_div = $(`<div class='share_graph scatter-plot' id='scatter-plot-3d-${x_name}_${y_name}_${z_name}' style='width:${get_width()}px;height:${get_height()}px;'></div>`);
					if($('#scatter_plot_3d_container').length) {
						$('#scatter_plot_3d_container').append(new_plot_div);

						// Plot the 3D scatter plot using Plotly
						Plotly.newPlot(`scatter-plot-3d-${x_name}_${y_name}_${z_name}`, [trace3d], layout3d);

						// Add the current key to the list of already existing plots
						already_existing_plots.push(_key);
					} else {
						error("Cannot find #scatter_plot_3d_container");
					}
				}
			}
		}
	}

	$('#scatter_plot_3d_container').data("md5", data_md5);
}

function scatter(_paramKeys, _results_csv_json, minResult, maxResult, resultValues, mappingKeyNameToIndex) {
	//debug_function("scatter()");
	var already_existing_plots = [];
	var data_md5 = md5(JSON.stringify(_results_csv_json));

	if($('#scatter_plot_2d_container').data("md5") == data_md5) {
		return;
	}

	$('#scatter_plot_2d_container').html("");

	// Function to map string values to unique negative numbers starting just below the minimum numeric value
	function mapStrings(values, minNumericValue) {
		var uniqueStrings = [...new Set(values.filter(v => isNaN(parseFloat(v))))];
		uniqueStrings.sort(); // Alphabetically sort the strings
		var stringMapping = {};
		// Start string values just below the minimum numeric value
		var baseNegativeValue = minNumericValue - uniqueStrings.length - 1;
		uniqueStrings.forEach((str, idx) => {
			stringMapping[str] = baseNegativeValue - idx;
		});
		return stringMapping;
	}

	// Get minimum numeric value for each axis
	function getMinNumericValue(values) {
		return Math.min(...values.filter(v => !isNaN(parseFloat(v))));
	}

	// Function to reduce tick marks (only for numeric values)
	function reduceNumericTicks(tickvals, ticktext, maxTicks) {
		if (tickvals.length > maxTicks) {
			const step = Math.ceil(tickvals.length / maxTicks);
			tickvals = tickvals.filter((_, i) => i % step === 0);
			ticktext = ticktext.filter((_, i) => i % step === 0);
		}
		return { tickvals, ticktext };
	}

	for (var i = 0; i < _paramKeys.length; i++) {
		for (var j = i + 1; j < _paramKeys.length; j++) {
			if(!$("#scatter_plot_2d_container").length) {
				add_tab("scatter_plot_2d", "2d-Scatter-Plot", "<div id='scatter_plot_2d_container'></div>")
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

			// Map strings to negative values
			var stringMappingX = mapStrings(xValuesRaw, minXValue);
			var stringMappingY = mapStrings(yValuesRaw, minYValue);

			var xValues = [];
			var yValues = [];
			var hoverText = [];

			_results_csv_json.forEach(function(row) {
				// Handle x-axis
				var xParsed = parseFloat(row[map_x]);
				var xValue = isNaN(xParsed) ? stringMappingX[row[map_x]] : xParsed;
				xValues.push(xValue);

				// Handle y-axis
				var yParsed = parseFloat(row[map_y]);
				var yValue = isNaN(yParsed) ? stringMappingY[row[map_y]] : yParsed;
				yValues.push(yValue);

				// Hover text with the original values
				hoverText.push(`x: ${row[map_x]}, y: ${row[map_y]}`);
			});

			// Color function for markers
			function color_curried(value) {
				return getColor(value, minResult, maxResult);
			}
			var colors = resultValues.map(color_curried);

			// Create a custom colorscale from the unique values of resultValues and their corresponding colors
			var uniqueValues = Array.from(new Set(resultValues)).sort((a, b) => a - b);
			var customColorscale = uniqueValues.map(value => {
				return [(value - minResult) / (maxResult - minResult), color_curried(value)];
			});

			// Plotly trace for 2D scatter plot
			var trace2d = {
				x: xValues,
				y: yValues,
				mode: 'markers',
				type: 'scatter',
				marker: {
					color: colors
				},
				text: hoverText, // Show the original values in hover info
				hoverinfo: 'text'
			};

			// Dummy Trace for Color Legend with Custom Colorscale
			var colorScaleTrace = {
				x: [null], // Dummy data
				y: [null], // Dummy data
				type: 'scatter',
				mode: 'markers',
				marker: {
					color: [minResult, maxResult],
					colorscale: customColorscale,
					cmin: minResult,
					cmax: maxResult,
					showscale: true, // Show the color scale
					size: 60,
					colorbar: {
						title: 'Result Values',
						titleside: 'right'
					}
				},
				hoverinfo: 'none' // Hide hover info for this trace
			};

			// Custom axis labels: tickvals (numeric + mapped string) and ticktext (display string/number)
			function getAxisConfig(stringMapping, rawValues, minValue, isNumeric) {
				var tickvals = [];
				var ticktext = [];

				// Handle string values (always show all strings)
				Object.entries(stringMapping).forEach(([key, mappedValue]) => {
					tickvals.push(mappedValue);
					ticktext.push(key);
				});

				// Handle numeric values (only reduce ticks for numeric values)
				if (isNumeric) {
					rawValues.forEach(val => {
						var parsed = parseFloat(val);
						if (!isNaN(parsed)) {
							tickvals.push(parsed);
							ticktext.push(String(parsed));
						}
					});
					// Reduce tick count if too many numeric values
					return reduceNumericTicks(tickvals, ticktext, max_nr_ticks);
				}

				return { tickvals, ticktext };
			}

			var xAxisConfig = getAxisConfig(stringMappingX, xValuesRaw, minXValue, !isNaN(minXValue));
			var yAxisConfig = getAxisConfig(stringMappingY, yValuesRaw, minYValue, !isNaN(minYValue));

			// Layout for 2D scatter plot
			var layout2d = {
				title: `Scatter Plot: ${x_name} vs ${y_name}`,
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
				paper_bgcolor: 'rgba(0,0,0,0)',
				plot_bgcolor: 'rgba(0,0,0,0)',
				showlegend: false // We use the colorbar instead of a traditional legend
			};

			// Create a new div for the plot
			var new_plot_div = $(`<div class='share_graph scatter-plot' id='scatter-plot-${x_name}_${y_name}' style='width:${get_width()}px;height:${get_height()}px;'></div>`);
			$('#scatter_plot_2d_container').append(new_plot_div);

			if ($('#scatter_plot_2d_container').length) {
				// Plot the 2D scatter plot using Plotly
				Plotly.newPlot(`scatter-plot-${x_name}_${y_name}`, [trace2d, colorScaleTrace], layout2d);

				// Add the current key to the list of already existing plots
				already_existing_plots.push(_key);
			} else {
				error("Cannot find #scatter_plot_2d_container");
			}
		}
	}

	$('#scatter_plot_2d_container').data("md5", data_md5);
}


async function load_results () {
	//debug_function("load_results()");
	var data = await fetchJsonFromUrlFilenameOnly(`results.csv`);
	if(!data) {
		warn("load_results: Could not fetch results.csv");
		return;
	}

	if(!Object.keys(data).includes("raw")) {
		//warn(`load_results: Could not plot seemingly empty data: no raw found`);
		return;
	}

	add_tab("results", "Results", "<div id='results_csv'></div>");

	$("#results_csv").html(`<pre class="stdout_file invert_in_dark_mode autotable">${data.raw}</pre>${copy_button('stdout_file')}`);
}

function isFullyNumeric(values) {
	return values.every(value => !isNaN(parseFloat(value)) && isFinite(value));
}

async function plot_all_possible () {
	//debug_function("plot_all_possible()");

	var _results_csv_json = await fetchJsonFromUrlFilenameOnly(`results.csv`)

	if(!_results_csv_json) {
		return;
	}

	if(!Object.keys(_results_csv_json).includes("data")) {
		warn(`plot_all_possible: Could not plot seemingly empty _results_csv_json: no data found`);
		return;
	}

	if(!Object.keys(_results_csv_json).includes("data") && !results_csv_json.data.length) {
		warn(`plot_all_possible: Could not plot seemingly empty _results_csv_json`);
		return;
	}

	convertToIntAndFilter(_results_csv_json.data.map(Object.values))

	var header_line = _results_csv_json.data.shift();

	var mappingKeyNameToIndex = {};
	var paramKeys = [];

	for (var i = 0; i < header_line.length; i++) {
		var this_element = header_line[i];

		if(!['trial_index', 'arm_name', 'trial_status', 'generation_method', 'result'].includes(this_element)) {
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

	scatter(paramKeys, _results_csv_json.data, minResult, maxResult, resultValues, mappingKeyNameToIndex);
	scatter_3d(paramKeys, _results_csv_json.data, minResult, maxResult, resultValues, mappingKeyNameToIndex);

	apply_theme_based_on_system_preferences();
}

function convertUnixTimeToReadable(unixTime) {
	var date = new Date(unixTime * 1000);
	return date.toLocaleString();
}

async function load_parameter () {
	//debug_function("load_parameter()");
	var data = await fetchJsonFromUrlFilenameOnly(`parameters.txt`, 1)
	if(!data) {
		return;
	}

	if(!Object.keys(data).includes("raw")) {
		warn(`load_parameter: Could not plot seemingly empty data: no raw found`);
		return;
	}

	if (data.raw != "null" && data.raw !== null) {
		$(".parameters_txt").html(`<pre>${removeAnsiCodes(data.raw)}</pre>`);
	}
}

async function load_out_files () {
	//debug_function("load_out_files()");
	var urlParams = new URLSearchParams(window.location.search);

	var data = await fetchJsonFromUrl(`get_out_files.php?user_id=${urlParams.get('user_id')}&experiment_name=${urlParams.get('experiment_name')}&run_nr=${urlParams.get('run_nr')}`)
	if(!data) {
		return;
	}
	
	if(!Object.keys(data).includes("raw")) {
		warn(`load_out_files: Could not plot seemingly empty data: no raw found`);
		return;
	}

	if(data.raw) {
		add_tab("out_files", "Out-Files", "<div id='out_files_content'></div>");
		//log(data.raw);
		$("#out_files_content").html(data.raw);

		$("#out_files_tabs").tabs();

		convert_ansi_to_html();
	}
}

async function load_evaluation_errors_and_oo_errors () {
	//debug_function("load_evaluation_errors_and_oo_errors()");
	var p = [];
	p.push(_load_evaluation_errors_and_oo_errors("oo_errors", "Evaluation Errors", "oo_errors.txt", "oo_errors"));

	for (var i = 0; i < p.length; i++) {
		await p[i];
	}
}

async function _load_evaluation_errors_and_oo_errors (tab_div, title, _fn, _divname) {
	//debug_function("_load_evaluation_errors_and_oo_errors()");
	var data = await fetchJsonFromUrlFilenameOnly(`${_fn}`)
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
	//debug_function("load_progressbar_log()");
	var data = await fetchJsonFromUrlFilenameOnly(`progressbar`)
	if(!data) {
		return;
	}

	if(!Object.keys(data).includes("raw")) {
		//warn(`load_progressbar_log: Could not plot seemingly empty data: no raw found`);
		return;
	}

	add_tab("progressbar_log", "Progressbar-Log", `<div id='progressbar_log_element'></div>`);

	if($(`#progressbar_log_element`).length == 0) {
		error(`Could not find #progressbar_log_element`);
	} else {
		var converted = ansi_to_html(removeLinesStartingWith(data.raw, "P7;1;75", "-$$$$$-$$$$$"));
		const removeTrailingWhitespaces = (str) => str.split('\n').map(line => line.replace(/\s+$/, '')).join('\n');
		converted = removeTrailingWhitespaces(converted);
		$(`#progressbar_log_element`).html(`<pre class="progressbar_log_class" class='invert_in_dark_mode' style='color: white; background-color: black; white-space: break-spaces;'>${converted}</pre>${copy_button("progressbar_log_class")}`);
	}
}

async function load_trial_index_to_params_log () {
	//debug_function("load_trial_index_to_params_log()");
	var data = await fetchJsonFromUrlFilenameOnly(`trial_index_to_params`)
	if(!data) {
		return;
	}

	if(!Object.keys(data).includes("raw")) {
		//warn(`load_trial_index_to_params_log: Could not plot seemingly empty data: no raw found`);
		return;
	}

	add_tab("trial_index_to_params", "Trial Index to Param", `<div id='trial_index_to_params_element'></div>`);

	if($(`#trial_index_to_params_element`).length == 0) {
		error(`Could not find #trial_index_to_params_element`);
	} else {
		var converted = ansi_to_html(removeLinesStartingWith(data.raw, "P7;1;75", "-$$$$$-$$$$$"));
		const removeTrailingWhitespaces = (str) => str.split('\n').map(line => line.replace(/\s+$/, '')).join('\n');
		converted = removeTrailingWhitespaces(converted);
		$(`#trial_index_to_params_element`).html(`<pre class="trial_index_to_params_element_class" class='invert_in_dark_mode' style='color: white; background-color: black; white-space: break-spaces;'>${converted}</pre>${copy_button("trial_index_to_params_element_class")}`);
	}
}

async function load_install_errors() {
	//debug_function("load_install_errors()");
	var data = await fetchJsonFromUrlFilenameOnly(`install_errors`)
	if(!data) {
		return;
	}

	if(!Object.keys(data).includes("raw")) {
		//warn(`load_install_errors: Could not plot seemingly empty data: no raw found`);
		return;
	}

	add_tab("install_errors", "Install errors", `<div id='install_errors_element'></div>`);

	if($(`#install_errors_element`).length == 0) {
		error(`Could not find #install_errors_element`);
	} else {
		var converted = ansi_to_html(removeLinesStartingWith(data.raw, "P7;1;75", "-$$$$$-$$$$$"));
		const removeTrailingWhitespaces = (str) => str.split('\n').map(line => line.replace(/\s+$/, '')).join('\n');
		converted = removeTrailingWhitespaces(converted);
		$(`#install_errors_element`).html(`<pre class="install_errors_class" class='invert_in_dark_mode' style='color: white; background-color: black; white-space: break-spaces;'>${converted}</pre>${copy_button("install_errors_class")}`);
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
	});
	table.appendChild(headerRow);

	data.forEach(item => {
		const row = document.createElement("tr");
		const timeCell = document.createElement("td");
		const msgCell = document.createElement("td");
		const stackCell = document.createElement("td");
		stackCell.classList.add("stacktrace");

		timeCell.innerText = item.time;
		msgCell.innerText = item.msg;

		if(Object.keys(item).includes("function_stack")) {
			const formattedStack = item.function_stack
				.map(func => `${func.function} (Line ${func.line_number})`)
				.join('\n');

			stackCell.innerText = formattedStack;

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
	const lines = logData.split('\n');
	const jsonData = [];

	lines.forEach((line, index) => {
		if (line.trim() === "") return;
		try {
			const jsonObject = JSON.parse(line);
			jsonData.push(jsonObject);
		} catch (error) {
		}
	});

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
	//debug_function("load_debug_log()");
	var data = await fetchJsonFromUrlFilenameOnly(`log`)
	if(!data) {
		return;
	}

	if(!Object.keys(data).includes("raw")) {
		//warn(`load_debug_log: Could not plot seemingly empty data: no raw found`);
		return;
	}

	add_tab("internal_log", "Debug-Log", `<div id='internal_log_element'></div>`);

	if($(`#internal_log_element`).length == 0) {
		error(`Could not find #internal_log_element`);
	} else {
		var converted = ansi_to_html(removeLinesStartingWith(data.raw, "P7;1;75", "-$$$$$-$$$$$"));
		const removeTrailingWhitespaces = (str) => str.split('\n').map(line => line.replace(/\s+$/, '')).join('\n');
		converted = removeTrailingWhitespaces(converted);
		$(`#internal_log_element`).html(`<pre style="display: none" class="internal_log_element_class" class='invert_in_dark_mode' style='color: white; background-color: black; white-space: break-spaces;'>${converted}</pre>${copy_button("internal_log_element_class")}<div id="internal_log_table"></div>`);

		var id = 'internal_log_table';

		var parsed_log_data = parseLogData(data.raw);

		console.log(parsed_log_data);

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
	//debug_function("load_outfile()");
	var data = await fetchJsonFromUrlFilenameOnly(`outfile`)
	if(!data) {
		return;
	}

	if(!Object.keys(data).includes("raw")) {
		//warn(`load_outfile: Could not plot seemingly empty data: no raw found`);
		return;
	}

	add_tab("outfile", "Main Log", `<div id='outfile'></div>`);

	if($(`#outfile`).length == 0) {
		error(`Could not find #outfile`);
	} else {
		var converted = ansi_to_html(removeLinesStartingWith(data.raw, "P7;1;75", "-$$$$$-$$$$$"));
		const removeTrailingWhitespaces = (str) => str.split('\n').map(line => line.replace(/\s+$/, '')).join('\n');
		converted = removeTrailingWhitespaces(converted);
		$(`#outfile`).html(`<pre class="main_outfile" class='invert_in_dark_mode' style='color: white; background-color: black; white-space: break-spaces;'>${converted}</pre>${copy_button("main_outfile")}`);
	}
}

async function load_next_trials () {
	//debug_function("load_next_trials()");
	var urlParams = new URLSearchParams(window.location.search);

	var data = await fetchJsonFromUrl(`get_next_trials.php?user_id=${urlParams.get('user_id')}&experiment_name=${urlParams.get('experiment_name')}&run_nr=${urlParams.get('run_nr')}`)
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
	//debug_function("load_job_infos()");
	var data = await fetchJsonFromUrlFilenameOnly(`job_infos.csv`)
	if(!data) {
		return;
	}

	if(!Object.keys(data).includes("raw")) {
		//warn(`load_job_infos: Could not plot seemingly empty data: no raw found`);
		return;
	}

	add_tab("job_infos", "Job-Infos", "<div id='job_infos_csv'></div>");
	$("#job_infos_csv").html(`<pre class="stdout_file invert_in_dark_mode autotable">${data.raw}</pre>${copy_button('stdout_file')}`);
}

async function load_best_result () {
	//debug_function("load_best_result()");
	var data = await fetchJsonFromUrlFilenameOnly(`best_result.txt`)
	if(!data) {
		return;
	}

	if(!Object.keys(data).includes("raw")) {
		//warn(`load_best_result: Could not plot seemingly empty data: no raw found`);
		return;
	}

	if (data.raw != "null" && data.raw !== null) {
		$(".best_result_txt").html(`<pre>${removeAnsiCodes(data.raw)}</pre>`);
	}
}

async function plot_planned_vs_real_worker_over_time () {
	//debug_function("plot_planned_vs_real_worker_over_time()");
	var data = await fetchJsonFromUrlFilenameOnly(`worker_usage.csv`)
	if(!data) {
		return;
	}

	if(!Object.keys(data).includes("data")) {
		//warn(`plot_planned_vs_real_worker_over_time: Could not plot seemingly empty data: no data found`);
		return;
	}

	if(!data.data.length) {
		warn(`plot_planned_vs_real_worker_over_time: Could not plot seemingly empty data`);
		return;
	}

	convertToIntAndFilter(data.data.map(Object.values))

	replaceZeroWithNull(data.data);

	var unixTime = data.data.map(row => row[0]);
	var readableTime = unixTime.map(convertUnixTimeToReadable);
	var plannedWorkers = data.data.map(row => row[1]);
	var actualWorkers = data.data.map(row => row[2]);

	var tracePlanned = {
		x: readableTime,
		y: plannedWorkers,
		mode: 'lines',
		name: 'Planned Worker'
	};

	var traceActual = {
		x: readableTime,
		y: actualWorkers,
		mode: 'lines',
		name: 'Real Worker'
	};

	var layout = {
		title: 'Planned vs. real worker over time',
		xaxis: {
			title: 'Date'
		},
		yaxis: {
			title: 'Nr. Worker'
		},
		width: get_width(),
		height: get_height(),
		paper_bgcolor: 'rgba(0,0,0,0)',
		plot_bgcolor: 'rgba(0,0,0,0)',

		showlegend: true,
		legend: {
			x: 0.1,
			y: 1.1,
			orientation: 'h'
		},
	};

	add_tab("worker_usage", "Worker-Usage", "<div id='worker_usage_plot'></div><div id='worker_usage_raw'></div>");
	Plotly.newPlot('worker_usage_plot', [tracePlanned, traceActual], layout);

	$("#worker_usage_raw").html(`<pre class="stdout_file invert_in_dark_mode autotable">${data.raw}</pre>${copy_button("stdout_file")}`);
}

async function plot_cpu_ram_graph() {
    //debug_function("plot_cpu_ram_graph()");
    var cpu_ram_usage_json = await fetchJsonFromUrlFilenameOnly(`cpu_ram_usage.csv`);
    if (!cpu_ram_usage_json) {
        return;
    }

    if (!Object.keys(cpu_ram_usage_json).includes("data")) {
        warn(`plot_cpu_ram_graph: Could not plot seemingly empty cpu_ram_usage_json: no data found`);
        return;
    }

    if (!Object.keys(cpu_ram_usage_json).includes("data") || !cpu_ram_usage_json.data.length) {
        warn(`plot_cpu_ram_graph: Could not plot seemingly empty cpu_ram_usage_json`);
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
        type: 'scatter',
        mode: 'lines',
        name: 'RAM Usage (MB)',
        line: { color: 'lightblue' }
    };

    // CPU Usage Plot
    const cpuTrace = {
        x: timestamps_cpu,
        y: cpuUsage,
        type: 'scatter',
        mode: 'lines',
        name: 'CPU Usage (%)',
        line: { color: 'orange' }
    };

    const ramLayout = {
        title: 'RAM Usage Over Time by the main worker',
        xaxis: {
            title: 'Time',
            type: 'date'
        },
        yaxis: {
            title: 'RAM Usage (MB)',
            showline: true
        },
        showlegend: true,
        legend: {
            x: 0.1,
            y: 1.1,
            orientation: 'h'
        },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)'
    };

    const cpuLayout = {
        title: 'CPU Usage Over Time by the main worker',
        xaxis: {
            title: 'Time',
            type: 'date'
        },
        yaxis: {
            title: 'CPU Usage (%)',
            showline: true
        },
        showlegend: true,
        legend: {
            x: 0.1,
            y: 1.1,
            orientation: 'h'
        },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)'
    };

    add_tab("cpu_ram_usage", "CPU/RAM Usage", `
        <div id='cpuRamChartContainer'>
            <div id='ramChart'></div>
            <div id='cpuChart'></div>
            <div id='cpuRamChartRawData'></div>
        </div>
    `);

    if ($("#ramChart").length) {
        Plotly.newPlot('ramChart', [ramTrace], ramLayout);
    }

    if ($("#cpuChart").length) {
        Plotly.newPlot('cpuChart', [cpuTrace], cpuLayout);
    }

    $("#cpuRamChartRawData").html(`<pre class="stdout_file invert_in_dark_mode autotable">${cpu_ram_usage_json.raw}</pre>${copy_button("stdout_file")}`);
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
};

async function fetchJsonFromUrl(url) {
	//debug_function(`fetchJsonFromUrl("${url}")`);
	try {
		const response = await fetch(url);
		if (!response.ok) {
			throw new Error('Network response was not ok: ' + response.statusText);
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

	var _res = await fetchJsonFromUrl(`get_overview_data.php?user_id=${urlParams.get('user_id')}&experiment_name=${urlParams.get('experiment_name')}&run_nr=${urlParams.get('run_nr')}`)

	return _res;
}

async function load_overview_data() {
	//debug_function("load_overview_data()");

	add_tab("overview_data", "Overview", "<div class='best_result_txt'></div><div class='parameters_txt'></div><div class='overview_table'></div>");

	var res = await _get_overview_data();

	//log(res);
	if(!Object.keys(res).includes("error")) {
		// Create a table
		var table = document.createElement('table');
		table.style.width = '100%';
		table.style.borderCollapse = 'collapse';

		// Create table headers
		var headerRow = document.createElement('tr');
		['Failed', 'Succeeded', 'Total'].forEach(function (heading) {
			var th = document.createElement('th');
			th.style.border = '1px solid black';
			th.style.padding = '8px';
			th.classList.add("invert_in_dark_mode");
			th.textContent = heading;
			headerRow.appendChild(th);
		});
		table.appendChild(headerRow);

		// Create a data row
		var dataRow = document.createElement('tr');
		[res.failed, res.succeeded, res.total].forEach(function (value) {
			var td = document.createElement('td');
			td.style.border = '1px solid black';
			td.style.padding = '8px';
			td.classList.add("invert_in_dark_mode");
			td.textContent = value;
			dataRow.appendChild(td);
		});
		table.appendChild(dataRow);

		// Insert table into the #overview_table element
		$('.overview_table').html(table);
	} else {
		$('.overview_table').html(`Error: <span class="error_line">${res.error}</span>`);
	}
}

async function fetchJsonFromUrlFilenameOnly(filename, remove_ansi=false) {
	//debug_function(`fetchJsonFromUrlFilenameOnly('${filename}')`);
	var urlParams = new URLSearchParams(window.location.search);

	var url = `share_to_csv.php?user_id=${urlParams.get('user_id')}&experiment_name=${urlParams.get('experiment_name')}&run_nr=${urlParams.get('run_nr')}&filename=${filename}`;

	if(remove_ansi) {
		url = url + "&remove_ansi=1";
	}

	var _res = await fetchJsonFromUrl(url);

	return _res;
}

function convert_ansi_to_html () {
	$(".convert_ansi_to_html").each(function (i, e) {
		var html = e.innerHTML;

		var res = ansi_to_html(parseAnsiToVirtualTerminal(html));

		e.innerHTML = res;
	});
}

async function load_all_data() {
	//debug_function("load_all_data()");
	var urlParams = new URLSearchParams(window.location.search);

	if(urlParams.get("user_id") && urlParams.get("experiment_name") && !isNaN(parseInt(urlParams.get("run_nr")))) {
		showSpinnerOverlay("Loading data...")

		var promises = [];

		promises.push(load_overview_data());
		promises.push(load_evaluation_errors_and_oo_errors());
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

		promises.push(plot_all_possible());
		promises.push(plot_cpu_ram_graph());
		promises.push(plot_parallel_plot());
		promises.push(plot_planned_vs_real_worker_over_time());

		for (var i = 0; i < promises.length; i++) {
			await promises[i];
		}

		await load_out_files();

		initialize_autotables();

		removeSpinnerOverlay();

		//log("Loaded page");
	}
}

function copy_button (name_to_search_for) {
	if(!name_to_search_for) {
		error("Empty name_to_search_for in copy_button");
		console.trace();
		return "";
	}

	return `<button class='copy_to_clipboard_button invert_in_dark_mode' onclick='find_closest_element_behind_and_copy_content_to_clipboard(this, "${name_to_search_for}")'>📋 Copy raw data to clipboard</button>`;
}
