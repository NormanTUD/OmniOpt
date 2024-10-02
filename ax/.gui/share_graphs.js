var hashcache = [];

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

function parallel_plot(paramKeys, _results_csv_json, minResult, maxResult, resultValues, mappingKeyNameToIndex) {
	var dimensions = [...paramKeys, 'result'].map(function(key) {
		var idx = mappingKeyNameToIndex[key];

		var values = _results_csv_json.map(function(row) {
			return row[idx];
		});

		values = values.filter(value => value !== undefined && !isNaN(value));

		var numericValues = values.map(function(value) { return parseFloat(value); });
		numericValues = numericValues.filter(value => value !== undefined && !isNaN(value));

		if (numericValues.every(isNumeric)) {
			return {
				range: [Math.min(...numericValues), Math.max(...numericValues)],
				label: key,
				values: numericValues
			};
		} else {
			var uniqueValues = getUniqueValues(values);
			var valueIndices = values.map(function(value) { return uniqueValues.indexOf(value); });
			return {
				range: [0, uniqueValues.length - 1],
				label: key,
				values: valueIndices
			};
		}
	});

	var traceParallel = {
		type: 'parcoords',
		line: {
			color: resultValues,
			colorscale: 'Jet',
			showscale: true,
			cmin: minResult,
			cmax: maxResult
		},
		unselected: {
			line: {
				opacity: 0
			}
		},
		dimensions: dimensions
	};

	var layoutParallel = {
		title: 'Parallel Coordinates Plot',
		width: get_width(),
		height: get_height(),
		paper_bgcolor: 'rgba(0,0,0,0)',
		plot_bgcolor: 'rgba(0,0,0,0)',
		showlegend: false
	};

	var new_plot_div = $(`<div class='share_graph parallel-plot' id='parallel-plot' style='width:${get_width()}px;height:${get_height()}px;'></div>`);
	$('#parallel_plot_container').html(new_plot_div);

	Plotly.newPlot('parallel-plot', [traceParallel], layoutParallel);
}

function scatter_3d (_paramKeys, _results_csv_json, minResult, maxResult, resultValues, mappingKeyNameToIndex) {
	var already_existing_plots = [];
	$('#scatter_plot_3d_container').html("");

	if (_paramKeys.length >= 3 && _paramKeys.length <= 6) {
		for (var i = 0; i < _paramKeys.length; i++) {
			for (var j = i + 1; j < _paramKeys.length; j++) {
				for (var k = j + 1; k < _paramKeys.length; k++) {
					var map_x = mappingKeyNameToIndex[_paramKeys[i]];
					var map_y = mappingKeyNameToIndex[_paramKeys[j]];
					var map_z = mappingKeyNameToIndex[_paramKeys[k]];

					var x_name = _paramKeys[i];
					var y_name = _paramKeys[j];
					var z_name = _paramKeys[k];

					var _key = [x_name, y_name, z_name].sort().join("!!!");
					if (already_existing_plots.includes(_key)) {
						log(`Key already exists: ${_key}`);
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
							color: colors
						},
						text: hoverText, // Show the original values in hover info
						hoverinfo: 'text'
					};

					log(trace3d);

					// Custom axis labels: tickvals (numeric + mapped string) and ticktext (display string/number)
					function getAxisConfig(stringMapping, rawValues, minValue) {
						var tickvals = [];
						var ticktext = [];
						
						// Handle numeric values
						rawValues.forEach(val => {
							var parsed = parseFloat(val);
							if (!isNaN(parsed)) {
								if (!tickvals.includes(parsed)) {
									tickvals.push(parsed);
									ticktext.push(String(parsed));
								}
							}
						});

						// Handle string values
						Object.entries(stringMapping).forEach(([key, mappedValue]) => {
							tickvals.push(mappedValue);
							ticktext.push(key);
						});

						return { tickvals, ticktext };
					}

					var xAxisConfig = getAxisConfig(stringMappingX, xValuesRaw, minXValue);
					var yAxisConfig = getAxisConfig(stringMappingY, yValuesRaw, minYValue);
					var zAxisConfig = getAxisConfig(stringMappingZ, zValuesRaw, minZValue);

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
					$('#scatter_plot_3d_container').append(new_plot_div);

					// Plot the 3D scatter plot using Plotly
					Plotly.newPlot(`scatter-plot-3d-${x_name}_${y_name}_${z_name}`, [trace3d], layout3d);

					// Add the current key to the list of already existing plots
					already_existing_plots.push(_key);
				}
			}
		}
	}
}

function scatter(_paramKeys, _results_csv_json, minResult, maxResult, resultValues, mappingKeyNameToIndex) {
	var already_existing_plots = [];

	$('#scatter_plot_2d_container').html("");

	for (var i = 0; i < _paramKeys.length; i++) {
		for (var j = i + 1; j < _paramKeys.length; j++) {
			var map_x = mappingKeyNameToIndex[_paramKeys[i]];
			var map_y = mappingKeyNameToIndex[_paramKeys[j]];

			var x_name = _paramKeys[i];
			var y_name = _paramKeys[j];

			var _key = [x_name, y_name].sort().join("!!!");

			if(already_existing_plots.includes(_key)) {
				log(`Key already exists: ${_key}`);
				continue;
			}
			var xValues = _results_csv_json.map(function(row) {
				var parsedValue = parseFloat(row[map_x]);
				return isNaN(parsedValue) ? row[map_x] : parsedValue;
			});

			var yValues = _results_csv_json.map(function(row) {
				var parsedValue = parseFloat(row[map_y]);
				return isNaN(parsedValue) ? row[map_y] : parsedValue;
			});

			function color_curried(value) {
				return getColor(value, minResult, maxResult);
			}

			var colors = resultValues.map(color_curried);

			// Create a custom colorscale from the unique values of resultValues and their corresponding colors
			var uniqueValues = Array.from(new Set(resultValues)).sort((a, b) => a - b);
			var customColorscale = uniqueValues.map(value => {
				return [(value - minResult) / (maxResult - minResult), color_curried(value)];
			});

			var trace2d = {
				x: xValues,
				y: yValues,
				mode: 'markers',
				type: 'scatter',
				marker: {
					color: colors
				}
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

			var layout2d = {
				title: `Scatter Plot: ${x_name} vs ${y_name}`,
				xaxis: { title: x_name },
				yaxis: { title: y_name },
				paper_bgcolor: 'rgba(0,0,0,0)',
				plot_bgcolor: 'rgba(0,0,0,0)',
				showlegend: false // We use the colorbar instead of a traditional legend
			};

			var new_plot_div = $(`<div class='share_graph scatter-plot' id='scatter-plot-${x_name}_${y_name}' style='width:${get_width()}px;height:${get_height()}px;'></div>`);
			$('#scatter_plot_2d_container').append(new_plot_div);
			Plotly.newPlot(`scatter-plot-${x_name}_${y_name}`, [trace2d, colorScaleTrace], layout2d);
			already_existing_plots.push(_key);
		}
	}
}

async function plot_parallel_plot () {
	var _results_csv_json = await fetchJsonFromUrlFilenameOnly(`job_infos.csv`)
	if(!_results_csv_json) {
		return;
	}

	convertToIntAndFilter(_results_csv_json.data.map(Object.values))

	replaceZeroWithNull(_results_csv_json.data);

	if(!Object.keys(_results_csv_json).includes("data")) {
		log(`plot_parallel_plot: Could not plot seemingly empty _results_csv_json: no data found`);
		return;
	}
	
	if(!_results_csv_json.data.length) {
		log(`plot_parallel_plot: Could not plot seemingly empty _results_csv_json`);
		return;
	}

	var header_line = _results_csv_json.data.shift();

	var mappingKeyNameToIndex = {};

	for (var i = 0; i < header_line.length; i++) {
		mappingKeyNameToIndex[header_line[i]] = i;
	}

	// Extract parameter names
	var paramKeys = header_line.filter(function(key) {
		return ![
			'trial_index',
			'arm_name',
			'run_time',
			'trial_status',
			'generation_method',
			'result',
			'start_time',
			'end_time',
			'program_string',
			'hostname',
			'signal',
			'exit_code'
		].includes(key);
	});

	var result_idx = header_line.indexOf("result");

	// Get result values for color mapping
	var resultValues = _results_csv_json.data.map(function(row) {
		return parseFloat(row[result_idx]);
	});

	resultValues = resultValues.filter(value => value !== undefined && !isNaN(value));

	var minResult = Math.min.apply(null, resultValues);
	var maxResult = Math.max.apply(null, resultValues);

	parallel_plot(paramKeys, _results_csv_json.data, minResult, maxResult, resultValues, mappingKeyNameToIndex);

	apply_theme_based_on_system_preferences();

	$("#out_files_tabs").tabs();
}

async function load_results () {
	var data = await fetchJsonFromUrlFilenameOnly(`results.csv`);
	if(!data) {
		log("load_results: Could not fetch results.csv");
		return;
	}

	if(!Object.keys(data).includes("raw")) {
		log(`load_results: Could not plot seemingly empty data: no raw found`);
		return;
	}


	$("#results_csv").html(`<pre class="stdout_file invert_in_dark_mode autotable">${data.raw}</pre>${copy_button('stdout_file')}`);
}

async function plot_all_possible () {
	var _results_csv_json = await fetchJsonFromUrlFilenameOnly(`results.csv`)

	if(!_results_csv_json) {
		return;
	}

	convertToIntAndFilter(_results_csv_json.data.map(Object.values))

	if(!Object.keys(_results_csv_json).includes("data")) {
		log(`plot_all_possible: Could not plot seemingly empty _results_csv_json: no data found`);
		return;
	}

	if(!_results_csv_json.data.length) {
		log(`plot_all_possible: Could not plot seemingly empty _results_csv_json`);
		return;
	}

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
		console.error("Cannot find result column index!");
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

	$("#out_files_tabs").tabs();
}

function convertUnixTimeToReadable(unixTime) {
	var date = new Date(unixTime * 1000);
	return date.toLocaleString();
}

async function load_parameter () {
	var data = await fetchJsonFromUrlFilenameOnly(`parameters.txt`)
	if(!data) {
		return;
	}

	if(!Object.keys(data).includes("raw")) {
		log(`load_parameter: Could not plot seemingly empty data: no raw found`);
		return;
	}


	$("#parameters_txt").html(`<pre>${data.raw}</pre>`);
}

async function load_out_files () {
	var urlParams = new URLSearchParams(window.location.search);

	var data = await fetchJsonFromUrl(`get_out_files.php?user_id=${urlParams.get('user_id')}&experiment_name=${urlParams.get('experiment_name')}&run_nr=${urlParams.get('run_nr')}`)
	if(!data) {
		return;
	}

	$("#out_files_content").html(data.raw);
}

async function load_evaluation_errors_and_oo_errors () {
	var p = [];
	p.push(_load_evaluation_errors_and_oo_errors("evaluation_errors.log", "evaluation_errors"));
	p.push(_load_evaluation_errors_and_oo_errors("oo_errors.txt", "oo_errors"));

	for (var i = 0; i < p.length; i++) {
		await p[i];
	}
}

async function _load_evaluation_errors_and_oo_errors (_fn, _divname) {
	var data = await fetchJsonFromUrlFilenameOnly(`${_fn}`)
	if(!data) {
		return;
	}

	if(!Object.keys(data).includes("raw")) {
		log(`_load_evaluation_errors_and_oo_errors: Could not plot seemingly empty data: no raw found`);
		return;
	}


	if($(`#${_divname}`).length == 0) {
		console.error(`Could not find #${_divname}`);
	} else {
		$(`#${_divname}`).html(`<pre>${data.raw}</pre>`);
	}
}

async function load_next_trials () {
	var urlParams = new URLSearchParams(window.location.search);

	var data = await fetchJsonFromUrl(`get_next_trials.php?user_id=${urlParams.get('user_id')}&experiment_name=${urlParams.get('experiment_name')}&run_nr=${urlParams.get('run_nr')}`)
	if(!data) {
		return;
	}

	if(!Object.keys(data).includes("raw")) {
		log(`load_next_trials: Could not plot seemingly empty data: no raw found`);
		return;
	}

	$("#next_trials_csv").html(`${data.raw}`);
}

async function load_job_infos () {
	var data = await fetchJsonFromUrlFilenameOnly(`job_infos.csv`)
	if(!data) {
		return;
	}

	if(!Object.keys(data).includes("raw")) {
		log(`load_job_infos: Could not plot seemingly empty data: no raw found`);
		return;
	}

	$("#job_infos_csv").html(`<pre class="stdout_file invert_in_dark_mode autotable">${data.raw}</pre>`);
}

async function load_best_result () {
	var data = await fetchJsonFromUrlFilenameOnly(`best_result.txt`)
	if(!data) {
		return;
	}

	if(!Object.keys(data).includes("raw")) {
		log(`load_best_result: Could not plot seemingly empty data: no raw found`);
		return;
	}


	$("#best_result_txt").html(`<pre>${data.raw}</pre>`);
}

async function plot_planned_vs_real_worker_over_time () {
	var data = await fetchJsonFromUrlFilenameOnly(`worker_usage.csv`)
	if(!data) {
		return;
	}

	if(!Object.keys(data).includes("data")) {
		log(`plot_planned_vs_real_worker_over_time: Could not plot seemingly empty data: no data found`);
		return;
	}

	if(!data.data.length) {
		log(`plot_planned_vs_real_worker_over_time: Could not plot seemingly empty data`);
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

	Plotly.newPlot('worker_usage_plot', [tracePlanned, traceActual], layout);

	initialize_autotables();
}

async function plot_cpu_gpu_graph() {
	var cpu_ram_usage_json = await fetchJsonFromUrlFilenameOnly(`cpu_ram_usage.csv`)
	if(!cpu_ram_usage_json) {
		return;
	}

	convertToIntAndFilter(cpu_ram_usage_json.data.map(Object.values))

	replaceZeroWithNull(cpu_ram_usage_json.data);

	if(!Object.keys(cpu_ram_usage_json).includes("data")) {
		log(`plot_cpu_gpu_graph: Could not plot seemingly empty cpu_ram_usage_json: no data found`);
		return;
	}
	
	if(!cpu_ram_usage_json.data.length) {
		log(`plot_cpu_gpu_graph: Could not plot seemingly empty cpu_ram_usage_json`);
		return;
	}

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
		line: { color: 'lightblue' },
		yaxis: 'y1'
	};

	// CPU Usage Plot
	const cpuTrace = {
		x: timestamps_cpu,
		y: cpuUsage,
		type: 'scatter',
		mode: 'lines',
		name: 'CPU Usage (%)',
		line: { color: 'orange' },
		yaxis: 'y2'
	};

	const layout = {
		title: 'CPU and RAM Usage Over Time',
		xaxis: {
			title: 'Time',
			type: 'date'
		},
		yaxis: {
			title: 'RAM Usage (MB)',
			showline: true,
			side: 'left'
		},
		yaxis2: {
			title: 'CPU Usage (%)',
			overlaying: 'y',
			side: 'right',
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

	const data = [ramTrace, cpuTrace];

	Plotly.newPlot('cpuRamChart', data, layout);
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

async function fetchJsonFromUrlFilenameOnly(filename) {
	var urlParams = new URLSearchParams(window.location.search);

	var _res = await fetchJsonFromUrl(`share_to_csv.php?user_id=${urlParams.get('user_id')}&experiment_name=${urlParams.get('experiment_name')}&run_nr=${urlParams.get('run_nr')}&filename=${filename}`)

	return _res;
}

async function load_all_data() {
	showSpinnerOverlay("Loading data...")

	var promises = [];

	promises.push(plot_all_possible());
	promises.push(plot_cpu_gpu_graph());
	promises.push(plot_parallel_plot());
	promises.push(plot_planned_vs_real_worker_over_time());

	promises.push(load_evaluation_errors_and_oo_errors());
	promises.push(load_out_files());
	promises.push(load_best_result());
	promises.push(load_job_infos());
	promises.push(load_next_trials());
	promises.push(load_results());
	promises.push(load_parameter());

	for (var i = 0; i < promises.length; i++) {
		await promises[i];
	}

	initialize_autotables();

	removeSpinnerOverlay();
}

function copy_button (name_to_search_for) {
	if(!name_to_search_for) {
		console.error("Empty name_to_search_for in copy_button");
		console.trace();
		return "";
	}

	return `<button class='copy_to_clipboard_button invert_in_dark_mode' onclick='find_closest_element_behind_and_copy_content_to_clipboard(this, "${name_to_search_for}")'>📋 Copy raw data to clipboard</button>`;
}
