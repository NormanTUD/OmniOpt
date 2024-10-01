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

function parallel_plot(_paramKeys, _results_csv_json, minResult, maxResult, resultValues) {
	var dimensions = [..._paramKeys, 'result'].map(function(key) {
		var values = _results_csv_json.map(function(row) { return row[key]; });
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

function scatter_3d (_paramKeys, _results_csv_json, minResult, maxResult, resultValues) {
	var already_existing_plots = [];

	if (_paramKeys.length >= 3 && _paramKeys.length <= 6) {
		for (var i = 0; i < _paramKeys.length; i++) {
			for (var j = i + 1; j < _paramKeys.length; j++) {
				for (var k = j + 1; k < _paramKeys.length; k++) {
					var x_name = _paramKeys[i];
					var y_name = _paramKeys[j];
					var z_name = _paramKeys[k];

					var _key = [x_name, y_name, z_name].sort().join("!!!");
					if(!already_existing_plots.includes(_key)) {
						var xValues = _results_csv_json.map(function(row) {
							var parsedValue = parseFloat(row[i]);
							return isNaN(parsedValue) ? row[i] : parsedValue;
						});

						var yValues = _results_csv_json.map(function(row) {
							var parsedValue = parseFloat(row[j]);
							return isNaN(parsedValue) ? row[j] : parsedValue;
						});

						var zValues = _results_csv_json.map(function(row) {
							var parsedValue = parseFloat(row[k]);
							return isNaN(parsedValue) ? row[k] : parsedValue;
						});


						function color_curried (value) {
							return getColor(value, minResult, maxResult)
						}

						var colors = resultValues.map(color_curried);

						var trace3d = {
							x: xValues,
							y: yValues,
							z: zValues,
							mode: 'markers',
							type: 'scatter3d',
							marker: {
								color: colors
							}
						};

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
								xaxis: { title: x_name },
								yaxis: { title: y_name },
								zaxis: { title: z_name }
							},
							paper_bgcolor: 'rgba(0,0,0,0)',
							plot_bgcolor: 'rgba(0,0,0,0)',

							showlegend: false,
							legend: {
								x: 0.1,
								y: 1.1,
								orientation: 'h'
							},
						};

						var new_plot_div = $(`<div class='share_graph scatter-plot' id='scatter-plot-3d-${i}_${j}_${k}' style='width:${get_width()}px;height:${get_height()}px;'></div>`);
						$('#scatter_plot_3d_container').append(new_plot_div);
						Plotly.newPlot(`scatter-plot-3d-${i}_${j}_${k}`, [trace3d], layout3d);

						already_existing_plots.push(_key);
					}
				}
			}
		}
	}
}

function scatter(_paramKeys, _results_csv_json, minResult, maxResult, resultValues) {
	// 2D Scatter Plot
	var already_existing_plots = [];

	for (var i = 0; i < _paramKeys.length; i++) {
		for (var j = i + 1; j < _paramKeys.length; j++) {
			var x_name = _paramKeys[i];
			var y_name = _paramKeys[j];

			var _key = [x_name, y_name].sort().join("!!!");

			if(!already_existing_plots.includes(_key)) {
				var xValues = _results_csv_json.map(function(row) {
					var parsedValue = parseFloat(row[i]);
					return isNaN(parsedValue) ? row[i] : parsedValue;
				});

				var yValues = _results_csv_json.map(function(row) {
					var parsedValue = parseFloat(row[j]);
					return isNaN(parsedValue) ? row[j] : parsedValue;
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

				// Scatter Plot Trace
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

				var new_plot_div = $(`<div class='share_graph scatter-plot' id='scatter-plot-${i}_${j}' style='width:${get_width()}px;height:${get_height()}px;'></div>`);
				$('#scatter_plot_2d_container').append(new_plot_div);
				Plotly.newPlot(`scatter-plot-${i}_${j}`, [trace2d, colorScaleTrace], layout2d);
				already_existing_plots.push(_key);
			}
		}
	}
}

async function plot_parallel_plot () {
	const urlParams = new URLSearchParams(window.location.search);

	var _results_csv_json = await fetchJsonFromUrl(`share_to_csv.php?user_id=${urlParams.get('user_id')}&experiment_name=${urlParams.get('experiment_name')}&run_nr=${urlParams.get('run_nr')}&filename=results.csv`)

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
	// Extract parameter names
	var paramKeys = Object.keys(_results_csv_json.data[0]).filter(function(key) {
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

	// Get result values for color mapping
	var resultValues = _results_csv_json.data.map(function(row) {
		return parseFloat(row.result);
	});

	resultValues = resultValues.filter(value => value !== undefined && !isNaN(value));

	var minResult = Math.min.apply(null, resultValues);
	var maxResult = Math.max.apply(null, resultValues);

	parallel_plot(paramKeys, _results_csv_json.data, minResult, maxResult, resultValues);

	apply_theme_based_on_system_preferences();

	$("#out_files_tabs").tabs();
}

async function plot_all_possible () {
	const urlParams = new URLSearchParams(window.location.search);

	var _results_csv_json = await fetchJsonFromUrl(`share_to_csv.php?user_id=${urlParams.get('user_id')}&experiment_name=${urlParams.get('experiment_name')}&run_nr=${urlParams.get('run_nr')}&filename=results.csv`)

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

	var paramKeys = [];

	for (var i = 0; i < header_line.length; i++) {
		var this_element = header_line[i];

		if(!['trial_index', 'arm_name', 'trial_status', 'generation_method', 'result'].includes(this_element)) {
			paramKeys.push(this_element);
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

	scatter(paramKeys, _results_csv_json.data, minResult, maxResult, resultValues);
	scatter_3d(paramKeys, _results_csv_json.data, minResult, maxResult, resultValues);

	apply_theme_based_on_system_preferences();

	$("#out_files_tabs").tabs();
}

function convertUnixTimeToReadable(unixTime) {
	var date = new Date(unixTime * 1000);
	return date.toLocaleString();
}

async function load_parameter () {
	const urlParams = new URLSearchParams(window.location.search);

	var data = await fetchJsonFromUrl(`share_to_csv.php?user_id=${urlParams.get('user_id')}&experiment_name=${urlParams.get('experiment_name')}&run_nr=${urlParams.get('run_nr')}&filename=parameters.txt`)

	if(!Object.keys(data).includes("raw")) {
		log(`load_parameter: Could not plot seemingly empty data: no raw found`);
		return;
	}
	

	$("#parameters_txt").html(`<pre>${data.raw}</pre>`);
}

async function load_best_result () {
	const urlParams = new URLSearchParams(window.location.search);

	var data = await fetchJsonFromUrl(`share_to_csv.php?user_id=${urlParams.get('user_id')}&experiment_name=${urlParams.get('experiment_name')}&run_nr=${urlParams.get('run_nr')}&filename=best_result.txt`)

	if(!Object.keys(data).includes("raw")) {
		log(`load_best_result: Could not plot seemingly empty data: no raw found`);
		return;
	}
	

	$("#best_result_txt").html(`<pre>${data.raw}</pre>`);
}

async function plot_planned_vs_real_worker_over_time () {
	const urlParams = new URLSearchParams(window.location.search);

	var data = await fetchJsonFromUrl(`share_to_csv.php?user_id=${urlParams.get('user_id')}&experiment_name=${urlParams.get('experiment_name')}&run_nr=${urlParams.get('run_nr')}&filename=worker_usage.csv`)

	convertToIntAndFilter(data.data.map(Object.values))

	replaceZeroWithNull(data.data);

	if(!Object.keys(data).includes("data")) {
		log(`plot_planned_vs_real_worker_over_time: Could not plot seemingly empty data: no data found`);
		return;
	}

	if(!data.data.length) {
		log(`plot_planned_vs_real_worker_over_time: Could not plot seemingly empty data`);
		return;
	}

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
	const urlParams = new URLSearchParams(window.location.search);

	var cpu_ram_usage_json = await fetchJsonFromUrl(`share_to_csv.php?user_id=${urlParams.get('user_id')}&experiment_name=${urlParams.get('experiment_name')}&run_nr=${urlParams.get('run_nr')}&filename=cpu_ram_usage.csv`)

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
		return data;
	} catch (error) {
		console.error('Fetch error:', error);
		return null;
	}
}

async function load_all_data() {
	showSpinnerOverlay("Loading data...")

	var promises = [];

	promises.push(plot_all_possible());
	promises.push(plot_cpu_gpu_graph());
	promises.push(plot_parallel_plot());
	promises.push(plot_planned_vs_real_worker_over_time());

	promises.push(load_best_result());
	promises.push(load_parameter());

	for (var i = 0; i < promises.length; i++) {
		await promises[i];
	}

	removeSpinnerOverlay();
}
