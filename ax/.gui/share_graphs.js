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
	if (_paramKeys.length >= 3 && _paramKeys.length <= 6) {
		for (var i = 0; i < _paramKeys.length; i++) {
			for (var j = i + 1; j < _paramKeys.length; j++) {
				for (var k = j + 1; k < _paramKeys.length; k++) {
					var xValues = _results_csv_json.map(function(row) { return parseFloat(row[_paramKeys[i]]); });
					var yValues = _results_csv_json.map(function(row) { return parseFloat(row[_paramKeys[j]]); });
					var zValues = _results_csv_json.map(function(row) { return parseFloat(row[_paramKeys[k]]); });

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
						title: `3D Scatter Plot: ${_paramKeys[i]} vs ${_paramKeys[j]} vs ${_paramKeys[k]}`,
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
							xaxis: { title: _paramKeys[i] },
							yaxis: { title: _paramKeys[j] },
							zaxis: { title: _paramKeys[k] }
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
					$('#scatter_plot_3d_container').html(new_plot_div);
					Plotly.newPlot(`scatter-plot-3d-${i}_${j}_${k}`, [trace3d], layout3d);
				}
			}
		}
	}
}

function scatter(_paramKeys, _results_csv_json, minResult, maxResult, resultValues) {
	// 2D Scatter Plot
	for (var i = 0; i < _paramKeys.length; i++) {
		for (var j = i + 1; j < _paramKeys.length; j++) {
			var xValues = _results_csv_json.map(function(row) { return parseFloat(row[_paramKeys[i]]); });
			var yValues = _results_csv_json.map(function(row) { return parseFloat(row[_paramKeys[j]]); });

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
					colorbar: {
						title: 'Result Values',
						titleside: 'right'
					}
				},
				hoverinfo: 'none' // Hide hover info for this trace
			};

			var layout2d = {
				title: `Scatter Plot: ${_paramKeys[i]} vs ${_paramKeys[j]}`,
				xaxis: { title: _paramKeys[i] },
				yaxis: { title: _paramKeys[j] },
				paper_bgcolor: 'rgba(0,0,0,0)',
				plot_bgcolor: 'rgba(0,0,0,0)',
				showlegend: false // We use the colorbar instead of a traditional legend
			};

			var new_plot_div = $(`<div class='share_graph scatter-plot' id='scatter-plot-${i}_${j}' style='width:${get_width()}px;height:${get_height()}px;'></div>`);
			$('#scatter_plot_2d_container').html(new_plot_div);
			Plotly.newPlot(`scatter-plot-${i}_${j}`, [trace2d, colorScaleTrace], layout2d);
		}
	}
}

function hex_scatter(_paramKeys, _results_csv_json, minResult, maxResult, resultValues) {
	// Hexbin Scatter Plot
	if (_paramKeys.length >= 2) {
		for (var i = 0; i < _paramKeys.length; i++) {
			for (var j = i + 1; j < _paramKeys.length; j++) {
				try {
					var xValues = _results_csv_json.map(function(row) { return parseFloat(row[_paramKeys[i]]); });
					var yValues = _results_csv_json.map(function(row) { return parseFloat(row[_paramKeys[j]]); });
					var resultValues = _results_csv_json.map(function(row) { return parseFloat(row["result"]); });

					// Create a custom colorscale based on resultValues
					var colorscale = [];
					var steps = 10; // Number of color steps
					for (var k = 0; k <= steps; k++) {
						var value = minResult + (maxResult - minResult) * (k / steps);
						colorscale.push([
							k / steps,
							`rgb(${255 * k / steps}, ${255 * (1 - k / steps)}, 0)`
						]);
					}

					var traceHexbin = {
						x: xValues,
						y: yValues,
						z: resultValues,
						type: 'histogram2dcontour',
						colorscale: colorscale,
						showscale: true,
						colorbar: {
							title: 'Avg Result',
							titleside: 'right'
						},
						contours: {
							coloring: 'heatmap'
						}
					};

					var layoutHexbin = {
						title: `Contour Plot: ${_paramKeys[i]} vs ${_paramKeys[j]}`,
						xaxis: { title: _paramKeys[i] },
						yaxis: { title: _paramKeys[j] },
						width: get_width(),
						height: get_height(),
						paper_bgcolor: 'rgba(0,0,0,0)',
						plot_bgcolor: 'rgba(0,0,0,0)'
					};

					var new_plot_div = $(`<div class='share_graph hexbin-plot' id='hexbin-plot-${i}_${j}' style='width: ${get_width()}px;height:${get_height()}px;'></div>`);
					$('body').append(new_plot_div);
					Plotly.newPlot(`hexbin-plot-${i}_${j}`, [traceHexbin], layoutHexbin);
				} catch (error) {
					log(error, `Error in hex_scatter function for parameters: ${_paramKeys[i]}, ${_paramKeys[j]}`);
				}
			}
		}
	}
}

function createHexbinData(data, minResult, maxResult) {
	var hexbin = d3.hexbin()
		.x(function(d) { return d.x; })
		.y(function(d) { return d.y; })
		.radius(20);

	var hexbinPoints = hexbin(data);

	var x = [];
	var y = [];
	var avgResults = [];
	var colors = [];

	hexbinPoints.forEach(function(bin) {
		var avgResult = d3.mean(bin, function(d) { return d.result; });
		x.push(d3.mean(bin, function(d) { return d.x; }));
		y.push(d3.mean(bin, function(d) { return d.y; }));
		avgResults.push(avgResult);
		colors.push(getColor(avgResult, minResult, maxResult));
	});

	return {
		x: x,
		y: y,
		avgResults: avgResults,
		colors: colors
	};
}

function plot_parallel_plot (_results_csv_json) {
	// Extract parameter names
	var paramKeys = Object.keys(_results_csv_json[0]).filter(function(key) {
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
	var resultValues = _results_csv_json.map(function(row) {
		return parseFloat(row.result);
	});

	resultValues = resultValues.filter(value => value !== undefined && !isNaN(value));

	var minResult = Math.min.apply(null, resultValues);
	var maxResult = Math.max.apply(null, resultValues);

	parallel_plot(paramKeys, _results_csv_json, minResult, maxResult, resultValues);

	apply_theme_based_on_system_preferences();

	$("#out_files_tabs").tabs();
}

function plot_all_possible (_results_csv_json) {
	// Extract parameter names
	var paramKeys = Object.keys(_results_csv_json[0]).filter(function(key) {
		return !['trial_index', 'arm_name', 'trial_status', 'generation_method', 'result'].includes(key);
	});

	// Get result values for color mapping
	var resultValues = _results_csv_json.map(function(row) { return parseFloat(row.result); });
	var minResult = Math.min.apply(null, resultValues);
	var maxResult = Math.max.apply(null, resultValues);

	scatter(paramKeys, _results_csv_json, minResult, maxResult, resultValues);
	scatter_3d(paramKeys, _results_csv_json, minResult, maxResult, resultValues);

	apply_theme_based_on_system_preferences();

	$("#out_files_tabs").tabs();
}

function convertUnixTimeToReadable(unixTime) {
	var date = new Date(unixTime * 1000); // Unix-Zeit ist in Sekunden, daher * 1000 um Millisekunden zu erhalten
	return date.toLocaleString(); // Konvertiere zu einem lesbaren Datum und Uhrzeit
}

function plotLineChart(data) {
	// Extrahiere die Unix-Zeit, die geplanten Worker und die tatsächlichen Worker
	var unixTime = data.map(row => row[0]);
	var readableTime = unixTime.map(convertUnixTimeToReadable); // Konvertiere Unix-Zeit in menschenlesbares Format
	var plannedWorkers = data.map(row => row[1]);
	var actualWorkers = data.map(row => row[2]);

	// Erstelle den Trace für geplante Worker
	var tracePlanned = {
		x: readableTime,
		y: plannedWorkers,
		mode: 'lines',
		name: 'Planned Worker'
	};

	// Erstelle den Trace für tatsächliche Worker
	var traceActual = {
		x: readableTime,
		y: actualWorkers,
		mode: 'lines',
		name: 'Real Worker'
	};

	// Layout des Diagramms
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

	var new_plot_div = document.createElement('div');
	new_plot_div.id = 'line-plot';
	new_plot_div.style.width = get_width() + 'px';
	new_plot_div.style.height = get_height() + 'px';
	new_plot_div = $(new_plot_div).addClass("share_graph")[0];
	document.body.appendChild(new_plot_div);
	
	// Erstelle das Diagramm
	Plotly.newPlot('line-plot', [tracePlanned, traceActual], layout);
}

function plot_cpu_gpu_graph(cpu_ram_usage_json) {
	// Initialize arrays to store valid values
	const validCpuEntries = cpu_ram_usage_json.filter(entry => entry[2] !== null && entry[2] !== undefined);

	// Filtered timestamps and CPU usage data
	const timestamps_cpu = validCpuEntries.map(entry => new Date(entry[0] * 1000));
	const cpuUsage = validCpuEntries.map(entry => entry[2]);

	// RAM data remains the same
	const timestamps_ram = cpu_ram_usage_json.map(entry => new Date(entry[0] * 1000));
	const ramUsage = cpu_ram_usage_json.map(entry => entry[1]);

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

	var new_plot_div = document.createElement('div');
	new_plot_div.id = 'cpuRamChart';
	new_plot_div.style.width = get_width() + 'px';
	new_plot_div.style.height = get_height() + 'px';
	new_plot_div = $(new_plot_div).addClass("share_graph")[0];
	document.body.appendChild(new_plot_div);

	Plotly.newPlot('cpuRamChart', data, layout);
}

function replaceZeroWithNull(arr) {
	// Überprüfen, ob arr ein Array ist
	if (Array.isArray(arr)) {
		for (let i = 0; i < arr.length; i++) {
			// Wenn das aktuelle Element ein Array ist, rekursiv aufrufen
			if (Array.isArray(arr[i])) {
				replaceZeroWithNull(arr[i]);
			} else if (arr[i] === 0) {
				// Wenn das aktuelle Element 0 ist, durch null ersetzen
				arr[i] = null;
			}
		}
	}
};
