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
		} else {
			console.warn('Skipping non-convertible row:', obj);
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
		var numericValues = values.map(function(value) { return parseFloat(value); });

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
				tickvals: valueIndices,
				ticktext: uniqueValues,
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
		dimensions: dimensions
	};

	var layoutParallel = {
		title: 'Parallel Coordinates Plot',
		width: 1200,
		height: 800
	};

	var new_plot_div = $(`<div class='parallel-plot' id='parallel-plot' style='width:1200px;height:800px;'></div>`);
	$('body').append(new_plot_div);
	Plotly.newPlot('parallel-plot', [traceParallel], layoutParallel);
}

function scatter_3d (_paramKeys, _results_csv_json, minResult, maxResult, resultValues) {
	// 3D Scatter Plot
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
						width: 1200,
						height: 800,
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
						}
					};

					var new_plot_div = $(`<div class='scatter-plot' id='scatter-plot-3d-${i}_${j}_${k}' style='width:1200px;height:800px;'></div>`);
					$('body').append(new_plot_div);
					Plotly.newPlot(`scatter-plot-3d-${i}_${j}_${k}`, [trace3d], layout3d);
				}
			}
		}
	}
}

function scatter (_paramKeys, _results_csv_json, minResult, maxResult, resultValues) {
	// 2D Scatter Plot
	for (var i = 0; i < _paramKeys.length; i++) {
		for (var j = i + 1; j < _paramKeys.length; j++) {
			var xValues = _results_csv_json.map(function(row) { return parseFloat(row[_paramKeys[i]]); });
			var yValues = _results_csv_json.map(function(row) { return parseFloat(row[_paramKeys[j]]); });

			function color_curried (value) {
				return getColor(value, minResult, maxResult)
			}

			var colors = resultValues.map(color_curried);

			var trace2d = {
				x: xValues,
				y: yValues,
				mode: 'markers',
				type: 'scatter',
				marker: {
					color: colors
				}
			};

			var layout2d = {
				title: `Scatter Plot: ${_paramKeys[i]} vs ${_paramKeys[j]}`,
				xaxis: { title: _paramKeys[i] },
				yaxis: { title: _paramKeys[j] }
			};

			var new_plot_div = $(`<div class='scatter-plot' id='scatter-plot-${i}_${j}' style='width:1200px;height:800px;'></div>`);
			$('body').append(new_plot_div);
			Plotly.newPlot(`scatter-plot-${i}_${j}`, [trace2d], layout2d);
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
						width: 1200,
						height: 800
					};

					var new_plot_div = $(`<div class='hexbin-plot' id='hexbin-plot-${i}_${j}' style='width:1200px;height:800px;'></div>`);
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

function plot_all_possible (_results_csv_json) {
	// Extract parameter names
	var paramKeys = Object.keys(results_csv_json[0]).filter(function(key) {
		return !['trial_index', 'arm_name', 'trial_status', 'generation_method', 'result'].includes(key);
	});

	// Get result values for color mapping
	var resultValues = _results_csv_json.map(function(row) { return parseFloat(row.result); });
	var minResult = Math.min.apply(null, resultValues);
	var maxResult = Math.max.apply(null, resultValues);

	scatter(paramKeys, results_csv_json, minResult, maxResult, resultValues);
	scatter_3d(paramKeys, results_csv_json, minResult, maxResult, resultValues);
	parallel_plot(paramKeys, results_csv_json, minResult, maxResult, resultValues);
	//hex_scatter(paramKeys, _results_csv_json, minResult, maxResult, resultValues);
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
		width: 1200,
		height: 800
	};

	var new_plot_div = document.createElement('div');
	new_plot_div.id = 'line-plot';
	new_plot_div.style.width = '1200px';
	new_plot_div.style.height = '800px';
	document.body.appendChild(new_plot_div);
	
	// Erstelle das Diagramm
	Plotly.newPlot('line-plot', [tracePlanned, traceActual], layout);
}
