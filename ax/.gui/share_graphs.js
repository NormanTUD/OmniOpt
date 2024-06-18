function getColor(value) {
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

function parallel_plot(_paramKeys, _results_csv_json) {
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

function scatter_3d (_paramKeys, _results_csv_json) {
	// 3D Scatter Plot
	if (_paramKeys.length >= 3 && _paramKeys.length <= 6) {
		for (var i = 0; i < _paramKeys.length; i++) {
			for (var j = i + 1; j < _paramKeys.length; j++) {
				for (var k = j + 1; k < _paramKeys.length; k++) {
					var xValues = _results_csv_json.map(function(row) { return parseFloat(row[_paramKeys[i]]); });
					var yValues = _results_csv_json.map(function(row) { return parseFloat(row[_paramKeys[j]]); });
					var zValues = _results_csv_json.map(function(row) { return parseFloat(row[_paramKeys[k]]); });
					var colors = resultValues.map(getColor);

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

function scatter (_paramKeys, _results_csv_json) {
	// 2D Scatter Plot
	for (var i = 0; i < _paramKeys.length; i++) {
		for (var j = i + 1; j < _paramKeys.length; j++) {
			var xValues = _results_csv_json.map(function(row) { return parseFloat(row[_paramKeys[i]]); });
			var yValues = _results_csv_json.map(function(row) { return parseFloat(row[_paramKeys[j]]); });
			var colors = resultValues.map(getColor);

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
