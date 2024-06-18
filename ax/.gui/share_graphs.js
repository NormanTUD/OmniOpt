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
