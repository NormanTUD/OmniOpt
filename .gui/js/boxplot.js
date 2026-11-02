function plotBoxplot() {
	if ($("#plotBoxplot").data("loaded") == "true") {
		return;
	}

	var headers = tab_results_headers_json;
	var rows = tab_results_csv_json;
	var headerIndex = new Map();

	for (var i = 0; i < headers.length; i++) {
		headerIndex.set(headers[i], i);
	}

	var specialSet = new Set(special_col_names);
	var resultSet = new Set(result_names);
	var numericColumns = [];

	for (var c = 0; c < headers.length; c++) {
		var col = headers[c];

		if (specialSet.has(col) || resultSet.has(col) || col.startsWith("OO_Info")) {
			continue;
		}

		var colIndex = headerIndex.get(col);
		var columnData = new Array(rows.length);
		var isNumeric = true;

		for (var r = 0; r < rows.length; r++) {
			var value = parseFloat(rows[r][colIndex]);

			if (isNaN(value)) {
				isNumeric = false;
				break;
			}

			columnData[r] = value;
		}

		if (isNumeric) {
			numericColumns.push({
				name: col,
				index: colIndex,
				data: columnData
			});
		}
	}

	if (numericColumns.length < 1) {
		console.error("Not enough numeric columns for Boxplot");
		return;
	}

	var lowerResultNames = result_names.map(function (name) {
		return String(name).toLowerCase();
	});
	var resultIndex = headers.findIndex(function (header) {
		return result_names.indexOf(header) !== -1 || lowerResultNames.indexOf(String(header).toLowerCase()) !== -1;
	});

	if (resultIndex < 0) {
		console.error("No result column found for Boxplot");
		return;
	}

	var resultValues = rows
		.map(function (row) { return parseFloat(row[resultIndex]); })
		.filter(function (value) { return !isNaN(value); });
	var minResult = Math.min.apply(null, resultValues);
	var maxResult = Math.max.apply(null, resultValues);

	var plotDiv = document.getElementById("plotBoxplot");
	var maxBoxplotPoints = 3000;

	var traces = numericColumns.map(function (column) {
		var values = column.data;

		if (values.length > maxBoxplotPoints) {
			var sampled = new Array(maxBoxplotPoints);
			var stride = values.length / maxBoxplotPoints;

			for (var s = 0; s < maxBoxplotPoints; s++) {
				sampled[s] = values[Math.floor(s * stride)];
			}

			values = sampled;
		}

		return {
			y: values,
			type: 'box',
			name: column.name,
			boxmean: 'sd',
			boxpoints: false,
			marker: {
				color: 'rgb(0, 255, 0)'
			},
		};
	});

	var layout = {
		title: 'Boxplot of Numerical Columns',
		xaxis: {
			title: get_axis_title_data("Columns")
		},
		yaxis: {
			title: get_axis_title_data("Value")
		},
		showlegend: false
	};

	if (typeof Plotly.react === "function" && plotDiv.classList.contains("js-plotly-plot")) {
		Plotly.react(plotDiv, traces, add_default_layout_data(layout));
	} else {
		Plotly.newPlot(plotDiv, traces, add_default_layout_data(layout));
	}

	$("#plotBoxplot").data("loaded", "true");
}
