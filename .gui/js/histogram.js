function plotHistogram() {
	if ($("#plotHistogram").data("loaded") == "true") {
		return;
	}

	var plotDiv = document.getElementById("plotHistogram");
	if (!plotDiv) {
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

	// Collect numeric columns once.  The old version called indexOf() inside
	// every row of every column, which is O(columns^2 * rows) on large runs.
	var numericColumns = [];
	for (var c = 0; c < headers.length; c++) {
		var col = headers[c];
		if (specialSet.has(col) || resultSet.has(col) || col.startsWith("OO_Info")) {
			continue;
		}

		var colIndex = headerIndex.get(col);
		var data = new Array(rows.length);
		var isNumeric = true;
		for (var r = 0; r < rows.length; r++) {
			var value = parseFloat(rows[r][colIndex]);
			if (isNaN(value)) {
				isNumeric = false;
				break;
			}
			data[r] = value;
		}

		if (isNumeric) {
			numericColumns.push({ name: col, data: data });
		}
	}

	if (numericColumns.length < 1) {
		console.error("Not enough columns for Histogram");
		return;
	}

	plotDiv.innerHTML = "";

	var colorPalette = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6'];
	// Keep the number of SVG bars bounded for very large runs.  Without a
	// cap Plotly may emit thousands of bars per trace, which dominates layout
	// and paint time even though the distribution is visually unreadable.
	var binCount = Math.min(60, Math.max(10, Math.ceil(Math.sqrt(rows.length))));

	var traces = numericColumns.map(function(column, index) {
		return {
			x: column.data,
			type: 'histogram',
			name: column.name,
			opacity: 0.7,
			marker: {
				color: colorPalette[index % colorPalette.length]
			},
			nbinsx: binCount,
			autobinx: false
		};
	});

	var layout = {
		title: 'Histogram of Numerical Columns',
		xaxis: {
			title: get_axis_title_data("Value")
		},
		yaxis: {
			title: get_axis_title_data("Frequency")
		},
		showlegend: true,
		barmode: 'overlay'
	};

	Plotly.newPlot(plotDiv, traces, add_default_layout_data(layout));
	$("#plotHistogram").data("loaded", "true");
}
