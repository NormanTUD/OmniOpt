async function plot_parallel_plot () {
	//debug_function("plot_parallel_plot()");
	var _results_csv_json = await fetchJsonFromUrlFilenameOnly(`job_infos.csv`)
	if(!_results_csv_json || !_results_csv_json.data) {
		return;
	}

	if(!Object.keys(_results_csv_json).includes("data")) {
		error(`plot_parallel_plot: Could not plot seemingly empty _results_csv_json: no data found`);
		return;
	}
	
	if(!_results_csv_json.data.length) {
		error(`plot_parallel_plot: Could not plot seemingly empty _results_csv_json`);
		return;
	}

	convertToIntAndFilter(_results_csv_json.data.map(Object.values))

	replaceZeroWithNull(_results_csv_json.data);

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
}

function parallel_plot(paramKeys, _results_csv_json, minResult, maxResult, resultValues, mappingKeyNameToIndex) {
	var data_md5 = md5(JSON.stringify(_results_csv_json));

	if ($('#parallel_plot_container').data("md5") == data_md5) {
		return;
	}

	// Helper function to handle null/undefined and convert to valid strings or numeric values
	function cleanValue(value) {
		if (value === null || value === undefined || value === '') {
			return 'N/A'; // Placeholder for missing data
		}
		return value;
	}

	// Function to map string values to unique indices, including 'N/A' if present
	function mapStrings(values) {
		var cleanedValues = values.map(cleanValue);
		var uniqueStrings = [...new Set(cleanedValues)];
		uniqueStrings.sort(); // Sort alphabetically
		return uniqueStrings.reduce((acc, str, idx) => {
			acc[str] = idx;
			return acc;
		}, {});
	}

	// Function to create dimensions for the parallel plot
	var dimensions = paramKeys.map(function(key) {
		var idx = mappingKeyNameToIndex[key];

		// Extract and clean values from the results
		var values = _results_csv_json.map(function(row) {
			return cleanValue(row[idx]);
		});

		var stringMapping = mapStrings(values);

		if (!every_array_element_is_a_number(values)) {
			// Map all values (strings) to their corresponding indices
			var valueIndices = values.map(function(value) {
				return stringMapping[cleanValue(value)];
			});

			var uniqueValues = Object.keys(stringMapping).sort();
			return {
				range: [0, uniqueValues.length - 1],
				label: key,
				values: valueIndices,
				tickvals: Object.values(stringMapping).slice(0, max_nr_ticks),
				ticktext: uniqueValues.slice(0, max_nr_ticks)
			};
		}

		// For other parameters, continue normal numeric handling or string mapping
		var numericValues = values.filter(value => !isNaN(parseFloat(value))).map(parseFloat);

		// If fully numeric, handle normally
		if (numericValues.length === values.length) {
			return {
				range: [Math.min(...numericValues), Math.max(...numericValues)],
				label: key,
				values: numericValues,
				tickvals: createTicks(numericValues, max_nr_ticks), // Create ticks
				ticktext: createTickText(createTicks(numericValues, max_nr_ticks)) // Create tick labels
			};
		}

		// Otherwise, map non-numeric values as strings
		var valueIndices = values.map(function(value) {
			var parsedValue = parseFloat(value);
			if (!isNaN(parsedValue)) {
				return parsedValue;
			} else {
				return stringMapping[cleanValue(value)];
			}
		});

		return {
			range: [0, Object.keys(stringMapping).length - 1],
			label: key,
			values: valueIndices,
			tickvals: Object.values(stringMapping).slice(0, max_nr_ticks),
			ticktext: Object.keys(stringMapping).slice(0, max_nr_ticks)
		};
	});

	// Add the result dimension (color scale)
	dimensions.push({
		range: [minResult, maxResult],
		label: 'result',
		values: resultValues,
		colorscale: 'Jet',
		tickvals: createTicks(resultValues, max_nr_ticks), // Create ticks for results
		ticktext: createTickText(createTicks(resultValues, max_nr_ticks)) // Create tick labels for results
	});

	// Parallel coordinates trace
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

	// Layout for the parallel plot
	var layoutParallel = {
		title: 'Parallel Coordinates Plot',
		width: get_width(),
		height: get_height(),
		paper_bgcolor: 'rgba(0,0,0,0)',
		plot_bgcolor: 'rgba(0,0,0,0)',
		showlegend: false
	};

	// Create a new div for the plot
	var new_plot_div = $(`<div class='share_graph parallel-plot' id='parallel-plot' style='width:${get_width()}px;height:${get_height()}px;'></div>`);
	$('#parallel_plot_container').html(new_plot_div);

	if (!$("#parallel_plot").length) {
		add_tab("parallel_plot", "Parallel Plot", "<div id='parallel_plot_container'><div id='parallel-plot'></div></div>");
	}

	// Render the plot with Plotly
	if ($("#parallel_plot_container").length) {
		Plotly.newPlot('parallel-plot', [traceParallel], layoutParallel).then(function() {
		}).catch(function(err) {
			error("Creating the plot failed:", err);
		});
	} else {
		error("Cannot find #parallel_plot_container");
	}

	$('#parallel_plot_container').data("md5", data_md5);
}

// Function to create tick values dynamically
function createTicks(values, maxTicks) {
	const min = Math.min(...values);
	const max = Math.max(...values);

	const step = (max - min) / (maxTicks - 1);

	let ticks = [];
	for (let i = 0; i < maxTicks; i++) {
		ticks.push(min + step * i);
	}

	ticks[ticks.length - 1] = max;

	return ticks;
}

// Function to create tick text
function createTickText(ticks) {
	return ticks.map(v => v.toLocaleString()); // Format large numbers
}
