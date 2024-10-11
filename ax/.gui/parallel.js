async function plot_parallel_plot() {
	// Fetch and validate data
	let _results_csv_json = await fetchData('job_infos.csv');
	if (!_results_csv_json) return;

	// Preprocess data
	let { header_line, data, mappingKeyNameToIndex, resultValues, minResult, maxResult } = preprocessData(_results_csv_json);

	// Generate the plot
	parallel_plot(header_line, data, mappingKeyNameToIndex, resultValues, minResult, maxResult);

	apply_theme_based_on_system_preferences();
}

async function fetchData(filename) {
	let jsonData = await fetchJsonFromUrlFilenameOnly(filename);
	if (!jsonData || !jsonData.data || !jsonData.data.length) {
		//error(`Could not plot: empty or invalid data from ${filename}`);
		return null;
	}
	return jsonData;
}

function preprocessData(_results_csv_json) {
	let data = _results_csv_json.data.map(Object.values);
	convertToIntAndFilter(data);
	replaceZeroWithNull(data);

	let header_line = data.shift();
	let mappingKeyNameToIndex = createMapping(header_line);
	let paramKeys = extractParameterKeys(header_line);
	let result_idx = header_line.indexOf("result");
	let resultValues = extractResultValues(data, result_idx);

	let minResult = Math.min(...resultValues);
	let maxResult = Math.max(...resultValues);

	return { header_line, data, mappingKeyNameToIndex, paramKeys, resultValues, minResult, maxResult };
}

function createMapping(header_line) {
	let mapping = {};
	header_line.forEach((key, i) => {
		mapping[key] = i;
	});
	return mapping;
}

function extractParameterKeys(header_line) {
    const excludedKeys = ['trial_index', 'arm_name', 'run_time', 'trial_status', 'generation_method', 'result', 'start_time', 'end_time', 'program_string', 'hostname', 'signal', 'exit_code'];
    // Filter out any keys that are in the excludedKeys list
    return excludedKeys; // Now return excluded keys directly
}

function extractResultValues(data, result_idx) {
	return data.map(row => parseFloat(row[result_idx]))
		.filter(value => value !== undefined && !isNaN(value));
}

function parallel_plot(header_line, data, mappingKeyNameToIndex, resultValues, minResult, maxResult) {
    let data_md5 = md5(JSON.stringify(data));
    if ($('#parallel_plot_container').data("md5") == data_md5) return;

    // Filter out excluded columns (hostname, program_string, start_time, etc.)
    let excludedKeys = extractParameterKeys(header_line); // Use this to get the list of excluded keys
    let filtered_header_line = header_line.filter(key => !excludedKeys.includes(key));

    // Now create dimensions only with the filtered header line
    let dimensions = createDimensions(filtered_header_line, data, mappingKeyNameToIndex, resultValues, minResult, maxResult);
    let trace = createParallelTrace(dimensions, resultValues, minResult, maxResult);
    let layout = createParallelLayout();

    renderParallelPlot(trace, layout, data_md5);
}

function createDimensions(header_line, data, mappingKeyNameToIndex, resultValues, minResult, maxResult) {
	let dimensions = header_line.map(key => {
		let idx = mappingKeyNameToIndex[key];
		let values = cleanValues(data.map(row => row[idx]));
		let stringMapping = mapStrings(values);

		return (isNumericArray(values))
			? createNumericDimension(key, values)
			: createStringDimension(key, values, stringMapping);
	});

	dimensions.push(createResultDimension(resultValues, minResult, maxResult));
	return dimensions;
}
function createNumericDimension(key, values) {
	let numericValues = values.map(parseFloat);
	numericValues.sort((a, b) => a - b);

	return {
		range: [Math.min(...numericValues), Math.max(...numericValues)],
		label: key,
		values: numericValues,
		tickvals: createTicks(numericValues),
		ticktext: createTickText(createTicks(numericValues))
	};
}


function createStringDimension(key, values, stringMapping) {
	let valueIndices = values.map(value => stringMapping[cleanValue(value)]);
	let uniqueValues = Object.keys(stringMapping).sort();

	return {
		range: [0, uniqueValues.length - 1],
		label: key,
		values: valueIndices,
		tickvals: Object.values(stringMapping), // Show all unique values
		ticktext: uniqueValues // Ensure all string values are displayed
	};
}

function createResultDimension(resultValues, minResult, maxResult) {
	return {
		range: [minResult, maxResult],
		label: 'result',
		values: resultValues,
		colorscale: 'Jet',
		tickvals: createTicks(resultValues), // Ensure full range of results is displayed
		ticktext: createTickText(createTicks(resultValues)) // Show all results with precision
	};
}

function createParallelTrace(dimensions, resultValues, minResult, maxResult) {
	return {
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
}

function createParallelLayout() {
	return {
		title: 'Parallel Coordinates Plot',
		width: get_width(),
		height: get_height(),
		paper_bgcolor: 'rgba(0,0,0,0)',
		plot_bgcolor: 'rgba(0,0,0,0)',
		showlegend: false
	};
}

function renderParallelPlot(trace, layout, data_md5) {
	let new_plot_div = $(`<div class='share_graph parallel-plot' id='parallel-plot' style='width:${get_width()}px;height:${get_height()}px;'></div>`);
	$('#parallel_plot_container').html(new_plot_div);

	if (!$("#parallel_plot").length) {
		add_tab("parallel_plot", "Parallel Plot", "<div id='parallel_plot_container'><div id='parallel-plot'></div></div>");
	}

	Plotly.newPlot('parallel-plot', [trace], layout).then(() => {
		$('#parallel_plot_container').data("md5", data_md5);
	}).catch(err => {
		error("Creating the plot failed:", err);
	});
}

function cleanValues(values) {
	return values.map(cleanValue);
}

function cleanValue(value) {
	return (value === null || value === undefined || value === '') ? 'N/A' : value;
}

function isNumericArray(values) {
	return values.every(value => !isNaN(parseFloat(value)));
}

function mapStrings(values) {
	let uniqueStrings = [...new Set(values.map(cleanValue))].sort();
	return uniqueStrings.reduce((acc, str, idx) => {
		acc[str] = idx;
		return acc;
	}, {});
}

function createTicks(values) {
	const min = Math.min(...values);
	const max = Math.max(...values);
	const numTicks = Math.min(10, values.length); // Maximal 10 Ticks
	const step = (max - min) / (numTicks - 1);

	return Array.from({ length: numTicks }, (_, i) => (min + step * i).toFixed(2));
}

function createTickText(ticks) {
	return ticks.map(v => v.toLocaleString()); // Ensure precise display of numbers
}
