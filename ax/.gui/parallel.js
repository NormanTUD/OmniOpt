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
