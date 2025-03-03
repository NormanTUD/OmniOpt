"use strict";

async function plot_new_parallel_plot() {
	var urlParams = new URLSearchParams(window.location.search);
	var filename = "results.csv";

	var url = `share_to_csv.php?user_id=${urlParams.get("user_id")}&experiment_name=${urlParams.get("experiment_name")}&run_nr=${urlParams.get("run_nr")}&filename=${filename}&only_raw_data=1`;

	log(`url: ${url}`);

	$("body").append("<div id='example1'></div>");

	// linear color scale
	var blue_to_brown = d3.scaleLinear()
		.domain([9, 50])
		.range(["steelblue", "brown"])
		.interpolate(d3.interpolateLab);

	// interact with this variable from a javascript console
	var pc1;

	// load csv file and create the chart
	d3.csv(url, function(data) {
		pc1 = d3.parcoords()("#example1")
			.data(data)
			.hideAxis(["name"])
			.composite("darken")
			.color(function(d) { return blue_to_brown(d['result']); })  // quantitative color scale
			.alpha(0.35)
			.render()
			.brushMode("1D-axes")  // enable brushing
			.interactive()  // command line mode

		var explore_count = 0;
		var exploring = {};
		var explore_start = false;
		pc1.svg
			.selectAll(".dimension")
			.style("cursor", "pointer")
			.on("click", function(d) {
				exploring[d] = d in exploring ? false : true;
				event.preventDefault();
				if (exploring[d]) d3.timer(explore(d,explore_count));
			});

		function explore(dimension,count) {
			if (!explore_start) {
				explore_start = true;
				d3.timer(pc1.brush);
			}
			var speed = (Math.round(Math.random()) ? 1 : -1) * (Math.random()+0.5);
			return function(t) {
				if (!exploring[dimension]) return true;
				var domain = pc1.yscale[dimension].domain();
				var width = (domain[1] - domain[0])/4;

				var center = width*1.5*(1+Math.sin(speed*t/1200)) + domain[0];

				pc1.yscale[dimension].brush.extent([
					d3.max([center-width*0.01, domain[0]-width/400]),
					d3.min([center+width*1.01, domain[1]+width/100])
				])(pc1.g()
					.filter(function(d) {
						return d == dimension;
					})
				);
			};
		};

	});
}

async function plot_parallel_plot() {
	var resnames = await get_result_names_data();

	showSpinnerOverlay("Plotting parallel-plot");
	//async function fetchJsonFromUrlFilenameOnly(filename, remove_ansi=false, parse_ansi=false, no_raw_data=false, only_raw_data=false) {
	let _results_csv_json = await fetchData("job_infos.csv", false, false, false, false);
	if (!_results_csv_json) {
		log("C");
		return;
	}

	let { header_line, data, mappingKeyNameToIndex, resultValues, minResult, maxResult } = await preprocessData(_results_csv_json, resnames);

	parallel_plot(header_line, data, mappingKeyNameToIndex, resultValues, minResult, maxResult);

	apply_theme_based_on_system_preferences();
}

async function fetchData(filename, remove_ansi=false, parse_ansi=false, no_raw_data=false, only_raw_data=false) {
	let jsonData = await fetchJsonFromUrlFilenameOnly(filename, remove_ansi, parse_ansi, no_raw_data, only_raw_data);
	if (!jsonData || !jsonData.data || !jsonData.data.length) {
		//error(`Could not plot: empty or invalid data from ${filename}`);
		return null;
	}
	return jsonData;
}

async function preprocessData(_results_csv_json, resnames) {
	let data = _results_csv_json.data.map(Object.values);
	convertToIntAndFilter(data);

	let header_line = data.shift();
	let mappingKeyNameToIndex = createMapping(header_line);
	let paramKeys = extractParameterKeys(header_line);

	let minResults = {};
	let maxResults = {};
	let resultValues = {};

	for (var i = 0; i < resnames.length; i++) {
		var resname = resnames[i];

		let result_idx = header_line.findIndex(header => header.toLowerCase() === resname.toLowerCase());

		if(result_idx == -1) {
			error(`Could not find '${resname}' column`)
		}

		let resultValue = extractResultValues(data, result_idx);

		let minResult = Math.min(...resultValue);
		let maxResult = Math.max(...resultValue);

		minResults[resname] = minResult;
		maxResults[resname] = maxResult;
		resultValues[resname] = resultValue;
	}

	return { header_line, data, mappingKeyNameToIndex, paramKeys, resultValues, minResults, maxResults };
}

function createMapping(header_line) {
	let mapping = {};
	header_line.forEach((key, i) => {
		mapping[key] = i;
	});
	return mapping;
}

function extractParameterKeys(header_line) {
	const excludedKeys = ["trial_index", "arm_name", "run_time", "trial_status", "generation_method", "start_time", "end_time", "program_string", "hostname", "signal", "exit_code"];
	// Filter out any keys that are in the excludedKeys list
	return excludedKeys; // Now return excluded keys directly
}

function extractResultValues(data, result_idx) {
	return data.map(row => parseFloat(row[result_idx]))
		.filter(value => value !== undefined && !isNaN(value));
}

function parallel_plot(header_line, data, mappingKeyNameToIndex, resultValues, minResult, maxResult) {
	let data_md5 = md5(JSON.stringify(data));
	if ($("#parallel_plot_container").data("md5") == data_md5) return;

	let excludedKeys = extractParameterKeys(header_line); // Use this to get the list of excluded keys
	let filtered_header_line = header_line.filter(key => !excludedKeys.includes(key));

	let dimensions = createDimensions(filtered_header_line, data, mappingKeyNameToIndex, resultValues, minResult, maxResult);
	let trace = createParallelTrace(dimensions, resultValues, minResult, maxResult);
	let layout = createParallelLayout();

	renderParallelPlot(trace, layout, data_md5);
}

function createDimensions(header_line, data, mappingKeyNameToIndex, resultValues, minResult, maxResult) {
	let dimensions = header_line.map(key => {
		let idx = mappingKeyNameToIndex[key];
		let values = cleanValues(data.map(row => row[idx]));

		if(isNumericArray(values)) {
			return createNumericDimension(key, values);
		} else {
			let stringMapping = mapStringsParallel(values);

			return createStringDimension(key, values, stringMapping);
		}
	});

	var resnames = Object.keys(resultValues);

	for (var i = 0; i < resnames; i++) {
		var resname = resnames[i];
		dimensions.push(createResultDimension(resultValues[resname], minResult[resname], maxResult[resname]));
	}

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

	var res = {
		range: [0, uniqueValues.length - 1],
		label: key,
		values: valueIndices,
		tickvals: Object.values(stringMapping),
		ticktext: uniqueValues
	};

	return res;
}

function createResultDimension(resultValues, minResult, maxResult) {
	return {
		range: [minResult, maxResult],
		label: "result",
		values: resultValues,
		colorscale: "Jet",
		tickvals: createTicks(resultValues), // Ensure full range of results is displayed
		ticktext: createTickText(createTicks(resultValues)) // Show all results with precision
	};
}

function createParallelTrace(dimensions, resultValues, minResult, maxResult) {
	return {
		type: "parcoords",
		line: {
			color: resultValues,
			colorscale: "Jet",
			showscale: true,
			cmin: minResult,
			cmax: maxResult
		},
		dimensions: dimensions
	};
}

function createParallelLayout() {
	return {
		title: "Parallel Coordinates Plot",
		width: get_width(),
		height: get_height(),
		paper_bgcolor: "rgba(0,0,0,0)",
		plot_bgcolor: "rgba(0,0,0,0)",
		showlegend: false
	};
}

function renderParallelPlot(trace, layout, data_md5) {
	let new_plot_div = $(`<div class='share_graph parallel-plot' id='parallel-plot' style='width:${get_width()}px;height:${get_height()}px;'></div>`);
	$("#parallel_plot_container").html(new_plot_div);

	if (!$("#parallel_plot").length) {
		add_tab("parallel_plot", "Parallel Plot", "<div id='parallel_plot_container'><div id='parallel-plot'></div></div>");
	}

	Plotly.newPlot("parallel-plot", [trace], layout).then(() => {
		$("#parallel_plot_container").data("md5", data_md5);
	}).catch(err => {
		error("Creating the plot failed:", err);
	});
}

function cleanValues(values) {
	return values.map(cleanValue);
}

function cleanValue(value) {
	return (value === null || value === undefined || value === "") ? "N/A" : value;
}

function isNumericArray(values) {
	return values.every(value => !isNaN(parseFloat(value)));
}

function mapStringsParallel(values) {
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
