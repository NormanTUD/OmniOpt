"use strict";

function get_row_by_index(idx) {
	if (!Object.keys(window).includes("tab_results_csv_json")) {
		error("tab_results_csv_json is not defined");
		return;
	}

	if (!Object.keys(window).includes("tab_results_headers_json")) {
		error("tab_results_headers_json is not defined");
		return;
	}


	var trial_index_col_idx = tab_results_headers_json.indexOf("trial_index");

	if(trial_index_col_idx == -1) {
		error(`"trial_index" could not be found in tab_results_headers_json. Cannot continue`);

		return null;
	}

	for (var i = 0; i < tab_results_csv_json.length; i++) {
		var row = tab_results_csv_json[i];
		var trial_index = row[trial_index_col_idx];

		if (trial_index == idx) {
			return row;
		}
	}

	return null;
}

function load_pareto_graph_from_idxs () {
	if (!Object.keys(window).includes("pareto_idxs")) {
		error("pareto_idxs is not defined");
		return;
	}

	if (!Object.keys(window).includes("tab_results_csv_json")) {
		error("tab_results_csv_json is not defined");
		return;
	}

	if (!Object.keys(window).includes("tab_results_headers_json")) {
		error("tab_results_headers_json is not defined");
		return;
	}

	var table = get_pareto_table_data_from_idx();

	console.log(table);
}

function get_pareto_table_data_from_idx () {
	if (!Object.keys(window).includes("pareto_idxs")) {
		error("pareto_idxs is not defined");
		return;
	}

	if (!Object.keys(window).includes("tab_results_csv_json")) {
		error("tab_results_csv_json is not defined");
		return;
	}

	if (!Object.keys(window).includes("tab_results_headers_json")) {
		error("tab_results_headers_json is not defined");
		return;
	}

	var x_keys = Object.keys(pareto_idxs);

	var tables = {};

	for (var i = 0; i < x_keys.length; i++) {
		var x_key = x_keys[i];
		var y_keys = Object.keys(pareto_idxs[x_key]);

		for (var j = 0; j < y_keys.length; j++) {
			var y_key = y_keys[j];

			log(`x_key: ${x_key}, y_key: ${y_key}`);

			var indices = pareto_idxs[x_key][y_key];

			for (var k = 0; k < indices.length; k++) {
				var idx = indices[k];
				var row = get_row_by_index(idx);

				if(row === null) {
					error(`Error getting the row for index ${idx}`);
					return;
				}

				var row_dict = {
					"results": {},
					"values": {},
				};

				for (var l = 0; l < tab_results_headers_json.length; l++) {
					var header = tab_results_headers_json[l];

					if (!special_col_names.includes(header) || header == "trial_index") {
						var val = row[l];

						if (result_names.includes(header)) {
							if (!Object.keys(row_dict["results"]).includes(header)) {
								row_dict["results"][header] = [];
							}

							row_dict["results"][header].push(val);
						} else {
							if (!Object.keys(row_dict["values"]).includes(header)) {
								row_dict["values"][header] = [];
							}
							row_dict["values"][header].push(val);
						}
					}
					
				}

				var table_key = `Pareto front for ${x_key}/${y_key}`;

				if(!Object.keys(tables).includes(table_key)) {
					tables[table_key] = [];
				}

				tables[table_key].push(row_dict);
			}

		}
	}

	return tables;
}
