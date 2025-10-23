function createParallelPlot(dataArray, headers, resultNames, ignoreColumns = [], reload = false) {
	try {
		if ($("#parallel-plot").data("loaded") === "true" && !reload) {
			return;
		}

		dataArray = filterNonEmptyRows(dataArray);

		const ignoreSet = new Set(ignoreColumns);
		const numericalCols = [];
		const categoricalCols = [];
		const categoryMappings = {};
		const enable_slurm_id_if_exists = $("#enable_slurm_id_if_exists").is(":checked");

		parallel_classify_columns(headers, dataArray, ignoreSet, enable_slurm_id_if_exists, numericalCols, categoricalCols, categoryMappings);

		const precomputedMappings = parallel_precompute_category_mappings(dataArray, categoricalCols);

		const { controlContainer, columnVisibility, minMaxLimits, headerIndex } = parallel_create_controls(
			headers,
			dataArray,
			ignoreSet,
			enable_slurm_id_if_exists,
			numericalCols
		);

		const { resultSelect, colorValuesRef } = parallel_create_result_selector(resultNames, numericalCols, dataArray);

		function updatePlot() {
			parallel_update_plot({
				dataArray,
				numericalCols,
				categoricalCols,
				columnVisibility,
				minMaxLimits,
				precomputedMappings,
				colorValuesRef,
				resultSelect
			});
		}

		parallel_bind_min_max_and_checkbox_events(headers, dataArray, numericalCols, columnVisibility, minMaxLimits, updatePlot);

		resultSelect.off("change").on("change", function () {
			parallel_handle_color_change(this, numericalCols, dataArray, resultNames, colorValuesRef, updatePlot);
		});

		if (resultNames.length === 1) {
			resultSelect.val(resultNames[0]).trigger("change");
		} else {
			resultSelect.val("none").trigger("change");
		}

		updatePlot();
		$("#parallel-plot").data("loaded", "true");
	} catch (err) {
		console.error("Error in createParallelPlot:", err);
	}
}

function parallel_classify_columns(headers, dataArray, ignoreSet, enable_slurm_id_if_exists, numericalCols, categoricalCols, categoryMappings) {
	headers.forEach((header, colIndex) => {
		if (ignoreSet.has(header)) return;
		if (!enable_slurm_id_if_exists && header === "OO_Info_SLURM_JOB_ID") return;

		const values = dataArray.map(row => row[colIndex]);
		if (values.every(val => !isNaN(parseFloat(val)))) {
			numericalCols.push({ name: header, index: colIndex });
		} else {
			categoricalCols.push({ name: header, index: colIndex });
			const uniqueValues = [...new Set(values)];
			categoryMappings[header] = Object.fromEntries(uniqueValues.map((val, i) => [val, i]));
		}
	});
}

function parallel_precompute_category_mappings(dataArray, categoricalCols) {
	const precomputedMappings = {};
	categoricalCols.forEach(col => {
		const uniqueValues = [...new Set(dataArray.map(row => row[col.index]))];
		precomputedMappings[col.name] = Object.fromEntries(uniqueValues.map((val, i) => [val, i]));
	});
	return precomputedMappings;
}

function parallel_create_controls(headers, dataArray, ignoreSet, enable_slurm_id_if_exists, numericalCols) {
	const controlContainerId = "parallel-plot-controls";
	let controlContainer = $("#" + controlContainerId);
	if (controlContainer.length === 0) {
		controlContainer = $('<div id="' + controlContainerId + '" style="margin-bottom:10px; display: flex;"></div>');
		$("#parallel-plot").before(controlContainer);
	} else {
		controlContainer.empty();
	}

	const columnVisibility = {};
	const minMaxLimits = {};
	const headerIndex = {};
	headers.forEach((h, i) => headerIndex[h] = i);

	headers.forEach(header => {
		try {
			if (ignoreSet.has(header)) return;
			if (!enable_slurm_id_if_exists && header === "OO_Info_SLURM_JOB_ID") return;
			const isNumerical = numericalCols.some(col => col.name === header);
			const box = parallel_create_control_box(header, dataArray, headerIndex, isNumerical, columnVisibility, minMaxLimits);
			controlContainer.append(box);
		} catch (error) {
			console.error(`Fehler bei Header '${header}':`, error);
		}
	});

	return { controlContainer, columnVisibility, minMaxLimits, headerIndex };
}

function parallel_create_control_box(header, dataArray, headerIndex, isNumerical, columnVisibility, minMaxLimits) {
	const checkboxId = `chk_${header}`;
	const minInputId = `min_${header}`;
	const maxInputId = `max_${header}`;

	columnVisibility[header] = true;
	minMaxLimits[header] = { min: null, max: null };

	const boxWrapper = $('<div></div>').css({
		border: "1px solid #ddd",
		borderRadius: "8px",
		padding: "12px 16px",
		marginBottom: "12px",
		boxShadow: "0 2px 6px rgba(0,0,0,0.1)",
		backgroundColor: "#fff",
		display: "flex",
		flexWrap: "wrap",
		alignItems: "center",
		gap: "15px",
		maxWidth: "350px",
		width: "100%",
		boxSizing: "border-box"
	});

	const container = $('<div></div>').css({
		display: "flex",
		alignItems: "center",
		gap: "10px",
		flexWrap: "wrap",
		flexGrow: 1,
		minWidth: "0"
	});

	const checkbox = $(`<input type="checkbox" id="${checkboxId}" checked />`);
	const label = $(`<label for="${checkboxId}" style="font-weight: 600; min-width: 140px; cursor: pointer; white-space: nowrap;">${header}</label>`);
	container.append(checkbox).append(label);

	if (isNumerical) {
		const { minWrapper, maxWrapper } = parallel_create_min_max_inputs(header, dataArray, headerIndex);
		container.append(minWrapper).append(maxWrapper);
	}

	boxWrapper.append(container);
	return boxWrapper;
}

function parallel_create_min_max_inputs(header, dataArray, headerIndex) {
	const numericValues = dataArray.map(row => parseFloat(row[headerIndex[header]])).filter(val => !isNaN(val));
	const minVal = numericValues.length > 0 ? Math.min(...numericValues) : 0;
	const maxVal = numericValues.length > 0 ? Math.max(...numericValues) : 100;

	function makeInput(labelText, id, placeholder, valMin, valMax) {
		const wrapper = $('<div></div>').css({
			display: "flex",
			flexDirection: "column",
			alignItems: "flex-start",
			minWidth: "90px"
		});
		const lbl = $('<label></label>').attr("for", id).text(labelText).css({
			fontSize: "0.75rem",
			color: "#555",
			marginBottom: "2px"
		});
		const input = $(`<input type="number" id="${id}" placeholder="${placeholder}" />`).css({
			width: "80px",
			padding: "5px 8px",
			borderRadius: "5px",
			border: "1px solid #ccc",
			boxShadow: "inset 0 1px 3px rgba(0,0,0,0.1)",
			transition: "border-color 0.3s ease"
		});
		input.attr("min", valMin);
		input.attr("max", valMax);
		input.on("focus", function () { $(this).css("border-color", "#007BFF"); });
		input.on("blur", function () { $(this).css("border-color", "#ccc"); });
		wrapper.append(lbl).append(input);
		return wrapper;
	}

	return {
		minWrapper: makeInput("Min", `min_${header}`, "min", minVal, maxVal),
		maxWrapper: makeInput("Max", `max_${header}`, "max", minVal, maxVal)
	};
}

function parallel_bind_min_max_and_checkbox_events(headers, dataArray, numericalCols, columnVisibility, minMaxLimits, updatePlot) {
	headers.forEach(header => {
		const minInput = $(`#min_${header}`);
		const maxInput = $(`#max_${header}`);
		const checkbox = $(`#chk_${header}`);

		if (minInput.length > 0) {
			minInput.on("input", function () {
				const val = parseFloat($(this).val());
				minMaxLimits[header].min = isNaN(val) ? null : val;
				updatePlot();
			});
		}

		if (maxInput.length > 0) {
			maxInput.on("input", function () {
				const val = parseFloat($(this).val());
				minMaxLimits[header].max = isNaN(val) ? null : val;
				updatePlot();
			});
		}

		if (checkbox.length > 0) {
			checkbox.on("change", function () {
				columnVisibility[header] = $(this).is(":checked");
				updatePlot();
			});
		}
	});
}

function parallel_create_result_selector(resultNames, numericalCols, dataArray) {
	const resultSelectId = "result-select";
	let resultSelect = $(`#${resultSelectId}`);
	if (resultSelect.length === 0) {
		resultSelect = $(`<select id="${resultSelectId}"></select>`);
		$("#parallel-plot-controls").before(resultSelect);
	} else {
		resultSelect.empty();
	}
	resultSelect.append('<option value="none">No color</option>');

	for (let i = 0; i < resultNames.length; i++) {
		let minMaxInfo = "min [auto]";
		if (typeof result_min_max !== "undefined" && result_min_max[i] !== undefined) {
			minMaxInfo = result_min_max[i];
		}
		resultSelect.append(`<option value="${resultNames[i]}">${resultNames[i]} (${minMaxInfo})</option>`);
	}
	const colorValuesRef = { values: null, scale: null };
	return { resultSelect, colorValuesRef };
}

function parallel_handle_color_change(selectElem, numericalCols, dataArray, resultNames, colorValuesRef, updatePlot) {
	const selectedResult = $(selectElem).val();
	if (selectedResult === "none") {
		colorValuesRef.values = null;
		colorValuesRef.scale = null;
		updatePlot();
		return;
	}
	const col = numericalCols.find(c => c.name.toLowerCase() === selectedResult.toLowerCase());
	if (!col) {
		colorValuesRef.values = null;
		colorValuesRef.scale = null;
		updatePlot();
		return;
	}
	colorValuesRef.values = dataArray.map(row => parseFloat(row[col.index]));

	let invertColor = false;
	if (typeof result_min_max !== "undefined") {
		const idx = resultNames.indexOf(selectedResult);
		if (idx !== -1) {
			invertColor = result_min_max[idx] === "max";
		}
	}
	colorValuesRef.scale = invertColor ? [[0, 'red'], [1, 'green']] : [[0, 'green'], [1, 'red']];
	updatePlot();
}

function parallel_update_plot(ctx) {
	try {
		const { dataArray, numericalCols, categoricalCols, columnVisibility, minMaxLimits, precomputedMappings, colorValuesRef, resultSelect } = ctx;

		const filteredNumericalCols = numericalCols.filter(col => columnVisibility[col.name]);
		const filteredCategoricalCols = categoricalCols.filter(col => columnVisibility[col.name]);

		const filteredData = dataArray.filter(row => {
			for (let col of filteredNumericalCols) {
				const val = parseFloat(row[col.index]);
				if (isNaN(val)) return false;
				const limits = minMaxLimits[col.name];
				if (limits.min !== null && val < limits.min) return false;
				if (limits.max !== null && val > limits.max) return false;
			}
			return true;
		});

		const dimensions = [];
		filteredNumericalCols.forEach(col => {
			let vals = filteredData.map(row => parseFloat(row[col.index]));
			let realMin = Infinity, realMax = -Infinity;
			for (let v of vals) {
				if (v < realMin) realMin = v;
				if (v > realMax) realMax = v;
			}
			if (!isFinite(realMin)) { realMin = 0; realMax = 100; }
			dimensions.push({ label: col.name, values: vals, range: [realMin, realMax] });
		});

		filteredCategoricalCols.forEach(col => {
			const vals = filteredData.map(row => precomputedMappings[col.name][row[col.index]]);
			dimensions.push({
				label: col.name,
				values: vals,
				tickvals: Object.values(precomputedMappings[col.name]),
				ticktext: Object.keys(precomputedMappings[col.name])
			});
		});

		let filteredColorValues = null;
		if (colorValuesRef.values) {
			filteredColorValues = filteredData.map(row => {
				const col = numericalCols.find(c => c.name.toLowerCase() === resultSelect.val().toLowerCase());
				return col ? parseFloat(row[col.index]) : null;
			});
		}

		const trace = {
			type: 'parcoords',
			dimensions: dimensions,
			line: filteredColorValues ? { color: filteredColorValues, colorscale: colorValuesRef.scale } : {},
			unselected: {
				line: {
					color: get_text_color(),
					opacity: 0
				}
			},
		};

		dimensions.forEach(dim => {
			if (!dim.line) dim.line = {};
			if (!dim.line.color) dim.line.color = 'rgba(169,169,169, 0.01)';
		});

		Plotly.react('parallel-plot', [trace], add_default_layout_data({ uirevision: 'static' }));
		make_text_in_parallel_plot_nicer();
	} catch (error) {
		console.error("Fehler in parallel_update_plot():", error);
	}
}

function make_text_in_parallel_plot_nicer() {
	$(".parcoords g > g > text").each(function() {
		if (theme == "dark") {
			$(this)
				.css("text-shadow", "unset")
				.css("font-size", "0.9em")
				.css("fill", "white")
				.css("stroke", "black")
				.css("stroke-width", "2px")
				.css("paint-order", "stroke fill");
		} else {
			$(this)
				.css("text-shadow", "unset")
				.css("font-size", "0.9em")
				.css("fill", "black")
				.css("stroke", "unset")
				.css("stroke-width", "unset")
				.css("paint-order", "stroke fill");
		}
	});
}
