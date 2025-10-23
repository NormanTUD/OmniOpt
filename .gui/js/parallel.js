function createParallelPlot(dataArray, headers, resultNames, ignoreColumns = [], reload = false) {
	try {
		if ($("#parallel-plot").data("loaded") === "true" && !reload) {
			return;
		}

		// Filter rows ohne leere Werte (wie in deinem Originalcode)
		dataArray = filterNonEmptyRows(dataArray);

		const ignoreSet = new Set(ignoreColumns);
		const numericalCols = [];
		const categoricalCols = [];
		const categoryMappings = {};
		const enable_slurm_id_if_exists = $("#enable_slurm_id_if_exists").is(":checked");

		// Spalten einteilen in numerisch oder kategorisch + category mappings aufbauen
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

		const precomputedMappings = {};
		categoricalCols.forEach(col => {
			const uniqueValues = [...new Set(dataArray.map(row => row[col.index]))];
			precomputedMappings[col.name] = Object.fromEntries(uniqueValues.map((val, i) => [val, i]));
		});

		// Erzeuge UI für Checkboxen und Min/Max Inputs für numerische Spalten
		const controlContainerId = "parallel-plot-controls";
		let controlContainer = $("#" + controlContainerId);
		if (controlContainer.length === 0) {
			controlContainer = $('<div id="' + controlContainerId + '" style="margin-bottom:10px; display: flex;"></div>');
			$("#parallel-plot").before(controlContainer);
		} else {
			controlContainer.empty();
		}

		// Map um Checkbox-Zustände und Min/Max-Werte zu speichern
		const columnVisibility = {};
		const minMaxLimits = {};

		const headerIndex = {};
		headers.forEach((h,i) => headerIndex[h] = i);

		// Checkboxen + Min/Max Felder generieren mit Boxen, max-Breite, Umbruch und Zeilenumbruch nach jeder Box
		headers.forEach((header) => {
			try {
				if (ignoreSet.has(header)) return;
				if (!enable_slurm_id_if_exists && header === "OO_Info_SLURM_JOB_ID") return;

				const isNumerical = numericalCols.some(col => col.name === header);

				const checkboxId = `chk_${header}`;
				const minInputId = `min_${header}`;
				const maxInputId = `max_${header}`;

				columnVisibility[header] = true;
				minMaxLimits[header] = { min: null, max: null };

				// Wrapper Box mit max-Breite, Umbruch, Block-Level-Element für newline nach jeder Box
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
					width: "100%", // damit bei kleinen Screens die Box maximal voll breit ist
					boxSizing: "border-box"
				});

				// Innerer Container mit Flexbox für Ausrichtung der Elemente, flex-grow damit Inputs genug Platz bekommen
				const container = $('<div></div>').css({
					display: "flex",
					alignItems: "center",
					gap: "10px",
					flexWrap: "wrap",
					flexGrow: 1,
					minWidth: "0" // wichtig für flexbox Overflow Handling
				});

				// Checkbox mit Label
				const checkbox = $(`<input type="checkbox" id="${checkboxId}" checked />`);
				const label = $(`<label for="${checkboxId}" style="font-weight: 600; min-width: 140px; cursor: pointer; white-space: nowrap;">${header}</label>`);

				container.append(checkbox).append(label);

				if (isNumerical) {
					// Werte ermitteln (nur gültige Zahlen)
					const numericValues = dataArray
						.map(row => parseFloat(row[headerIndex[header]]))
						.filter(val => !isNaN(val));

					const minVal = numericValues.length > 0 ? Math.min(...numericValues) : 0;
					const maxVal = numericValues.length > 0 ? Math.max(...numericValues) : 100;

					// Min Input mit Label
					const minWrapper = $('<div></div>').css({
						display: "flex",
						flexDirection: "column",
						alignItems: "flex-start",
						minWidth: "90px"
					});
					const minLabel = $('<label></label>').attr("for", minInputId).text("Min").css({
						fontSize: "0.75rem",
						color: "#555",
						marginBottom: "2px"
					});
					const minInput = $(`<input type="number" id="${minInputId}" placeholder="min" />`).css({
						width: "80px",
						padding: "5px 8px",
						borderRadius: "5px",
						border: "1px solid #ccc",
						boxShadow: "inset 0 1px 3px rgba(0,0,0,0.1)",
						transition: "border-color 0.3s ease"
					});
					minInput.attr("min", minVal);
					minInput.attr("max", maxVal);

					minInput.on("focus", function () {
						$(this).css("border-color", "#007BFF");
					});
					minInput.on("blur", function () {
						$(this).css("border-color", "#ccc");
					});

					minWrapper.append(minLabel).append(minInput);

					// Max Input mit Label
					const maxWrapper = $('<div></div>').css({
						display: "flex",
						flexDirection: "column",
						alignItems: "flex-start",
						minWidth: "90px"
					});
					const maxLabel = $('<label></label>').attr("for", maxInputId).text("Max").css({
						fontSize: "0.75rem",
						color: "#555",
						marginBottom: "2px"
					});
					const maxInput = $(`<input type="number" id="${maxInputId}" placeholder="max" />`).css({
						width: "80px",
						padding: "5px 8px",
						borderRadius: "5px",
						border: "1px solid #ccc",
						boxShadow: "inset 0 1px 3px rgba(0,0,0,0.1)",
						transition: "border-color 0.3s ease"
					});
					maxInput.attr("min", minVal);
					maxInput.attr("max", maxVal);

					maxInput.on("focus", function () {
						$(this).css("border-color", "#007BFF");
					});
					maxInput.on("blur", function () {
						$(this).css("border-color", "#ccc");
					});

					maxWrapper.append(maxLabel).append(maxInput);

					// Events für min/max Eingaben
					minInput.on("input", function () {
						const val = parseFloat($(this).val());
						minMaxLimits[header].min = isNaN(val) ? null : val;
						updatePlot();
					});

					maxInput.on("input", function () {
						const val = parseFloat($(this).val());
						minMaxLimits[header].max = isNaN(val) ? null : val;
						updatePlot();
					});

					container.append(minWrapper).append(maxWrapper);
				}

				// Checkbox Change Event
				checkbox.on("change", function () {
					columnVisibility[header] = $(this).is(":checked");
					updatePlot();
				});

				boxWrapper.append(container);

				// Jede Box bekommt ihren eigenen Block (also newline)
				controlContainer.append(boxWrapper);
			} catch (error) {
				console.error(`Fehler bei Header '${header}':`, error);
			}
		});



		// Erzeuge Ergebnis-Auswahl für Farbskala (color by result)
		const resultSelectId = "result-select";
		let resultSelect = $(`#${resultSelectId}`);
		if (resultSelect.length === 0) {
			resultSelect = $(`<select id="${resultSelectId}"></select>`);
			controlContainer.before(resultSelect);
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

		let colorValues = null;
		let colorScale = null;

		resultSelect.off("change").on("change", function () {
			const selectedResult = $(this).val();
			if (selectedResult === "none") {
				colorValues = null;
				colorScale = null;
			} else {
				const col = numericalCols.find(c => c.name.toLowerCase() === selectedResult.toLowerCase());
				if (!col) {
					colorValues = null;
					colorScale = null;
					updatePlot();
					return;
				}
				colorValues = dataArray.map(row => parseFloat(row[col.index]));

				let invertColor = false;
				if (typeof result_min_max !== "undefined") {
					const idx = resultNames.indexOf(selectedResult);
					if (idx !== -1) {
						invertColor = result_min_max[idx] === "max";
					}
				}

				colorScale = invertColor
					? [[0, 'red'], [1, 'green']]
					: [[0, 'green'], [1, 'red']];
			}
			updatePlot();
		});

		if (resultNames.length === 1) {
			resultSelect.val(resultNames[0]).trigger("change");
		} else {
			resultSelect.val("none").trigger("change");
		}

		function updatePlot() {
			try {
				// Filter Spalten nach Checkboxen
				const filteredNumericalCols = numericalCols.filter(col => columnVisibility[col.name]);
				const filteredCategoricalCols = categoricalCols.filter(col => columnVisibility[col.name]);

				// Filtere die Datenzeilen, um nur die zu behalten, die innerhalb aller gesetzten Min/Max Limits liegen
				const filteredData = dataArray.filter(row => {
					for (let col of filteredNumericalCols) {
						const val = parseFloat(row[col.index]);
						if (isNaN(val)) return false; // ungültiger Wert raus

						const limits = minMaxLimits[col.name];
						if (limits.min !== null && val < limits.min) return false;
						if (limits.max !== null && val > limits.max) return false;
					}
					// Kategorische Werte ignorieren Filter (könntest hier evtl. erweitern)
					return true;
				});

				const dimensions = [];

				// Füge numerische Dimensionen hinzu mit Min/Max Limits (Range anhand gefilterter Daten)
				filteredNumericalCols.forEach(col => {
					let vals = filteredData.map(row => parseFloat(row[col.index]));

					// Fallback falls alle Werte NaN (sollte eigentlich nicht vorkommen)
					let realMin = Infinity, realMax = -Infinity;
					for (let v of vals) {
						if (v < realMin) realMin = v;
						if (v > realMax) realMax = v;
					}
					if (!isFinite(realMin)) { realMin = 0; realMax = 100; }

					dimensions.push({
						label: col.name,
						values: vals,
						range: [realMin, realMax]
					});
				});

				// Kategorische Dimensionen (aus gefilterten Daten)
				filteredCategoricalCols.forEach(col => {
					const vals = filteredData.map(row => precomputedMappings[col.name][row[col.index]]);
					dimensions.push({
						label: col.name,
						values: vals,
						tickvals: Object.values(precomputedMappings[col.name]),
						ticktext: Object.keys(precomputedMappings[col.name])
					});
				});

				// Linienfarbe bestimmen, falls Farbskala gesetzt ist
				let filteredColorValues = null;
				if (colorValues) {
					// Da colorValues für alle Daten sind, filtere sie auch entsprechend
					filteredColorValues = filteredData.map(row => {
						const col = numericalCols.find(c => c.name.toLowerCase() === resultSelect.val().toLowerCase());
						return col ? parseFloat(row[col.index]) : null;
					});
				}

				const trace = {
					type: 'parcoords',
					dimensions: dimensions,
					line: filteredColorValues ? { color: filteredColorValues, colorscale: colorScale } : {},
					unselected: {
						line: {
							color: get_text_color(),
							opacity: 0
						}
					},
				};

				dimensions.forEach(dim => {
					if (!dim.line) {
						dim.line = {};
					}
					if (!dim.line.color) {
						dim.line.color = 'rgba(169,169,169, 0.01)';
					}
				});

				Plotly.react('parallel-plot', [trace], add_default_layout_data({uirevision: 'static'}));

				make_text_in_parallel_plot_nicer();
			} catch (error) {
				console.error("Fehler in updatePlot():", error);
			}
		}

		updatePlot();

		$("#parallel-plot").data("loaded", "true");

		make_text_in_parallel_plot_nicer();
	} catch (err) {
		console.error("Error in createParallelPlot:", err);
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
