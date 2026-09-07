var invalid_names = ["generation_node", "start_time", "end_time", "hostname", "signal", "exit_code", "run_time", "program_string", "arm_name", "trial_index", "generation_method", "trial_status", "idxs", "submit_time", "queue_time", "worker_generator_uuid"];
var document_is_ready = false;
var fadeTime = 0;
var fadeTimeAfterLoading = 300;

function is_visible(e) {
	if (!e || e.nodeType !== 1) return false;
	if (!document.body.contains(e)) return false;

	let style = e.ownerDocument.defaultView.getComputedStyle(e);
	if (style.display === "none" || style.visibility === "hidden" || style.opacity === "0") return false;

	let parent = e.parentElement;
	while (parent && parent !== document.body) {
		let ps = parent.ownerDocument.defaultView.getComputedStyle(parent);
		if (ps.display === "none" || ps.visibility === "hidden" || ps.opacity === "0") return false;
		parent = parent.parentElement;
	}

	return true;
}

function normalizeFloat(value) {
	if (!isFinite(value)) {
		return '';
	}

	var str = value.toString();

	if (str.includes('e')) {
		var fixed = value.toFixed(20);
		fixed = fixed.replace(/(\.\d*?[1-9])0+$/, '$1');
		fixed = fixed.replace(/\.0+$/, '');
		return fixed;
	}

	return str;
}

function get_invalid_names () {
	var gin = JSON.parse(JSON.stringify(invalid_names));

	let _element = document.getElementById("result_names");

	if (_element) {
		let content = $(_element).val().trim();
		let parts = content.split(/\s+/);
		let new_gin = parts.map(item => item.includes('=') ? item.split('=')[0] : item);

		for (var i = 0; i < new_gin.length; i++) {
			gin.push(new_gin[i]);
		}
	}

	return gin;
}

var initialized = false;
var shown_operation_insecure_without_server = false;

var l = typeof log === "function" ? log : console.log;

function input_to_time_picker (input_id) {
	var $input = $("#" + input_id);
	var $parent = $($input).parent();

	if (
		$parent.find(".time_picker_container").length ||
		$parent.find(".time_picker_minutes").length ||
		$parent.find(".time_picker_hours").length
	) {
		log(".time_picker_minutes or .time_picker_hours already found. Not reinstantiating for id " + input_id);
		return;
	}

	var minutes = $input.val();
	var _hours = 0;
	var _minutes = 0;

	if (minutes) {
		_hours = Math.floor(minutes / 60);
		_minutes = minutes % 60;
	}

	var $div = $(`
		    <div class='time_picker_container'>
			<input type='number' min=-1 max=159 class="time_picker_hours" value='${_hours}' onchange='update_original_time_element("${input_id}", this)'></input> Hours,
			<input type='number' min=-1 step=31 class="time_picker_minutes" value='${_minutes}' onchange='update_original_time_element("${input_id}", this)'></input> Minutes
		    </div>
		`);

	$parent.prepend($div);

	$input.hide();
}

function update_original_time_element (original_element_id, new_element) {
	var $parent = $(new_element).parent();
	var $time_picker_minutes = $parent.find(".time_picker_minutes");
	var $time_picker_hours = $parent.find(".time_picker_hours");

	var _minutes = parseInt($time_picker_minutes.val());
	var _hours = parseInt($time_picker_hours.val());

	if (_minutes == -1) {
		if (_hours > 0) {
			_hours = _hours - 1;
			_minutes = 55;
		} else {
			_hours = 0;
			_minutes = 5;
		}
	} else if (_minutes >= 60) {
		_hours = _hours + 1;
		_minutes = 0;
	}

	if (_hours == -1) {
		if (_hours > 1) {
			_hours = _hours - 1;
		} else {
			_hours = 0;
		}
	}

	$time_picker_hours.val(_hours);
	$time_picker_minutes.val(_minutes);

	var new_val = (parseInt(_hours) * 60) + parseInt(_minutes);

	$("#" + original_element_id).val(new_val).trigger("change");
}

function highlight_bash (code) {
	return Prism.highlight(code, Prism.languages.bash, "bash");
}

function highlight_all_bash () {
	$(".highlight_me").each(function (i, e) {
		$(e).html(highlight_bash($(e).text()));
	});
}

function update_partition_options() {
	var partitionSelect = $("#partition");
	partitionSelect.empty();

	$.each(partition_data, function(key, value) {
		partitionSelect.append($("<option></option>")
			.attr("value", key)
			.text(value.name));
	});

	partitionSelect.on("change", function() {
		var partition = $(this).val();
		if(Object.keys(partition_data).includes(partition)) {
			var partitionInfo = partition_data[partition];

			if (partitionInfo) {
				$("#mem_gb").attr("max", Math.floor(partitionInfo.max_mem_per_core / 1000)).each(function() {
					if ($(this).val() > $(this).attr("max")) {
						$(this).val($(this).attr("max"));
					}
				});

				$("#time").attr("max", partitionInfo.computation_time).each(function() {
					if ($(this).val() > $(this).attr("max")) {
						$(this).val($(this).attr("max"));
					}
				});

				$("#worker_timeout").attr("max", partitionInfo.computation_time).each(function() {
					if ($(this).val() > $(this).attr("max")) {
						$(this).val($(this).attr("max"));
					}
				});

				$("#max_eval").attr("min", 1).each(function() {
					if ($(this).val() < $(this).attr("min")) {
						$(this).val($(this).attr("min"));
					}
				});

				$("#num_parallel_jobs").attr("min", 1).each(function() {
					if ($(this).val() < $(this).attr("min")) {
						$(this).val($(this).attr("min"));
					}
				});

				$("#num_parallel_jobs").attr("max", partitionInfo.number_of_workers).each(function() {
					if ($(this).val() > $(this).attr("max")) {
						$(this).val($(this).attr("max"));
					}
				});

				$("#gpus").attr("max", partitionInfo.max_number_of_gpus).each(function() {
					if ($(this).val() > $(this).attr("max")) {
						$(this).val($(this).attr("max"));
					}
				});

				$("#gpus").attr("min", partitionInfo.min_number_of_gpus).each(function() {
					if ($(this).val() < $(this).attr("min")) {
						$(this).val($(this).attr("min"));
					}
				});
			} else {
				error("No partition info");
			}
		} else {
			error(`Cannot find ${partition} in partition_data.`);
		}

		update_url();
	});

	update_url();
}

function set_min_max () {
	document.querySelectorAll("input").forEach(input => {
		if (input.hasAttribute("min") || input.hasAttribute("max")) {
			var _min = input.hasAttribute("min") ? parseFloat(input.getAttribute("min")) : null;
			var _max = input.hasAttribute("max") ? parseFloat(input.getAttribute("max")) : null;

			let value = parseFloat(input.value);

			let red = "#FFE2DE";

			if (_min !== null && (isNaN(value) || value < _min)) {
				if(isNaN(value)) {
					$(input).parent().find("[id$='_error']").html("Value is empty or invalid");
				}

				if(value < _min) {
					$(input).val(_min);
				}
			} else if (_max !== null && value > _max) {
				$(input).val(_max);
			} else {
				$(input).parent().find("[id$='_error']").html("");
			}
		}
	});
}

function quote_variables(input) {
	return input.replace(/(["'])(.*?)\1|%(\((\w+)\)|(\w+))/g, function(match, quotes, insideQuotes, p1, p2, p3) {
		if (quotes) {
			return match;
		} else {
			var variable = p2 || p3;
			return "'%(" + variable + ")'";
		}
	});
}

function get_var_names_from_run_program(run_program_string) {
	const pattern = /(?:\$|\%)?\([a-zA-Z_]+\)|(?:\$|%)[a-zA-Z_]+/g;
	const variableNames = [];

	let match;
	while ((match = pattern.exec(run_program_string)) !== null) {
		let varName = match[0];
		varName = varName.replace(/^(\$|%)/, "");
		varName = varName.replace(/^(\$|%)?\(|\)$/g, "");
		if (/^[a-zA-Z_]+$/.test(varName)) {
			variableNames.push(varName);
		}
	}

	return variableNames;
}

function update_table_row (item, errors, warnings, command) {
	var value = $("#" + item.id).val();

	if(item.regex) {
		var re = new RegExp(item.regex, "i");

		var text = $("#" + item.id).val();

		if(!text.match(re)) {
			var this_error = `The element "${item.id}" does not match regex /${item.regex}/.`;
			if(Object.keys(item).includes("regex_does_not_match_text")) {
				this_error = item.regex_does_not_match_text;
			}
			errors.push(this_error);
			$("#" + item.id + "_error").html(`<img src='i/warning.svg' style='height: 1em' /> ${this_error}`).show()
		} else {
			$("#" + item.id + "_error").html("").hide()
		}
	}

	if (item.type === "checkbox") {
		value = $("#" + item.id).is(":checked") ? "1" : "0";
		if (value === "1") {
			command += " --" + item.id;
		}
	} else if ((item.type === "textarea" || item.type === "text") && value === "") {
		if(item.required) {
			var this_error = "<img src='i/warning.svg' style='height: 1em' /> Field '" + item.label + "' is required.";
			$("#" + item.id + "_error").html(this_error).show();
			$("#" + item.id).css("background-color", "#FFCCCC !important");

			errors.push(this_error);
		}
	} else if (item.id == "time") {
		var worker_timeout_larger_than_global_timeout = parseInt($("#worker_timeout").val()) > parseInt($("#time").val());
		var new_errors = [];
		var numValue = parseFloat(value);

		if (worker_timeout_larger_than_global_timeout) {
			new_errors.push("Worker timeout is larger than global time. Increase global time or decrease worker time.");
		} else if (isNaN(numValue) && ((Object.keys(item).includes("required") && item.required) || !Object.keys(item).includes("required"))) {
			new_errors.push("Invalid value for '" + item.label + "'. Must be a number.");
		} else if (item.min && item.max && (numValue < item.min || numValue > item.max)) {
			new_errors.push("Value for '" + item.label + "' must be between " + item.min + " and " + item.max + ".");
		} else if (item.min && (numValue < item.min)) {
			new_errors.push("Value for '" + item.label + "' must be larger than " + item.min + ".");
		} else if (item.max && (numValue > item.max)) {
			new_errors.push("Value for '" + item.label + "' must be smaller than" + item.max + ".");
		}

		if(new_errors.length) {
			$("#time_error").html(string_or_array_to_list(new_errors)).show();
			errors.push(...new_errors);
		} else {
			$("#time_error").html("").hide()
			command += " --" + item.id + "=" + value;
		}
	} else if (item.id == "max_eval") {
		var parallel_evaluations = parseInt($("#num_parallel_jobs").val());
		var max_eval = parseInt($("#max_eval").val());
		var num_random_steps = parseInt($("#num_random_steps").val());

		if (parallel_evaluations <= 0) {
			$("#num_parallel_jobs").val(1);
		}

		if (max_eval < parallel_evaluations) {
			$("#num_parallel_jobs").val(max_eval);
		}

		if (max_eval < num_random_steps) {
			$("#num_random_steps").val(max_eval);
		}

		if (num_random_steps <= 0) {
			$("#num_random_steps").val(1);
		}

		command += " --" + item.id + "=" + value;
	} else if (item.id == "worker_timeout") {
		var worker_timeout_larger_than_global_timeout = parseInt($("#worker_timeout").val()) > parseInt($("#time").val());
		var new_errors = [];
		var numValue = parseFloat(value);

		if (worker_timeout_larger_than_global_timeout) {
			new_errors.push("Worker timeout is larger than global time. Increase global time or decrease worker time.");
		} else if (isNaN(numValue) && ((Object.keys(item).includes("required") && item.required) || !Object.keys(item).includes("required"))) {
			new_errors.push("Invalid value for '" + item.label + "'. Must be a number.");
		} else if (item.min && item.max && (numValue < item.min || numValue > item.max)) {
			new_errors.push("Value for '" + item.label + "' must be between " + item.min + " and " + item.max + ".");
		} else if (item.min && (numValue < item.min)) {
			new_errors.push("Value for '" + item.label + "' must be larger than " + item.min + ".");
		} else if (item.max && (numValue > item.max)) {
			new_errors.push("Value for '" + item.label + "' must be smaller than" + item.max + ".");
		}

		if(new_errors.length) {
			$("#worker_timeout_error").html(string_or_array_to_list(new_errors)).show();
			errors.push(...new_errors);
		} else {
			$("#worker_timeout_error").html("").hide()
			command += " --" + item.id + "=" + value;
		}
	} else if (item.type === "number") {
		var numValue = parseFloat(value);

		if (isNaN(numValue) && ((Object.keys(item).includes("required") && item.required) || !Object.keys(item).includes("required"))) {
			errors.push("Invalid value for '" + item.label + "'. Must be a number.");
		} else if (item.min && item.max && (numValue < item.min || numValue > item.max)) {
			errors.push("Value for '" + item.label + "' must be between " + item.min + " and " + item.max + ", is " + numValue + ".");
		} else if (item.min && (numValue < item.min)) {
			errors.push("Value for '" + item.label + "' must be larger than " + item.min + ".");
		} else if (item.max && (numValue > item.max)) {
			errors.push("Value for '" + item.label + "' must be smaller than" + item.max + ".");
		} else {
			value = numValue.toString();
			if(value != "NaN") {
				if (item.type == "number" || value.matches(/^[a-zA-Z0-9=_]+$/)) {
					command += " --" + item.id + "=" + value;
				} else {
					command += " --" + item.id + "='" + value + "'";
				}
			}
		}
	} else if (item.id == "external_generator") {
		command += " --" + item.id + "='" + encode_base64(value) + "'";
	} else if (item.id == "run_program_once") {
		command += " --" + item.id + "='" + encode_base64(value) + "'";
	} else if (item.id == "run_program") {
		// When a formula is active, the run_program is auto-generated from
		// it, so the placeholder-vs-parameter cross-check is misleading.
		// Skip it entirely in that case so the scientist can leave an
		// old run_program around without seeing red.
		var formulaIsActive = ($("#formula").val() || "").trim() !== "";
		if (!formulaIsActive) {
			var variables_in_run_program = get_var_names_from_run_program(value);
			//value = quote_variables(value);

			var existing_parameter_names = $(".parameterName").map(function() {
				const val = $(this).val();
				var ret = null;
				if(!/^\s*$/.test(val) && /^[a-zA-Z_]+$/.test(val)) {
					ret = val;
				}
				return ret;
			}).get().filter(Boolean);

			var new_errors = [];

			for (var k = 0; k < variables_in_run_program.length; k++) {
				var test_this_var_name = variables_in_run_program[k];

				if(!existing_parameter_names.includes(test_this_var_name)) {
					var err_msg = `<img src='i/warning.svg' style='height: 1em' /> <code>%(${test_this_var_name})</code> not in existing defined parameters.`;
					new_errors.push(err_msg);
				}
			}

			for (var k = 0; k < existing_parameter_names.length; k++) {
				var test_this_var_name = existing_parameter_names[k];

				if(!variables_in_run_program.includes(test_this_var_name)) {
					var err_msg = `<img src='i/warning.svg' style='height: 1em' /> <code>%(${test_this_var_name})</code> is defined but not used.`;
					new_errors.push(err_msg);
				}
			}

			if(new_errors.length) {
				$("#run_program_error").html(string_or_array_to_list(new_errors)).show()
				errors.push(...new_errors);
			} else {
				$("#run_program_error").html("").hide()
			}
		} else {
			$("#run_program_error").html("").hide()
		}

		value = encode_base64(value);

		command += " --" + item.id + "='" + value + "'";
		$("#" + item.id).css("background-color", "");
	} else {
		if(!errors.length) {
			if (item.id != "constraints") {
				if (item.id == "result_names") {
					value = value.replace(/\s+/g, ' ').trim()
					command += ` --${item.id} '${value}'`
				} else {
					command += " --" + item.id + "=" + value;
				}
				$("#" + item.id + "_error").html("").hide()
				$("#" + item.id).css("background-color", "");
			}
		}
	}

	return [command, errors, warnings];
}

function set_row_background_color_red_color(_row) {
	log("_row:", _row);
	console.trace();
	$(_row).css("background-color", "#ffabab").addClass("invert_in_dark_mode");
}

function is_invalid_parameter_name(name) {
	if(name.startsWith("OO_Info_")) {
		return true;
	}

	var gin = get_invalid_names();

	if (gin.includes(name)) {
		return true;
	}

	return false;
}

function show_warnings_and_errors(warnings, errors) {
	var warnings_html = "";
	var errors_html = "";

	var img_warning = "<img src='i/warning.svg' width=16 />";

	function formatMessages(messages, title, color) {
		if (!Array.isArray(messages) || messages.length === 0) return '';
		return `<h2 class="invert_in_dark_mode" style="color: ${color};">${title}:</h2><ul>` + 
			messages.map(msg => {
				if (msg.startsWith('<img')) {
					msg = msg.replace(/<img[^>]*>/, '');
				}
				return `<li style="color: ${color};">${img_warning} <span class="invert_in_dark_mode">${msg}</span></li>`;
			}).join('') +
			'</ul>';
	}

	warnings_html = formatMessages(warnings, 'Warnings', 'orange');
	errors_html = formatMessages(errors, 'Errors', 'red');

	var content = warnings_html + errors_html;
	var warnings_element = $('#warnings');

	warnings_element.html(content);

	if (content.length > 0) {
		warnings_element.show();
	} else {
		warnings_element.hide();
	}

	apply_theme_based_on_system_preferences();
}

function update_command() {
	set_min_max();

	var errors = [];
	var warnings = [];
	var command = "./omniopt";

	if($("#installation_method").val() == "pip") {
		command = "omniopt";
	}

	var curl_options = "";

	if ($("#run_mode").val() == "docker") {
		command = "bash omniopt_docker omniopt";
	}

	var formulaHandledIds = new Set(["formula", "formula_mode"]);
	function processTableData(_tableData) {
		_tableData.forEach(function(item) {
			if (formulaHandledIds.has(item.id)) return;
			if (!item.use_in_curl_bash) {
				var command_error_and_warning = update_table_row(item, errors, warnings, command);
				command = command_error_and_warning[0];
				errors = command_error_and_warning[1];
				warnings = command_error_and_warning[2];
			} else {
				if (item.type == "select") {
					curl_options = ` --${item.id}=${$(`#${item.id}`).val()} `;
				} else if (item.type == "checkbox" && $(`#${item.id}`).is(":checked")) {
					curl_options = ` --${item.id} `;
				} else {
					error("use_in_curl_bash currently only supports select and checkbox");
				}
			}
		});
	}

	processTableData(tableData);
	processTableData(hiddenTableData);

	// Custom handling for the formula editor (lives outside tableData).
	(function handleFormula() {
		var formulaText = ($("#formula").val() || "").trim();
		var formulaMode = ($("#formula_mode").val() || "auto");
		if (formulaText) {
			try {
				var b64 = btoa(unescape(encodeURIComponent(formulaText)));
				command += " --formula='" + b64 + "'";
				if (formulaMode && formulaMode !== "auto") {
					command += " --formula_mode=" + formulaMode;
				}
			} catch (e) {
				console.error("Base64 encoding failed for formula:", e);
			}
		}
	})();

	var parameters = [];

	var i = 0;
	var parameter_names = [];

	$(".parameterRow").each(function() {
		var option = $(this).find(".optionSelect").val();
		var parameterName = $(this).find(".parameterName").val().trim();
		var _value;

		var warn_msg = [];

		if(parameter_names.includes(parameterName)) {
			var err_msg = `Parameter name "${parameterName}" already exists. Can only be defined once!`;
			warn_msg.push(err_msg);

			$($(".parameterRow")[i]).css("background-color", "#e57373");
		} else if(parameterName && !parameterName.match(/^[a-zA-Z_]+$/)) {
			warn_msg.push("Name contains invalid characters. Must be all-letters.");
		} else if(is_invalid_parameter_name(parameterName)) {
			warn_msg.push(`Name is or contains a reserved keyword, cannot be any of those: <tt>${invalid_names.join(', ')}</tt>, or any of the names specified in the results-names.`);
		} else if(parameterName.match(/^[a-zA-Z_]+$/)) {
			if (option === "range") {
				var $this = $(this);
				//log("$this.find('.minValue').val():", $this.find(".minValue").val());

				var minValue = normalizeFloat(parseFloat($this.find(".minValue").val()));
				var maxValue = normalizeFloat(parseFloat($this.find(".maxValue").val()));

				var numberType = $($(".parameterRow")[i]).find(".numberTypeSelect").val();

				if (minValue === maxValue) {
					warn_msg.push("Warning: The minimum and maximum values for parameter " + parameterName + " are equal.");
				}

				var is_ok = true;

				//log("minValue:", minValue);

				if(isNaN(minValue)) {
					warn_msg.push("<img src='i/warning.svg' style='height: 1em' /><i>minValue</i> for parameter <i>" + parameterName + "</i> is not a number.");
					is_ok = false;
				}

				if(isNaN(maxValue)) {
					warn_msg.push("<img src='i/warning.svg' style='height: 1em' /><i>maxValue</i> for parameter <i>" + parameterName + "</i> is not a number.");
					is_ok = false;
				}

				if(maxValue == "") {
					warn_msg.push("<img src='i/warning.svg' style='height: 1em' /><i>maxValue</i> for parameter <i>" + parameterName + "</i> is empty or not a number.");
					is_ok = false;
				}

				if(minValue == "") {
					warn_msg.push("<img src='i/warning.svg' style='height: 1em' /><i>minValue</i> for parameter <i>" + parameterName + "</i> is empty or not a number.");
					is_ok = false;
				}

				if (numberType == "int") {
					var parsed_int_max = parseInt(maxValue);
					if (parsed_int_max != maxValue) {
						warn_msg.push("maxValue is not an integer");
					}

					var parsed_int_min = parseInt(minValue);
					if (parsed_int_min != minValue) {
						warn_msg.push("minValue is not an integer");
					}
				}

				var log_scale = $($(".parameterRow")[i]).find(".log_scale").is(":checked") ? "true" : "false";

				if(is_ok) {
					_value = `${parameterName} range ${minValue} ${maxValue} ${numberType} ${log_scale}`;
				}
			} else if (option === "choice") {
				var choiceValues = $(this).find(".choiceValues").val();

				if(choiceValues !== undefined) {
					choiceValues = choiceValues.replaceAll(/\s/g, ",");
					choiceValues = choiceValues.replaceAll(/,,*/g, ",");
					choiceValues = choiceValues.replaceAll(/,,*$/g, "");
					choiceValues = choiceValues.replaceAll(/^,,*/g, "");

					choiceValues = [...new Set(choiceValues.split(","))].join(",");

					_value = `${parameterName} choice ${choiceValues}`;

					if(!choiceValues.match(/./)) {
						warn_msg.push("Values are missing.");
					}
				} else {
					warn_msg.push("choiceValues not defined.");
				}
			} else if (option === "fixed") {
				var fixedValue = $(this).find(".fixedValue").val();

				if(typeof(fixedValue) == "string") {
					fixedValue = fixedValue.replace(/,*$/g, "");
					fixedValue = fixedValue.replace(/,+/g, ",");

					fixedValue = Array.from(new Set(fixedValue.split(","))).join(",");

					_value = `${parameterName} fixed ${fixedValue}`;
				}

				if(fixedValue === undefined) {
					warn_msg.push("<img src='i/warning.svg' style='height: 1em' /> <i>Value</i> is missing.");
				} else if(!fixedValue.match(/./)) {
					warn_msg.push("<img src='i/warning.svg' style='height: 1em' /> <i>Value</i> is missing.");
				} else if(!fixedValue.match(/^[a-zA-Z0-9\.,_]+$/)) {
					warn_msg.push("Invalid values. Must match Regex /[a-zA-Z0-9,_\.]/.");

				}
			}

			if (parameterName && _value) {
				parameters.push(_value);
				parameter_names.push(parameterName);

				if(!warn_msg.length) {
					$($(".parameterRow")[i]).css("background-color", "");
				}
			} else {
				if(!parameterName) {
					warn_msg.push("No parameter name");
				}
			}
		} else {
			warn_msg.push("<img src='i/warning.svg' style='height: 1em' /> Parameter option <i>Name</i> is missing.");
		}

		if(warn_msg.length) {
			$($(".parameterError")[i]).html(string_or_array_to_list(warn_msg)).show()
		} else {
			$($(".parameterError")[i]).html("").hide();
		}

		errors.push(...warn_msg);

		i++;
	});

	if (parameters.length > 0) {
		command += " --parameter " + parameters.join(" --parameter ");
	}

	if ($("#constraints").val()) {
		var constraints_string = $("#constraints").val();

		constraints_string = constraints_string.replaceAll(/;;*/g, ";");
		constraints_string = constraints_string.replace(/;;*$/, "");
		constraints_string = constraints_string.replace(/^;;*/, "");

		var _constraints = constraints_string.split(";");
		_constraints = _constraints.filter(function(entry) { return entry.trim() != ""; }).map(function (el) {
			return el.trim();
		});
		for (var r = 0; r < _constraints.length; r++) {
			command += " --experiment_constraints '" + encode_base64(add_equation_spaces(_constraints[r])) + "'";
		}

		var constraints_string = $("#constraints").val();
		var errors_string = is_valid_constraints_string(constraints_string);

		if(isAnyLogScaleSet()) {
			errors_string += "Cannot set constraints if one or more log-scale parameters are there."
		}

		if (errors_string != "") {
			errors.push("Something was wrong in the constraints parameter");
			$("#constraints_error").html(errors_string).show()
		} else {
			$("#constraints_error").html("").hide()
		}
	} else {
		$("#constraints_error").html("");
	}

	var errors_visible = false;
	$(".parameterError").each(function (i, e) {
		if(is_visible($(e))) {
			errors_visible = true;
		}
	});

	// Cross-field: either Run program OR the formula must be filled.
	var rp = ($("#run_program").val() || "").trim();
	var fm = ($("#formula").val() || "").trim();
	// Highlight the offending field(s) in red so the user immediately
	// sees which one they need to fill.  The class is removed as soon
	// as the field has content, so the indicator is reactive.
	$("#run_program").toggleClass("field_missing", !rp && !fm);
	$("#formula").toggleClass("field_missing", !rp && !fm);
	$("#formula_pane_text, #formula_pane_python").toggleClass("field_missing", !rp && !fm);
	if (!rp && !fm) {
		errors.push("<img src='i/warning.svg' style='height: 1em' /> Either <i>Run program</i> or the <i>Formula editor</i> must be filled.");
	}

	if (!errors.length && $(".optionSelect").length && !errors_visible) {
		var base_url = location.protocol + "//" + location.host + "/" + location.pathname + "/";

		base_url = base_url.replaceAll(/\/+/g, "/");

		base_url = base_url.replace(/^http:\//, "http://");
		base_url = base_url.replace(/^https:\//, "https://");
		base_url = base_url.replace(/^file:\//, "file://");

		var base_64_string = encode_base64(command);

		var curl_or_cat = "curl";

		if (base_url.startsWith("file://")) {
			curl_or_cat = "cat";

			var filename = location.pathname.substring(location.pathname.lastIndexOf("/")+1);

			var _re_ = new RegExp(`${filename}/?$`);

			base_url = base_url.replace(_re_, "");

			base_url = base_url.replace(/^file:\//, "/");
			base_url = base_url.replace(/^\/\//, "/");
		}

		base_url = base_url.replace(/\/index(?:.php)?/, "");
		base_url = base_url.replace(/\/gui(?:.php)?/, "");

		var curl_command = "";
		var command_end = ` | bash -l -s -- "${base_64_string}"${curl_options}`;

		if(curl_or_cat == "curl") {
			curl_command = `${curl_or_cat} ${base_url}install_omniax.sh 2>/dev/null${command_end}`;
		} else {
			curl_command = `${curl_or_cat} ${base_url}install_omniax.sh${command_end}`;
		}

		nicer_command = addBase64DecodedVersions(command);

		$("#curl_command_highlighted").text(curl_command).show();
		$("#command_element_highlighted").text(nicer_command).show();
		$("#install_and_run").show();
		$("#only_run").show();

		$("#curl_command").text(curl_command);
		$("#command_element").text(nicer_command);
	} else {
		$("#command_element_highlighted").hide();
		$("#curl_command_highlighted").hide();
		$("#install_and_run").hide();
		$("#only_run").hide();

		$("#curl_command").text("");
		$("#command_element").text("");
	}

	show_warnings_and_errors(warnings, errors);

	update_url();

	_schedule_formula_preview_refresh();
}

function encode_base64 (v) {
	return btoa(v);
}

function decode_base64 (input) {
	decoded = atob(input);

	return decoded;
}

function is_base64_like(s) {
	// Only base64-decode strings that consist entirely of base64
	// characters.  This protects against the recursive re-decode
	// path catching ``$(echo ...`` (the previous output of this
	// function) and trying to decode it.
	return typeof s === "string" && s.length >= 4 && /^[A-Za-z0-9+/]+={0,2}$/.test(s);
}

function addBase64DecodedVersions(cmdString) {
	return cmdString.replace(/(--[a-zA-Z0-9_]+)=('([^']+)'|"([^"]+)"|([^\s]+))/g, (match, key, _, singleQuoted, doubleQuoted, bare) => {
		const value = singleQuoted || doubleQuoted || bare;

		let decoded = null;
		try {
			if (
				(key === "--run_program" || key === "--run_program_once" || key === "--formula") &&
				is_base64_like(value)
			) {
				decoded = decode_base64(value);
			}
		} catch (e) {
			console.error(e);
		}

		if (decoded) {
			var safeDecoded = decoded.replace(/\x27/g, `'\\''`).trim();
			return ` ${key}=$(printf '%s' '${safeDecoded}' | base64 -w0)`;
		} else {
			return match;
		}
	});
}

// ---------------------------------------------------------------------------
// Formula card (left: tabs + textarea, right: live parameter suggestions)
// ---------------------------------------------------------------------------

function build_formula_card_html() {
	return (
		"<div style='display: flex; gap: 12px; flex-wrap: wrap; margin-top: 6px;'>" +
		"<div id='formula_card_left' style='flex: 1 1 380px; min-width: 320px;'>" +
		"<div style='display: flex; gap: 6px; margin-bottom: 6px; align-items: center;'>" +
		"<button type='button' id='formula_tab_text' class='formula_tab' data-mode='text'>Formula</button>" +
		"<button type='button' id='formula_tab_python' class='formula_tab' data-mode='python'>Python</button>" +
		"</div>" +
		"<div id='formula_panel_text'>" +
		"<textarea id='formula_pane_text' placeholder=\"f(x, y) = \\frac{x}{y} + \\sin(a)   or   f(x) = 2*x + y\" style='width: 100%; min-height: 80px; font-family: monospace;'></textarea>" +
		"</div>" +
		"<div id='formula_panel_python' style='display: none;'>" +
		"<textarea id='formula_pane_python' placeholder=\"def evaluate(params):&#10;    return math.sin(params['a']*params['x']) + params['b']\" style='width: 100%; min-height: 110px; font-family: monospace;'></textarea>" +
		"<div style='margin-top: 4px; font-size: 0.85em; color: #555;'>" +
		"Python tab: define <code>evaluate(params)</code> returning a float. The <code>params</code> dict also has a <code>'_raw'</code> key with the raw values." +
		"</div>" +
		"</div>" +
		"<div id='formula_hint' style='margin-top: 6px; font-size: 0.85em; color: #555;'></div>" +
		"<div id='formula_preview' style='margin-top: 8px; padding: 8px 8px 16px 8px; background: #fff; border: 1px dashed #c0c0d0; border-radius: 8px; min-height: 50px; font-size: 1.05em; overflow: visible;'></div>" +
		"<div id='formula_param_legend' style='margin-top: 4px; font-size: 0.82em; color: #444; min-height: 1.2em;'></div>" +
		"<div id='formula_error' style='margin-top: 4px; font-size: 0.85em; color: #b00020;'></div>" +
		"</div>" +
		"<div id='formula_card_right' style='flex: 0 0 380px; min-width: 320px; padding: 10px; background: #fafaff; border: 1px solid #d6d6e6; border-radius: 10px;'>" +
		"<h4 style='margin-top: 0; margin-bottom: 6px;'>Suggested parameters</h4>" +
		"<div id='formula_suggestions' style='font-size: 0.9em; color: #444;'><em>Type a formula to see suggestions…</em></div>" +
		"<div style='margin-top: 10px; display: flex; gap: 6px;'>" +
		"<button type='button' id='formula_apply_btn' class='formula_tab' style='background:#4a90d9;color:#fff;'>Apply → add as parameters</button>" +
		"<button type='button' id='formula_clear_btn' class='formula_tab'>Clear formula</button>" +
		"</div>" +
		"<div style='margin-top: 10px; font-size: 0.8em; color: #777;'>" +
		"<b>Parameters</b> are variables that appear on the left-hand side of the formula (e.g. <code>x</code> in <code>f(x) = …</code>).<br>" +
		"<b>Constants</b> are variables that appear only on the right-hand side — they keep their default value.<br>" +
		"Variables bound by <code>\\sum</code> or <code>\\prod</code> are excluded automatically." +
		"</div>" +
		"</div>" +
		"</div>"
	);
}


// Very small JS-side formula parser used to extract suggested parameter
// names.  It is intentionally permissive: it identifies every identifier-like
// token and removes ones that are obviously sympy built-ins or known math
// constants.  The Python side does the authoritative parse; this is just for
// quick client-side feedback.
function _strip_underbraces(text) {
	var result = "";
	var i = 0;
	var marker = "\\underbrace{";
	while (i < text.length) {
		var idx = text.indexOf(marker, i);
		if (idx === -1) {
			result += text.substring(i);
			break;
		}
		result += text.substring(i, idx);
		var pos = idx + marker.length;
		var depth = 1;
		while (pos < text.length && depth > 0) {
			if (text[pos] === "{") depth++;
			else if (text[pos] === "}") depth--;
			pos++;
		}
		var content = text.substring(idx + marker.length, pos - 1);
		result += content;
		if (pos < text.length && text[pos] === "_") pos++;
		if (pos < text.length && text[pos] === "{") {
			depth = 1;
			pos++;
			while (pos < text.length && depth > 0) {
				if (text[pos] === "{") depth++;
				else if (text[pos] === "}") depth--;
				pos++;
			}
		}
		i = pos;
	}
	return result;
}

var FORMULA_RESERVED = new Set([
	// sympy constants
	"pi", "E", "I", "oo", "inf", "infty", "nan", "NaN", "True", "False",
	// common mathematical functions
	"sin", "cos", "tan", "asin", "acos", "atan",
	"sinh", "cosh", "tanh",
	"exp", "log", "ln", "sqrt", "abs", "Min", "Max",
	"Sum", "Product", "Integral", "Derivative",
	"max", "min", "norm", "trace", "det", "rank",
	"ReLU", "relu", "softmax", "sigmoid", "tanh",
	"clip", "clamp", "ceil", "floor", "round",
	"argmax", "argmin", "argsort",
	// python built-ins that often leak in
	"math", "numpy", "np", "self", "def", "return", "import", "from",
	"if", "else", "elif", "for", "while", "in", "and", "or", "not",
	"params", "evaluate", "_raw",
	// LaTeX commands and environments
	"underbrace", "overbrace", "substack", "mathbb", "text", "mathrm",
	"mathbf", "mathcal", "operatorname", "left", "right", "displaystyle",
	"quad", "qquad", "cdot", "times", "limits", "hat", "bar",
	"vec", "dot", "tilde", "widehat", "overline", "underline",
	"int", "oint", "iint", "forall", "exists", "partial",
	"frac", "dfrac", "tfrac", "sqrt", "begin", "end",
	"cases", "sim", "propto", "approx", "neq", "leq", "geq",
	"to", "rightarrow", "mapsto", "sum", "prod",
]);

var FORMULA_CONSTANTS = {
	pi: 3.141592653589793,
	e: 2.718281828459045,
	E: 2.718281828459045,
	PI: 3.141592653589793,
};

// Strip formatting/structural macros that obscure identifiers.
function _strip_macros(text) {
	// \frac{num}{den} → (num)/(den)
	text = text.replace(/\\(?:frac|dfrac|tfrac)\s*\{([^{}]*)\}\s*\{([^{}]*)\}/g, "($1)/($2)");
	// \sqrt[n]{x} or \sqrt{x} → (x)
	text = text.replace(/\\sqrt\s*(?:\[[^[\]]*\])?\s*\{((?:[^{}]|\{[^{}]*\})*)\}/g, "($1)");
	// \text{...}, \mathbb{...}, etc. → keep content
	text = text.replace(/\\(?:text|textit|textbf|mathrm|operatorname|mathbf|mathcal|mathbb|mathfrak|mathsf|mathtt|mbox|boldsymbol)\*?\s*\{((?:[^{}]|\{[^{}]*\})*)\}/g, "$1");
	// \| → remove (norm delimiters)
	text = text.replace(/\\\|/g, "");
	// \max, \min, \argmax, \argmin → remove name, keep args
	text = text.replace(/\\(?:max|min|argmax|argmin|norm|trace|det)\b/g, "");
	// \begin{...}, \end{...} → remove
	text = text.replace(/\\(?:begin|end)\s*\{[^{}]*\}/g, " ");
	// \sim, \propto, \approx, \neq, \leq, \geq, \to, \left, \right → remove
	text = text.replace(/\\(?:sim|propto|approx|neq|leq|geq|to|rightarrow|mapsto|left|right|displaystyle|textstyle)\b[|(\[.]?/g, " ");
	// \sin, \cos, etc. → keep name (filtered by FORMULA_RESERVED)
	text = text.replace(/\\(sin|cos|tan|asin|acos|atan|sinh|cosh|tanh|exp|log|ln|abs)\b/g, "$1");
	// Greek letters → ASCII names (these ARE parameter names)
	text = text.replace(/\\(alpha|beta|gamma|delta|epsilon|zeta|eta|theta|vartheta|iota|kappa|lambda|mu|nu|xi|omicron|rho|sigma|tau|upsilon|phi|varphi|chi|psi|omega|Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Phi|Psi|Omega)\b/g, "$1");
	// \sum, \prod, \int, etc. → remove (handled by _strip_sumprod_bodies)
	text = text.replace(/\\(?:sum|prod|int|oint|iint)\b/g, "");
	// Any remaining \command → space
	text = text.replace(/\\[A-Za-z]+/g, " ");
	// Clean up: {single_var} → var
	text = text.replace(/\{([A-Za-z_][A-Za-z0-9_]*)\}/g, "$1");
	return text;
}

// Collect identifiers from a piece of text, ignoring reserved ones and
// capturing duplicates (so we can detect overlap between left-hand and right-hand sides).
function _collect_identifiers(text) {
	var re = /[A-Za-z_][A-Za-z0-9_]*/g;
	var out = [];
	var seen = {};
	var m;
	while ((m = re.exec(text)) !== null) {
		var name = m[0];
		if (name === "_") continue;
		// Normalize indexed variables: x_1, x_2 → x
		var baseName = name.replace(/_\d+$/, "");
		if (baseName.length > 0 && baseName !== name) name = baseName;
		if (FORMULA_RESERVED.has(name)) continue;
		out.push(name);
		seen[name] = (seen[name] || 0) + 1;
	}
	return { list: out, set: Object.keys(seen) };
}

// Split a LaTeX/infix formula on `=` into {lhs, rhs}.  We use the FIRST `=`
// that is not inside a brace block and not a comparator (we only handle
// assignments here, so any `==`, `<=`, `>=` would also be picked up but the
// caller can deal with it).  Returns {lhs: "", rhs: text} if no `=` exists.
function _split_assignment(text) {
	// Walk through char-by-char, tracking brace depth.  Use the LAST
	// top-level ``=`` as the LHS / RHS separator so chained assignments
	// like ``a + b = c = c - d`` (mathematically nonsensical, but valid
	// Python) land as ``lhs = "a + b = c"``, ``rhs = "c - d"`` — the
	// outermost assignment is what controls the parameter list.
	var depth = 0;
	var lastIdx = -1;
	for (var i = 0; i < text.length; i++) {
		var ch = text[i];
		if (ch === "{") depth++;
		else if (ch === "}") depth--;
		else if (ch === "=" && depth === 0) {
			lastIdx = i;
		}
	}
	if (lastIdx >= 0) {
		return { lhs: text.substring(0, lastIdx), rhs: text.substring(lastIdx + 1) };
	}
	return { lhs: "", rhs: text };
}

// Strip the function name + parens from a left-hand side like `f(x, y)` or `f(g(x))`
// so we are left with the actual parameter list inside the outermost parens.
// Returns an empty list if the LHS isn't shaped like a function definition
// (no ``(`` present, or the LHS contains more than one identifier / operator).
function _lhs_parameter_names(lhs) {
	var open = lhs.indexOf("(");
	if (open < 0) return [];
	var close = lhs.lastIndexOf(")");
	if (close < open) return [];
	var name = lhs.substring(0, open).trim();
	// The LHS must look like ``f`` or ``f.f`` (a function name, not an
	// arithmetic expression).  This guards against ``a + b = c = c - d``
	// being interpreted as parameters.
	if (!/^[A-Za-z_][A-Za-z0-9_.]*$/.test(name)) return [];
	var inner = lhs.substring(open + 1, close);
	return _collect_identifiers(_strip_macros(inner)).list;
}

// Extract bound variable names from \sum_{...}, \prod_{...}, \int_{...},
// \mathbb{E}_{...}, \oint_{...}, \iint_{...}.
// Returns { text: text (unchanged), bound: {name: true, ...} }.
function _strip_sumprod_bodies(text) {
	var boundNames = {};
	// Match \sum_{...}, \prod_{...}, \int_{...}, \oint_{...}, \iint_{...}
	var re1 = /\\(?:sum|prod|int|oint|iint)\s*_\s*\{([^{}]+)\}/g;
	var m;
	while ((m = re1.exec(text)) !== null) {
		var sub = m[1];
		var name = sub.indexOf("=") >= 0 ? sub.split("=")[0].trim() : sub.trim();
		if (/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) boundNames[name] = true;
	}
	// Match \sum_var, \int_var (single char subscript without braces)
	var re2 = /\\(?:sum|prod|int|oint|iint)\s*_([A-Za-z_][A-Za-z0-9_]*)/g;
	while ((m = re2.exec(text)) !== null) {
		var name = m[1];
		if (name.indexOf("=") >= 0) name = name.split("=")[0].trim();
		if (/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) boundNames[name] = true;
	}
	// Match \mathbb{E}_{x\sim...} or E_{x~...}
	var re3 = /(?:\\mathbb\{E\}|E)\s*_\s*\{([^{}]+)\}/g;
	while ((m = re3.exec(text)) !== null) {
		var sub = m[1];
		var name = sub.split(/\\sim|~|=|,/)[0].trim();
		if (/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) boundNames[name] = true;
	}
	return { text: text, bound: boundNames };
}


// Extract parameter names from a Python-mode formula by scanning for
// ``params['x']`` / ``params["x"]`` / ``params.get('x')`` patterns.
function client_extract_python_params(text) {
	if (!text) return [];
	var seen = {};
	var out = [];
	// ``params\.get\(`` for the ``params.get('x')`` form, ``params\[`` for
	// ``params['x']``, with optional trailing ``)`` / ``]`` so we still
	// close the call.
	var re = /params(?:\s*\.\s*get\s*\()?\s*\[?\s*['"]([A-Za-z_][A-Za-z0-9_]*)['"]\s*\)?/g;
	var m;
	while ((m = re.exec(text)) !== null) {
		var n = m[1];
		if (!seen[n]) {
			seen[n] = true;
			out.push(n);
		}
	}
	return out.map(function (n) {
		if (n.endsWith("_int")) {
			return { name: n, kind: "range", lower: 0, upper: 10, value_type: "int", log_scale: false };
		}
		if (n.startsWith("lr_") || n.startsWith("log_") || n.endsWith("_log")) {
			return { name: n, kind: "range", lower: 1e-5, upper: 1e-1, value_type: "float", log_scale: true };
		}
		return { name: n, kind: "range", lower: -1, upper: 1, value_type: "float", log_scale: false };
	});
}


function client_extract_formula_params(text, mode) {
	// Guard rail: any thrown error must fall back to an empty suggestion
	// list so the rest of the GUI still works (no broken `apply` button).
	try {
		return client_extract_formula_params_impl(text, mode);
	} catch (e) {
		try {
			console.error("[formula extraction] failed:", e, "for text:", text);
		} catch (_) { /* swallow */ }
		return { parameters: [], constants: [], bound: [], error: (e && e.message) ? e.message : String(e) };
	}
}

function client_extract_formula_params_impl(text, mode) {
	if (!text || !text.trim()) return { parameters: [], constants: [], bound: [] };
	if (typeof text !== "string") return { parameters: [], constants: [], bound: [] };
	var raw = _strip_underbraces(text);

	var split;
	try { split = _split_assignment(raw); }
	catch (e) { return { parameters: [], constants: [], bound: [] }; }
	var lhsRaw = split.lhs;
	var rhsRaw = split.rhs;

	var boundInfo;
	try { boundInfo = _strip_sumprod_bodies(raw); }
	catch (e) { return { parameters: [], constants: [], bound: [] }; }
	var boundNames = (boundInfo && boundInfo.bound) || {};

	// Left-hand side: find the parameter names of the function (variables
	// that appear inside the outermost parentheses of the left side).
	// If there is no LHS, every free symbol in the RHS becomes a parameter
	// (otherwise bare expressions like ``a + sin(b)`` would lose all of
	// their variables to the ``constant`` bucket).
	var lhsIdents = [];
	if (lhsRaw.trim().length > 0) {
		try { lhsIdents = _lhs_parameter_names(lhsRaw) || []; }
		catch (e) { lhsIdents = []; }
	}
	var lhsSet = {};
	for (var i = 0; i < lhsIdents.length; i++) lhsSet[lhsIdents[i]] = true;

	// Right-hand side: strip the function macros, collect identifiers.
	var rhsClean = "";
	try { rhsClean = _strip_macros(boundInfo.text); } catch (e) {}
	// Re-locate `=` for the stripped right-hand side — the bound-stripping
	// may have shifted positions.
	var rhsOnly = rhsClean;
	try { rhsOnly = _split_assignment(rhsClean).rhs; } catch (e) {}
	var rhsInfo;
	try { rhsInfo = _collect_identifiers(rhsOnly); } catch (e) { rhsInfo = { list: [], set: [] }; }
	var rhsIdents = rhsInfo.list || [];

	// Build the parameter and constant lists.
	var parameters = [];
	var constants = [];
	var seenP = {};
	var seenC = {};
	var hasLhs = lhsIdents.length > 0;
	for (var j = 0; j < lhsIdents.length; j++) {
		var n = lhsIdents[j];
		if (!seenP[n]) {
			seenP[n] = true;
			parameters.push(n);
		}
	}
	for (var k = 0; k < rhsIdents.length; k++) {
		var rn = rhsIdents[k];
		if (!rn) continue;
		if (boundNames[rn]) continue;          // bound by sum/prod
		if (lhsSet[rn]) continue;              // already a parameter
		if (FORMULA_RESERVED && FORMULA_RESERVED.has && FORMULA_RESERVED.has(rn)) continue; // safety
		if (!seenP[rn] && !seenC[rn]) {
			seenP[rn] = true;
			seenC[rn] = true;
			// With an LHS, RHS-only identifiers are *constants* (so the user
			// sees the split between parameters and fixed values).
			// Without an LHS, every free symbol is a *parameter* (so a bare
			// expression like ``a + sin(b)`` ends up with ``a`` and ``b``
			// as optimisable ranges rather than mysterious constants).
			if (hasLhs) {
				constants.push(rn);
			} else {
				parameters.push(rn);
			}
		}
	}

	function asSuggestion(name, kind) {
		if (Object.prototype.hasOwnProperty.call(FORMULA_CONSTANTS, name)) {
			return {
				name: name,
				kind: "fixed",
				lower: FORMULA_CONSTANTS[name],
				upper: FORMULA_CONSTANTS[name],
				value_type: "float",
				log_scale: false,
				note: "well-known constant (default value preserved; switch to range to optimize)",
			};
		}
		if (name.endsWith("_int")) {
			return { name: name, kind: kind, lower: 0, upper: 10, value_type: "int", log_scale: false };
		}
		if (name.startsWith("lr_") || name.startsWith("log_") || name.endsWith("_log")) {
			return { name: name, kind: kind, lower: 1e-5, upper: 1e-1, value_type: "float", log_scale: true };
		}
		// Constants default to 1.0 (so multiplying by them has a clear effect).
		// Parameters default to a [-1, 1] range around 0.
		if (kind === "fixed") {
			return { name: name, kind: kind, lower: 1, upper: 1, value_type: "float", log_scale: false };
		}
		return { name: name, kind: kind, lower: -1, upper: 1, value_type: "float", log_scale: false };
	}

	return {
		parameters: parameters.map(function (n) { return asSuggestion(n, "range"); }),
		constants: constants.map(function (n) { return asSuggestion(n, "fixed"); }),
		bound: Object.keys(boundNames),
	};
}


function client_render_suggestions(result) {
	var $out = $("#formula_suggestions");
	if (!result || (!result.parameters.length && !result.constants.length)) {
		$out.html("<em>(no free symbols detected)</em>");
		return;
	}
	var html = "<table style='width:100%; border-collapse: collapse;'>";
	html += "<tr><th align='left'>name</th><th align='left'>type</th><th align='left'>value</th><th align='left'>role</th></tr>";
	function row(s, role) {
		var val = s.kind === "fixed"
			? s.lower
			: ("[" + s.lower + ", " + s.upper + "]");
		return "<tr><td><code>" + s.name + "</code></td><td>" + s.kind + "</td><td>" + val + "</td><td>" + role + "</td></tr>";
	}
	for (var i = 0; i < result.parameters.length; i++) {
		html += row(result.parameters[i], "<span style='color:#1b6e1b'>parameter</span>");
	}
	for (var j = 0; j < result.constants.length; j++) {
		html += row(result.constants[j], "<span style='color:#7a3e9e'>constant</span>");
	}
	if (result.bound && result.bound.length) {
		html += "<tr><td colspan='4' style='padding-top: 6px; color:#777; font-style: italic;'>" +
			"bound variables (excluded): " +
			result.bound.map(function (b) { return "<code>" + b + "</code>"; }).join(", ") +
			"</td></tr>";
	}
	html += "</table>";
	$out.html(html);
}


function client_apply_suggestions(result) {
	// For each parameter/constant suggestion, if a parameter row with the
	// same name already exists, leave it alone (user may have edited it).
	// Otherwise add a new row of the appropriate kind.
	if (!result) return;
	var all = [].concat(result.parameters || [], result.constants || []);
	if (!all.length) return;

	// Capture the row count BEFORE we add new ones so we can address them
	// by index even if other code paths mutate the DOM in between.
	var initialCount = $(".parameterRow").length;

	// Find the first existing empty row (the GUI always renders one
	// blank parameter row on load).  Filling it lets the first suggestion
	// land in row 0 instead of being pushed past an empty placeholder.
	var firstEmptyIdx = -1;
	for (var ei = 0; ei < initialCount; ei++) {
		var $probe = $(".parameterRow").eq(ei);
		var probeName = $probe.find(".parameterName").val();
		if (!probeName || probeName.trim() === "") {
			firstEmptyIdx = ei;
			break;
		}
	}

	for (var i = 0; i < all.length; i++) {
		(function (s, idx) {
			var existing = false;
			$(".parameterName").each(function () {
				if ($(this).val() === s.name) existing = true;
			});
			if (existing) return;

			var targetIndex;
			if (firstEmptyIdx >= 0) {
				// Reuse the blank row first.
				targetIndex = firstEmptyIdx;
				firstEmptyIdx = -1;
			} else {
				$("#main_add_row_button").click();
				targetIndex = $(".parameterRow").length - 1;
			}
			var $row = $(".parameterRow").eq(targetIndex);
			if ($row.length === 0) return;

			// Set the name BEFORE changing the kind so updateOptions picks
			// it up when re-rendering the value cell.
			$row.find(".parameterName").val(s.name);
			$row.find(".optionSelect").val(s.kind).trigger("change");

			// Now the value cell has been re-rendered for the new kind —
			// fill in the kind-specific fields synchronously.
			if (s.kind === "range") {
				$row.find(".minValue").val(s.lower);
				$row.find(".maxValue").val(s.upper);
				$row.find(".numberTypeSelect").val(s.value_type);
				$row.find(".log_scale").prop("checked", !!s.log_scale);
			} else if (s.kind === "fixed") {
				$row.find(".fixedValue").val(s.lower);
			}
		})(all[i], initialCount + i);
	}

	var suggestedNames = all.map(function (s) { return s.name; });
	$(".parameterRow").each(function () {
		var name = $(this).find(".parameterName").val().trim();
		if (name && suggestedNames.indexOf(name) === -1) {
			$(this).remove();
		}
	});
	var remaining = $(".parameterRow").length;
	if (remaining === 0) {
		$("#main_add_row_button").click();
	}

	update_command();
}

var _formula_preview_callback = null;
var _formulaPreviewTimer = null;

function _schedule_formula_preview_refresh() {
	if (_formulaPreviewTimer) clearTimeout(_formulaPreviewTimer);
	_formulaPreviewTimer = setTimeout(function () {
		_formulaPreviewTimer = null;
		if (typeof _formula_preview_callback === "function") {
			_formula_preview_callback();
		}
	}, 300);
}

function get_current_parameter_info() {
	var info = {};
	$(".parameterRow").each(function () {
		var name = $(this).find(".parameterName").val().trim();
		if (!name || !/^[a-zA-Z_]+$/.test(name)) return;
		var option = $(this).find(".optionSelect").val();
		if (option === "range") {
			info[name] = {
				kind: "range",
				min: $(this).find(".minValue").val() || "",
				max: $(this).find(".maxValue").val() || "",
				type: $(this).find(".numberTypeSelect").val() || "float",
				log_scale: $(this).find(".log_scale").is(":checked")
			};
		} else if (option === "fixed") {
			info[name] = {
				kind: "fixed",
				value: $(this).find(".fixedValue").val() || ""
			};
		} else if (option === "choice") {
			info[name] = {
				kind: "choice",
				values: $(this).find(".choiceValues").val() || ""
			};
		}
	});
	return info;
}

function _convert_infix_to_latex(text) {
	text = text.replace(/([a-zA-Z_)\d]|(\}))\s*\*\*\s*([a-zA-Z_]\w*|\d+(?:\.\d+)?)/g, "$1^{$3}");
	text = text.replace(/\*\*/g, "^");
	// Function names → LaTeX
	text = text.replace(/\b(sin|cos|tan|asin|acos|atan|sinh|cosh|tanh|exp|log|ln)\b/g, "\\$1");
	text = text.replace(/\bsqrt\s*\(/g, "\\sqrt{");
	text = text.replace(/\babs\s*\(/g, "\\left|");
	text = text.replace(/\b(max|min)\s*\(/g, "\\$1(");
	text = text.replace(/\b(ReLU|relu|sigmoid|softmax|clip|clamp|ceil|floor)\s*\(/g, "\\text{$1}(");
	// abs( closing → |
	text = text.replace(/\)\s*(?=[+\-*/^,)\s]|$)/g, ")");
	return text;
}

function _format_param_label(info) {
	if (info.kind === "range") {
		var min = info.min !== "" ? info.min : "?";
		var max = info.max !== "" ? info.max : "?";
		var numberSet = (info.type === "int") ? "\\mathbb{Z}" : "\\mathbb{R}";
		var line1 = "[" + min + ", " + max + "] \\in " + numberSet;
		var line2 = (info.type === "int") ? "discrete" : "continuous";
		if (info.log_scale) line2 += ", log";
		return "\\substack{" + line1 + " \\\\ \\text{" + line2 + "}}";
	} else if (info.kind === "fixed") {
		var val = info.value !== "" ? info.value : "?";
		return "\\substack{" + val + " \\\\ \\text{fixed}}";
	} else if (info.kind === "choice") {
		var vals = info.values ? info.values.split(",").map(function (v) { return "\\text{" + v.trim() + "}"; }).filter(Boolean).join(", ") : "?";
		return "\\substack{\\{" + vals + "\\} \\\\ \\text{choice}}";
	}
	return "";
}

function _protect_subsup(text) {
	var parts = [];
	var result = "";
	var i = 0;
	while (i < text.length) {
		if (text[i] === "_" || text[i] === "^") {
			var j = i + 1;
			if (j < text.length && text[j] === "{") {
				var depth = 1;
				j++;
				while (j < text.length && depth > 0) {
					if (text[j] === "{") depth++;
					else if (text[j] === "}") depth--;
					j++;
				}
			} else if (j < text.length) {
				j++;
			}
			parts.push(text.substring(i, j));
			result += "\x00" + (parts.length - 1) + "\x00";
			i = j;
		} else {
			result += text[i];
			i++;
		}
	}
	return { text: result, parts: parts };
}

function _restore_subsup(text, parts) {
	return text.replace(/\x00(\d+)\x00/g, function (m, idx) {
		return parts[parseInt(idx)];
	});
}

function _add_explicit_grouping(latex) {
	var re = /(\\(?:int|oint|iint|iiint|sum|prod))(?:\s*_{\s*[^{}]*\s*}|\s*_\s*[A-Za-z][A-Za-z0-9_]*\s*)?(?:\s*\^\s*\{[^{}]*\}|\s*\^\s*[A-Za-z][A-Za-z0-9_]*\s*)?/g;
	var result = "";
	var lastEnd = 0;
	var m;
	while ((m = re.exec(latex)) !== null) {
		var cmdEnd = m.index + m[0].length;
		result += latex.slice(lastEnd, cmdEnd);
		var pos = cmdEnd;
		while (pos < latex.length && latex[pos] === " ") pos++;
		if (pos >= latex.length) { lastEnd = cmdEnd; continue; }
		var ch = latex[pos];
		if (ch === "(" || ch === "{") { lastEnd = cmdEnd; continue; }
		if (ch === "\\") { lastEnd = cmdEnd; continue; }
		// If the integral/sum is inside user-written parens (i.e. the char
		// before the command is `(`), skip adding \left( \right) — the
		// user's parens already define the scope.
		var beforeIdx = m.index - 1;
		while (beforeIdx >= 0 && latex[beforeIdx] === " ") beforeIdx--;
		var inUserParens = beforeIdx >= 0 && latex[beforeIdx] === "(";
		if (inUserParens) {
			lastEnd = cmdEnd;
			continue;
		}
		var termEnd = _read_term_end(latex, pos);
		if (termEnd > pos) {
			var body = latex.slice(pos, termEnd);
			result += "\\left(" + body + "\\right)";
			lastEnd = termEnd;
		} else {
			lastEnd = cmdEnd;
		}
	}
	result += latex.slice(lastEnd);
	return result;
}

function _read_term_end(text, start) {
	var i = start;
	var depth = 0;
	var sawAny = false;
	while (i < text.length) {
		var ch = text[i];
		if (ch === "(" || ch === "{" || ch === "[") { depth++; sawAny = true; i++; continue; }
		if (ch === ")" || ch === "}" || ch === "]") {
			if (depth === 0) break;
			depth--; sawAny = true; i++; continue;
		}
		if (depth === 0 && (ch === "+" || ch === "-") && sawAny) {
			var nxt = text[i + 1] || "";
			if (nxt !== ch) break;
		}
		if (depth === 0 && ch === "=" && sawAny) break;
		i++;
		sawAny = true;
	}
	return i;
}

function _add_parameter_underbraces(latex, paramInfo) {
	var names = Object.keys(paramInfo);
	if (!names.length) return latex;

	var eqIdx = -1;
	var depth = 0;
	for (var i = 0; i < latex.length; i++) {
		var ch = latex[i];
		if (ch === "{") depth++;
		else if (ch === "}") depth--;
		else if (ch === "=" && depth === 0) eqIdx = i;
	}

	var lhs, rhs;
	if (eqIdx >= 0) {
		lhs = latex.substring(0, eqIdx + 1);
		rhs = latex.substring(eqIdx + 1);
	} else {
		lhs = "";
		rhs = latex;
	}

	var prot = _protect_subsup(rhs);
	var work = prot.text;

	names.sort(function (a, b) { return b.length - a.length; });
	var escaped = names.map(function (n) { return n.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); });
	var re = new RegExp("(?<![A-Za-z0-9_])(?:" + escaped.join("|") + ")(?![A-Za-z0-9_])", "g");

	work = work.replace(re, function (match) {
		var info = paramInfo[match];
		var label = _format_param_label(info);
		return "\\underbrace{" + match + "}_{" + label + "}";
	});

	return lhs + _restore_subsup(work, prot.parts);
}

function setup_formula_editor() {
	// The "Use formula editor" toggle button lives INSIDE the run_program
	// row of the table.  When clicked, we lazily build the formula card
	// and toggle mutual exclusivity: run_program is hidden while the
	// formula editor is open, and vice versa.
	$(document).on("click", "#formula_toggle_btn", function () {
		var $card = $("#formula_card");
		var $wrapper = $("#run_program_wrapper");
		if ($card.is(":visible")) {
			// Closing the formula editor reveals run_program.
			$card.hide();
			$wrapper.show();
			$(this).html("&#9881; Switch to formula editor");
		} else {
			// Opening the formula editor hides run_program and clears any
			// leftover run_program text — the formula will generate its
			// own run_program at invocation time, so the leftover would
			// just confuse the user.
			$("#run_program").val("");
			$wrapper.hide();
			if (!$card.data("built")) {
				$card.html(build_formula_card_html());
				$card.data("built", true);
				setup_formula_card_inner();
			}
			$card.show();
			// Ensure the Formula tab is active when opening.
			$card.find(".formula_tab").removeClass("active");
			$card.find("#formula_tab_text").addClass("active");
			$card.find("#formula_panel_text").show();
			$card.find("#formula_panel_python").hide();
			$(this).html("&#9881; Switch back to Run program");
		}
		update_command();
	});

	// If the user starts typing into run_program, hide the formula card.
	$(document).on("input", "#run_program", function () {
		var $card = $("#formula_card");
		if ($card.is(":visible") && $(this).val().trim() !== "") {
			$card.hide();
			$("#formula_toggle_btn").html("&#9881; Switch to formula editor");
		}
	});

	// If the user starts typing into the formula editor, hide run_program.
	$(document).on("input", "#formula", function () {
		var $wrapper = $("#run_program_wrapper");
		if ($wrapper.is(":visible") && $(this).val().trim() !== "") {
			$wrapper.hide();
			$("#formula_toggle_btn").html("&#9881; Switch back to Run program");
			// Make sure the card is built and visible.
			var $card = $("#formula_card");
			if (!$card.data("built")) {
				$card.html(build_formula_card_html());
				$card.data("built", true);
				setup_formula_card_inner();
			}
			if (!$card.is(":visible")) {
				$card.show();
			}
		}
		if (typeof update_command === "function") update_command();
	});

	// Persist the hidden #formula and #formula_mode on every change so the
	// URL always carries the latest values (a refresh restores them).
	$(document).on("change input", "#formula, #formula_mode", function () {
		if (typeof update_command === "function") update_command();
	});
}

function setup_formula_card_inner() {
	function set_active_tab(mode) {
		$("#formula_card .formula_tab").removeClass("active");
		$("#formula_card #formula_tab_" + mode).addClass("active");
		$("#formula_card #formula_panel_text").toggle(mode === "text");
		$("#formula_card #formula_panel_python").toggle(mode === "python");
	}

	function sync_to_main_textarea(text) {
		$("#formula").val(text).trigger("change");
		var $panes = $("#formula_card #formula_pane_text, #formula_card #formula_pane_python");
		if ($panes.length) {
			$panes.val(text);
		}
	}

	function auto_detect_mode(text) {
		if (!text) return "auto";
		if (/^\s*(def|import|from)\b/.test(text) || /\n/.test(text)) {
			return "python";
		}
		if (/\\(sin|cos|tan|sum|prod|frac|sqrt|text|textit|begin|end)\b/.test(text)) {
			return "latex";
		}
		return "infix";
	}

	function current_mode() {
		// Mode is derived from which tab is active; the tab is the source
		// of truth now (no more select pill).
		var $active = $("#formula_card .formula_tab.active");
		if ($active.length === 0) return "auto";
		return $active.data("mode") || "auto";
	}

	function update_hint() {
		var text = $("#formula").val();
		var det = auto_detect_mode(text);
		var active = current_mode();
		var hint = "Auto-detected mode: <b>" + det + "</b> &nbsp;·&nbsp; active tab: <b>" + active + "</b>";
		$("#formula_hint").html(hint);
	}

	function client_render_formula_preview(text) {
		var $prev = $("#formula_card #formula_preview");
		var $err = $("#formula_card #formula_error");

		// Guard rail: if the card isn't in the DOM yet, do nothing.
		if (!$prev.length || !$err.length) {
			return;
		}

		try {
			client_render_formula_preview_inner(text, $prev, $err);
		} catch (e) {
			// Last-resort guard rail — anything that throws here must
			// not break the rest of the page (form, command, etc.).
			try {
				console.error("[formula preview] uncaught error:", e);
				$err.text("Preview error: " + (e && e.message ? e.message : String(e)));
			} catch (_) { /* swallow */ }
		}
	}

	function client_render_formula_preview_inner(text, $prev, $err) {
		var $legend = $("#formula_card #formula_param_legend");

		if (!text || !text.trim()) {
			$prev.empty();
			$err.empty();
			if ($legend.length) $legend.html("");
			return;
		}

		var mode;
		try {
			mode = current_mode();
		} catch (e) {
			mode = "auto";
		}

		if (mode === "python") {
			$prev.html("<em style='color:#777'>Python mode — no preview.</em>");
			$err.empty();
			if ($legend.length) $legend.html("");
			return;
		}
		if (mode === "infix" || mode === "text" || mode === "auto") {
			var detected = auto_detect_mode(text);
			if (detected === "infix") {
				text = _convert_infix_to_latex(text);
			}
		}
		text = text.replace(/\*/g, "\\cdot ");
		text = text.replace(/([\d)])([a-zA-Z_])/g, "$1\\cdot $2");
		text = _add_explicit_grouping(text);

		var _pi = get_current_parameter_info();
		var _hasUnderbraces = text.indexOf("\\underbrace") !== -1;
		if (!_hasUnderbraces && Object.keys(_pi).length) {
			text = _add_parameter_underbraces(text, _pi);
			_hasUnderbraces = true;
		}

		if ($legend.length) {
			if (Object.keys(_pi).length) {
				var legendParts = [];
				for (var _ln in _pi) {
					var _li = _pi[_ln];
					if (_li.kind === "range") {
						legendParts.push("<code>" + _ln + "</code> ∈ [" + (_li.min || "?") + ", " + (_li.max || "?") + "] (" + (_li.type || "float") + (_li.log_scale ? ", log" : "") + ")");
					} else if (_li.kind === "fixed") {
						legendParts.push("<code>" + _ln + "</code> = " + (_li.value || "?") + " (fixed)");
					} else if (_li.kind === "choice") {
						legendParts.push("<code>" + _ln + "</code> ∈ {" + (_li.values || "?") + "}");
					}
				}
				$legend.html(legendParts.join(" &nbsp;·&nbsp; "));
			} else {
				$legend.html("");
			}
		}

		// Sanity-check the input: refuse to render empty / whitespace /
		// extreme-length LaTeX so we never break MathJax.
		if (typeof text !== "string") {
			$err.text("Preview error: formula is not a string");
			return;
		}
		if (text.length > 5000) {
			$err.text("Formula is too long to preview (>5000 chars)");
			$prev.empty();
			return;
		}

		// Detect obviously broken LaTeX before handing it to MathJax so we
		// can show a useful error message instead of a giant red ``?``.
		var stripped = text
			.replace(/\\underbrace\{[^{}]*\}\{[^{}]*\}/g, "")
			.replace(/^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)\s*=\s*/, "")
			.replace(/^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*/, "");
		var balanceErrors = [];
		if (/\\\s/.test(text) && !_hasUnderbraces) balanceErrors.push("stray space after a backslash");
		if (!_hasUnderbraces && /[{}]\s*[+\-*/^=]/.test(stripped)) balanceErrors.push("brace next to an operator (probably missing ``\\right``)");
		// Count opening and closing braces; an imbalance is almost
		// certainly a typo.
		var depth = 0, maxDepth = 0, bchar;
		for (var bi = 0; bi < text.length; bi++) {
			bchar = text[bi];
			if (bchar === "{") { depth++; if (depth > maxDepth) maxDepth = depth; }
			else if (bchar === "}") { depth--; if (depth < 0) break; }
		}
		if (depth !== 0) balanceErrors.push("unbalanced braces (depth=" + depth + ")");
		// Reject control characters / null bytes that MathJax can't digest.
		if (/[\x00-\x08\x0b\x0c\x0e-\x1f]/.test(text)) balanceErrors.push("control characters in formula");

		// HTML-escape the four characters that would otherwise let the
		// user's text break out of the preview div; keep ``\\`` intact so
		// LaTeX commands like ``\\sin`` survive.
		var safe;
		try {
			safe = text
				.replace(/&/g, "&amp;")
				.replace(/</g, "&lt;")
				.replace(/>/g, "&gt;")
				.replace(/[\x00-\x1f]/g, "");
		} catch (e) {
			$err.text("Preview error: failed to escape formula");
			return;
		}

		$err.empty();

		// Show a quick "Loading…" hint so the user sees feedback before
		// MathJax has finished typesetting (it's async, so there can be
		// a noticeable delay the first time around).
		try {
			$prev.empty().html("<span style='color:#777'>&#x2026;rendering preview&#x2026;</span>");
		} catch (e) {
			$err.text("Preview error: failed to clear preview area");
			return;
		}

		// Insert the display-math wrapper directly into the preview div
		// (a child wrapper would otherwise get lost when MathJax replaces
		// the element after typesetting).
		var renderNode;
		try {
			renderNode = document.getElementById("formula_preview");
			if (!renderNode) {
				throw new Error("preview node missing");
			}
			renderNode.innerHTML = "\\[" + safe + "\\]";
		} catch (e) {
			$err.text("Preview error: failed to insert formula into DOM");
			return;
		}

		if (balanceErrors.length) {
			$err.text("Possible issues: " + balanceErrors.join("; "));
		}

		// Ask MathJax to typeset.  Wait for its startup promise first so we
		// don't lose the first render when MathJax is still loading.
		client_mathjax_typeset(renderNode, $err);
	}

	function client_mathjax_typeset(node, $err) {
		// If MathJax isn't loaded at all, show a friendly fallback so
		// the user still sees *something* rather than a blank box.
		if (!window.MathJax) {
			if ($err) {
				$err.text($err.text() ? $err.text() + "; " : "" +
					"MathJax not loaded — showing raw formula.");
			}
			return;
		}
		// MathJax v3 exposes ``MathJax.typesetPromise``.  Earlier versions
		// used ``MathJax.Hub.Queue``; if neither is available we just
		// keep the rendered (but un-typeset) HTML.
		try {
			var run = function () {
				// Strip any previous MathJax output so the user always
				// sees the current formula, not a stale render.
				if (window.MathJax.typesetClear) {
					try { window.MathJax.typesetClear([node]); } catch (_) {}
				} else if (node) {
					var old = node.querySelectorAll("mjx-container, .mjx-container, [data-mathjax]");
					for (var oi = 0; oi < old.length; oi++) {
						if (old[oi].parentNode) old[oi].parentNode.removeChild(old[oi]);
					}
				}
				if (window.MathJax.typesetPromise) {
					window.MathJax.typesetPromise([node])
						.catch(function (err) {
							console.error("[formula preview] typeset failed:", err);
							if ($err) {
								$err.text("LaTeX render failed: " + (err && err.message ? err.message : "unknown"));
							}
						});
				} else if (window.MathJax.Hub && window.MathJax.Hub.Queue) {
					window.MathJax.Hub.Queue(["Typeset", window.MathJax.Hub, node]);
				} else {
					console.warn("[formula preview] no typeset API on MathJax");
				}
			};
			if (window.MathJax.startup && window.MathJax.startup.promise) {
				// Race the startup promise against a hard timeout so the
				// preview never gets stuck in "Loading…" forever.
				var startupTimeout = setTimeout(function () {
					console.warn("[formula preview] MathJax startup timed out, rendering raw formula");
					if ($err) {
						$err.text(($err.text() ? $err.text() + "; " : "") +
							"MathJax startup timed out");
					}
				}, 10000);
				window.MathJax.startup.promise.then(function () {
					clearTimeout(startupTimeout);
					run();
				}, function (err) {
					clearTimeout(startupTimeout);
					console.error("[formula preview] MathJax startup failed:", err);
				});
			} else {
				run();
			}
		} catch (e) {
			console.error("[formula preview] typeset threw:", e);
		}
	}

	function refresh_suggestions() {
		var text = $("#formula").val() || "";
		var mode = current_mode();
		if (mode === "python") {
			$("#formula_card #formula_suggestions").html("<em>Python code — parameters are read from the <code>params</code> dict; no auto-extraction.</em>");
			return;
		}
		var result;
		try {
			result = client_extract_formula_params(text, mode);
		} catch (e) {
			$("#formula_card #formula_suggestions").html(
				"<span style='color:#b00020'>Broken formula: " + client_escape_html(String(e)) + "</span>"
			);
			return;
		}
		client_render_suggestions(result);
	}

	function update_everything() {
		update_hint();
		refresh_suggestions();
		client_render_formula_preview($("#formula").val() || "");
	}

	function client_escape_html(s) {
		return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
			.replace(/"/g, "&quot;").replace(/'/g, "&#39;");
	}

	$("#formula_card #formula_tab_text").on("click", function () {
		set_active_tab("text");
		sync_to_main_textarea($("#formula_pane_text").val());
		$("#formula_mode").val("auto").trigger("change");
		update_everything();
	});
	$("#formula_card #formula_tab_python").on("click", function () {
		set_active_tab("python");
		sync_to_main_textarea($("#formula_pane_python").val());
		$("#formula_mode").val("python").trigger("change");
		update_everything();
	});

	$("#formula_card #formula_pane_text, #formula_card #formula_pane_python").on("input", function () {
		var text = $(this).val();
		$("#formula_card #formula_pane_text").val(text);
		$("#formula_card #formula_pane_python").val(text);
		sync_to_main_textarea(text);
		update_everything();
		if (typeof update_command === "function") update_command();
	});

	// Ctrl/Cmd + Enter on any formula pane applies the suggestions.
	$("#formula_card #formula_pane_text, #formula_card #formula_pane_python").on("keydown", function (ev) {
		if ((ev.ctrlKey || ev.metaKey) && (ev.key === "Enter" || ev.keyCode === 13)) {
			ev.preventDefault();
			$("#formula_card #formula_apply_btn").trigger("click");
		}
	});

	$("#formula_card #formula_apply_btn").on("click", function () {
		var text = $("#formula").val() || "";
		var mode = current_mode();
		if (mode === "python") {
			// For Python mode, scan for params['x'] / params["x"] patterns
			// to extract parameters from the user code.
			var pyParams = client_extract_python_params(text);
			client_apply_suggestions({ parameters: pyParams, constants: [], bound: [] });
		} else {
			var result = client_extract_formula_params(text, mode);
			client_apply_suggestions(result);
		}
		update_command();
	});

	$("#formula_card #formula_clear_btn").on("click", function () {
		$("#formula_card #formula_pane_text").val("");
		$("#formula_card #formula_pane_python").val("");
		sync_to_main_textarea("");
		update_everything();
		update_command();
	});

	// Initial population from the hidden #formula textarea.
	var initial = $("#formula").val() || "";
	if (initial) {
		$("#formula_card #formula_pane_text").val(initial);
		$("#formula_card #formula_pane_python").val(initial);
	}
	var initial_mode = $("#formula_mode").val() || "auto";
	// Default to the matching tab based on the mode (so the user sees
	// their original input).  "auto" lands on "infix" since most
	// scientists paste infix expressions.
	if (initial_mode === "latex") set_active_tab("text");
	else if (initial_mode === "python") set_active_tab("python");
	else set_active_tab("infix");
	update_everything();

	_formula_preview_callback = function () {
		client_render_formula_preview($("#formula").val() || "");
	};
}

function updateOptions(select) {
	var selectedOption = select.value;
	var valueCell = select.parentNode.nextSibling;
	var paramName = $(select).parent().parent().find(".parameterName").val();

	if(paramName === undefined) {
		paramName = "";
	}

	if (selectedOption === "range") {
		valueCell.innerHTML = `
			<table class='parameter_config_table'>
				<tr>
					<td>Name:</td>
					<td><input placeholder="Parameter name" onchange="update_command()" onkeyup="update_command()" onclick="update_command()" value="${paramName}" type='text' class='parameterName'></td>
				</tr>
				<tr>
					<td>Min:</td>
					<td><input placeholder="Minimum value" onchange="update_command()" onkeyup="update_command()" onclick="update_command()" type='number' class='minValue'></td>
				</tr>
				<tr>
					<td>Max:</td>
					<td><input placeholder="Maximum value" onchange="update_command()" onkeyup="update_command()" onclick="update_command()" type='number' class='maxValue'></td>
				</tr>
				<tr>
					<td>Type:</td>
					<td>
						<select onchange="update_command()" onkeyup="update_command()" onclick="update_command()" class="numberTypeSelect">
							<option value="float">Float</option>
							<option value="int">Integer</option>
						</select>
					</td>
				</tr>
				<tr>
					<td>Log-Scale<a class='tooltip invert_in_dark_mode' title='Ensures parameters (e.g., learning rate, weight decay, epsilon) are optimized in log-space, so all magnitudes are handled evenly.'><img src='i/help.svg' /></a>:</td>
					<td>
						<input onchange="update_command()" type="checkbox" class="log_scale" />
						
					</td>
				</tr>
			</table>
		    `;
	} else if (selectedOption === "choice") {
		valueCell.innerHTML = `
			<table class='parameter_config_table'>
				<tr>
					<td>Name:</td>
					<td><input placeholder="Parameter name" onchange="update_command()" onkeyup="update_command()" onclick="update_command()" value="${paramName}" type='text' class='parameterName'></td>
				</tr>
				<tr>
					<td>Values (comma separated):</td>
					<td><input placeholder="Comma-Seperated Values" onchange="update_command()" onkeyup="update_command()" onclick="update_command()" type='text' class='choiceValues'></td>
				</tr>
			</table>
		    `;
	} else if (selectedOption === "fixed") {
		valueCell.innerHTML = `
			<table class='parameter_config_table'>
				<tr>
					<td>Name:</td>
					<td><input placeholder="Parameter name" onchange="update_command()" onkeyup="update_command()" onclick="update_command()" value="${paramName}" type='text' class='parameterName'></td>
				</tr>
				<tr>
					<td>Value:</td>
					<td><input placeholder="Value" onchange="update_command()" onkeyup="update_command()" onclick="update_command()" type='text' class='fixedValue'></td>
				</tr>
			</table>
		    `;
	}

	valueCell.innerHTML += "<div style='display: none' class='error_element parameterError invert_in_dark_mode'></div>";

	update_command();

	apply_theme_based_on_system_preferences();
}

function add_parameter_row(button) {
	var table = document.getElementById("config_table");
	var rowIndex = button.parentNode.parentNode.rowIndex;
	var numberOfParams = $(".parameterRow").length;
	var newRow = table.insertRow(rowIndex + numberOfParams + 1);

	$(newRow).css("display", "none");

	var optionCell = newRow.insertCell(0);
	var valueCell = newRow.insertCell(1);
	var buttonCell = newRow.insertCell(2);

	optionCell.innerHTML = `<select onchange='updateOptions(this)' class='optionSelect'>
		<option value='range'>Range</option>
		<option value='choice'>Choice</option>
		<option value='fixed'>Fixed</option>
		</select>`;

	valueCell.innerHTML = "";

	buttonCell.innerHTML = "<button class='remove_parameter' onclick='remove_parameter_row(this)'><img class='invert_in_dark_mode' style='height: 1em' src='i/red_x.svg' />&nbsp;Remove</button>";

	updateOptions(optionCell.firstChild);

	newRow.classList.add("parameterRow");
	optionCell.firstChild.classList.add("optionSelect");

	$(newRow).show()

	update_command();

	toggle_disabled_status_of_remove_parameters_depending_on_if_there_are_more_than_one();
}

function remove_parameter_row(button) {
	var table = document.getElementById("config_table");
	var rowIndex = button.parentNode.parentNode.rowIndex;
	var rowCount = table.rows.length;
	if (rowCount > 2) {
		table.deleteRow(rowIndex);
		update_command();
	}

	toggle_disabled_status_of_remove_parameters_depending_on_if_there_are_more_than_one();
}

function string_or_array_to_list (input) {
	if (typeof input === "string") {
		return input;
	} else if (Array.isArray(input)) {
		if (input.length === 1) {
			return input[0];
		} else {
			const listItems = input.map(item => `<li>${item}</li>`);
			return `<ul>${listItems.join("")}</ul>`;
		}
	} else {
		throw new Error("Invalid input type. Only strings or arrays are allowed.");
	}
}

function create_table_row (table, tbody, item) {
	var row = $("<tr>");

	var left_side_content = item.label;

	if ("help" in item && item.help.length > 0) {
		function escapeQuotes(str) {
			return str.replace(/'/g, "&#039;");
		}

		left_side_content += `<a class='tooltip invert_in_dark_mode' title='${escapeQuotes(item.help)}'><img src='i/help.svg' /></a>`;
	}

	var labelCell = $("<td class='left_side'>").html(left_side_content);
	var valueCell = $("<td class='right_side'>").attr("colspan", "2");

	if (item.type === "select") {
		var $select = $("<select>").attr("id", item.id);

		$.each(item.options, function(index, option) {
			var $option = $("<option></option>")
				.attr("value", option.value)
				.text(option.text);

			if (index == 0) {
				$option.prop("selected", "selected");
			}

			$select.append($option);
		});

		$select.change(update_command);

		if (Object.keys(item).includes("onchange")) {
			$select.change(item.onchange);
		}

		valueCell.append($select);
	} else if (item.type === "textarea") {
		var input = $("<textarea>").attr({ id: item.id, type: item.type, value: item.value, placeholder: item.placeholder, min: item.min, max: item.max });
		$(input).css({"width": "95%", "height": "95%"});

		input.on({
			change: update_command,
			keyup: update_command,
			click: update_command
		});

		var $container = $("<div>").attr("id", item.id + "_wrapper");
		$container.append(input);
		valueCell.append($container);

		if (Object.keys(item).includes("onchange")) {
			$(input).change(item.onchange);
		}

		if (Object.keys(item).includes("append_html")) {
			valueCell.append($(item.append_html));
		}
	} else {
		var input = $("<input>").attr({ id: item.id, type: item.type, value: item.value, placeholder: item.placeholder, min: item.min, max: item.max, step: item.step }).css("width", "95%");

		if (item.type === "checkbox") {
			input.prop("checked", item.value);
		}

		input.on({
			change: update_command,
			keyup: update_command,
			click: update_command
		});

		valueCell.append(input);

		if (Object.keys(item).includes("onchange")) {
			$(input).change(item.onchange);
		}

		if (Object.keys(item).includes("onkeypress")) {
			$(input).keypress(item.onkeypress);
		}

		if (Object.keys(item).includes("onblur")) {
			$(input).blur(item.onblur);
		}

		if (Object.keys(item).includes("onfocus")) {
			$(input).focus(item.onchange);
		}
	}

	if (item.id !== "partition") {
		valueCell.append($(`<div class='error_element invert_in_dark_mode' id="${item.id}_error"></div>`));
	}

	if (item.info) {
		valueCell.append($(`<div class='info_element' id="${item.id}_info">${item.info}</div>`));
	}

	row.append(labelCell, valueCell);
	tbody.append(row);
}

function create_tables() {
	var table = $("#config_table");
	var tbody = table.find("tbody");

	tableData.forEach(function(item) {
		create_table_row(table, tbody, item);
	});

	tbody.append("<tr><td colspan=3><button onclick='add_parameter_row(this)' class='add_parameter' id='main_add_row_button'><img class='invert_in_dark_mode 'src='i/green_plus.svg' style='height: 1em' />&nbsp;Add variable</button></td></tr>");

	var hidden_table = $("#hidden_config_table");
	var hidden_tbody = hidden_table.find("tbody");

	hiddenTableData.forEach(function(item) {
		create_table_row(hidden_table, hidden_tbody, item);
	});

	highlight_all_bash();

	$("#site").show();
	$("#loader").remove();
}

function update_url() {
	var url = window.location.href;

	var index = url.indexOf("no_update_url");

	if (index !== -1) {
		return;
	}

	var params = [];

	// Field IDs that are emitted by ``pushFormula`` below; skip them
	// here so we don't write the same key twice into the URL.
	var FORMULA_FIELDS = {
		formula: true,
		formula_mode: true,
		formula_python_path: true,
	};

	function push_value(item) {
		if (FORMULA_FIELDS[item.id]) return;
		var element = $("#" + item.id);
		var value;

		if (element.is(":checkbox")) {
			value = element.is(":checked") ? 1 : 0;
		} else {
			value = element.val();

			// Base64-encode run_program, run_program_once and external_generator
			// for the URL so newlines and special characters survive a refresh
			// and we don't blow past the browser's URL-length limit.
			if (item.id === "run_program" || item.id === "run_program_once" || item.id === "external_generator") {
				if (value && value.trim() !== "") {
					try {
						// encodeURIComponent trick preserves multi-byte chars
						value = btoa(unescape(encodeURIComponent(value)));
					} catch (e) {
						console.error("Base64 encoding failed for " + item.id + ":", e);
					}
				}
			}

			value = encodeURIComponent(value);
		}

		// Skip empty / default values so the URL stays compact.
		if (value === "" || value === "0" || value === "false") {
			// Keep checkboxes (booleans default to 0 = off) but skip empty text fields.
			if (typeof value === "string" && value === "") {
				return;
			}
		}

		params.push(item.id + "=" + value);
	}

	tableData.forEach(function(item) {
		push_value(item);
	});

	hiddenTableData.forEach(function(item) {
		push_value(item);
	});

	// Custom URL params for the formula editor (lives outside tableData).
	(function pushFormula() {
		var formulaEl = $("#formula");
		var modeEl = $("#formula_mode");
		var pythonPathEl = $("#formula_python_path");
		if (formulaEl.length === 0 || modeEl.length === 0) return;
		var formulaVal = formulaEl.val() || "";
		var hasFormula = formulaVal.trim() !== "";
		if (hasFormula) {
			try {
				var b64 = btoa(unescape(encodeURIComponent(formulaVal)));
				params.push("formula=" + encodeURIComponent(b64));
			} catch (e) {
				console.error("Base64 encoding failed for formula:", e);
			}
			params.push("formula_mode=" + encodeURIComponent(modeEl.val() || "auto"));
		}
		// Only persist the Python interpreter override when the user
		// actually customised it.
		if (pythonPathEl.length > 0) {
			var pyPath = pythonPathEl.val() || "";
			if (pyPath.trim() !== "") {
				params.push("formula_python_path=" + encodeURIComponent(pyPath));
			}
		}
	})();

	var parameterIndex = 0;
	$(".parameterRow").each(function() {
		var option = $(this).find(".optionSelect").val();
		var parameterName = $(this).find(".parameterName").val();

		if(parameterName && !parameterName.match(/^\w+$/)) {
			//error(`Parameter name "${parameterName}" does have invalid characters. Must be all letters.`)
		} else if (parameterName) {
			var param_base = "parameter_" + parameterIndex;
			if (option === "range") {
				var minValue = $(this).find(".minValue").val();
				var maxValue = $(this).find(".maxValue").val();
				var numberType = $(this).find(".numberTypeSelect").val();
				var log_scale = $(this).find(".log_scale").is(":checked") ? "true" : "false";

				params.push(param_base + "_name=" + encodeURIComponent(parameterName));
				params.push(param_base + "_type=" + encodeURIComponent(option));
				params.push(param_base + "_min=" + encodeURIComponent(minValue));
				params.push(param_base + "_max=" + encodeURIComponent(maxValue));
				params.push(param_base + "_number_type=" + encodeURIComponent(numberType));
				params.push(param_base + "_log_scale=" + encodeURIComponent(log_scale));
			} else if (option === "choice") {
				var choiceValues = $(this).find(".choiceValues").val();

				params.push(param_base + "_name=" + encodeURIComponent(parameterName));
				params.push(param_base + "_type=" + encodeURIComponent(option));
				params.push(param_base + "_values=" + encodeURIComponent(choiceValues));
			} else if (option === "fixed") {
				var fixedValue = $(this).find(".fixedValue").val();

				params.push(param_base + "_name=" + encodeURIComponent(parameterName));
				params.push(param_base + "_type=" + encodeURIComponent(option));
				params.push(param_base + "_value=" + encodeURIComponent(fixedValue));
			}
			parameterIndex++;
		}
		// Rows without a parameter name are skipped but their index is
		// preserved so the URL stays compact (no gaps in numbering).
	});

	if (initialized) {
		var url = window.location.origin + window.location.pathname + "?" + params.join("&") + "&num_parameters=" + parameterIndex;

		try {
			window.history.replaceState(null, null, url);
		} catch (err) {
			err = "" + err;

			if(err.includes("The operation is insecure") && !shown_operation_insecure_without_server) {
				log(err);
				shown_operation_insecure_without_server = true;
			} else if (!err.includes("The operation is insecure")) {
				error(err);
			}
		}
	}
}

function copy_bashcommand_to_clipboard_main () {
	var serialized = $("#command_element").text();
	copy_to_clipboard(serialized);

	$("#copied_main").show();
	setTimeout(function() {
		$("#copied_main").fadeOut();
	}, 5000);
}

function copy_bashcommand_to_clipboard_curl () {
	var serialized = $("#curl_command").text();
	copy_to_clipboard(serialized);

	$("#copied_curl").show();
	setTimeout(function() {
		$("#copied_curl").fadeOut();
	}, 5000);
}

function get_parameter_names(only_these_types = []) {
	var values = $(".parameterName").map(function() {
		var parameterValue = $(this).val();
		var parameterType = $(this).closest('.parameterRow')
			.find(".optionSelect")
			.val();

		if (only_these_types.length > 0 && only_these_types.includes(parameterType)) {
			return parameterValue;
		} else if (only_these_types.length === 0) {
			return parameterValue;
		}
	}).get().filter(Boolean);

	return values;
}

function is_valid_constraints_string(input) {
	const parameter_names = get_parameter_names(["range"]);

	input = input.replaceAll(/;;*/g, ";");
	input = input.replace(/;;*\s*$/, "");
	input = input.replace(/^\s*;;*/, "");

	return input.split(";").map(part => test_if_equation_is_valid(part, parameter_names)).join("");
}

function isAnyLogScaleSet() {
    return $(".log_scale:checked").length > 0;
}

function run_when_document_ready () {
	create_tables();

	update_partition_options();

	setup_formula_editor();

	var urlParams = new URLSearchParams(window.location.search);

	// --- tableData loop ---
	tableData.forEach(function(item) {
		var paramValue = urlParams.get(item.id);

		if (paramValue !== null) {
			// Strip surrounding single quotes if present
			if (item.id === "result_names") {
				paramValue = paramValue.replace(/^'(.*)'$/, '$1');
			}

			// Base64-decode the long text fields.  We detect them by trying
			// to decode; if that fails we leave the value alone (so the
			// URL-encoded form still works for older / hand-crafted URLs).
			if (item.id === "run_program" || item.id === "run_program_once" || item.id === "external_generator") {
				if (paramValue !== "") {
					try {
						paramValue = decodeURIComponent(escape(atob(paramValue)));
					} catch (e) {
						// Not base64 — keep the raw URL-decoded value.
						try {
							paramValue = decodeURIComponent(paramValue);
						} catch (e2) { /* keep as-is */ }
					}
				}
			}

			var $element = $("#" + item.id);
			if ($element.is(":checkbox")) {
				var boolValue = /^(1|true)$/i.test(paramValue);
				$element.prop("checked", boolValue).trigger("change");
			} else {
				$element.val(paramValue).trigger("change");
			}
		}
	});

	// --- formula editor (lives outside tableData) ---
	(function restoreFormula() {
		var fm = urlParams.get("formula");
		var fmMode = urlParams.get("formula_mode") || "auto";
		var fmEmpty = (fm === null || fm === "");
		var rpEl = document.getElementById("run_program");
		var rpEmpty = !rpEl || !(rpEl.value || "").trim();
		// If the URL carries a formula, open the formula card immediately
		// and hide run_program so the scientist lands back in the same
		// workflow they left.
		if (!fmEmpty) {
			fm = fm.replace(/^'(.*)'$/, '$1');
			try {
				fm = decodeURIComponent(escape(atob(fm)));
			} catch (e) {
				console.error("Base64 decoding failed for formula:", e);
				fm = "";
			}
			// Validate the mode before assigning.
			var validModes = ["auto", "latex", "infix", "python"];
			if (validModes.indexOf(fmMode) < 0) fmMode = "auto";
			$("#formula").val(fm);
			$("#formula_mode").val(fmMode);
			// Open the formula card and hide run_program to mirror the
			// scientist's prior state.
			var $btn = $("#formula_toggle_btn");
			var $card = $("#formula_card");
			var $wrap = $("#run_program_wrapper");
			if ($card.length && !$card.data("built")) {
				$card.html(build_formula_card_html());
				$card.data("built", true);
				setup_formula_card_inner();
			}
			$card.show();
			$wrap.hide();
			// Clear any leftover run_program text — when the formula
			// editor is active we generate the run_program automatically,
			// so the leftover is just confusing.
			$("#run_program").val("");
			$btn.html("&#9881; Switch back to Run program");
		} else if (rpEmpty && fmEmpty) {
			// Nothing yet: keep run_program visible and don't auto-open
			// the formula editor — let the scientist pick.
		}
		// Defer so the formula card's input handlers are wired up.
		setTimeout(function () {
			if (!fmEmpty) {
				// Push the formula into all three panes and trigger the
				// active pane so ``update_everything`` runs and MathJax
				// gets a chance to render the preview.
				$("#formula_card #formula_pane_text, #formula_card #formula_pane_python").val(fm);
				$("#formula_pane_text").trigger("input");
				// Belt-and-suspenders: also call the renderer directly so
				// the preview shows up even if the input handler is
				// somehow shadowed by a third-party script.
				if (typeof _formula_preview_callback === "function") {
					_formula_preview_callback();
				}
				// Auto-apply the suggestions so the parameter table is
				// populated when restoring from URL.
				setTimeout(function () {
					$("#formula_card #formula_apply_btn").trigger("click");
				}, 20);
			}
		}, 0);
	})();

	// --- hiddenTableData loop ---
	hiddenTableData.forEach(function(item) {
		var paramValue = urlParams.get(item.id);
		if (paramValue !== null) {
			// Base64-decode for run_program_once and external_generator
			if (item.id === "run_program_once" || item.id === "external_generator") {
				// Strip surrounding single quotes FIRST (they come from the URL encoding)
				paramValue = paramValue.replace(/^'(.*)'$/, '$1');

				try {
					paramValue = decodeURIComponent(escape(atob(paramValue)));
				} catch (e) {
					console.error("Base64 decoding failed for " + item.id + ":", e);
				}
			}

			var $element = $("#" + item.id);
			if ($element.is(":checkbox")) {
				var boolValue = /^(1|true)$/i.test(paramValue);
				$element.prop("checked", boolValue).trigger("change");
			} else {
				$element.val(paramValue).trigger("change");
			}
		}
	});

	var num_parameters = urlParams.get("num_parameters");
	if (num_parameters) {
		for (var k = 0; k < num_parameters; k++) {
			$("#main_add_row_button").click();
		}
	} else {
		$("#main_add_row_button").click();
	}

	var parameterIndex = 0;
	$(".parameterRow").each(function(index) {
		var param_base = "parameter_" + parameterIndex
		var parameterName = urlParams.get(param_base + "_name");
		var option = urlParams.get(param_base + "_type");

		if (parameterName && option) {
			$(this).find(".parameterName").val(parameterName);
			$(this).find(".optionSelect").val(option).trigger('change');

			if (option === 'range') {
				$(this).find(".minValue").val(urlParams.get(param_base + "_min"));
				$(this).find(".maxValue").val(urlParams.get(param_base + "_max"));
				$(this).find(".numberTypeSelect").val(urlParams.get(param_base + "_number_type"));

				var log_scale_value = urlParams.get(param_base + "_log_scale") == "true" ? true : false;
				$(this).find(".log_scale").prop("checked", log_scale_value);
			} else if (option === 'choice') {
				$(this).find(".choiceValues").val(urlParams.get(param_base + "_values"));
			} else if (option === 'fixed') {
				$(this).find(".fixedValue").val(urlParams.get(param_base + "_value"));
			}
		}
		parameterIndex++;
	});

	document.getElementById("copytoclipboardbutton_curl").addEventListener(
		"click",
		copy_bashcommand_to_clipboard_curl,
		false
	);

	document.getElementById("copytoclipboardbutton_main").addEventListener(
		"click",
		copy_bashcommand_to_clipboard_main,
		false
	);

	input_to_time_picker("time")
	input_to_time_picker("worker_timeout")

	$('.tooltip').tooltipster();

	apply_theme_based_on_system_preferences();

	initialized = true;

	fadeTime = fadeTimeAfterLoading;

	setTimeout(update_command, 200);

	update_url();

	update_command();

	document_is_ready = true;
}

function test_if_equation_is_valid(str, names) {
	var errors = [];
	var isValid = true;

	if (!str.includes(">=") && !str.includes("<=")) {
		errors.push("<img src='i/warning.svg' style='height: 1em' /> Missing '>=' or '<=' operator. The equation should include a comparison operator.");
		isValid = false;
	}

	var splitted = str.includes(">=") ? str.split(">=") : str.split("<=");
	if (splitted.length !== 2) {
		errors.push("<img src='i/warning.svg' style='height: 1em' /> Equation format is incorrect. There should be exactly one comparison operator.");
		isValid = false;
	}

	var left_side = splitted[0].replace(/\s+/g, "");
	if (!left_side) {
		errors.push("<img src='i/warning.svg' style='height: 1em' /> Left side is empty or contains only whitespace. Please provide an expression on the left side.");
		isValid = false;
	}

	if (isValid) {
		var right_side = splitted[1].trim();

		if (names.includes(left_side) && names.includes(right_side)) {
			return "";
		}

		if (!/^[+-]?\d+(\.\d+)?$/.test(right_side)) {
			errors.push("<img src='i/warning.svg' style='height: 1em' /> The right side does not look like a constant. The right side should be a valid number.");
			isValid = false;
		}

		// Escape variable names for regex usage
		var escapedNames = names.map(n => n.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
		var namePattern = `(?:${escapedNames.join("|")})`;

		var numberPattern = "\\d+(?:\\.\\d+)?";
		var factorPattern = `(?:${numberPattern}|${namePattern})`;
		var productPattern = `${factorPattern}(?:\\*${factorPattern})*`;
		var termPattern = `[+-]?${productPattern}`;
		var fullPattern = `^${termPattern}(?:[+-]${termPattern})*$`;

		var regex = new RegExp(fullPattern);
		if (!regex.test(left_side)) {
			errors.push(`<img src='i/warning.svg' style='height: 1em' /> Left side does not match expected pattern. Invalid term or parameter format detected in '${left_side}'`);
			isValid = false;
		}

		// Check for multiple operators in a row (e.g., ++, --, **)
		if (/[*+-]{2,}/.test(left_side)) {
			errors.push("<img src='i/warning.svg' style='height: 1em' /> The left side contains multiple operators directly in a row. Ensure that operators are used correctly.");
			isValid = false;
		}

		// Check for number directly followed by variable without *
		var nr_re = "([+-]?\\d+(\\.\\d+)?)";
		var number_followed_by_varname = new RegExp(`${nr_re}(${namePattern})`);
		if (number_followed_by_varname.test(left_side)) {
			errors.push("<img src='i/warning.svg' style='height: 1em' /> A number is followed directly by a variable name without an operator. Example: '3x' is not valid, use '3*x' instead.");
			isValid = false;
		}

		// Check for starting with invalid operator
		if (/^[*+]/.test(left_side)) {
			errors.push("<img src='i/warning.svg' style='height: 1em' /> Left side starts with an operator. The equation cannot start with an operator.");
			isValid = false;
		}
	}

	function errorsToHtml(_errors) {
		if (_errors.length) {
			_errors.unshift(`<b>Equation: ${str}</b>`);
			return "<ul>" + _errors.map(error => `<li>${error}</li>`).join('') + "</ul>";
		}
		return "";
	}

	var ret_str = errorsToHtml(errors);
	return ret_str;
}

function equation_validation_test () {
	var param_names = ["hallo", "welt", "x", "y"];

	var failed = 0;
	var test_counter = 0;

	function internal_equation_checker(code, should_be) {
		var ret_str = test_if_equation_is_valid(code, param_names);

		if (should_be === true) {
			if (ret_str !== "") {
				console.error(`Error: ${code} failed. Should be: ${should_be}, is: ${ret_str}`);
				failed = failed + 1;
			}
		} else {
			if (ret_str === "") {
				console.error(`Error: ${code} failed. Should be: ${should_be}, is: ${ret_str}`);
				failed = failed + 1;
			}
		}

		test_counter++;
	}

	internal_equation_checker("x >= y", true);
	internal_equation_checker("x + y >= 5", true);
	internal_equation_checker("x + y <= 5", true);
	internal_equation_checker("2*x + 231*y <= 5", true);
	internal_equation_checker("x + 231*y <= 5", true);
	internal_equation_checker("2*x + y <= 5", true);
	internal_equation_checker("2*x+y<=5", true);
	internal_equation_checker("2*x+y>=5", true);
	internal_equation_checker("2+y >= 10", true);
	internal_equation_checker("3*x + 5*y >= 10", true);
	internal_equation_checker("10*hallo + 2*welt <= 100", true);
	internal_equation_checker("x - y >= 0", true);
	internal_equation_checker("5*x + 7*y + 9*hallo - 3*welt <= 50", true);
	internal_equation_checker("y + 10 >= 20", true);
	internal_equation_checker("x - 10 <= 5", true);
	internal_equation_checker("100*x + 50*y >= 1000", true);
	internal_equation_checker("2*x - 3*y + 4*hallo + 5*welt <= 42", true);
	internal_equation_checker("welt + 2*x + 3*y >= 0", true);
	internal_equation_checker("x + y + hallo + welt <= 99", true);
	internal_equation_checker("x + 2*y - 3*hallo + 4*welt >= -50", true);
	internal_equation_checker("x + 2*y + 3*hallo + 4*welt <= 1000000", true);
	internal_equation_checker("2*hallo - 3*welt + 4*x - 5*y >= -100", true);
	internal_equation_checker("100000*x + 200000*y <= 500000", true);
	internal_equation_checker("x - y + hallo - welt >= 1", true);
	internal_equation_checker("3*x - 5*y + 7*hallo - 9*welt <= 0", true);
	internal_equation_checker("hallo + welt + x + y >= 12345", true);
	internal_equation_checker("2*x - 2*y + 2*hallo - 2*welt >= -2", true);
	internal_equation_checker("x + y + hallo + welt <= -10", true);
	internal_equation_checker("3*x - 4*y + 5*hallo <= 30", true);
	internal_equation_checker("10*x + 2*y - 3*hallo + 4*welt >= -15", true);
	internal_equation_checker("x + y + 3*hallo - 2*welt <= 200", true);
	internal_equation_checker("2*x + 3*y - 4*hallo + 5*welt >= 10", true);
	internal_equation_checker("100*x + 50*y + 20*hallo - 30*welt <= 5000", true);
	internal_equation_checker("x - 2*y + 3*hallo - 4*welt >= -25", true);
	internal_equation_checker("4*x + 5*y + 6*hallo + 7*welt <= 1000", true);
	internal_equation_checker("2*hallo + 3*welt + 4*x + 5*y >= -500", true);
	internal_equation_checker("1*x + 2*y + 3*hallo + 4*welt <= 99999", true);
	internal_equation_checker("50*x - 25*y + 75*hallo - 125*welt >= 250", true);
	internal_equation_checker("3 * x + y >= 10", true);
	internal_equation_checker("999999*x + 888888*y - 777777*hallo + 666666*welt <= 555555", true);
	internal_equation_checker("0*x + 0*y + 0*hallo + 0*welt >= 0", true);
	internal_equation_checker("-3*x - 4*y + 5*hallo - 6*welt <= -100", true);
	internal_equation_checker("0002*x + 0003*y + 0004*hallo - 0005*welt >= 0006", true);
	internal_equation_checker("1*x + 2*y + 3*hallo + 4*welt <= 0", true);
	internal_equation_checker("1*x - 1*y + 1*hallo - 1*welt <= -1", true);
	internal_equation_checker("x + 2*y - 3*hallo + 4*welt >= -999999", true);
	internal_equation_checker("500000*x - 250000*y + 125000*hallo - 62500*welt <= 10", true);
	internal_equation_checker("123456789*x + 987654321*y >= 111111111", true);
	internal_equation_checker("x * y >= 10", true);
	internal_equation_checker("1000000*x + 1000000*y + 1000000*hallo + 1000000*welt >= 1000000", true);
	internal_equation_checker("-1*x + 2*y - 3*hallo + 4*welt >= -5", true);
	internal_equation_checker("x + y - 2*hallo + 3*welt <= 50", true);
	internal_equation_checker("1000*x + 999*y - 1234*hallo + 5555*welt >= 99999", true);
	internal_equation_checker("x + 1*y + hallo - 5*welt <= 20", true);
	internal_equation_checker("999999*x + 123456*y + 789101*hallo - 654321*welt <= 1000000000", true);
	internal_equation_checker("x*y + hallo - welt >= 100", true);
	internal_equation_checker("x + 2*welt - 3*hallo + 4*y >= -50", true);
	internal_equation_checker("0.0001*x + 0.0002*y >= 0.0003", true);
	internal_equation_checker("x + y + 5*hallo - 6*welt >= 100", true);
	internal_equation_checker("5*x + 3*y >= 15", true);
	internal_equation_checker("x + 2*y - 3*hallo + 4*welt <= 500", true);
	internal_equation_checker("100*x - 50*y + 75*hallo - 25*welt >= 1000", true);
	internal_equation_checker("1*x + 1*y + 1*hallo + 1*welt <= 10", true);
	internal_equation_checker("10*x + 20*y + 30*hallo - 40*welt >= -100", true);
	internal_equation_checker("123*x + 456*y - 789*hallo + 987*welt <= 654", true);
	internal_equation_checker("2*x - 3*y + 4*hallo + 5*welt >= -1000", true);
	internal_equation_checker("9999*x + 8888*y + 7777*hallo - 6666*welt <= 5555", true);
	internal_equation_checker("3*x - 5*y + 7*hallo - 9*welt >= -500", true);
	internal_equation_checker("0.5*x + 0.25*y - 0.75*hallo + 1.5*welt <= 2.5", true);
	internal_equation_checker("2*x+y>=5*4", false);
	internal_equation_checker("2*x+y", false);
	internal_equation_checker("2*x+y > 10", false);
	internal_equation_checker("2*x+y >= abc", false);
	internal_equation_checker("2*x+y >= welt", false);
	internal_equation_checker("2/x+y >= 10", false);
	internal_equation_checker("2+ASD >= 10", false);
	internal_equation_checker("x + y + 5*hallo - 2* >= 10", false);
	internal_equation_checker("3*x + y <= 10 + 5", false);
	internal_equation_checker("2*x ++ y >= 10", false);
	internal_equation_checker("x + y >= ", false);
	internal_equation_checker("10*x + y >= abc", false);
	internal_equation_checker("x + y ==> 10", false);
	internal_equation_checker("x * 2 >= 10", true);
	internal_equation_checker("3*x + y => 10", false);
	internal_equation_checker("2*x + y >= ", false);
	internal_equation_checker("x+y > 10", false);
	internal_equation_checker("x+abc >= 10", false);
	internal_equation_checker("x / 2 >= 5", false);
	internal_equation_checker("2*x+y>=5*4", false);
	internal_equation_checker("2**x + y >= 10", false);
	internal_equation_checker(">= 10", false);
	internal_equation_checker("x + y = 10", false);
	internal_equation_checker("x + y >== 10", false);
	internal_equation_checker("3x + y >= 10", false);
	internal_equation_checker("2* + y >= 10", false);
	internal_equation_checker("*x + y >= 10", false);
	internal_equation_checker("2**x + y >= 10", false);
	internal_equation_checker("2*x+y=10", false);
	internal_equation_checker("2*x+y >= ", false);
	internal_equation_checker(">= 10", false);
	internal_equation_checker("x + y >== 10", false);
	internal_equation_checker("3*x + + 5*y >= 10", false);
	internal_equation_checker("2*x / 3*y >= 10", false);
	internal_equation_checker("2*x + 5..y >= 10", false);
	internal_equation_checker("x + y + hallo*4 >= 20", true);
	internal_equation_checker("x + = y >= 5", false);
	internal_equation_checker("x + 2* + y >= 10", false);
	internal_equation_checker("x + 2*5*y >= ", false);
	internal_equation_checker("x + y +- 10 >= 10", false);
	internal_equation_checker("2*x / y + 5 >= 10", false);
	internal_equation_checker("2*x + y ** 3 >= 10", false);
	internal_equation_checker("x + + 2*y >= 5", false);
	internal_equation_checker("x + + 2 >= 5", false);
	internal_equation_checker("2*x+y >= 5 4", false);
	internal_equation_checker("x + 2*world - 3*hallo + 4*y >= -50", false);
	internal_equation_checker("x + y + z + 5*hallo - 6*welt >= 100", false);
	internal_equation_checker("5x + 3y >= 15", false);
	internal_equation_checker("2*x + y =>= 10", false);
	internal_equation_checker("x + y >== 5", false);
	internal_equation_checker("3x * y >= 10", false);
	internal_equation_checker("2**x + y <= 5", false);
	internal_equation_checker(">= 10", false);
	internal_equation_checker("5*x + 3*y =", false);
	internal_equation_checker("10*x + abc >= 50", false);
	internal_equation_checker("3*x + y 10", false);
	internal_equation_checker("x + y >> 10", false);
	internal_equation_checker("2*x + 3*y - 4*hallo + 5*welt <= 100", true);
	internal_equation_checker("1.5*hallo + 2.5*welt >= -3.14", true);
	internal_equation_checker("hallo + welt <= 10", true);
	internal_equation_checker("hallo - welt >= -5", true);
	internal_equation_checker("hallo + welt <= 𝟜𝟚", false);
	internal_equation_checker("𝟏𝟎*hallo + 𝟐𝟎*welt - 𝟑𝟎*x + 𝟒𝟎*y <= 𝟓𝟎", false);
	internal_equation_checker("hallo\t+\twelt \t<= 42", true);
	internal_equation_checker("1.2*x + 2.3*y - 3.4*hallo + 4.5*welt >= 6.7", true);
	internal_equation_checker("0*x + 0*y + 0*hallo + 0*welt <= 0", true);
	internal_equation_checker("1000000*x - 999999*y + 888888*hallo - 777777*welt >= 666666", true);
	internal_equation_checker("hallo + 0.0*welt - 0.0*x + 0.0*y <= 7", true);
	internal_equation_checker("hallo - welt - x - y >= -hallo", false);
	internal_equation_checker("𝒙 + 𝒚 - 𝒉𝒂𝒍𝒍𝒐 + 𝒘𝒆𝒍𝒕 <= 𝟏𝟎", false);
	internal_equation_checker("10**hallo + 20**welt - 30**x + 40**y <= 50", false);
	internal_equation_checker("hallo+welt <=+10", true);
	internal_equation_checker("  x  +   y  <=  15  ", true);
	internal_equation_checker("5 * hallo + 6 * welt - 7 * x + 8 * y >= 9", true);
	internal_equation_checker("hallo + 0.0000001*welt <= 42", true);
	internal_equation_checker("0.00000000000000001*x + 0.00000000000000002*y - 0.00000000000000003*hallo <= 0", true);
	internal_equation_checker("1000000000000000000000*hallo + 2000000000000000000000*welt >= 3000000000000000000000", true);
	internal_equation_checker("hallo - (welt) + x - (y) <= 10", false);
	internal_equation_checker("x/2 + y/3 - hallo/4 + welt/5 <= 1", false);
	internal_equation_checker("𝕙𝕒𝕝𝕝𝕠 + 𝕨𝕖𝕝𝕥 - 𝕩 + 𝕪 <= 𝟙𝟘", false);
	internal_equation_checker("hallo + welt + x + y <= 1_000_000", false);


	internal_equation_checker("hällo + welt <= 10", false);
	internal_equation_checker("hallo + welt <= ", false);
	internal_equation_checker("x + y = 10", false);
	internal_equation_checker("x + y << 10", false);
	internal_equation_checker("hallo + welt => 20", false);
	internal_equation_checker("𝑥 + 𝑦 ≤ 10", false);
	internal_equation_checker("x++y <= 10", false);
	internal_equation_checker("hallo / 0 <= 10", false);
	internal_equation_checker("x + y <= '10'", false);
	internal_equation_checker("hallo + welt + <= 10", false);
	internal_equation_checker("2hallo + 3welt <= 10", false);
	internal_equation_checker("hallo + welt <== 10", false);
	internal_equation_checker("x + y <= (10", false);
	internal_equation_checker("hallo ** welt <= 10", false);
	internal_equation_checker("hallo + €welt <= 10", false);
	internal_equation_checker("hallo, welt <= 10", false);
	internal_equation_checker("hallo + 1.2.3 <= 10", false);
	internal_equation_checker("hallo + 'welt' <= 10", false);
	internal_equation_checker("hallo + !welt <= 10", false);
	internal_equation_checker("hallo + x +- y <= 10", false);
	internal_equation_checker("hallo + --welt <= 10", false);
	internal_equation_checker("hallo x + welt y <= 10", false);
	internal_equation_checker("hallo+welt=<10", false);
	internal_equation_checker("hallo + {welt} <= 10", false);
	internal_equation_checker("hallo 𝙥𝙡𝙪𝙨 welt <= 10", false);

	console.log(`Ran ${test_counter} tests (${failed} failed)`);
}

function toggle_disabled_status_of_remove_parameters_depending_on_if_there_are_more_than_one() {
	var nr_params = $(".parameterName").length;

	if (nr_params <= 1) {
		$(".remove_parameter").prop("disabled", true);
	} else {
		$(".remove_parameter").prop("disabled", false);
	}
}

function toggleHiddenConfigTableIfError() {
	const table = document.getElementById("hidden_config_table");
	if (table.style.display !== "none") return;

	const errors = table.querySelectorAll(".error_element");
	for (let el of errors) {
		if (el.offsetParent !== null && el.textContent.trim() !== "") {
			table.style.display = "";
			return;
		}
	}
}

function show_warning_for_model_when_custom_generation_strategy_is_set() {
	$("#model_error").html("Custom generation strategy is set, so --model is ignored.").show();
}

function hide_warning_when_custom_custom_generation_strategy_isnt_set() {
	$("#model_error").html("").hide()
}

function toggle_model_warning_for_custom_generation_strategy() {
	if($("#generation_strategy").val() == "") {
		hide_warning_when_custom_custom_generation_strategy_isnt_set();
	} else {
		show_warning_for_model_when_custom_generation_strategy_is_set();
	}
}

function add_equation_spaces(expression) {
	const operators = {
		'>=': '__GE__',
		'<=': '__LE__',
		'==': '__EQ__',
		'!=': '__NE__',
		'=>': '__AR__',
	};

	for (const [op, placeholder] of Object.entries(operators)) {
		expression = expression.replaceAll(op, placeholder);
	}

	expression = expression.replace(/([+\-*/()=<>])/g, ' $1 ');

	for (const [op, placeholder] of Object.entries(operators)) {
		expression = expression.replaceAll(placeholder, ` ${op} `);
	}

	return expression.replace(/\s+/g, ' ').trim();
}
